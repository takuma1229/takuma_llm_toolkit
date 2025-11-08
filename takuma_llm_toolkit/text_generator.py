"""LLM推論のためのテキスト生成クラスを定義するモジュール。"""

from __future__ import annotations

import logging
import time
import os
from typing import Any, Dict, Callable
import warnings
import re

import torch
import transformers
from dotenv import load_dotenv, find_dotenv
import openai
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from vllm import LLM, SamplingParams

from .utils.load import Config, is_base_model
from .utils.llm_config import estimate_gpu_utilization


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)


class TextGenerator:
    """テキスト生成を担うクラス。

    Parameters
    ----------
    inference_engine : {"normal", "vllm"}
        推論エンジンの種別。``"normal"`` は従来挙動（公式実装優先、
        ただし Llama のベースモデルは従来どおり vLLM を使用）。
        ``"vllm"`` は vLLM による推論を優先（未対応モデルは公式へフォールバック）。
    config : Config or dict, optional
        互換性維持のための引数（非推奨）。
        個別パラメータでの明示指定を推奨します。
    gpu_memory_utilization : float, optional
        vLLMエンジン使用時の GPU メモリ使用率（0.0〜1.0）。
        ``None`` の場合、モデル名から概算した値を初回利用時に自動設定します。
    tensor_parallel_size : int, optional
        vLLMエンジン使用時のテンソル並列数。``None`` の場合、
        利用可能な GPU 台数（最低1）を初回利用時に自動設定します。
    max_model_len : int, optional
        vLLM 初期化時に指定する最大シーケンス長。未指定時の既定は ``2048``。
    max_new_tokens : int, optional
        生成する最大トークン数。
    repetition_penalty : float, optional
        反復抑制の係数。
    temperature : float, optional
        サンプリング温度。
    do_sample : bool, optional
        サンプリングを行うかどうか。
    top_p : float, optional
        nucleus サンプリングの確率質量。
    top_k : int, optional
        サンプリング対象の上位トークン数。

    Attributes
    ----------
    inference_engine : str
        選択された推論エンジン。
    openai_api_key : str or None
        OpenAI APIのキー。
    hf_token : str
        Hugging Faceのトークン。
    deepseek_api_key : str or None
        DeepSeekのAPIキー。
    device : torch.device
        推論に使用するデバイス。
    gpu_memory_utilization : float | None
        vLLM の ``gpu_memory_utilization``。明示しない場合は ``None``（自動推定）。
    tensor_parallel_size : int | None
        vLLM の ``tensor_parallel_size``。明示しない場合は ``None``（自動決定）。
    """

    def __init__(
        self,
        inference_engine: str,
        config: Config | Dict[str, Any] | None = None,
        *,
        gpu_memory_utilization: float | None = None,
        tensor_parallel_size: int | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_model_len: int | None = 2048,
    ):
        # 引数バリデーション
        if inference_engine not in {"normal", "vllm"}:
            raise ValueError(
                "inference_engine は 'normal' または 'vllm' を指定してください。"
            )
        self.inference_engine = inference_engine
        if isinstance(config, dict):
            warnings.warn(
                "TextGenerator(config=dict) は非推奨です。個別パラメータで指定してください。",
                DeprecationWarning,
                stacklevel=2,
            )
            self.config = Config(overrides=config)
        elif isinstance(config, Config):
            warnings.warn(
                "TextGenerator(config=Config) は非推奨です。個別パラメータで指定してください。",
                DeprecationWarning,
                stacklevel=2,
            )
            self.config = config
        else:
            self.config = Config()

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        # HFトークンはHFモデル利用時にのみ必要。OpenAI経路等では不要のためここでは強制しない。
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm: LLM | None = None
        # vLLM関連パラメータをクラス属性として保持
        self.gpu_memory_utilization: float | None = gpu_memory_utilization
        self.tensor_parallel_size: int | None = tensor_parallel_size
        # None で明示的に渡された場合も 2048 を既定として採用
        self.max_model_len: int | None = (
            2048 if max_model_len is None else int(max_model_len)
        )
        # bitsandbytes / triton が無い環境では自動で非量子化へフォールバック
        self.enable_bnb = self._should_enable_bnb()
        self.quantization_config = (
            BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            if self.enable_bnb
            else None
        )
        generation = self.config["generation"]

        def _resolve(name: str, explicit, cast):
            return cast(explicit) if explicit is not None else cast(generation[name])

        self.max_new_tokens = _resolve("max_new_tokens", max_new_tokens, int)
        self.repetition_penalty = _resolve(
            "repetition_penalty", repetition_penalty, float
        )
        # 互換のため内部表記は self.temprature を継続
        self.temprature = _resolve("temperature", temperature, float)
        self.temperature = self.temprature
        self.do_sample = _resolve("do_sample", do_sample, bool)
        self.top_p = _resolve("top_p", top_p, float)
        self.top_k = _resolve("top_k", top_k, int)
        self.execution_time_history: list[float] = []
        logger.info(
            "TextGenerator init: engine=%s, tp=%s, gmu=%s, max_model_len=%s",
            self.inference_engine,
            self.tensor_parallel_size,
            self.gpu_memory_utilization,
            self.max_model_len,
        )

    def run(self, model_name: str, prompt: str) -> str:
        """指定したモデルでテキスト生成を実行する。

        Parameters
        ----------
        model_name : str
            使用するモデルの名前。
        prompt : str
            テキスト生成用のプロンプト。

        Returns
        -------
        str
            生成されたテキスト。

        Notes
        -----
        本メソッドの実行時間（秒）を ``logging`` の INFO レベルで出力します。
        また、計測結果は ``execution_time_history`` に追記されます。
        """
        t0 = time.perf_counter()
        is_base_model_flag = is_base_model(model_name)
        logger.info("TextGenerator.run() called")
        logger.info("device: %s", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                logger.info(
                    "Using device[0]: %s | visible_gpus=%d",
                    torch.cuda.get_device_name(0),
                    torch.cuda.device_count(),
                )
            except Exception:
                logger.info("visible_gpus=%d", torch.cuda.device_count())
        logger.info("model_name: %s", model_name)
        logger.info("inference_engine: %s", self.inference_engine)
        safe_keys = {
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "temprature": self.temprature,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        logger.info("runtime_config: %s", safe_keys)

        # エンジン選択・モデル種別に応じたハンドラへディスパッチ
        handler = self._select_generation_handler(model_name, is_base_model_flag)
        response: str | None = None
        try:
            response = handler(model_name, prompt)
            return response
        except Exception as e:
            # GPU互換性やビルド要件に起因する失敗をヒント付きでログ
            self._log_gpu_compatibility_hint(model_name, e)
            logger.error("generation_failed model=%s error=%s", model_name, e)
            raise
        finally:
            elapsed = time.perf_counter() - t0
            self.execution_time_history.append(elapsed)
            logger.info("run_elapsed_sec=%.3f", elapsed)
            if response is not None:
                logger.info(f"Input Prompt: \n\n{prompt}")
                logger.info(f"Generated Text: \n\n{response}")
            else:
                logger.debug("No generated text due to earlier failure.")

    def _classify_model_name(self, model_name: str) -> str:
        """モデル名から大まかなファミリを判定する。

        Parameters
        ----------
        model_name : str
            判定対象のモデル名。

        Returns
        -------
        str
            判定結果（``"llama"``, ``"qwen"``, ``"phi"``, ``"gemma"``, ``"mistral"``, ``"deepseek"``, ``"unknown"``）。
        """
        name = (model_name or "").lower()
        # 優先度: deepseek を先に判定（例: DeepSeek-R1-Distill-Llama-*）
        if "deepseek" in name:
            return "deepseek"
        if "llama" in name:
            return "llama"
        if "qwen" in name:
            return "qwen"
        if "phi" in name:
            return "phi"
        if "gemma" in name:
            return "gemma"
        if "mistral" in name:
            return "mistral"
        return "unknown"

    def _select_generation_handler(
        self, model_name: str, is_base_model: bool
    ) -> Callable[[str, str], str]:
        """推論エンジンとモデル種別に応じて生成ハンドラを返す。

        Parameters
        ----------
        model_name : str
            モデル名。
        is_base_model : bool
            ベースモデルかどうか（Llama系の分岐に利用）。

        Returns
        -------
        Callable[[str, str], str]
            ``(model_name, prompt) -> text`` の関数。OpenAI 経路は
            内部で `run_openai_gpt(prompt, model)` へラップします。

        Notes
        -----
        - vLLM 未対応の組合せは、公式実装や OpenAI 経路へフォールバックします。
        - 挙動は従来の if/elif 分岐と同一になるよう維持しています。
        """
        family = self._classify_model_name(model_name)

        def _openai_wrapper(m: str, p: str) -> str:
            return self.run_openai_gpt(p, m)

        if self.inference_engine == "vllm":
            if family == "llama":
                return self.llama_base_vllm if is_base_model else self.llama_vllm
            if family == "qwen":
                return self.qwen_vllm
            if family == "phi":
                logger.warning(
                    "phi 系で vLLM は未対応のため、公式実装へフォールバックします。"
                )
                return self.phi_official
            if family == "gemma":
                logger.warning(
                    "gemma 系で vLLM はまだ私が実装していないため、公式実装へフォールバックします。"
                )
                return self.gemma_official
            if family == "mistral":
                logger.warning(
                    "mistral 系で vLLM は未対応扱いとし、公式実装へフォールバックします。"
                )
                return self.mistral_official
            if family == "deepseek":
                logger.warning(
                    "DeepSeek はリモートAPIのため vLLM を無視して API 経路を利用します。"
                )
                return self.run_deepseek
            logger.warning(
                "vLLM未対応モデルのためOpenAI経路にフォールバックします: %s",
                model_name,
            )
            return _openai_wrapper
        elif self.inference_engine == "normal":
            # normal: 従来挙動（公式実装優先）
            if family == "llama":
                return self.llama_base_vllm if is_base_model else self.llama_official
            if family == "qwen":
                return self.qwen_official
        if family == "phi":
            return self.phi_official
        if family == "gemma":
            # Gemma 系は公式実装（transformers/Gemma3ForConditionalGeneration）で推論
            return self.gemma_official
        if family == "mistral":
            return self.mistral_official
        if family == "deepseek":
            return self.run_deepseek
        logger.warning("Unexpected model name: %s", model_name)
        return _openai_wrapper

    def llama_official(self, model_name: str, prompt: str) -> str:
        """Llamaの公式実装を用いてテキスト生成を行う。

        Parameters
        ----------
        model_name : str
            使用するモデルの名前。
        prompt : str
            テキスト生成用のプロンプト。

        Returns
        -------
        str
            生成されたテキスト。
        """

        self._require_hf_token()
        if self.pipeline is None:
            try:
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={
                        "dtype": torch.bfloat16,
                        **(
                            {"quantization_config": self.quantization_config}
                            if self.quantization_config
                            else {}
                        ),
                    },
                    device_map="auto",
                )
            except Exception as e:
                logger.warning(
                    "BitsAndBytes量子化でのロードに失敗したため、非量子化で再試行します: %s",
                    e,
                )
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={"dtype": torch.bfloat16},
                    device_map="auto",
                )

        messages = [{"role": "user", "content": prompt}]
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=self.do_sample,
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        return outputs[0]["generated_text"][-1]["content"]

    def llama_vllm(self, model_name: str, prompt: str) -> str:
        """vLLMを用いてLlamaのテキスト生成を行う。

        Parameters
        ----------
        model_name : str
            使用するモデルの名前。
        prompt : str
            テキスト生成用のプロンプト。

        Returns
        -------
        str
            生成されたテキスト。
        """

        self._require_hf_token()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sampling_params = SamplingParams(
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if self.llm is None:
            # クラス属性が未設定なら初回利用時に自動決定して確定させる
            if self.tensor_parallel_size is None:
                self.tensor_parallel_size = max(1, torch.cuda.device_count())
            if self.gpu_memory_utilization is None:
                try:
                    self.gpu_memory_utilization = min(
                        0.95, float(estimate_gpu_utilization(model_name))
                    )
                except Exception:
                    self.gpu_memory_utilization = 0.9

            llm_kwargs: Dict[str, Any] = dict(
                model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=True,
            )
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = int(self.max_model_len)
                llm_kwargs["max_seq_len"] = int(self.max_model_len)
            self.llm = self._safe_create_llm(llm_kwargs)

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.llm.generate([text], sampling_params)

        for output in outputs:
            prompt_text = output.prompt
            generated_text = output.outputs[0].text

        return generated_text

    def _safe_create_llm(self, kwargs: Dict[str, Any]) -> LLM:
        """vLLM の `max_model_len`/`max_seq_len` 差異にフォールバック対応して初期化する。

        Parameters
        ----------
        kwargs : dict
            LLM 初期化時に渡すキーワード引数。

        Returns
        -------
        LLM
            初期化済みの vLLM LLM インスタンス。

        Notes
        -----
        vLLM のバージョンによって受理されるキーが異なるため、
        以下の順序で安全に初期化を試みます。
        1) そのまま渡す → 2) max_seq_len のみ → 3) max_model_len のみ → 4) どちらも除外
        """
        # 近年の vLLM は `max_seq_len` を受け付けないケースが多いため、
        # まずは `max_model_len` のみで試行 → それで失敗したら `max_seq_len` のみ → 両方 → どちらも無し。
        model_only = dict(kwargs)
        model_only.pop("max_seq_len", None)
        try:
            logger.info(
                "LLM init kwargs(model_only)=%s",
                {k: v for k, v in model_only.items() if k != "model"},
            )
            return LLM(**model_only)
        except TypeError as e1:
            seq_only = dict(kwargs)
            seq_only.pop("max_model_len", None)
            try:
                logger.warning("Retry LLM init with max_seq_len only due to: %s", e1)
                return LLM(**seq_only)
            except TypeError as e2:
                both = dict(kwargs)
                try:
                    logger.warning("Retry LLM init with both lens due to: %s", e2)
                    return LLM(**both)
                except TypeError as e3:
                    no_len = dict(kwargs)
                    no_len.pop("max_model_len", None)
                    no_len.pop("max_seq_len", None)
                    logger.warning("Fallback LLM init without max len due to: %s", e3)
                    return LLM(**no_len)

    def llama_base_vllm(self, model_name: str, prompt: str) -> str:
        """vLLMを用いてLlamaベースモデルで推論を行う。

        Parameters
        ----------
        model_name : str
            使用するモデルの名前。
        prompt : str
            テキスト生成用のプロンプト。

        Returns
        -------
        str
            生成されたテキスト。
        """

        self._require_hf_token()
        sampling_params = SamplingParams(
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if self.llm is None:
            if self.tensor_parallel_size is None:
                self.tensor_parallel_size = max(1, torch.cuda.device_count())
            if self.gpu_memory_utilization is None:
                try:
                    self.gpu_memory_utilization = min(
                        0.95, float(estimate_gpu_utilization(model_name))
                    )
                except Exception:
                    self.gpu_memory_utilization = 0.9
            llm_kwargs: Dict[str, Any] = dict(
                model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=True,
            )
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = int(self.max_model_len)
                llm_kwargs["max_seq_len"] = int(self.max_model_len)
            self.llm = self._safe_create_llm(llm_kwargs)

        outputs = self.llm.generate([prompt], sampling_params)
        try:
            return outputs[0].outputs[0].text
        except Exception:
            return ""

    def qwen_official(self, model_name: str, prompt: str) -> str:
        """Qwenの公式実装を用いてテキスト生成を行う。

        Parameters
        ----------
        model_name : str
            使用するモデルの名前。
        prompt : str
            テキスト生成用のプロンプト。

        Returns
        -------
        str
            生成されたテキスト。
        """

        self._require_hf_token()
        if self.model is None:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    **(
                        {"quantization_config": self.quantization_config}
                        if self.quantization_config
                        else {}
                    ),
                )
            except Exception as e:
                logger.warning(
                    "量子化ロードに失敗したため、非量子化で再試行します: %s", e
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temprature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids, strict=False
            )
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response

    def qwen_vllm(self, model_name: str, prompt: str) -> str:
        """vLLMを用いてQwenでのテキスト生成を行う。

        Parameters
        ----------
        model_name : str
            使用するモデルの名前。
        prompt : str
            テキスト生成用のプロンプト。

        Returns
        -------
        str
            生成されたテキスト。
        """

        self._require_hf_token()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sampling_params = SamplingParams(
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if self.llm is None:
            if self.tensor_parallel_size is None:
                self.tensor_parallel_size = max(1, torch.cuda.device_count())
            if self.gpu_memory_utilization is None:
                try:
                    self.gpu_memory_utilization = min(
                        0.95, float(estimate_gpu_utilization(model_name))
                    )
                except Exception:
                    self.gpu_memory_utilization = 0.9
            llm_kwargs: Dict[str, Any] = dict(
                model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
                enforce_eager=True,
            )
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = int(self.max_model_len)
                llm_kwargs["max_seq_len"] = int(self.max_model_len)
            self.llm = self._safe_create_llm(llm_kwargs)

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.llm.generate([text], sampling_params)

        for output in outputs:
            prompt_text = output.prompt
            generated_text = output.outputs[0].text

        return generated_text

    def phi_official(self, model_name: str, prompt: str) -> str:
        """Phiの公式実装を用いてテキスト生成を行う。

        Parameters
        ----------
        model_name : str
            使用するモデルの名前。
        prompt : str
            テキスト生成用のプロンプト。

        Returns
        -------
        str
            生成されたテキスト。
        """

        self._require_hf_token()
        if self.pipeline is None:
            try:
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={
                        "dtype": "auto",
                        **(
                            {"quantization_config": self.quantization_config}
                            if self.quantization_config
                            else {}
                        ),
                    },
                    device_map="auto",
                )
            except Exception as e:
                logger.warning(
                    "量子化ロードに失敗したため、非量子化で再試行します: %s", e
                )
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={"dtype": "auto"},
                    device_map="auto",
                )

        messages = [
            {
                "role": "system",
                "content": "You are a medieval knight and must provide explanations to modern people.",
            },
            {"role": "user", "content": prompt},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        return outputs[0]["generated_text"][-1]["content"]

    def gemma_official(self, model_name: str, prompt: str) -> str:
        """Gemma 3（Instruction/Turbo等）の公式実装でテキストを生成する。

        Parameters
        ----------
        model_name : str
            使用する Gemma 3 系モデル（例: ``"google/gemma-3-4b-it"``）。
        prompt : str
            ユーザーからの入力テキスト。

        Returns
        -------
        str
            生成結果のテキスト。

        Notes
        -----
        - 画像処理用の ``AutoProcessor`` ではなく、テキスト専用の ``AutoTokenizer`` を利用します。
        - チャットテンプレートはトークナイザの ``apply_chat_template`` を用いて文字列化し、
          通常の ``tokenizer(...)`` でテンソル化します。
        - モデル重みは ``bfloat16``、デバイス割当は ``device_map=\"auto\"`` を採用します。
        """

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # チャットテンプレートを文字列として展開
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # テンソル化してモデルデバイスへ
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temprature,
                repetition_penalty=self.repetition_penalty,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            # 追記部分のみ取り出し
            generation = generation[0][input_len:]

        decoded = tokenizer.decode(generation, skip_special_tokens=True)
        return decoded

    def mistral_official(self, model_name: str, prompt: str) -> str:
        """Mistral 公式実装（Tokenizer/Transformer/generate）で推論を行う。

        Parameters
        ----------
        model_name : str
            使用するモデルを指すパス。ローカルの Mistral フォルダパス、
            もしくは環境変数 ``MISTRAL_MODELS_PATH`` で指すフォルダを利用します。
        prompt : str
            ユーザー入力のテキスト。

        Returns
        -------
        str
            生成されたテキスト。

        Notes
        -----
        - 下記の Mistral 公式スタックに準拠します。
          ``MistralTokenizer.from_file`` → ``Transformer.from_folder`` → ``generate``。
        - 必要パッケージ（例: ``mistral-common``, ``mistral-inference``）が未導入の環境では、
          自動的に vLLM 互換経路へフォールバックします。
        - 生成ハイパラは本クラスの設定値（``max_new_tokens``/``temperature`` 等）を反映します。
        """

        # 公式実装（mistral-*）が利用可能なら優先する
        try:
            # 動的 import（パッケージ命名揺れに備え複数候補を試行）
            try:
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # type: ignore
            except Exception:
                # 旧来/別構成の可能性に備えたフォールバック
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # type: ignore

            try:
                from mistral_inference import (
                    Transformer,
                    ChatCompletionRequest,
                    UserMessage,
                    generate,
                )  # type: ignore
            except Exception:
                from mistral_inference.model import Transformer  # type: ignore
                from mistral_inference.chat import ChatCompletionRequest, UserMessage  # type: ignore
                from mistral_inference.generate import generate  # type: ignore

            # モデルフォルダの解決
            def _resolve_models_path(name: str) -> str:
                if name and os.path.isdir(name):
                    return name
                env_path = os.getenv("MISTRAL_MODELS_PATH", "").strip()
                if env_path and os.path.isdir(env_path):
                    return env_path
                raise ValueError(
                    "Mistral のモデルフォルダが見つかりません。model_name にローカルフォルダを指定するか、"
                    "環境変数 MISTRAL_MODELS_PATH を設定してください。"
                )

            mistral_models_path = _resolve_models_path(model_name)

            # トークナイザ/モデルのロード
            tokenizer = MistralTokenizer.from_file(
                f"{mistral_models_path}/tokenizer.model.v3"
            )
            model = Transformer.from_folder(mistral_models_path)

            # プロンプトをユーザ指定で構成
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )

            tokens = tokenizer.encode_chat_completion(completion_request).tokens

            # 生成（ハイパラは本クラスの設定値を反映）
            temperature = float(self.temprature) if self.temprature is not None else 0.0
            max_tokens = (
                int(self.max_new_tokens) if self.max_new_tokens is not None else 256
            )
            eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id

            out_tokens, _ = generate(
                [tokens],
                model,
                max_tokens=max_tokens,
                temperature=temperature,
                eos_id=eos_id,
            )
            result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            return result

        except (ImportError, ModuleNotFoundError) as e:
            # ライブラリ未導入時は transformers 経路を先に試す（vLLM 依存を避ける）
            logger.warning(
                "Mistral公式ライブラリが見つからないため、transformers 経路へフォールバックします: %s",
                e,
            )
            try:
                return self.mistral_hf_official(model_name, prompt)
            except Exception as e2:
                logger.warning(
                    "transformers 経路でも失敗したため、最終手段として vLLM へフォールバックします: %s",
                    e2,
                )
                return self.mistral_vllm_compat(model_name, prompt)
        except ValueError as e:
            # モデルフォルダ未検出など利用者設定が必要なケースはそのまま通知
            logger.error("Mistralモデルパスの解決に失敗しました: %s", e)
            raise
        except Exception as e:
            # それ以外は明示的に失敗を知らせる（不用意に vLLM へ逃げない）
            logger.error("Mistral公式実装での推論に失敗しました: %s", e)
            raise

    def mistral_vllm_compat(self, model_name: str, prompt: str) -> str:
        """vLLM を用いた Mistral 推論（後方互換用）。

        Parameters
        ----------
        model_name : str
            使用する Mistral モデル（例: ``"mistralai/Mistral-Small-3.1-24B-Instruct-2503"``）。
        prompt : str
            ユーザー入力のテキスト。

        Returns
        -------
        str
            生成されたテキスト。
        """

        self._require_hf_token()

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temprature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )

        if self.llm is None:
            if self.tensor_parallel_size is None:
                self.tensor_parallel_size = max(1, torch.cuda.device_count())
            if self.gpu_memory_utilization is None:
                try:
                    self.gpu_memory_utilization = min(
                        0.95, float(estimate_gpu_utilization(model_name))
                    )
                except Exception:
                    self.gpu_memory_utilization = 0.9
            llm_kwargs: Dict[str, Any] = dict(
                model=model_name,
                tokenizer_mode="mistral",
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=True,
            )
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = int(self.max_model_len)
                llm_kwargs["max_seq_len"] = int(self.max_model_len)
            self.llm = self._safe_create_llm(llm_kwargs)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        try:
            return outputs[0].outputs[0].text
        except Exception:
            return ""

    def mistral_hf_official(self, model_name: str, prompt: str) -> str:
        """transformers を用いた Mistral 推論を行う。

        Parameters
        ----------
        model_name : str
            使用する Mistral モデル（例: ``"mistralai/Mistral-7B-Instruct-v0.3"``）。
        prompt : str
            ユーザー入力のテキスト。

        Returns
        -------
        str
            生成されたテキスト。

        Notes
        -----
        - `AutoTokenizer.apply_chat_template` でプロンプトを整形し、
          `AutoModelForCausalLM.generate` で生成します。
        - 量子化が利用可能なら BitsAndBytes を用い、それ以外は非量子化で自動デバイス割当します。
        """

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **(
                {"quantization_config": self.quantization_config}
                if self.quantization_config
                else {}
            ),
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        gen = generated_ids[0, inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(gen, skip_special_tokens=True)

    def run_openai_gpt(
        self, prompt: str, model: str = "gpt-4o-mini", temperature: float | None = None
    ) -> str:
        """OpenAIのGPTモデルを用いてテキスト生成を行う。

        Parameters
        ----------
        prompt : str
            テキスト生成用のプロンプト。
        model : str, optional
            使用するモデルの名前。既定値は"gpt-4o-mini"。
        temperature : float, optional
            テキスト生成時のtemperature。

        Returns
        -------
        str
            生成されたテキスト。

        Notes
        -----
        一部のモデル（例: GPT-5 系）では ``temperature`` の任意値が未対応で、
        既定値のみ受け付ける場合があります。その場合は 400 のBadRequestが返るため、
        本メソッドでは例外内容を検出して ``temperature`` を除去した上で自動再試行します。
        """

        client = OpenAI()
        sampling_temperature = self.temprature if temperature is None else temperature

        # 送信ペイロードの組み立て
        if self._is_openai_gpt5_or_newer(model):
            # GPT-5系はパラメータ制約が変わる可能性があるため極力シンプルに
            payload = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            # 温度指定がある場合のみ付与（未対応時は下の例外分岐で自動除去）
            if sampling_temperature is not None:
                payload["temperature"] = sampling_temperature
        else:
            # GPT-4系
            payload = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            if sampling_temperature is not None:
                payload["temperature"] = sampling_temperature
            # 追加のサンプリング関連（互換性のある範囲で送る）
            payload.update(
                dict(
                    seed=9,
                    top_p=self.top_p,
                    max_tokens=self.max_new_tokens,
                    presence_penalty=self.repetition_penalty,
                )
            )

        # 送信＆フォールバック（temperature未対応時は自動で削除して再送）
        try:
            response = client.chat.completions.create(**payload)
        except Exception as e:
            err_text = str(e)
            # OpenAIのBadRequestで temperature 未対応を検出
            is_bad_request = (
                isinstance(e, openai.BadRequestError) or "BadRequestError" in err_text
            )
            temperature_involved = "temperature" in err_text.lower()
            unsupported_value = (
                "unsupported_value" in err_text or "does not support" in err_text
            )
            if (
                is_bad_request
                and temperature_involved
                and unsupported_value
                and "temperature" in payload
            ):
                # 温度パラメータを除外して再送
                temp_val = payload.pop("temperature")
                logger.warning(
                    "model=%s は temperature=%s を受け付けないため無視して再試行します。",
                    model,
                    temp_val,
                )
                response = client.chat.completions.create(**payload)
            else:
                raise

        return response.choices[0].message.content

    def run_deepseek(self, model_name: str, prompt: str) -> str:
        """DeepSeek を Hugging Face 経由で推論する。

        Parameters
        ----------
        model_name : str
            利用する DeepSeek のモデル名（例: ``"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"`` 等）。
        prompt : str
            生成の入力テキスト。

        Returns
        -------
        str
            生成されたテキスト。

        Notes
        -----
        - `inference_engine` が ``"vllm"`` の場合は vLLM 経路、
          それ以外は transformers 公式経路を用います。
        - トークナイザの ``apply_chat_template`` を優先的に利用し、
          チャットフォーマットはモデル付属のテンプレートに従います。
        """

        if self.inference_engine == "vllm":
            return self.deepseek_vllm(model_name, prompt)
        return self.deepseek_official(model_name, prompt)

    def deepseek_official(self, model_name: str, prompt: str) -> str:
        """transformers を用いて DeepSeek のテキスト生成を行う。

        Parameters
        ----------
        model_name : str
            DeepSeek 系のモデル名（Hugging Face のリポジトリパス）。
        prompt : str
            入力テキスト。

        Returns
        -------
        str
            生成結果のテキスト。
        """

        self._require_hf_token()

        # モデル/トークナイザ読み込み（量子化が可能なら利用）。
        # DeepSeek-V3 は finegrained-FP8 量子化のため、GPU の SM が 8.9 未満だと失敗します。
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **({"quantization_config": self.quantization_config} if self.quantization_config else {}),
            )
        except Exception as e:
            msg = str(e)
            if "FP8 quantized models is only supported" in msg:
                try:
                    sm_major, sm_minor = torch.cuda.get_device_capability(0)
                    sm = f"{sm_major}.{sm_minor}"
                except Exception:
                    sm = "unknown"
                raise RuntimeError(
                    "DeepSeek の FP8 量子化モデルは SM 8.9 以上が必要です。" \
                    f"(検出GPUのSM={sm})\n" \
                    "回避策: 1) Ada/Hopper 世代GPU(例: RTX 4090/H100)で実行, " \
                    "2) DeepSeek-R1 系など bf16/fp16 の代替チェックポイントを使用, " \
                    "3) vLLM + 対応環境に切替。"
                ) from e
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 可能ならチャットテンプレートで整形
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temprature,
                repetition_penalty=self.repetition_penalty,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        gen = generated[0, inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(gen, skip_special_tokens=True)

    def deepseek_vllm(self, model_name: str, prompt: str) -> str:
        """vLLM を用いて DeepSeek のテキスト生成を行う。

        Parameters
        ----------
        model_name : str
            DeepSeek 系のモデル名（Hugging Face のリポジトリパス）。
        prompt : str
            入力テキスト。

        Returns
        -------
        str
            生成結果のテキスト。
        """

        self._require_hf_token()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sampling_params = SamplingParams(
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if self.llm is None:
            if self.tensor_parallel_size is None:
                self.tensor_parallel_size = max(1, torch.cuda.device_count())
            if self.gpu_memory_utilization is None:
                try:
                    self.gpu_memory_utilization = min(
                        0.95, float(estimate_gpu_utilization(model_name))
                    )
                except Exception:
                    self.gpu_memory_utilization = 0.9
            llm_kwargs: Dict[str, Any] = dict(
                model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=True,
            )
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = int(self.max_model_len)
                llm_kwargs["max_seq_len"] = int(self.max_model_len)
            self.llm = self._safe_create_llm(llm_kwargs)

        # チャットテンプレート（可能なら）
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = prompt

        outputs = self.llm.generate([text], sampling_params)
        try:
            return outputs[0].outputs[0].text
        except Exception:
            return ""

    def _require_hf_token(self) -> None:
        """HFモデル利用時にトークンが設定されているか検証する。

        Raises
        ------
        ValueError
            `HF_TOKEN` が未設定の場合に送出。
        """
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN が設定されていません。Hugging Face のモデルを利用する場合は環境変数 HF_TOKEN を設定してください。"
            )

    def _is_openai_gpt5_or_newer(self, model_name: str) -> bool:
        """OpenAIのモデル名からGPT-5以降かどうかを判定する。

        Parameters
        ----------
        model_name : str
            判定対象のモデル名（例: 'gpt-4o-mini', 'gpt-5', 'gpt-5.1' 等）。

        Returns
        -------
        bool
            GPT-5 以上と判定された場合に True。

        Notes
        -----
        モデル名に含まれる数値のメジャーバージョン（`gpt-(\d+)`）を抽出して判定します。
        """
        if not model_name:
            return False
        m = re.search(r"gpt-(\d+)", model_name)
        if not m:
            return False
        try:
            major = int(m.group(1))
        except ValueError:
            return False
        return major >= 5

    def _should_enable_bnb(self) -> bool:
        """bitsandbytes量子化の利用可否を判定する。

        Returns
        -------
        bool
            量子化を有効にするなら True。`LLM_DISABLE_BNB` が真値、
            もしくは `bitsandbytes` / `triton` が無ければ False。
        """
        flag = os.getenv("LLM_DISABLE_BNB", "").strip().lower()
        if flag in {"1", "true", "t", "yes", "y", "on"}:
            logger.info("LLM_DISABLE_BNB が有効のため、量子化を無効化します。")
            return False
        try:
            import bitsandbytes  # noqa: F401
            import triton  # noqa: F401
        except Exception as e:
            logger.warning(
                "bitsandbytes/triton を検出できないため非量子化にします: %s", e
            )
            return False
        return True

    def _log_gpu_compatibility_hint(self, model_name: str, exc: Exception) -> None:
        """GPU互換性やビルド要件に関する代表的な失敗を検出し、ヒントを出力する。

        Parameters
        ----------
        model_name : str
            実行しようとしたモデル名。
        exc : Exception
            捕捉した例外。

        Notes
        -----
        - 代表例:
          - FP8 量子化モデル（例: DeepSeek-V3）における SM 要件未満。
          - Triton/Inductor の `-lcuda` リンク失敗。
          - KV キャッシュ不足（参考）。
        - 本メソッドはログ出力のみを行い、例外は再送出元に委ねます。
        """
        msg = str(exc)
        # GPU情報の収集
        try:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:
            gpu_name = "unknown"
        try:
            cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
            sm = f"{cc[0]}.{cc[1]}"
        except Exception:
            sm = "unknown"

        # FP8 / SM 要件
        if "FP8 quantized models is only supported" in msg or \
           ("compute capability" in msg and (">=" in msg or "以上" in msg)):
            logger.error(
                (
                    "GPU非対応の可能性: model=%s | gpu=%s | sm=%s | details=%s\n"
                    "対処: 1) Ada/Hopper 世代GPUで実行 (SM>=8.9), 2) bf16/fp16 の代替モデルに切替, 3) 量子化無しで利用可能なチェックポイントを選択"
                ),
                model_name,
                gpu_name,
                sm,
                msg,
            )
            return

        # Triton/Inductor の CUDA リンクエラー
        if "cannot find -lcuda" in msg or "libcuda.so" in msg or "triton" in msg:
            logger.error(
                (
                    "Triton/Inductor のビルド要件未充足: model=%s | gpu=%s | sm=%s | details=%s\n"
                    "対処: libcuda を解決可能にする、または vLLM 初期化で enforce_eager=True（既定で有効）/ normal エンジンへ切替"
                ),
                model_name,
                gpu_name,
                sm,
                msg,
            )
            return

        # KV キャッシュ不足のヒント（汎用）
        if "KV cache" in msg or "max seq len" in msg or "max_model_len" in msg:
            logger.error(
                (
                    "メモリ不足の可能性: model=%s | gpu=%s | sm=%s | details=%s\n"
                    "対処: gpu_memory_utilization を上げる、max_model_len を下げる、tensor_parallel_size を増やす"
                ),
                model_name,
                gpu_name,
                sm,
                msg,
            )
            return

        # その他のエラーは情報ログに GPU 情報を添えて出す
        logger.info(
            "error_with_gpu_context model=%s gpu=%s sm=%s details=%s",
            model_name,
            gpu_name,
            sm,
            msg,
        )


__all__ = ["TextGenerator"]
