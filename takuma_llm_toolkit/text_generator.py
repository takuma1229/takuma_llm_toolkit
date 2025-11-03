"""LLM推論のためのテキスト生成クラスを定義するモジュール。"""

from __future__ import annotations

import logging
import time
import os
from typing import Any, Dict
import warnings
import re

import torch
import transformers
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        self.max_model_len: int | None = 2048 if max_model_len is None else int(max_model_len)
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

        # エンジン選択に基づく分岐
        if self.inference_engine == "vllm":
            if "llama" in model_name or "Llama" in model_name:
                if is_base_model_flag:
                    response = self.llama_base_vllm(model_name, prompt)
                else:
                    response = self.llama_vllm(model_name, prompt)
            elif "Qwen" in model_name:
                response = self.qwen_vllm(model_name, prompt)
            elif "phi" in model_name or "Phi" in model_name:
                logger.warning(
                    "phi 系で vLLM は未対応のため、公式実装へフォールバックします。"
                )
                response = self.phi_official(model_name, prompt)
            else:
                logger.warning(
                    "vLLM未対応モデルのためOpenAI経路にフォールバックします: %s",
                    model_name,
                )
                response = self.run_openai_gpt(prompt, model_name)
        else:  # normal: 従来挙動を維持（公式実装優先）
            if "llama" in model_name or "Llama" in model_name:
                if is_base_model_flag:
                    # 従来どおり、Llamaベースモデルは vLLM を使用
                    response = self.llama_base_vllm(model_name, prompt)
                else:
                    response = self.llama_official(model_name, prompt)
            elif "Qwen" in model_name:
                response = self.qwen_official(model_name, prompt)
            elif "phi" in model_name or "Phi" in model_name:
                response = self.phi_official(model_name, prompt)
            else:
                logger.warning("Unexpected model name: %s", model_name)
                response = self.run_openai_gpt(prompt, model_name)

        elapsed = time.perf_counter() - t0
        self.execution_time_history.append(elapsed)
        logger.info("run_elapsed_sec=%.3f", elapsed)
        logger.info(f"Input Prompt: \n\n{prompt}")
        logger.info(f"Generated Text: \n\n{response}")
        return response

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
        try:
            logger.info("LLM init kwargs=%s", {k: v for k, v in kwargs.items() if k != "model"})
            return LLM(**kwargs)
        except TypeError as e:
            seq_only = dict(kwargs)
            seq_only.pop("max_model_len", None)
            try:
                logger.warning("Retry LLM init with max_seq_len only due to: %s", e)
                return LLM(**seq_only)
            except TypeError as e2:
                model_only = dict(kwargs)
                model_only.pop("max_seq_len", None)
                try:
                    logger.warning("Retry LLM init with max_model_len only due to: %s", e2)
                    return LLM(**model_only)
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
            do_sample=self.do_sample,
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
        """

        client = OpenAI()
        sampling_temperature = self.temprature if temperature is None else temperature

        if self._is_openai_gpt5_or_newer(model):
            # GPT-5 以降: 未対応/制約のあるパラメータを避け、最小構成で送信
            payload = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=sampling_temperature,
            )
        else:
            # GPT-4 系: 既存パラメータを活用
            payload = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=sampling_temperature,
                seed=9,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                presence_penalty=self.repetition_penalty,
            )

        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content

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


__all__ = ["TextGenerator"]
