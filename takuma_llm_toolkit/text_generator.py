"""LLM推論のためのテキスト生成クラスを定義するモジュール。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import torch
import transformers
from dotenv import load_dotenv
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

load_dotenv(override=True)


class TextGenerator:
    """テキスト生成を担うクラス。

    Parameters
    ----------
    config : Config or dict, optional
        生成設定を上書きするための辞書。

    Attributes
    ----------
    openai_api_key : str or None
        OpenAI APIのキー。
    hf_token : str
        Hugging Faceのトークン。
    deepseek_api_key : str or None
        DeepSeekのAPIキー。
    device : torch.device
        推論に使用するデバイス。
    """

    def __init__(self, config: Config | Dict[str, Any] | None = None):
        if isinstance(config, Config):
            self.config = config
        elif isinstance(config, dict):
            self.config = Config(config)
        else:
            self.config = Config()

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.hf_token:
            raise ValueError("Hugging Face token is not specified.")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm: LLM | None = None
        self.quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        generation = self.config["generation"]
        self.max_new_tokens = int(generation["max_new_tokens"])
        self.repetition_penalty = float(generation["repetition_penalty"])
        self.temprature = float(generation["temperature"])
        self.do_sample = bool(generation["do_sample"])
        self.top_p = float(generation["top_p"])
        self.top_k = int(generation["top_k"])
        self.execution_time_history: list[float] = []

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
        """

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
        logger.info(", ".join(f"{key}: {value}" for key, value in self.__dict__.items()))
        logger.info("locals(): %s", locals())

        if "llama" in model_name or "Llama" in model_name:
            if is_base_model_flag:
                response = self.llama_base_vllm(model_name, prompt)
            else:
                response = self.llama_official(model_name, prompt)
            response = self.qwen_official(model_name, prompt)
        elif "Qwen" in model_name:
            response = self.qwen_official(model_name, prompt)
        elif "phi" in model_name or "Phi" in model_name:
            response = self.phi_official(model_name, prompt)
        else:
            logger.warning("Unexpected model name: %s", model_name)
            response = self.run_openai_gpt(prompt, model_name)

        print(f"Generated Text: \n\n{response}")
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

        if self.pipeline is None:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": self.quantization_config},
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

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sampling_params = SamplingParams(
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if self.llm is None:
            tps = max(1, torch.cuda.device_count())
            try:
                gmu = min(0.95, float(estimate_gpu_utilization(model_name)))
            except Exception:
                gmu = 0.9
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tps,
                gpu_memory_utilization=gmu,
                max_model_len=15000,
            )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.llm.generate([text], sampling_params)

        for output in outputs:
            prompt_text = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt_text!r}, Generated text: {generated_text!r}")

        return generated_text

    def llama_base_vllm(self, model_name: str, prompt: str) -> None:
        """vLLMを用いてLlamaベースモデルで推論を行う。

        Parameters
        ----------
        model_name : str
            使用するモデルの名前。
        prompt : str
            テキスト生成用のプロンプト。

        Returns
        -------
        None
            生成結果は標準出力に表示される。
        """

        sampling_params = SamplingParams(
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if self.llm is None:
            tps = max(1, torch.cuda.device_count())
            try:
                gmu = min(0.95, float(estimate_gpu_utilization(model_name)))
            except Exception:
                gmu = 0.9
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tps,
                gpu_memory_utilization=gmu,
            )

        outputs = self.llm.generate(prompt, sampling_params)
        print(outputs)
        return None

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

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                quantization_config=self.quantization_config,
            )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sampling_params = SamplingParams(
            temperature=self.temprature,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if self.llm is None:
            tps = max(1, torch.cuda.device_count())
            try:
                gmu = min(0.95, float(estimate_gpu_utilization(model_name)))
            except Exception:
                gmu = 0.9
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tps,
                gpu_memory_utilization=gmu,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
            )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.llm.generate([text], sampling_params)

        for output in outputs:
            prompt_text = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt_text!r}, Generated text: {generated_text!r}")

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

        if self.pipeline is None:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model="microsoft/phi-4",
                model_kwargs={"torch_dtype": "auto", "quantization_config": self.quantization_config},
                device_map="auto",
            )

        messages = [
            {"role": "system", "content": "You are a medieval knight and must provide explanations to modern people."},
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

    def run_openai_gpt(self, prompt: str, model: str = "gpt-4o-mini", temperature: float | None = None) -> str:
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
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=sampling_temperature,
            seed=9,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            presence_penalty=self.repetition_penalty,
        )
        return response.choices[0].message.content


__all__ = ["TextGenerator"]
