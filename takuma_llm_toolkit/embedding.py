"""テキスト埋め込みを生成するクラスを提供するモジュール。"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Sequence

import torch
from dotenv import find_dotenv, load_dotenv
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)


class EmbeddingGenerator:
    """指定したモデルで埋め込みベクトルを生成するクラス。

    Parameters
    ----------
    model_name : str, optional
        利用する埋め込みモデル名。既定は ``"Qwen/Qwen3-Embedding-8B"``。
    max_length : int, optional
        トークナイズ時の最大長。既定は ``512`` トークン。
    normalize : bool, optional
        出力ベクトルを L2 正規化するかどうか。既定は ``True``。
    device : str or torch.device, optional
        利用するデバイス。省略時は CUDA が利用可能なら ``"cuda"``、
        それ以外は ``"cpu"``。
    trust_remote_code : bool, optional
        モデルのカスタムコードを信頼するか。Qwen 系では ``True`` 推奨。

    Notes
    -----
    - HF の gated モデルを扱う場合は環境変数 ``HF_TOKEN`` を設定してください。
    - モデルのロードはコンストラクタ実行時に1度だけ行われます。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        *,
        max_length: int = 512,
        normalize: bool = True,
        device: str | torch.device | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hf_token = os.getenv("HF_TOKEN")
        self.trust_remote_code = trust_remote_code

        logger.info("Initializing EmbeddingGenerator with model=%s, device=%s", model_name, self.device)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()

    def embed(self, texts: str | Sequence[str]) -> List[float] | List[List[float]]:
        """テキストから埋め込みベクトルを生成する。

        Parameters
        ----------
        texts : str or Sequence[str]
            埋め込み対象の文字列、または文字列のシーケンス。

        Returns
        -------
        list of float or list of list of float
            単一文字列の場合は 1 ベクトル、複数文字列の場合はベクトルのリスト。
        """

        if isinstance(texts, str):
            is_single = True
            batch = [texts]
        else:
            if not isinstance(texts, Iterable):
                raise TypeError("texts は文字列または文字列のシーケンスを指定してください。")
            batch = list(texts)
            is_single = len(batch) == 1
            if not batch:
                raise ValueError("空の入力はサポートしていません。")

        logger.debug("Encoding %d texts", len(batch))
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        target_device = getattr(self.model, "device", self.device)
        encoded = {k: v.to(target_device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = outputs[0]
            pooled = self._mean_pooling(hidden_states, encoded["attention_mask"])
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        result = pooled.cpu().tolist()
        if is_single:
            return result[0]
        return result

    def _load_tokenizer(self):
        tok_kwargs = {"trust_remote_code": self.trust_remote_code}
        if self.hf_token:
            tok_kwargs["token"] = self.hf_token
        return AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)

    def _load_model(self):
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
        }
        if self.hf_token:
            model_kwargs["token"] = self.hf_token
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        if "device_map" not in model_kwargs:
            model.to(self.device)
        return model

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """attention_mask を考慮した mean pooling を行う。"""

        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


__all__ = ["EmbeddingGenerator"]
