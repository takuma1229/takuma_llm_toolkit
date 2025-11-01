"""設定読み込みとモデル種別判定のヘルパー群。"""

from __future__ import annotations

import os
from typing import Any, Dict


_GENERATION_DEFAULTS: Dict[str, Any] = {
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.95,
    "top_k": 40,
}

_GENERATION_ENV_KEYS = {
    "max_new_tokens": "LLM_MAX_NEW_TOKENS",
    "repetition_penalty": "LLM_REPETITION_PENALTY",
    "temperature": "LLM_TEMPERATURE",
    "do_sample": "LLM_DO_SAMPLE",
    "top_p": "LLM_TOP_P",
    "top_k": "LLM_TOP_K",
}


class Config(dict):
    """LLM推論に用いる設定を保持する辞書風のクラス。

    Parameters
    ----------
    overrides : dict, optional
        既定値を上書きするための辞書。

    Raises
    ------
    ValueError
        環境変数の値を適切な型に変換できなかった場合。
    """

    def __init__(self, overrides: Dict[str, Any] | None = None):
        data = self._build_defaults()
        if overrides:
            data = self._merge_dicts(data, overrides)
        super().__init__(data)

    def _build_defaults(self) -> Dict[str, Dict[str, Any]]:
        generation = dict(_GENERATION_DEFAULTS)
        for key, env_name in _GENERATION_ENV_KEYS.items():
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            generation[key] = self._coerce_generation_value(key, raw_value)
        return {"generation": generation}

    def _coerce_generation_value(self, key: str, raw_value: str) -> Any:
        if key in {"max_new_tokens", "top_k"}:
            return int(raw_value)
        if key in {"repetition_penalty", "temperature", "top_p"}:
            return float(raw_value)
        if key == "do_sample":
            lowered = raw_value.strip().lower()
            if lowered in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "f", "no", "n", "off"}:
                return False
            raise ValueError(f"do_sampleに変換できない値です: {raw_value}")
        return raw_value

    def _merge_dicts(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in overrides.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged


def is_base_model(model_name: str) -> bool:
    """モデル名からベースモデルかどうかを判定する。

    Parameters
    ----------
    model_name : str
        判定対象のモデル名。

    Returns
    -------
    bool
        ベースモデルであればTrue、それ以外はFalse。
    """

    if not model_name:
        return False
    lowered = model_name.lower()
    if "instruct" in lowered or "chat" in lowered or "aligned" in lowered:
        return False
    if "base" in lowered:
        return True
    return any(token in lowered for token in ("-7b", "-8b", "-13b", "llama"))


__all__ = ["Config", "is_base_model"]
