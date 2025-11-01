"""LLM推論向けのGPU利用率推定ロジック。"""

from __future__ import annotations

import re
from typing import Optional


_SIZE_PATTERN = re.compile(r"(?:^|[-_/])(\d+(?:\.\d+)?)\s*(?:b|billion)", re.IGNORECASE)


def estimate_gpu_utilization(model_name: str) -> float:
    """モデル名からGPUメモリ利用率を概算する。

    Parameters
    ----------
    model_name : str
        対象となるモデルの名前。

    Returns
    -------
    float
        推定されたGPUメモリ利用率。0から1の範囲で返す。

    Notes
    -----
    推定はパラメータ数に基づく経験則であり、実際の値と一致しない場合がある。
    """

    size = _extract_model_size(model_name)
    if size is None:
        return 0.9
    if size <= 2:
        return 0.45
    if size <= 4:
        return 0.6
    if size <= 8:
        return 0.75
    if size <= 14:
        return 0.85
    return 0.9


def _extract_model_size(model_name: str) -> Optional[float]:
    match = _SIZE_PATTERN.search(model_name or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


__all__ = ["estimate_gpu_utilization"]
