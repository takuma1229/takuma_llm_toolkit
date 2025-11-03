"""設定読み込みとモデル種別判定のヘルパー群。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover - 3.11未満では依存を追加する想定はしない
    _toml = None  # type: ignore


_GENERATION_DEFAULTS: Dict[str, Any] = {
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    "temperature": 1.0,
    "do_sample": True,
    "top_p": 0.95,
    "top_k": 40,
}


class Config(dict):
    """LLM推論に用いる設定を保持する辞書風のクラス。

    本クラスは次の優先度で設定をマージします。

    1. インスタンス化時に与えた `overrides`
    2. `Config.toml` の内容（見つかった場合）
    3. コード内のデフォルト値

    Parameters
    ----------
    overrides : dict, optional
        既定値やファイル設定を上書きする辞書。
    config_path : str or Path, optional
        読み込む設定ファイルのパス。省略時は ``find_config_path()`` で探索します。

    Notes
    -----
    旧仕様のような環境変数による上書きは行いません。
    必要な場合は `overrides` または `Config.toml` を用いてください。
    """

    def __init__(self, overrides: Dict[str, Any] | None = None, config_path: str | Path | None = None):
        data = self._build_defaults()
        # 2) Config.toml の読み込み（存在すれば）
        cfg_path = self._resolve_config_path(config_path)
        if cfg_path is not None:
            file_data = self._load_toml(cfg_path)
            if isinstance(file_data, dict):
                data = self._merge_dicts(data, file_data)
        # 1) 明示的上書き
        if overrides:
            data = self._merge_dicts(data, overrides)
        super().__init__(data)

    def _build_defaults(self) -> Dict[str, Dict[str, Any]]:
        generation = dict(_GENERATION_DEFAULTS)
        return {"generation": generation}

    def _resolve_config_path(self, config_path: str | Path | None) -> Optional[Path]:
        if config_path is not None:
            p = Path(config_path)
            return p if p.is_file() else None
        return find_config_path()

    def _load_toml(self, path: Path) -> Dict[str, Any]:
        """TOML ファイルを読み込み辞書で返す。

        Parameters
        ----------
        path : Path
            読み込む ``Config.toml`` のパス。

        Returns
        -------
        dict
            TOML の内容を辞書として返します。読み込みや解析に失敗した場合は空辞書。
        """
        if _toml is None:
            return {}
        try:
            with path.open('rb') as f:
                data = _toml.load(f)
            return data or {}
        except Exception:
            return {}

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


def find_config_path(filename: str = "Config.toml") -> Optional[Path]:
    """`Config.toml` を探索して最初に見つかったパスを返す。

    Parameters
    ----------
    filename : str, optional
        探索する設定ファイル名。既定は ``"Config.toml"``。

    Returns
    -------
    Path or None
        見つかった場合はファイルパス、見つからなければ ``None``。

    Notes
    -----
    カレントディレクトリから親ディレクトリへ向けて順次探索します。
    仮想環境やインストール先から呼び出されるケースでも、
    実行時の作業ディレクトリ直下の構成を優先的に取り込みます。
    """
    start = Path.cwd()
    for p in [start, *start.parents]:
        candidate = p / filename
        if candidate.is_file():
            return candidate
    return None


__all__ = ["Config", "is_base_model", "find_config_path"]
