"""共通定数を定義するモジュール。

Notes
-----
本モジュールは CLI 関連から参照され、`TextGenerator` 本体には依存しません。
"""

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_PROMPT = "日本語で3行自己紹介して"

__all__ = ["DEFAULT_MODEL", "DEFAULT_PROMPT"]

