from __future__ import annotations

import argparse
from dotenv import load_dotenv, find_dotenv

from . import TextGenerator


def main(argv: list[str] | None = None) -> int:
    """
    コマンドラインから推論を実行するエントリーポイント。

    Parameters
    ----------
    argv : list of str, optional
        コマンドライン引数。既定では `sys.argv[1:]` が使用されます。

    Returns
    -------
    int
        正常終了時は 0、エラー時は 1。
    """

    load_dotenv(find_dotenv(), override=True)

    parser = argparse.ArgumentParser(
        prog="takuma-llm",
        description="takuma-llm-toolkit のCLIから手早く推論します。",
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="モデル名（例: gpt-4o-mini, Qwen/Qwen2.5-7B-Instruct など）",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default="日本語で3行自己紹介して",
        help="入力プロンプト",
    )

    args = parser.parse_args(argv)

    # 既定は従来挙動を維持するため "normal" を指定
    gen = TextGenerator(inference_engine="normal")
    text = gen.run(args.model_name, args.prompt)
    if text is None:
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
