from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv, find_dotenv
from takuma_llm_toolkit import TextGenerator


def main(argv: list[str] | None = None) -> int:
    """
    コマンドラインから手早く推論を実行するエントリポイント。

    Parameters
    ----------
    argv : list of str, optional
        コマンドライン引数（テストや埋め込み用途向け）。既定は `sys.argv[1:]`。

    Returns
    -------
    int
        プロセスの終了コード。正常終了時は0。
    """

    # .env を最優先で読み込む（OS環境変数より優先）
    load_dotenv(find_dotenv(), override=True)

    parser = argparse.ArgumentParser(
        prog="run_llm.py",
        description="LLM推論をワンコマンドで実行します。"
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help="使用するモデル名（例: gpt-4o-mini, Qwen/Qwen2.5-7B-Instruct など）",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="入力プロンプト（未指定時は既定のサンプルを使用）",
    )

    # `argv` が None の場合は `sys.argv[1:]` が用いられる
    args = parser.parse_args(argv)

    # 既定値
    model_name = args.model_name or "meta-llama/Meta-Llama-3-8B-Instruct"
    prompt = args.prompt or "Backpropagation（誤差逆伝播法）の概要を日本語で説明してください。"

    # 互換: 旧インタラクティブ挙動も維持
    if args.model_name is None and sys.stdin.isatty():
        entered = input("Enter the model name (empty = default): ").strip()
        if entered:
            model_name = entered

    generator = TextGenerator()
    text = generator.run(model_name, prompt)
    if text is None:
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
