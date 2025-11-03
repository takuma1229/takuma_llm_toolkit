from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv, find_dotenv
from takuma_llm_toolkit import TextGenerator
import inspect
from takuma_llm_toolkit.cli_args import add_common_args, make_generator_from_args
from takuma_llm_toolkit.constants import DEFAULT_MODEL, DEFAULT_PROMPT


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

    Notes
    -----
    生成系・vLLM系のパラメータはすべてコマンドライン引数から明示指定可能です。
    未指定項目は Config.toml → 既定値 の順で補完されます。
    """

    # .env を最優先で読み込む（APIキー等）。生成パラメータは読みません。
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
    add_common_args(parser)

    # `argv` が None の場合は `sys.argv[1:]` が用いられる
    args = parser.parse_args(argv)

    # 既定値
    model_name = args.model_name or DEFAULT_MODEL
    prompt = args.prompt or DEFAULT_PROMPT

    # 互換: 旧インタラクティブ挙動も維持
    if args.model_name is None and sys.stdin.isatty():
        entered = input("Enter the model name (empty = default): ").strip()
        if entered:
            model_name = entered

    # デバッグ: どの TextGenerator が使われているか明示
    try:
        path = inspect.getsourcefile(TextGenerator) or "<unknown>"
        print(f"[DEBUG] TextGenerator loaded from: {path}")
    except Exception:
        pass

    generator = make_generator_from_args(args)
    text = generator.run(model_name, prompt)
    if text is None:
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
