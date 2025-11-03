from __future__ import annotations

import argparse
from dotenv import load_dotenv, find_dotenv
from .cli_args import add_common_args, make_generator_from_args
from .constants import DEFAULT_MODEL, DEFAULT_PROMPT


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
    parser.add_argument("model_name", nargs="?", default=DEFAULT_MODEL, help="モデル名")
    add_common_args(parser)

    args = parser.parse_args(argv)

    gen = make_generator_from_args(args)
    text = gen.run(args.model_name, args.prompt or DEFAULT_PROMPT)
    if text is None:
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
