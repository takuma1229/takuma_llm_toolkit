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
    parser.add_argument("model_name", nargs="?", default="meta-llama/Meta-Llama-3-8B-Instruct", help="モデル名")
    parser.add_argument("-p", "--prompt", default="日本語で3行自己紹介して", help="入力プロンプト")
    parser.add_argument("-e", "--inference-engine", choices=["normal", "vllm"], default="normal", help="推論エンジン")

    # 生成系
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=None)

    # vLLM系
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)

    args = parser.parse_args(argv)

    gen = TextGenerator(
        inference_engine=args.inference_engine,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    text = gen.run(args.model_name, args.prompt)
    if text is None:
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
