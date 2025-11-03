from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv, find_dotenv
from takuma_llm_toolkit import TextGenerator
import inspect


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
    parser.add_argument(
        "-p", "--prompt", default=None,
        help="入力プロンプト（未指定時は既定のサンプルを使用）",
    )
    parser.add_argument(
        "-e", "--inference-engine",
        choices=["normal", "vllm"], default="normal",
        help="推論エンジン（normal|vllm）。既定: normal",
    )

    # 生成系
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大生成トークン数")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="反復抑制係数")
    parser.add_argument("--temperature", type=float, default=None, help="サンプリング温度")
    parser.add_argument("--top-p", type=float, default=None, help="nucleusサンプリング確率質量")
    parser.add_argument("--top-k", type=int, default=None, help="上位kサンプリング")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", help="サンプリングを有効化")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false", help="サンプリングを無効化")
    parser.set_defaults(do_sample=None)

    # vLLM系
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="テンソル並列数")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPUメモリ利用率(0-1)")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLMの最大シーケンス長")

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

    # デバッグ: どの TextGenerator が使われているか明示
    try:
        path = inspect.getsourcefile(TextGenerator) or "<unknown>"
        print(f"[DEBUG] TextGenerator loaded from: {path}")
    except Exception:
        pass

    generator = TextGenerator(
        inference_engine=args.inference_engine,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    text = generator.run(model_name, prompt)
    if text is None:
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
