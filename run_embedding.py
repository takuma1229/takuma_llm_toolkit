from __future__ import annotations

import argparse
import json
import sys
from typing import List

import torch

from dotenv import find_dotenv, load_dotenv

from takuma_llm_toolkit import EmbeddingGenerator


DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"


def _resolve_pair_texts(args: argparse.Namespace) -> List[str]:
    """類似度計算用に 2 本のテキストを取得する。"""

    if args.texts and len(args.texts) >= 2:
        return args.texts[:2]

    if not sys.stdin.isatty():
        lines = [line.rstrip("\n") for line in sys.stdin.readlines() if line.strip()]
        if len(lines) >= 2:
            return lines[:2]

    if sys.stdin.isatty():
        s1 = input("Enter text s1: ").strip()
        s2 = input("Enter text s2: ").strip()
        if s1 and s2:
            return [s1, s2]

    raise SystemExit("類似度計算には s1 と s2 の2つのテキストが必要です。")


def _resolve_texts(args: argparse.Namespace) -> List[str]:
    """CLI引数と標準入力から埋め込み対象テキストのリストを組み立てる。"""

    if args.texts:
        return args.texts

    # パイプ入力があればそれを優先
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return [data.rstrip("\n")]

    if sys.stdin.isatty():
        entered = input("Enter text to embed (empty to exit): ").strip()
        if entered:
            return [entered]

    raise SystemExit("埋め込み対象のテキストが指定されていません。")


def main(argv: list[str] | None = None) -> int:
    """簡易的に埋め込みを生成して標準出力へ JSON で返す。"""

    load_dotenv(find_dotenv(), override=True)

    parser = argparse.ArgumentParser(
        prog="run_embedding.py",
        description="埋め込みモデルでベクトルを生成します。",
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default=DEFAULT_EMBED_MODEL,
        help=f"利用するモデル名（既定: {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--mode",
        choices=["embed", "sim"],
        default="embed",
        help="embed: 埋め込みを出力 (既定), sim: s1 と s2 の類似度を計算",
    )
    parser.add_argument(
        "texts",
        nargs="*",
        help="埋め込みたいテキスト。未指定の場合は標準入力かプロンプト入力を利用。",
    )
    parser.add_argument("--max-length", type=int, default=512, help="トークナイズ上限長")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="L2 正規化を無効化（既定は有効）",
    )
    parser.add_argument("--device", default=None, help="使用デバイス。例: cuda, cuda:1, cpu")
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        default=True,
        help="カスタムコードを信頼しない（既定は信頼する）",
    )

    args = parser.parse_args(argv)

    if args.mode == "sim":
        texts = _resolve_pair_texts(args)
    else:
        texts = _resolve_texts(args)

    embedder = EmbeddingGenerator(
        model_name=args.model_name,
        max_length=args.max_length,
        normalize=not args.no_normalize,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    if args.mode == "sim":
        vecs = embedder.embed(texts)
        if not isinstance(vecs, list) or len(vecs) != 2:
            raise SystemExit("類似度計算に必要なベクトルが取得できませんでした。")
        t = torch.tensor(vecs)
        sim = torch.nn.functional.cosine_similarity(t[0], t[1], dim=0).item()
        output = {
            "model": args.model_name,
            "s1": texts[0],
            "s2": texts[1],
            "similarity": sim,
        }
        print(json.dumps(output, ensure_ascii=False))
    else:
        vectors = embedder.embed(texts)
        print(json.dumps(vectors, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
