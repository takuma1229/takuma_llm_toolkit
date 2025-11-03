"""CLI 引数定義と `TextGenerator` 生成の共通化ユーティリティ。

Notes
-----
`text_generator.py` 本体は変更しないため、本モジュールで引数の集約と
インスタンス生成の橋渡しを行います。
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from . import TextGenerator


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """CLI に共通の引数を追加する。

    Parameters
    ----------
    parser : argparse.ArgumentParser
        引数を追加したいパーサ。

    Returns
    -------
    argparse.ArgumentParser
        引数追加後の同一インスタンス。
    """

    parser.add_argument("-p", "--prompt", default=None, help="入力プロンプト")
    parser.add_argument(
        "-e",
        "--inference-engine",
        choices=["normal", "vllm"],
        default="normal",
        help="推論エンジン（normal|vllm）",
    )

    # 生成系
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=None)

    # vLLM 系
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    return parser


def build_generator_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """`TextGenerator` のコンストラクタ引数を `argparse` 結果から構築する。

    Parameters
    ----------
    args : argparse.Namespace
        `argparse` で解釈された引数オブジェクト。

    Returns
    -------
    dict
        `TextGenerator` のキーワード引数辞書。
    """

    return dict(
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


def make_generator_from_args(args: argparse.Namespace) -> TextGenerator:
    """`argparse` 結果から `TextGenerator` を生成する。

    Parameters
    ----------
    args : argparse.Namespace
        `argparse` で解釈された引数オブジェクト。

    Returns
    -------
    TextGenerator
        構築済みのジェネレータ。
    """

    kwargs = build_generator_kwargs_from_args(args)
    return TextGenerator(**kwargs)


__all__ = [
    "add_common_args",
    "build_generator_kwargs_from_args",
    "make_generator_from_args",
]

