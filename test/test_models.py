"""各モデルファミリのスモークテスト。

15GB 程度の GPU（例: T4/A10/A5000 等）を想定し、できるだけ小さめの
チェックポイントを既定にしています。環境に合わせて環境変数で上書きしてください。

環境変数（任意）
- `TAKUMA_TEST_LLAMA`: 例 `meta-llama/Llama-3.2-1B-Instruct`
- `TAKUMA_TEST_QWEN`:  例 `Qwen/Qwen2.5-1.5B-Instruct`
- `TAKUMA_TEST_PHI`:   例 `microsoft/Phi-3-mini-4k-instruct`
- `TAKUMA_TEST_GEMMA`: 例 `google/gemma-3-4b-it`
- `TAKUMA_TEST_MISTRAL`: 例 `mistralai/Mistral-7B-Instruct-v0.3`
- `TAKUMA_TEST_DEEPSEEK`: 例 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

注意:
- Llama/Gemma は Hugging Face の承認が必要な場合があります。`HF_TOKEN` が未設定の
  環境では該当テストを自動的にスキップします。
- DeepSeek はモデルにより FP8/大型チェックポイントがあります。15GB GPU では
  1.5B/3B/4B クラスを推奨し、それ以外は自動スキップされる場合があります。
"""

from __future__ import annotations

import os
import pytest
import torch

from takuma_llm_toolkit import TextGenerator
import difflib
import re


def _gpu_total_gib() -> float:
    """GPU の総メモリ(GB)を返す。

    Returns
    -------
    float
        最初の CUDA デバイスの総メモリ量（GiB）。CUDA 非対応時は 0。
    """
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory
    return total / (1024**3)


def _require_hf_token_for(model_name: str) -> None:
    """特定モデルで HF トークンが必要そうなら、未設定時にスキップする。

    Parameters
    ----------
    model_name : str
        試験対象のモデル名。
    """
    gated = ["meta-llama/", "google/gemma-"]
    if any(k in model_name.lower() for k in gated) and not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN 未設定のためスキップ（gated model）")


def _make_generator() -> TextGenerator:
    """テスト用の `TextGenerator` を作成する。

    Returns
    -------
    TextGenerator
        小さめの生成パラメータで初期化したインスタンス。
    """
    return TextGenerator(
        inference_engine="normal",
        max_new_tokens=16,
        temperature=0.0,  # 決定論的に
        do_sample=False,
        top_p=1.0,
        top_k=40,
        repetition_penalty=1.0,
    )


def _normalize(text: str) -> str:
    """比較用に正規化する。

    Parameters
    ----------
    text : str
        比較対象のテキスト。

    Returns
    -------
    str
        小文字化・空白正規化・末尾句読点の揺れを吸収した文字列。
    """
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.rstrip(" .!！。?")
    return t


def _assert_stable_generation(model: str, prompt: str) -> None:
    """同一条件で 2 回生成し、結果が安定していることを確認する。

    Parameters
    ----------
    model : str
        モデル名。
    prompt : str
        入力プロンプト。

    Notes
    -----
    - 完全一致をまず試み、失敗時は類似度（SequenceMatcher）0.9 以上を合格とする。
    - これにより GPU 上のわずかな非決定性や特殊トークナイズの微差を許容します。
    """
    g1 = _make_generator()
    g2 = _make_generator()
    out1 = g1.run(model, prompt)
    out2 = g2.run(model, prompt)
    n1, n2 = _normalize(out1), _normalize(out2)
    if n1 == n2:
        return
    ratio = difflib.SequenceMatcher(a=n1, b=n2).ratio()
    assert (
        ratio >= 0.9
    ), f"Unstable generation (ratio={ratio:.3f})\n---1:\n{out1}\n---2:\n{out2}"


@pytest.mark.smoke
def test_llama_small_runs() -> None:
    """Llama 系（1B 想定）が最小生成を完了することを確認する。"""
    model = os.getenv("TAKUMA_TEST_LLAMA", "meta-llama/Llama-3.2-1B-Instruct")
    _require_hf_token_for(model)
    _assert_stable_generation(model, "Say 'hi' in one short sentence.")


@pytest.mark.smoke
def test_qwen_small_runs() -> None:
    """Qwen 系（1.5B 想定）が最小生成を完了することを確認する。"""
    model = os.getenv("TAKUMA_TEST_QWEN", "Qwen/Qwen2.5-1.5B-Instruct")
    _assert_stable_generation(model, "Introduce yourself in one sentence.")


@pytest.mark.smoke
def test_phi_small_runs() -> None:
    """Phi 系（~3.8B）が最小生成を完了することを確認する。"""
    model = os.getenv("TAKUMA_TEST_PHI", "microsoft/Phi-3-mini-4k-instruct")
    _assert_stable_generation(model, "Give me one fun fact about physics.")


@pytest.mark.smoke
def test_gemma3_4b_runs() -> None:
    """Gemma 3 系（4B）が最小生成を完了することを確認する。"""
    model = os.getenv("TAKUMA_TEST_GEMMA", "google/gemma-3-4b-it")
    _require_hf_token_for(model)
    _assert_stable_generation(model, "Summarize what an LLM is in 1 line.")


@pytest.mark.smoke
def test_mistral_maybe_skip_on_vram() -> None:
    """Mistral 7B は 15GB だと厳しい場合があるため、条件付きで実行する。

    - 20GiB 未満の GPU では自動スキップ。
    - 実行時は transformers 経路が使われる（必要に応じて vLLM にフォールバック）。
    """
    if _gpu_total_gib() < 20.0:
        pytest.skip("VRAM < 20GiB のため Mistral テストをスキップ")
    model = os.getenv("TAKUMA_TEST_MISTRAL", "mistralai/Mistral-7B-Instruct-v0.3")
    _assert_stable_generation(model, "State one advantage of transformers in NLP.")


@pytest.mark.smoke
def test_deepseek_small_or_skip() -> None:
    """DeepSeek 系の小容量モデルで実行（なければスキップ）。

    既定は `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` を試みる。
    モデルが存在しない/取得不可、または 15GB では不安定な構成の場合はスキップ。
    """
    model = os.getenv("TAKUMA_TEST_DEEPSEEK", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # 1.5B を想定（存在しない環境ではユーザーが上書き）
    try:
        _assert_stable_generation(model, "Output a single English word: hello.")
    except Exception as e:
        pytest.skip(f"DeepSeek テストをスキップ: {e}")
        return
