## takuma-llm-toolkit の使い方

このライブラリは自作LLM推論モジュールを提供します。Gitリポジトリ経由で`pip install`し、`TextGenerator`クラスを利用して推論を実行できます。

### インストール

1. プライベートリポジトリにこのプロジェクトをpushします。
2. 利用側の環境で以下のコマンドを実行します。

   ```bash
   pip install "git+ssh://git@github.com/<OWNER>/<REPO>.git"
   ```

   HTTPS＋PATを利用する場合は `pip install "git+https://<TOKEN>@github.com/<OWNER>/<REPO>.git"` のように指定してください。

### 最新バージョンへのアップデート (クライアント側)
```bash
$ uv pip install -U "git+ssh://git@github.com/takuma1229/takuma_llm_toolkit.git"
$ uv lock --upgrade-package takuma-llm-toolkit
$ uv sync
```

### 環境変数（APIキー等）

`.env` または環境変数として以下を設定します（`python-dotenv` が自動読み込み）。

- `HF_TOKEN` : 必要に応じて。Hugging Face のアクセストークン。
- `OPENAI_API_KEY` : OpenAI API を利用する場合に設定。
  
- `MISTRAL_MODELS_PATH` : 任意。Mistral 公式実装（Tokenizer/Transformer）を使う場合に、
  ローカルのモデルフォルダ（`tokenizer.model.v3` 等が含まれるディレクトリ）を指します。

生成パラメータ（max_new_tokens 等）の上書きを環境変数では行いません。下記「設定の優先度」に従い、明示引数または Config.toml を利用してください。

### 設定の優先度

高い → 低い の順に適用されます。

1. TextGenerator のコンストラクタ引数（明示指定）
2. Config.toml の値
3. ライブラリ内のデフォルト

Config.toml はカレントディレクトリから親に向かって探索され、最初に見つかったものが読み込まれます。

サンプル（Config.toml）:

```toml
[generation]
max_new_tokens = 256
temperature = 0.7
top_p = 0.9
top_k = 40
repetition_penalty = 1.1
do_sample = true
```

### Pythonからの利用例

```python
from takuma_llm_toolkit import TextGenerator

# 必須: inference_engine（"normal" または "vllm"）
gen = TextGenerator(
    inference_engine="vllm",  # "normal" も可
    # 生成系（任意）: 未指定は Config.toml → 既定
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1,
    do_sample=True,
    # vLLM 系（任意）
    # tensor_parallel_size=2,
    # gpu_memory_utilization=0.85,
)

response = gen.run(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt="バックプロパゲーションの概要を教えて下さい。",
)
print(response)
```

### 対応モデルとルーティング方針（重要）

- Llama 系
  - `normal`: transformers（公式実装）
  - `vllm`: vLLM（ベースモデルは vLLM、Instruct は transformers）
- Qwen 系
  - `normal`: transformers
  - `vllm`: vLLM
- Phi 系
  - `normal`: transformers
- Mistral 系
  - 優先順: 公式実装（mistral-inference）→ transformers → vLLM
  - 公式実装を使う場合は `MISTRAL_MODELS_PATH` でローカルフォルダを指定
- Gemma 3 系（例: `google/gemma-3-4b-it`）
  - 常に transformers（`Gemma3ForConditionalGeneration` + `AutoTokenizer`）。
  - OpenAI API へは送信しません。
- DeepSeek 系（例: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` など）
  - `normal`: transformers 経由でローカル推論
  - `vllm`: vLLM 経由（チャットテンプレートに対応）

備考: vLLM 利用時は内部で `enforce_eager=True` を指定しており、Torch.compile/Triton の JIT を使わずに安定動作を優先しています。

### 付属スクリプトの実行

このリポジトリ内の `run_llm.py` はエントリーポイントの例です。Pythonモジュールとしてインストール後に直接実行しても同様に利用できます。

```bash
# 例: vLLM で推論、生成パラメータを明示指定
python run_llm.py -e vllm \
  --max-new-tokens 256 --temperature 0.7 --top-p 0.9 --top-k 40 --repetition-penalty 1.1 --do-sample \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 12000 \
  meta-llama/Meta-Llama-3-8B-Instruct -p "こんにちは！"
```

#### 実行例（Gemma 3 / transformers 経路）

```bash
uv run run_llm.py google/gemma-3-4b-it --prompt "分析哲学について教えて"
```

#### 実行例（Mistral 公式実装: ローカルフォルダ指定）

```bash
export MISTRAL_MODELS_PATH=/path/to/mistral_model_folder
uv run run_llm.py /path/to/mistral_model_folder --prompt "分析哲学について教えて"
```

#### 実行例（Mistral / transformers 経路）

```bash
uv run run_llm.py mistralai/Mistral-7B-Instruct-v0.3 --prompt "分析哲学について教えて"
```

#### 実行例（DeepSeek / transformers 経路）

```bash
uv run run_llm.py deepseek-ai/DeepSeek-R1-Distill-Llama-8B --prompt "分析哲学について教えて"
```

注意（DeepSeek-V3 について）
- `deepseek-ai/DeepSeek-V3` は finegrained-FP8 量子化のため、GPU の Compute Capability が 8.9 以上（例: RTX 4090/H100）でのみ動作します。
- Ampere 世代（例: RTX A6000, SM=8.6）ではロード時に失敗します。代替として `DeepSeek-R1-Distill-*` 系の bf16/fp16 チェックポイントをご利用ください。

SLURM に直接投げる場合は `shell_scripts/run_llm_once.sh` を使います。

```bash
sbatch --gres=gpu:1 shell_scripts/run_llm_once.sh \
  -m meta-llama/Meta-Llama-3-8B-Instruct -e vllm -p "こんにちは！" \
  --max-new-tokens 256 --temperature 0.7 --top-p 0.9 --top-k 40 --repetition-penalty 1.1 --do-sample \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 12000

トラブルシュート（vLLM の KV キャッシュ不足エラー）

vLLM 初期化時に下記のようなエラーが出る場合は、GPU メモリに対して `max_model_len` が大きすぎるか、`gpu_memory_utilization` が小さすぎます。

```
ValueError: ... max seq len (...) ... KV cache is needed ... available ... Try increasing `gpu_memory_utilization` or decreasing `max_model_len` ...
```

対処例:
- `--gpu-memory-utilization 0.95` に上げる
- `--max-model-len 12000` のようにコンテキスト長を下げる
- `--tensor-parallel-size` を増やして GPU を分割活用する

備考: 本ツールの既定の `max_model_len` は 2048 です。必要に応じて上記オプションで調整してください。
```

#### トラブルシュート（vLLM/Triton の `-lcuda` リンクエラー）

Torch.compile/Inductor → Triton のビルドで `-lcuda` が見つからず失敗する環境があります。
本ツールでは vLLM 初期化に `enforce_eager=True` を付けており、既定でコンパイル経路を無効化して回避します。

コンパイル経路を有効化したい場合は、以下のいずれかを満たしてください。
- システムに `libcuda.so` が解決できるよう適切に配置（例: `/usr/lib/x86_64-linux-gnu/libcuda.so`）
- CUDA 開発用パッケージの導入（環境に応じて）

それでも解決しない場合は、`normal` エンジンや transformers 経路をご利用ください。

### ログ出力

- 実行時間: `run_elapsed_sec`（秒）を INFO ログで出力します。
- 入力/出力: `Input Prompt` と `Generated Text` を DEBUG/INFO ログで記録します。

- エラー時: GPU 互換性やビルド要件に関わる典型的な失敗（例: FP8/SM 要件未満、Triton の `-lcuda` 未解決、KV キャッシュ不足）について、
  GPU 名や Compute Capability（SM）を含むヒントを ERROR ログに出力します。

例外発生時は実行時間の出力まで到達しない場合があります。必要であれば `try/finally` による計測出力へ改善可能です。

### バージョン自動更新

`main` ブランチへ変更がマージされるたびに、GitHub Actions が `bump2version` を利用して `pyproject.toml` のバージョンを自動でパッチ更新し、タグを作成します。自動コミットは `github-actions[bot]` によって行われ、ワークフロー内で循環実行を避ける設定済みです。
