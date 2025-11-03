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
pip install -U "git+ssh://git@github.com/takuma1229/takuma_llm_toolkit.git"
```

### 環境変数（APIキー等）

`.env` または環境変数として以下を設定します（`python-dotenv` が自動読み込み）。

- `HF_TOKEN` : 必須。Hugging Face のアクセストークン。
- `OPENAI_API_KEY` : OpenAI API を利用する場合に設定。
- `DEEPSEEK_API_KEY` : DeepSeek API を利用する場合に設定。

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
    inference_engine="vllm",
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

### 付属スクリプトの実行

このリポジトリ内の `run_llm.py` はエントリーポイントの例です。Pythonモジュールとしてインストール後に直接実行しても同様に利用できます。

```bash
# 例: vLLM で推論、生成パラメータを明示指定
python run_llm.py -e vllm \
  --max-new-tokens 256 --temperature 0.7 --top-p 0.9 --top-k 40 --repetition-penalty 1.1 --do-sample \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 12000 \
  meta-llama/Meta-Llama-3-8B-Instruct -p "こんにちは！"
```

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

### バージョン自動更新

`main` ブランチへ変更がマージされるたびに、GitHub Actions が `bump2version` を利用して `pyproject.toml` のバージョンを自動でパッチ更新し、タグを作成します。自動コミットは `github-actions[bot]` によって行われ、ワークフロー内で循環実行を避ける設定済みです。
