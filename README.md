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

### 環境変数の設定

`.env` または環境変数として以下を設定します（`python-dotenv` が自動読み込み）。

- `HF_TOKEN` : 必須。Hugging Faceのアクセストークン。
- `OPENAI_API_KEY` : OpenAI APIを利用する場合に設定。
- `DEEPSEEK_API_KEY` : DeepSeek APIを利用する場合に設定。
- 必要に応じて生成パラメータを上書きする環境変数：
  - `LLM_MAX_NEW_TOKENS`
  - `LLM_REPETITION_PENALTY`
  - `LLM_TEMPERATURE`
  - `LLM_DO_SAMPLE`
  - `LLM_TOP_P`
  - `LLM_TOP_K`

### Pythonからの利用例

```python
from takuma_llm_toolkit import TextGenerator

generator = TextGenerator()
response = generator.run(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt="バックプロパゲーションの概要を教えて下さい。",
)
print(response)
```

### 付属スクリプトの実行

このリポジトリ内の `run_llm.py` はエントリーポイントの例です。Pythonモジュールとしてインストール後に直接実行しても同様に利用できます。

```bash
python run_llm.py
```

プロンプト入力後、指定モデルで推論が実行されます。

### バージョン自動更新

`main` ブランチへ変更がマージされるたびに、GitHub Actions が `bump2version` を利用して `pyproject.toml` のバージョンを自動でパッチ更新し、タグを作成します。自動コミットは `github-actions[bot]` によって行われ、ワークフロー内で循環実行を避ける設定済みです。
