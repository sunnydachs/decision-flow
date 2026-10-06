# decision-flow

企業システムの判断処理を「テキスト生成 LLM」から「Decision Model(構造化された判断結果を返すモデル)」へ
切り出すときの trade-off を、**同一データ・同一タスク定義・同一実行条件・同一指標**で実測する評価基盤です。

比較する方式:

| 方式 | 実装 | 確率の扱い |
|---|---|---|
| Rule-based | `src/adapters/rule.py`(キーワードルール) | なし(0/1 のリスク信号のみ) |
| General LLM | `src/adapters/llm.py`(prompt モード / json_schema 強制モード) | 自己申告値(**較正の対象外**) |
| Decision Model | `src/adapters/mercury.py`, `pplx_decider.py`, `jev.py` | 公式に確率と定義された値(**較正の対象**) |
| Hybrid | `src/hybrid/pipeline.py`(ルール → 判断モデル → 閾値 → auto/review/block) | 上記の確率 |

Decision Model が「速い・安い・安全」であることは前提にしていません。数字が示す範囲だけをレポートします。

## 環境構築

Python 3.11 以上(CI と本番は 3.12)。依存は extras で分けています。

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"          # pytest / numpy / scikit-learn
uv pip install -e ".[embed]"        # ローカル埋め込み(fastembed)を使う場合
uv pip install -e ".[jev]"          # Jev の SDK を使う場合(任意)
```

API キーは `.env`(git-ignored)に置きます。解決順は **プロセス環境変数 → リポジトリ直下の `.env`** です。

```bash
cp .env.example .env   # OPENROUTER_API_KEY / TYPESAFE_API_KEY / PERPLEXITY_API_KEY
```

キーの値はコード・ログ・結果ファイル・レポートに**出しません**(存在確認だけなら `key_status()`)。

## 実行コマンド

```bash
# 1) 全方式を同一条件で実行(合成データのスモーク)
.venv/bin/python -m runners.benchmark \
  --dataset data/synthetic/support_classification.jsonl \
  --adapters rule,embedding_lr,mercury,llm_prompt,llm_json_schema,pplx_decider \
  --train-on data/synthetic/support_classification.jsonl

# 2) calib から閾値を決める(test では閾値を探さない)
.venv/bin/python -m runners.calibrate --calib data/calib/support_classification.jsonl

# 3) 生応答から指標を再計算してレポートを出力
.venv/bin/python -m runners.report --dataset data/test/support_classification.jsonl \
  --calib-raw-dir results/raw_calib --calib-dataset data/calib/support_classification.jsonl \
  --hybrid results/hybrid_pplx_decider_A_rulecommit.json,results/hybrid_pplx_decider_B_nocommit.json

# 3b) Hybrid を単体で評価(層2の判断モデルを指定。--no-rule-commit で層1の即確定を止める)
.venv/bin/python -m runners.hybrid_eval --dataset data/test/support_classification.jsonl \
  --raw-dir results/raw --model-method pplx_decider --no-rule-commit

# 3c) calib の生応答をキャッシュから復元(calib と test の出力名が衝突したときの保険)
.venv/bin/python -m runners.recover_raw --dataset data/calib/support_classification.jsonl \
  --out results/raw_calib

# 4) Jev のドライラン(実 API を呼ばない。リクエスト形と推定コストを提示)
.venv/bin/python -m runners.jev_dry_run --token-estimate 300 --out results/jev_dry_run.json

# 5) Jev の本番実行(事前確認のうえで。--allow-paid-models が必須)
.venv/bin/python -m runners.benchmark --dataset data/test/support_classification.jsonl \
  --adapters jev --allow-paid-models

# 6) shadow モード(既存の判定結果と並べて差分を出す)
.venv/bin/python -m runners.shadow --existing existing_decisions.jsonl \
  --model-raw results/raw/support_classification__jev.jsonl

# 7) 安定性(同一入力の反復 + 言い換え・選択肢順序の変異版)
.venv/bin/python -m runners.stability --subset 50 --runs 20 --mode runs
.venv/bin/python -m runners.stability --subset 50 --mode variants

# 8) スループット実測(同一並列数。キャッシュを避ける一意な run_index で実行)
.venv/bin/python -m runners.throughput --n 60 --concurrency 12 \
  --methods llm_prompt,llm_json_schema,pplx_decider

# 9) 再較正の前後(calib で学習 → test で評価)
.venv/bin/python -m runners.recalibrate

# 10) 最終レポート(results/report.md、8章立て)を組み立てる
.venv/bin/python -m runners.report --dataset data/test/support_classification.jsonl \
  --raw-dir results/raw --calib-raw-dir results/raw_calib \
  --hybrid results/hybrid_pplx_decider_A_rulecommit.json,results/hybrid_pplx_decider_B_nocommit.json \
  --throughput results/throughput.json --out results/report_test_partial.md
.venv/bin/python -m runners.final_report

# テスト
.venv/bin/python -m pytest -q
```

## 設定(`config/default.toml`)

ローカル上書きは `config/local.toml`(git-ignored)に置くと deep-merge されます。

| セクション | 内容 |
|---|---|
| `run` | 並列数・timeout・リトライ方針・seed(全方式で統一する実行条件) |
| `evaluation` | 高リスクのラベル/severity・目標 Recall・ブートストラップ設定・安定性テストの N |
| `free_tier` | 無料枠の日次上限(全モデル共有)・毎分上限・枯渇時の挙動 |
| `budget` | 有料モデルの推定コスト上限(Perplexity $0.5 / Jev $1)・外部送信の確認要否 |
| `models.*` | モデルID・エンドポイント・単価。**コードにハードコードしない**。`models.llm` は `provider`/`endpoint`/`key_env`/`tier` で経路を差し替え可能 |
| `embedding` | ローカル埋め込みモデル(fastembed の対応モデル) |

### 比較用 LLM の経路(外部ルーティング の無料枠を消費しない)

`models.llm` は既定で **OpenAI 互換エンドポイント**(`llm-generic-20b`)を使います。外部ルーティング の `:free` 無料枠は
mercury(外部ルーティング 経由のみ)のために温存するためです。`tier = "external_free"` は
「外部 API だが 外部ルーティング の無料枠カウンタを消費しない」の意味で、日次枠・毎分制限は適用されません。

外部ルーティング の `:free` を使いたい場合は `config/local.toml` で上書きします:

```toml
[models.llm]
provider = "外部ルーティング"
model = "llm-free-a"
endpoint = "https://外部ルーティング.ai/api/v1/chat/completions"
key_env = "OPENROUTER_API_KEY"
tier = "free"
```

外部エンドポイント経由の `json_schema` 強制は実測で動作を確認済み(`llm-generic-20b`、2026-10-05)。
第三者性のため、生成モデル(generator)と評価 LLM(llm-generic)は**系統を分けています**。

タスク定義は `config/tasks/*.toml`(カテゴリと説明・risk_label・タスク指示)。**コードを変えずにタスクを差し替え**できます。
ルールのパターンは `config/rules/*.toml`。閾値は `runners.calibrate` が `config/thresholds.json` に出力します。

## データフォーマット

JSONL(1 行 = 1 件)。`label` と `severity` は**独立した列**です。

```json
{"id": "sc-001", "text": "問い合わせ本文", "label": "urgent_claim", "severity": "high",
 "language": "ja", "ambiguous": false, "annotator_labels": ["urgent_claim", "billing"]}
```

データセットのディレクトリには `manifest.json` を置きます(`source` / `license` / `data_class` /
`external_ok` / `provenance`)。`external_ok=false`(実データ)を外部 API に送るには `--confirm-external` が必要です。
詳細は `data/schema.md`。

| ディレクトリ | 用途 |
|---|---|
| `data/synthetic/` | 動作確認専用(レポートの評価に使わない) |
| `data/calib/` | 閾値決定・再較正専用 |
| `data/test/` | 最終評価専用(調整に使わない) |
| `data/auxiliary/` | 補助データ(jev-ja-eval。外部妥当性の確認) |
| `data/validation/` | 公開データ(MASSIVE ja-JP。品質・較正のみ) |

## コストガードと安全弁

- **`--allow-paid-models`**: Jev の実行に必須(誤実行の防止)。Perplexity は予算上限のみで制御。
- **予算上限**: `budget.perplexity_usd`(既定 $0.5)/ `budget.jev_usd`(既定 $1)。超える見込みで停止します。
  コストは API が返すトークン数 × 公式単価からの**推定値**であり、実際の請求額ではありません。
- **`--confirm-external`**: `external_ok=false` のデータを外部 API に送るときに必須。
- **無料枠**: 外部ルーティング の `:free` は全モデル共有(実測 1000 req/day、20 req/min)。
  `free_tier.daily_limit` に達した方式は pending に退避し、他の方式・ローカル計算・レポートは止まりません。
  `GET https://外部ルーティング.ai/api/v1/key` の `free_model_daily_requests` で残量を確認できます。
- **キャッシュ**: `results/cache/` に (method, model, task, item, 反復回, 指示文ハッシュ) で保存。
  日をまたいだ再開や、指示文を変えた後の取り違えを防ぎます。
- **監査ログ**: `results/audit/` に全リクエストの入力・確率・リクエストID・実行日時(UTC)・モデルIDを保存します。

## ディレクトリ

```
config/     default.toml, tasks/, rules/, local.toml(任意), thresholds.json(生成物)
data/       synthetic/ calib/ test/ auxiliary/ validation/, schema.md
src/
  tasks/      タスク定義(カテゴリと説明の読み込み)
  adapters/   base, rule, embedding_lr, llm, mercury, pplx_decider, span, jev, registry
  hybrid/     ルール→判断モデル→閾値→auto/review/block
  evaluation/ classification, calibration, risk_coverage, bootstrap
  runners/    benchmark, calibrate, report, final_report, hybrid_eval, stability, throughput,
              recalibrate, recover_raw, jev_dry_run, shadow
  common/     env, config, dataset, http, ratelimit, budget, cache
results/    raw/(test), raw_calib/(calib), aggregate/, audit/, cache/, pending/, report.md
tests/
```

**calib と test の生応答は必ず別ディレクトリに出す**(`--raw-dir`)。出力ファイル名はデータセット名から作るため、
同じ名前にすると後から実行した方が上書きします(実際に一度起きました)。失っても
`runners.recover_raw` でキャッシュから復元できます。

## 出典

各 API の契約・制限・価格は公式ドキュメントで確認したものをコード内の docstring に出典URL付きで記録しています
(TypeSafe: docs.typesafe.ai / Perplexity: docs.perplexity.ai / 外部ルーティング: 外部ルーティング.ai/docs)。
確認できなかった仕様は推測せず、レポートの「制約と注意点」に明記します。
