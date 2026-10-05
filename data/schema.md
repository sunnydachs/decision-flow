# データフォーマット

## 入力(JSONL、1 行 = 1 件)

| 列 | 型 | 必須 | 説明 |
|---|---|---|---|
| `id` | str | ✓ | 一意な ID |
| `text` | str | ✓ | 問い合わせ本文 |
| `label` | str | ✓ | 正解カテゴリ(`label` と `severity` は独立した列) |
| `severity` | str | ✓ | `high` \| `normal`。高リスク候補を中心に付与 |
| `language` | str | | `ja` など(既定 `ja`) |
| `ambiguous` | bool | | 曖昧な文。通常評価とは別集計する |
| `annotator_labels` | list[str] | | 任意。複数ラベルの一致率算出用 |
| `meta` | object | | 任意。難易度タグなど |

## manifest.json(データセットディレクトリ直下)

| キー | 説明 |
|---|---|
| `source` | 来歴(例 `llm-generated-v1`, `jev-ja-eval`, `MASSIVE`) |
| `license` | ライセンス |
| `data_class` | `synthetic` \| `public` \| `real` |
| `external_ok` | 外部 API に送ってよいか(`real` は `false`) |
| `provenance` | 生成方法・モデル名などの説明 |

`external_ok=false` のデータを外部 API に送るには `--confirm-external` が必要。

## ディレクトリ

- `data/synthetic/` … 動作確認専用(レポートの評価に使わない)
- `data/calib/` … 閾値決定・再較正専用(calib と test は別実行で生成)
- `data/test/` … 最終評価専用(test は調整に使わない)
- `data/auxiliary/` … 補助データ(外部妥当性の確認、例: jev-ja-eval)
- `data/validation/` … 検証データ(品質・較正のみ、例: MASSIVE / BANKING77)
