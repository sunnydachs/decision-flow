# 評価レポート(集計)

- データ: `data/synthetic/support_classification.jsonl`(30 件)
- 高リスクの定義: severity = `high`
- 目標 Recall: 0.95
- 閾値の出所: same_as_test (smoke only)

## 1. 分類品質

| method | n | accuracy | macro F1 | urgent_claim P/R/F1 |
|---|---|---|---|---|
| rule | 30 | 0.800 | 0.807 | 0.700/0.778/0.737 |
| mercury | 30 | 0.900 | 0.905 | 0.889/0.889/0.889 |
| llm_prompt | 29 | 0.966 | 0.974 | 0.900/1.000/0.947 |
| llm_json_schema | 30 | 0.967 | 0.974 | 0.900/1.000/0.947 |
| pplx_decider | 30 | 0.867 | 0.851 | 0.900/1.000/0.947 |
| embedding_lr | 30 | 1.000 | 1.000 | 1.000/1.000/1.000 |

## 2. 確率の較正

| method | ECE | Brier | log loss | AUROC |
|---|---|---|---|---|
| rule | - | - | - | 0.847 |
| mercury | 0.092 | 0.089 | 0.612 | 0.912 |
| llm_prompt | - | - | - | 0.988 |
| llm_json_schema | - | - | - | 0.963 |
| pplx_decider | 0.095 | 0.075 | 0.392 | 0.949 |
| embedding_lr | 0.157 | 0.088 | 0.319 | 0.935 |

(ECE/Brier/log loss は value_type=probability の方式のみ。LLM の自己申告確率は較正の対象外。AUROC は全方式)

## 3. 判断効率(固定 Recall)

| method | calib 閾値 | test recall | test 自動化率 | 自動処理の誤り率 |
|---|---|---|---|---|
| rule | 0.000 | 1.000 | 0.000 | - |
| mercury | 0.000 | 1.000 | 0.233 | 0.000 |
| llm_prompt | 0.050 | 1.000 | 0.414 | 0.000 |
| llm_json_schema | 0.030 | 1.000 | 0.400 | 0.000 |
| pplx_decider | 0.002 | 1.000 | 0.233 | 0.000 |
| embedding_lr | 0.040 | 1.000 | 0.133 | 0.000 |

## 4. 運用性能(レイテンシ・コスト)

| method | p50 ms | p95 ms | 推定コスト / 1000件 |
|---|---|---|---|
| rule | 0.0 | 0.0 | - |
| mercury | 402.4 | 638.5 | 0.0000 |
| llm_prompt | 8533.0 | 47770.9 | - |
| llm_json_schema | 5835.4 | 29698.4 | - |
| pplx_decider | 357.1 | 675.2 | 0.0142 |
| embedding_lr | 0.0 | 0.0 | - |

## 5. 信頼性(エラー率・パース失敗率)

| method | 記録数 | 成功 | パース失敗 | エラー内訳 |
|---|---|---|---|---|
| rule | 30 | 30 | 0 | - |
| mercury | 30 | 30 | 0 | - |
| llm_prompt | 30 | 29 | 0 | {"http_429": 1} |
| llm_json_schema | 30 | 30 | 0 | - |
| pplx_decider | 30 | 30 | 0 | - |
| embedding_lr | 30 | 30 | 0 | - |
