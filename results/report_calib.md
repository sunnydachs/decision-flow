# 評価レポート(集計)

- データ: `data/calib/support_classification.jsonl`(300 件)
- 高リスクの定義: severity = `high`
- 目標 Recall: 0.95
- 閾値の出所: same_as_test (smoke only)

## 1. 分類品質

| method | n | accuracy | macro F1 | urgent_claim P/R/F1 |
|---|---|---|---|---|
| rule | 300 | 0.743 | 0.721 | 0.800/0.916/0.854 |
| embedding_lr | 300 | 0.863 | 0.832 | 0.920/0.964/0.941 |
| mercury | 300 | 0.890 | 0.888 | 0.806/1.000/0.892 |
| llm_prompt | 282 | 0.894 | 0.889 | 0.857/0.986/0.917 |
| llm_json_schema | 282 | 0.911 | 0.905 | 0.888/1.000/0.940 |
| pplx_decider | 300 | 0.880 | 0.866 | 0.865/1.000/0.927 |

## 2. 確率の較正

| method | ECE | Brier | log loss | AUROC |
|---|---|---|---|---|
| rule | - | - | - | 0.903 |
| embedding_lr | 0.101 | 0.086 | 0.339 | 0.955 |
| mercury | 0.066 | 0.073 | 0.460 | 0.949 |
| llm_prompt | - | - | - | 0.953 |
| llm_json_schema | - | - | - | 0.961 |
| pplx_decider | 0.060 | 0.064 | 0.356 | 0.939 |

(ECE/Brier/log loss は value_type=probability の方式のみ。LLM の自己申告確率は較正の対象外。AUROC は全方式)

### 曖昧な文(ambiguous)の別集計と、誤りの重要度

| method | overall acc | ambiguous acc (n) | 非ambiguous acc | 誤り件数 | うち severity=high |
|---|---|---|---|---|---|
| rule | 0.743 | 0.500 (18) | 0.759 | 77 | 30 |
| embedding_lr | 0.863 | 0.500 (18) | 0.887 | 41 | 13 |
| mercury | 0.890 | 0.722 (18) | 0.901 | 33 | 13 |
| llm_prompt | 0.894 | 0.778 (18) | 0.902 | 30 | 15 |
| llm_json_schema | 0.911 | 0.765 (17) | 0.921 | 25 | 11 |
| pplx_decider | 0.880 | 0.667 (18) | 0.894 | 36 | 14 |

## 3. 判断効率(固定 Recall)

| method | calib 閾値 | test recall | test 自動化率 | 自動処理の誤り率 |
|---|---|---|---|---|
| rule | 0.000 | 1.000 | 0.000 | - |
| embedding_lr | 0.016 | 0.956 | 0.467 | 0.093 |
| mercury | 0.000 | 0.956 | 0.387 | 0.026 |
| llm_prompt | 0.020 | 0.961 | 0.358 | 0.069 |
| llm_json_schema | 0.050 | 0.952 | 0.468 | 0.053 |
| pplx_decider | 0.002 | 0.956 | 0.293 | 0.011 |

## 4. 運用性能(レイテンシ・コスト)

| method | p50 ms | p95 ms | 推定コスト / 1000件 |
|---|---|---|---|
| rule | 0.0 | 0.0 | - |
| embedding_lr | 0.0 | 0.0 | - |
| mercury | 324.7 | 399.7 | 0.0000 |
| llm_prompt | 3469.2 | 38255.3 | - |
| llm_json_schema | 5832.8 | 26681.4 | - |
| pplx_decider | 340.6 | 442.3 | 0.0144 |

## 5. 信頼性(エラー率・パース失敗率)

| method | 記録数 | 成功 | パース失敗 | エラー内訳 |
|---|---|---|---|---|
| rule | 300 | 300 | 0 | - |
| embedding_lr | 300 | 300 | 0 | - |
| mercury | 300 | 300 | 0 | - |
| llm_prompt | 300 | 282 | 2 | {"http_429": 16, "parse": 2} |
| llm_json_schema | 300 | 282 | 4 | {"parse": 4, "http_429": 14} |
| pplx_decider | 300 | 300 | 0 | - |
