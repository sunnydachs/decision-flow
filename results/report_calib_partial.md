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
| pplx_decider | 300 | 0.880 | 0.866 | 0.865/1.000/0.927 |

## 2. 確率の較正

| method | ECE | Brier | log loss | AUROC |
|---|---|---|---|---|
| rule | - | - | - | 0.903 |
| embedding_lr | 0.101 | 0.086 | 0.339 | 0.955 |
| pplx_decider | 0.060 | 0.064 | 0.356 | 0.939 |

(ECE/Brier/log loss は value_type=probability の方式のみ。LLM の自己申告確率は較正の対象外。AUROC は全方式)

### 曖昧な文(ambiguous)の別集計と、誤りの重要度

| method | overall acc | ambiguous acc (n) | 非ambiguous acc | 誤り件数 | うち severity=high |
|---|---|---|---|---|---|
| rule | 0.743 | 0.500 (18) | 0.759 | 77 | 30 |
| embedding_lr | 0.863 | 0.500 (18) | 0.887 | 41 | 13 |
| pplx_decider | 0.880 | 0.667 (18) | 0.894 | 36 | 14 |

## 3. 判断効率(固定 Recall)

| method | calib 閾値 | test recall | test 自動化率 | 自動処理の誤り率 |
|---|---|---|---|---|
| rule | 0.000 | 1.000 | 0.000 | - |
| embedding_lr | 0.016 | 0.956 | 0.467 | 0.093 |
| pplx_decider | 0.002 | 0.956 | 0.293 | 0.011 |

## 4. 運用性能(レイテンシ・コスト)

| method | p50 ms | p95 ms | 推定コスト / 1000件 |
|---|---|---|---|
| rule | 0.0 | 0.0 | - |
| embedding_lr | 0.0 | 0.0 | - |
| pplx_decider | 340.6 | 442.3 | 0.0144 |

## 5. 信頼性(エラー率・パース失敗率)

| method | 記録数 | 成功 | パース失敗 | エラー内訳 |
|---|---|---|---|---|
| rule | 300 | 300 | 0 | - |
| embedding_lr | 300 | 300 | 0 | - |
| pplx_decider | 300 | 300 | 0 | - |
