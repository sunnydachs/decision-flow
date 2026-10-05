# 評価レポート(集計)

- データ: `data/test/support_classification.jsonl`(500 件)
- 高リスクの定義: severity = `high`
- 目標 Recall: 0.95
- 閾値の出所: separate

## 1. 分類品質

| method | n | accuracy | macro F1 | urgent_claim P/R/F1 |
|---|---|---|---|---|
| rule | 500 | 0.732 | 0.705 | 0.807/0.928/0.863 |
| embedding_lr | 500 | 0.878 | 0.845 | 0.913/0.961/0.936 |
| llm_prompt | 499 | 0.908 | 0.892 | 0.898/0.980/0.938 |
| llm_json_schema | 499 | 0.904 | 0.892 | 0.902/0.961/0.930 |
| pplx_decider | 500 | 0.902 | 0.893 | 0.889/0.993/0.938 |

## 2. 確率の較正

| method | ECE | Brier | log loss | AUROC |
|---|---|---|---|---|
| rule | - | - | - | 0.920 |
| embedding_lr | 0.082 | 0.073 | 0.335 | 0.941 |
| llm_prompt | - | - | - | 0.954 |
| llm_json_schema | - | - | - | 0.938 |
| pplx_decider | 0.046 | 0.055 | 0.281 | 0.960 |

(ECE/Brier/log loss は value_type=probability の方式のみ。LLM の自己申告確率は較正の対象外。AUROC は全方式)

### 曖昧な文(ambiguous)の別集計と、誤りの重要度

| method | overall acc | ambiguous acc (n) | 非ambiguous acc | 誤り件数 | うち severity=high |
|---|---|---|---|---|---|
| rule | 0.732 | 0.647 (34) | 0.738 | 134 | 40 |
| embedding_lr | 0.878 | 0.676 (34) | 0.893 | 61 | 17 |
| llm_prompt | 0.908 | 0.735 (34) | 0.920 | 46 | 17 |
| llm_json_schema | 0.904 | 0.824 (34) | 0.910 | 48 | 20 |
| pplx_decider | 0.902 | 0.824 (34) | 0.908 | 49 | 17 |

## 3. 判断効率(固定 Recall)

| method | calib 閾値 | test recall | test 自動化率 | 自動処理の誤り率 |
|---|---|---|---|---|
| rule | 0.000 | 1.000 | 0.000 | - |
| embedding_lr | 0.016 | 0.922 | 0.472 | 0.085 |
| llm_prompt | 0.050 | 0.990 | 0.142 | 0.028 |
| llm_json_schema | 0.050 | 0.984 | 0.202 | 0.030 |
| pplx_decider | 0.002 | 0.979 | 0.286 | 0.021 |

### Hybrid(ルール → 判断モデル → 閾値 → auto / review / block)

閾値は calib 由来。recall_high は auto に回した分だけで測った高リスク recall(見逃し=auto に入った severity=high)。

| 構成 | 自動化率 | 人間レビュー率 | block 率 | 高リスク recall(auto) | 見逃し | 自動処理の誤り率 |
|---|---|---|---|---|---|---|
| pplx_decider_A_rulecommit | 0.534 | 0.104 | 0.362 | 0.906 | 18 | 0.067 |
| pplx_decider_B_nocommit | 0.282 | 0.342 | 0.376 | 0.979 | 4 | 0.021 |
| embedding_lr_A_rulecommit | 0.580 | 0.060 | 0.360 | 0.896 | 20 | 0.114 |
| embedding_lr_B_nocommit | 0.450 | 0.174 | 0.376 | 0.964 | 7 | 0.089 |

## 4. 運用性能(レイテンシ・コスト)

| method | p50 ms | p95 ms | 推定コスト / 1000件 |
|---|---|---|---|
| rule | 0.0 | 0.0 | - |
| embedding_lr | 0.0 | 0.0 | - |
| llm_prompt | 4133.1 | 14803.2 | - |
| llm_json_schema | 3371.9 | 9172.8 | - |
| pplx_decider | 324.6 | 351.6 | 0.0144 |

## 5. 信頼性(エラー率・パース失敗率)

| method | 記録数 | 成功 | パース失敗 | エラー内訳 |
|---|---|---|---|---|
| rule | 500 | 500 | 0 | - |
| embedding_lr | 500 | 500 | 0 | - |
| llm_prompt | 500 | 499 | 1 | {"parse": 1} |
| llm_json_schema | 500 | 499 | 1 | {"parse": 1} |
| pplx_decider | 500 | 500 | 0 | - |
