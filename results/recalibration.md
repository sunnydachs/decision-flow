# 再較正の前後(test)

- 正例の定義: severity = `high`
- 学習は calib のみ。test は評価だけに使う。
- 対象は value_type=probability の方式のみ(LLM の自己申告確率は対象外)。

| method | 温度 | 指標 | 前 | 温度後 | isotonic 後 |
|---|---|---|---|---|---|
| embedding_lr | 1.630 | ece | 0.0824 | 0.0774 | 0.0462 |
| embedding_lr | 1.630 | brier | 0.0732 | 0.0759 | 0.0669 |
| embedding_lr | 1.630 | log_loss | 0.3348 | 0.2906 | 0.4034 |
| mercury | 2.902 | ece | 0.0641 | 0.0540 | 0.0318 |
| mercury | 2.902 | brier | 0.0650 | 0.0604 | 0.0554 |
| mercury | 2.902 | log_loss | 0.3761 | 0.2191 | 0.2940 |
| pplx_decider | 1.943 | ece | 0.0460 | 0.0521 | 0.0248 |
| pplx_decider | 1.943 | brier | 0.0550 | 0.0573 | 0.0522 |
| pplx_decider | 1.943 | log_loss | 0.2807 | 0.2296 | 0.2880 |
| jev | 6.473 | ece | 0.0433 | 0.0611 | 0.0228 |
| jev | 6.473 | brier | 0.0562 | 0.0698 | 0.0579 |
| jev | 6.473 | log_loss | 0.7467 | 0.2544 | 0.3134 |

温度・isotonic はどちらも calib で学習した写像。isotonic は単調なので順位(AUROC)は変わらない。