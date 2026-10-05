# 安定性テスト

- データ: `/home/arari/projects/decision-flow/data/test/support_classification.jsonl` の先頭 50 件
- 反復回数: 20

| method | 項目数 | 確率の平均SD | 確率の平均IQR | 閾値反転(平均) | 反転した項目 | 常に不確実(≥0.3) | ラベル不一致率 |
|---|---|---|---|---|---|---|---|
| rule | 50 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0.000 |
| embedding_lr | 50 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0.000 |
| llm_prompt | 50 | 0.014 | 0.011 | 0.064 | 18 | 5 | 0.080 |
| llm_json_schema | 50 | 0.013 | 0.020 | 0.054 | 13 | 4 | 0.040 |
| pplx_decider | 50 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0.000 |

確率を返さない方式(rule)はラベル不一致率のみ。閾値は calib 由来(`config/thresholds.json`)。