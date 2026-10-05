# 安定性テスト(言い換え・選択肢順序)

- データ: `/home/arari/projects/decision-flow/data/test/support_classification.jsonl` の先頭 50 件
- 変異版: base, paraphrase_1, paraphrase_2, order_rotated

| method | 変異版間でラベル不一致 | 確率の平均SD | 閾値反転(平均) |
|---|---|---|---|
| rule | 0 / 50 (0.000) | 0.000 | 0.000 |
| embedding_lr | 0 / 50 (0.000) | 0.000 | 0.000 |
| llm_prompt | 4 / 50 (0.080) | 0.034 | 0.120 |
| llm_json_schema | 5 / 50 (0.100) | 0.045 | 0.175 |
| pplx_decider | 1 / 50 (0.020) | 0.002 | 0.040 |
