# スループット実測

- データ: `<repo-root>/data/test/support_classification.jsonl` の先頭 60 件
- 並列数: 12(全方式で統一)
- run_index: 207427(キャッシュに当たらない一意な値)
- キャッシュを避けるため専用の run_index で新規に呼び出した実測値。

| method | 成功 | エラー | 壁時計秒 | 件/秒 |
|---|---|---|---|---|
| llm_prompt | 60 | 0 | 30.82 | 1.95 |
| llm_json_schema | 60 | 0 | 21.85 | 2.75 |
| pplx_decider | 59 | 1 | 8.94 | 6.60 |
