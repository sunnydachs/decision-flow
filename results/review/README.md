# 目視確認の進め方(100件)

生成ラベル(= モデルの自己ラベル)が「第二の人間アノテータ」として妥当かを確認する作業です。

## 統計の根拠(なぜ一致率だけでは足りないか)

生の一致率(percentage agreement)は、カテゴリが偏っていると信頼性を**過大評価**します
(全て「billing」と答えても 70% 一致する)。そのため **Cohen's κ(チャンス補正済み)を併記**します
(arXiv:2603.06865「Counting on Consensus」)。

## 人間アノテータ同士の参考値(文献アンカー)

| 指標 | 参考値 | 出典 |
|---|---|---|
| 訓練されたアノテータ同士の一致率 | 約 79% | Gilardi et al. 2023, PNAS 120(30), n=6,183 |
| 専門家 2 名の Cohen's κ | 0.788 | CODA-19 再検証, arXiv:2402.16795(5クラス) |
| クラウドワーカー同士の一致率 | 約 56% | Gilardi et al. 2023, PNAS 120(30) |

**あなたの確認結果がこの範囲(一致率 約79%前後 / κ 約0.79前後)に入れば、
「生成ラベルは第二の人間アノテータ相当」と読めます。**
スコア出力にはこのアンカーが同封されるので、比較は自動です。

## 手順(所要 約1時間)

1. `results/review/review_sample.csv` を開く(101件。層化サンプル: urgent 31 / technical 24 /
   billing 23 / sales 14 / other 9)
2. 各行に記入:
   - `reviewed_label` … その文をあなたが分類するなら 5 カテゴリのどれか
     (`urgent_claim` / `billing` / `technical` / `sales` / `other`)
   - `reviewed_severity` … `high`(即時対応が必要)or `normal`
   - `note` … 任意。迷った理由(例: 「緊急だがクレームではない」「ambiguous に近い」)
3. 保存して採点:

```bash
cd /home/arari/projects/decision-flow
.venv/bin/python -m runners.review_sample --score results/review/review_sample.csv
```

4. 出力の見方:
   - `label_agreement`(生の一致率)+ `label_cohen_kappa`(κ)
   - `literature_anchors` … 上の文献値(自動で同封)
   - `disagreements` … 一致しなかった項目の一覧(note つき)

5. このフォルダに `agreement.md` として結果を保存すると、レポート第8章に反映できます
   (`runners.final_report` が読み込みます)。

## 不一致の扱い

不一致は「ノイズ」ではなく分析対象です(文献も同じ立場):
- `ambiguous=true` の項目での不一致は、分類の失敗ではなく境界事例の証拠
- `urgent_claim` の境界(緊急だがクレームではない等)での不一致は、カテゴリ定義の改善候補
- レポートでは「一致率/κ + 不一致のパターン」の 3 点セットで報告する
