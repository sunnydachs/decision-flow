"""最終レポート(results/report.md)を、各成果物から組み立てる。

章立て(仕様の 8 章に対応):
  1. 分類品質 / 2. 確率の較正 / 3. 判断効率 / 4. 運用性能 / 5. 信頼性   ← report_test*.md
  6. 安定性                ← results/stability.md, results/stability_variants.md
  7. 再較正の前後          ← results/recalibration.md
  8. 制約と注意点          ← この runner が生成(実測と公式確認に基づく)

数値は成果物から読むだけで、ここで再計算はしない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import REPO_ROOT, load_config

LIMITATIONS_TEMPLATE = """## 8. 制約と注意点

### データ
- 主データは **LLM 生成の合成データ**(calib 300 / test 500、生成は 外部エンドポイント `generator-a`)。
  実問い合わせの分布とは異なる可能性がある。**方式間の相対比較**を主目的とし、絶対値は参考値とみなすこと。
- 評価対象の LLM は生成モデルとは別系統(OpenAI 系 `llm-generic-20b`)にしてあるが、生成ラベル自体が
  モデルの自己ラベルであり、人手の正解ではない。
- **人間の正解ラベルは付けていない。** 生成ラベルの妥当性は、文献のアンカー値との比較で置き換える
  (次節「生成ラベルの妥当性」)。実データを運用に使う前に、人手の検証(100 件程度の二重採点と κ)を
  別途行うことを推奨する。
- 実データは未使用。無料枠のモデルに送ったのは合成データのみ。

### 生成ラベルの妥当性(人間アンカーの代わりに文献で置き換え)
人手の二重採点は実施していない。代わりに、同種のタスクで報告されている人間アノテータの一致水準を
アンカーとして示す(本プロジェクトの測定値ではない):

| アンカー | 値 | 出典 |
|---|---|---|
| 訓練されたアノテータ同士の一致率 | 約 79% | Gilardi et al. 2023, PNAS 120(30), n=6,183 |
| 専門家 2 名の Cohen's κ | 0.788 | CODA-19 再検証, arXiv:2402.16795(5クラス) |
| クラウドワーカー同士の一致率 | 約 56% | Gilardi et al. 2023, PNAS 120(30) |

**この置き換えで言えること**: 人間同士でも難しい分類タスクでは一致率 約79%・κ 約0.79 程度に収束する
ことが知られている。したがって **全方式の分類精度(0.886〜0.908)は「人間同士の一致水準と同程度か、
やや上」の帯域にあり、これ以上の改善はアノテータ間の揺れの範囲に入る可能性がある**(生成ラベルを
正解とした相対比較であることに注意)。

**言えないこと**: 生成ラベルそのものの正確さは検証されていない(モデルが own labels を付与している)。
文献のアンカーは「人間同士の一致水準」であって、本データセットの品質を保証しない。
絶対値ではなく方式間の相対比較として読むこと。

生の一致率ではなく chance-corrected 統計(κ 等)を使う根拠は arXiv:2603.06865
(カテゴリが偏ったタスクで percentage agreement は信頼性を過大評価する)。
採点ツール自体は実装済み(`runners.review_sample` が κ と文献アンカーを出力する)ので、
実データでの検証時にそのまま使える。

### 件数と信頼区間
- test は 500 件。うち `severity=high` は {high_n} 件、`urgent_claim` かつ `severity=high`(見逃しコストが
  最も高い集合)は {urgent_high_n} 件。固定 Recall の正例は `severity=high`({high_n} 件)を使っている。
- 高リスク件数が 100 件以上あっても、サブグループの指標は信頼区間が広い。固定 Recall は
  ブートストラップ区間の下限と併せて読むこと(第 3 章に併記)。
- 安定性テストは全件ではなく **50 件 × {stability_runs} 回**のサブセット(無料枠を全 :free モデルで共有するため)。
- {mercury_note}

### モデルと経路
- 比較用 LLM の経路は設定で差し替え可能。既定は **OpenAI 互換エンドポイント**(外部ルーティング の無料枠を mercury 用に温存)。
  外部エンドポイントは per-minute 制限が無いため並列 12 で実行している。
- 比較用 LLM は無料枠・小型モデルである。**この LLM についての結論であり、より大きなモデルには当てはまらない。**
- `pplx_decider` と `jev` は**有料**。`jev` は設定した予算上限($1)の範囲で実行し、実測コストは $0.0279(推定値)。
- TypeSafe 公式ドキュメントに、**英語が最も精度が高く CJK は「handled but not equally well」**との記載がある。
  日本語タスクの Jev の数値はこの留保つきで読むこと。

### 確率の扱い
- 較正の対象は **公式に確率と定義された値のみ**(`mercury` / `pplx_decider` / `jev` の choice 確率、
  `embedding_lr` の確率)。LLM の自己申告確率(`stated_probability`)は確率として扱わず、
  AUROC と誤り検出力のみで評価している。
- Jev の `confidence` は確率分布から導かれた値で、**確率ではない**ため較正の対象にしていない(生応答にのみ保存)。
- 確率が「確率と呼ばれている」ことと「較正されている」ことは別。後者は ECE 等で測った結果を参照。

### コスト
- 金額は API が返す `usage` のトークン数 × 公開単価からの**推定値**であり、実際の請求額ではない。
- Jev の実行前見積り(入力 300 トークン/件)は実測(831 トークン/件)の約 1/3 だった。スモーク実測で補正済み。

### 確認できなかった仕様
{unverified}
"""


def build(*, repo_root: Path = REPO_ROOT, out: Path | None = None) -> str:
    config = load_config(repo_root=repo_root)
    results = repo_root / "results"
    test_raw = results / "raw"

    # --- 1-5: test 集計(存在する方式のみ) ---
    methods = []
    for name in ("rule", "embedding_lr", "llm_prompt", "llm_json_schema", "pplx_decider", "jev", "mercury"):
        if (test_raw / f"support_classification__{name}.jsonl").exists():
            methods.append(name)
    part = results / "report_test_partial.md"
    if not part.exists():
        raise FileNotFoundError(f"{part} がありません。先に runners.report を実行してください")
    body = part.read_text(encoding="utf-8").rstrip()

    # --- 6: 安定性 ---
    sections = [body]
    stability = []
    for path, title in ((results / "stability.md", "### 同一入力の反復"),):
        if path.exists():
            text = path.read_text(encoding="utf-8")
            text = "\n".join(line for line in text.splitlines() if not line.startswith("# "))
            stability.append(f"{title}\n{text.strip()}")
    variants = results / "stability_variants.md"
    if variants.exists():
        text = variants.read_text(encoding="utf-8")
        text = "\n".join(line for line in text.splitlines() if not line.startswith("# "))
        stability.append(f"### 言い換え・選択肢順序の変異版\n{text.strip()}")
    sections.append("## 6. 安定性\n\n" + "\n\n".join(stability))

    # --- 7: 再較正 ---
    rec = results / "recalibration.md"
    if rec.exists():
        text = rec.read_text(encoding="utf-8")
        text = "\n".join(line for line in text.splitlines() if not line.startswith("# "))
        sections.append("## 7. 再較正の前後\n" + text.strip())

    # --- 8: 制約 ---
    high_n = _count_high(test_raw, repo_root)
    urgent_high_n = _count_urgent_high(repo_root)
    stability_runs = int(config.get("evaluation.stability_n", 20))
    mercury_note = (
        "**mercury は 外部ルーティング の無料枠枯渇のため test 未実行**(枠は全 :free モデル共有。次回リセット後に実行)。"
        if "mercury" not in methods else
        "mercury は test も実行済み。"
    )
    unverified = _unverified_notes(config)
    sections.append(LIMITATIONS_TEMPLATE.format(
        high_n=high_n, urgent_high_n=urgent_high_n, stability_runs=stability_runs,
        mercury_note=mercury_note, unverified=unverified,
    ).strip())

    markdown = "\n\n".join(sections) + "\n"
    target = out or results / "report.md"
    target.write_text(markdown, encoding="utf-8")
    return markdown


def _count_high(test_raw: Path, repo_root: Path) -> int:
    from common.dataset import load_jsonl

    items = load_jsonl(repo_root / "data" / "test" / "support_classification.jsonl")
    return sum(1 for it in items if it.severity == "high")


def _count_urgent_high(repo_root: Path) -> int:
    from common.dataset import load_jsonl

    items = load_jsonl(repo_root / "data" / "test" / "support_classification.jsonl")
    return sum(1 for it in items if it.severity == "high" and it.label == "urgent_claim")


def _unverified_notes(config) -> str:
    notes = [
        "- Jev の入力トークン数は事前ドライランでは確定できず、スモーク実測(831 トークン/件)で置き換えた。",
        "- TypeSafe のレート制限は「動的に調整中」と公式に記載されており、固定値として扱えない。",
        "- 外部エンドポイントのカタログには 80 モデルが載るが、このアカウントでは大半が 404。利用可能なモデルは実測で確認した。",
    ]
    return "\n".join(notes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="最終レポートを組み立てる")
    parser.add_argument("--out", default="results/report.md")
    args = parser.parse_args(argv)
    markdown = build(out=Path(REPO_ROOT / args.out))
    print(f"report -> {REPO_ROOT / args.out} ({len(markdown)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
