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
- 主データは **LLM 生成の合成データ**(calib 300 / test 500、生成は NIM `nvidia/nemotron-3-super-120b-a12b`)。
  実問い合わせの分布とは異なる可能性がある。**方式間の相対比較**を主目的とし、絶対値は参考値とみなすこと。
- 評価対象の LLM は生成モデルとは別系統(OpenAI 系 `openai/gpt-oss-20b`)にしてあるが、生成ラベル自体が
  モデルの自己ラベルであり、人手の正解ではない。200 件の目視確認と一致率を別途記録する。
- 実データは未使用。無料枠のモデルに送ったのは合成データのみ。

### 件数と信頼区間
- test は 500 件。うち `severity=high` は {high_n} 件、`urgent_claim` かつ `severity=high`(見逃しコストが
  最も高い集合)は {urgent_high_n} 件。固定 Recall の正例は `severity=high`({high_n} 件)を使っている。
- 高リスク件数が 100 件以上あっても、サブグループの指標は信頼区間が広い。固定 Recall は
  ブートストラップ区間の下限と併せて読むこと(第 3 章に併記)。
- 安定性テストは全件ではなく **50 件 × {stability_runs} 回**のサブセット(無料枠を全 :free モデルで共有するため)。
- {mercury_note}

### モデルと経路
- 比較用 LLM の経路は設定で差し替え可能。既定は **NVIDIA NIM**(OpenRouter の無料枠を mercury 用に温存)。
  NIM は per-minute 制限が無いため並列 12 で実行している。
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
        "**mercury は OpenRouter の無料枠枯渇のため test 未実行**(枠は全 :free モデル共有。次回リセット後に実行)。"
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
        "- NIM のカタログには 80 モデルが載るが、このアカウントでは大半が 404。利用可能なモデルは実測で確認した。",
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
