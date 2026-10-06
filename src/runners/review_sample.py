"""目視確認用のサンプリングと、確認後の一致率算出(Cohen's κ つき)。

1) サンプル作成(層化):
     .venv/bin/python -m runners.review_sample --dataset data/test/support_classification.jsonl \
       --n 100 --out results/review/review_sample.csv
   CSV に reviewed_label / reviewed_severity / note の空列を用意する(生成時のラベルは同封)。

2) 記入後の一致率:
     .venv/bin/python -m runners.review_sample --score results/review/review_sample.csv
   生成ラベルと reviewed_label の一致率と Cohen's κ、クラス別一致率、severity の一致率を出す。

統計の根拠: 生の一致率(percentage agreement)はカテゴリが偏っていると信頼性を過大評価するため、
チャンス補正済みの κ を併記する(arXiv:2603.06865)。人間アノテータ同士の参考値は
Gilardi et al. 2023 (PNAS 120(30)): 訓練アノテータ同士の一致率 約79% /
CODA-19 再検証 (arXiv:2402.16795): 専門家2名の κ = 0.788。
これらと同程度なら「生成ラベルは第二の人間アノテータ相当」と読める。
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from common.config import REPO_ROOT
from common.dataset import load_jsonl

FIELDS = ["id", "text", "generated_label", "generated_severity", "ambiguous",
          "bucket", "reviewed_label", "reviewed_severity", "note"]

# 文献アンカー(レポートに併記する。出典は docstring と results/review/README を参照)
LITERATURE_ANCHORS = {
    "trained_annotators_agreement": {
        "value": 0.79, "source": "Gilardi et al. 2023, PNAS 120(30), n=6,183",
        "note": "訓練されたアノテータ同士の intercoder agreement(タスク平均)",
    },
    "expert_kappa": {
        "value": 0.788, "source": "CODA-19 re-evaluation, arXiv:2402.16795",
        "note": "専門家2名の Cohen's kappa(5クラス、学術論文)",
    },
    "crowd_agreement": {
        "value": 0.56, "source": "Gilardi et al. 2023, PNAS 120(30)",
        "note": "MTurk クラウドワーカー同士の一致率(下限の目安)",
    },
}


def stratified_sample(items, n: int, seed: int):
    rng = random.Random(seed)
    by_label: dict[str, list] = defaultdict(list)
    for item in items:
        by_label[item.label].append(item)
    total = len(items)
    picked = []
    for label, group in by_label.items():
        want = max(1, round(n * len(group) / total))
        rng.shuffle(group)
        picked.extend(group[:want])
    # 端数は全体から補充
    seen = {it.id for it in picked}
    rest = [it for it in items if it.id not in seen]
    rng.shuffle(rest)
    picked.extend(rest[: max(0, n - len(picked))])
    picked.sort(key=lambda it: it.id)
    return picked


def write_sample(items, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "id": item.id,
                "text": item.text,
                "generated_label": item.label,
                "generated_severity": item.severity,
                "ambiguous": int(item.ambiguous),
                "bucket": (item.meta or {}).get("bucket", ""),
                "reviewed_label": "",
                "reviewed_severity": "",
                "note": "",
            })
    return {"n": len(items), "out": str(out_path), "labels": dict(Counter(it.label for it in items))}


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    """Cohen's κ(2者のチャンス補正済み一致)。カテゴリが偏っていると生の一致率は
    信頼性を過大評価するため、一致率と併せて必ず出す(Cohen 1960 / arXiv:2603.06865)。"""
    if not labels_a or len(labels_a) != len(labels_b):
        return None
    n = len(labels_a)
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    marg_a, marg_b = Counter(labels_a), Counter(labels_b)
    pe = sum((marg_a[c] / n) * (marg_b[c] / n) for c in set(marg_a) | set(marg_b))
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def score(path: Path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    reviewed = [r for r in rows if (r.get("reviewed_label") or "").strip()]
    label_hits = sum(1 for r in reviewed if r["reviewed_label"].strip() == r["generated_label"].strip())
    sev_rows = [r for r in reviewed if (r.get("reviewed_severity") or "").strip()]
    sev_hits = sum(1 for r in sev_rows if r["reviewed_severity"].strip() == r["generated_severity"].strip())
    per_label = Counter()
    per_label_total = Counter()
    for r in reviewed:
        per_label_total[r["generated_label"]] += 1
        if r["reviewed_label"].strip() == r["generated_label"].strip():
            per_label[r["generated_label"]] += 1
    kappa = cohen_kappa(
        [r["reviewed_label"].strip() for r in reviewed],
        [r["generated_label"].strip() for r in reviewed],
    )
    sev_kappa = cohen_kappa(
        [r["reviewed_severity"].strip() for r in sev_rows],
        [r["generated_severity"].strip() for r in sev_rows],
    )
    return {
        "rows": len(rows),
        "reviewed": len(reviewed),
        "label_agreement": (label_hits / len(reviewed)) if reviewed else None,
        "label_cohen_kappa": kappa,
        "severity_agreement": (sev_hits / len(sev_rows)) if sev_rows else None,
        "severity_cohen_kappa": sev_kappa,
        "literature_anchors": LITERATURE_ANCHORS,
        "per_label_agreement": {
            label: (per_label[label] / per_label_total[label]) for label in sorted(per_label_total)
        },
        "disagreements": [
            {"id": r["id"], "generated": r["generated_label"], "reviewed": r["reviewed_label"].strip(),
             "note": (r.get("note") or "").strip()}
            for r in reviewed if r["reviewed_label"].strip() != r["generated_label"].strip()
        ][:50],
    }


def main(argv: list[str] | None = None) -> int:
    import json

    parser = argparse.ArgumentParser(description="目視確認のサンプリング / 一致率の算出")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--out", default="results/review/review_sample.csv")
    parser.add_argument("--score", default=None, help="記入済み CSV のパス(この場合は採点のみ)")
    args = parser.parse_args(argv)

    if args.score:
        print(json.dumps(score(Path(args.score)), ensure_ascii=False, indent=2))
        return 0
    if not args.dataset:
        raise SystemExit("--dataset か --score のどちらかが必要です")
    items = load_jsonl(Path(REPO_ROOT / args.dataset))
    sample = stratified_sample(items, args.n, args.seed)
    summary = write_sample(sample, Path(REPO_ROOT / args.out))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
