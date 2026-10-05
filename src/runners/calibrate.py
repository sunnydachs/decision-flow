"""calib から閾値を決める(hybrid / 固定 Recall)。

test では閾値を探さない。ここで calib のみから決めて config/thresholds.json に保存し、
評価時はその値を固定適用する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import REPO_ROOT, load_config
from common.dataset import load_jsonl
from evaluation.risk_coverage import fixed_recall_threshold, evaluate_threshold


def load_raw(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["item_id"]] = rec
    return out


def block_threshold(p, y) -> float:
    """誤ブロック(高リスクでないものを block)を出さない最小の閾値。無ければ 1.0。"""
    pairs = sorted(zip(p, y), key=lambda t: t[0], reverse=True)
    best = 1.0
    tp = fp = 0
    for value, label in pairs:
        if label == 1:
            tp += 1
            if fp == 0:
                best = value
        else:
            fp += 1
            break
    return float(best) if tp else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="calib から閾値を決める")
    parser.add_argument("--calib", required=True, help="calib データ(JSONL)")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--methods", default="mercury,pplx_decider,llm_prompt,llm_json_schema,rule,embedding_lr")
    parser.add_argument("--out", default="config/thresholds.json")
    args = parser.parse_args(argv)

    config = load_config(repo_root=REPO_ROOT)
    severity = str(config.get("evaluation.risk_severity", "high"))
    target = float(config.get("evaluation.recall_target", 0.95))
    items = {it.id: it for it in load_jsonl(REPO_ROOT / args.calib)}
    stem = Path(args.calib).stem

    out: dict = {"source": args.calib, "risk_severity": severity, "recall_target": target, "methods": {}}
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        records = load_raw(REPO_ROOT / args.raw_dir / f"{stem}__{method}.jsonl")
        p, y = [], []
        for iid, r in records.items():
            item = items.get(iid)
            if item is None or r.get("error_type") or r.get("risk_score") is None:
                continue
            p.append(float(r["risk_score"]))
            y.append(1 if item.severity == severity else 0)
        if not p or len(set(y)) < 2:
            continue
        review_above = fixed_recall_threshold(p, y, target=target)
        block_above = max(review_above, block_threshold(p, y))
        out["methods"][method] = {
            "review_above": review_above,
            "block_above": block_above,
            "calib_metrics": evaluate_threshold(p, y, threshold=review_above),
            "n": len(p),
        }
    path = Path(REPO_ROOT / args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nthresholds -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
