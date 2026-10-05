"""shadow モードの雛形。

既存の判定結果(例: 現行システムや人手のラベル)を別途読み込み、Decision Model の判定と並べて差分を出す。
本番導入前に「今の運用とどこが違うか」を見るための道具。実運用では件数が増えるので、
ここでは集計のみを行い、差分の明細は JSONL で保存する。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[str(rec.get("id") or rec.get("item_id"))] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="既存判定と Decision Model の差分を出す(雛形)")
    parser.add_argument("--existing", required=True, help='JSONL: {"id":..., "decision":"..."} 既存の判定')
    parser.add_argument("--model-raw", required=True, help="Decision Model の生応答 JSONL(results/raw/*.jsonl)")
    parser.add_argument("--out", default="results/shadow_diff.jsonl")
    args = parser.parse_args(argv)

    existing = load_jsonl(Path(args.existing))
    model = load_jsonl(Path(args.model_raw))
    common = sorted(set(existing) & set(model))
    diffs = []
    counter: Counter = Counter()
    for iid in common:
        old = str(existing[iid].get("decision") or existing[iid].get("label"))
        new = str(model[iid].get("label"))
        if old != new:
            counter[(old, new)] += 1
            diffs.append({"id": iid, "existing": old, "decision_model": new,
                          "risk_score": model[iid].get("risk_score"),
                          "request_id": model[iid].get("request_id")})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for d in diffs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    summary = {
        "n_existing": len(existing),
        "n_model": len(model),
        "n_common": len(common),
        "n_diff": len(diffs),
        "diff_rate": (len(diffs) / len(common)) if common else None,
        "top_transitions": [{"existing": k[0], "model": k[1], "n": v} for k, v in counter.most_common(10)],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"diff -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
