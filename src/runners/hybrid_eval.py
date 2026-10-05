"""Hybrid 方式の評価。

層1: ルールの明確一致(risk ラベル一致のみ block/auto を即確定)
層2: 判断モデルの確率(mercury / pplx_decider / embedding_lr から選択)
層3: 閾値(config/thresholds.json の calib 決定値)で auto / review / block

calib で review_above / block_above を決め、test に固定適用する。
判断不能(エラー)時の既定動作は fail-closed(review)。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from adapters.rule import load_rules
from common.config import REPO_ROOT, load_config
from common.dataset import load_jsonl
from hybrid.pipeline import ACTION_AUTO, ACTION_BLOCK, ACTION_REVIEW, HybridPipeline, Thresholds


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


def evaluate(
    *,
    dataset: Path,
    raw_dir: Path,
    model_method: str,
    thresholds_path: Path,
    fail_open: bool,
    rules_path: Path,
    rule_commit: bool = True,
) -> dict:
    config = load_config(repo_root=REPO_ROOT)
    from tasks.base import load_task

    task = load_task(REPO_ROOT / "config" / "tasks" / "support_classification.toml")
    items = {it.id: it for it in load_jsonl(dataset)}
    stem = dataset.stem
    model_records = load_raw(raw_dir / f"{stem}__{model_method}.jsonl")
    threshold_data = json.loads(thresholds_path.read_text(encoding="utf-8"))
    entry = threshold_data["methods"][model_method]
    thresholds = Thresholds(
        block_above=float(entry["block_above"]), review_above=float(entry["review_above"])
    )
    rules = load_rules(rules_path)
    pipeline = HybridPipeline(rules=rules, thresholds=thresholds, risk_label=task.risk_label,
                              fail_open=fail_open, rule_commit=rule_commit)

    actions = Counter()
    outcomes: Counter = Counter()
    auto_high_missed = 0
    auto_errors = 0
    auto_n = 0
    review_n = 0
    block_n = 0
    for item_id, item in items.items():
        record = model_records.get(item_id, {})
        pred_label = record.get("label")
        risk_score = record.get("risk_score")
        error_type = record.get("error_type")
        decision = pipeline.run(
            type("P", (), {
                "item_id": item_id, "label": pred_label, "risk_score": risk_score,
                "probabilities": record.get("probabilities"), "request_id": record.get("request_id"),
                "error_type": error_type,
            })(),
            text=item.text,
        )
        actions[decision.action] += 1
        is_high = item.severity == threshold_data.get("risk_severity", "high")
        if decision.action == ACTION_AUTO:
            auto_n += 1
            if is_high:
                auto_high_missed += 1
            if pred_label is not None and pred_label != item.label:
                auto_errors += 1
        elif decision.action == ACTION_REVIEW:
            review_n += 1
        elif decision.action == ACTION_BLOCK:
            block_n += 1
        outcomes[f"{decision.action}|{'high' if is_high else 'normal'}"] += 1

    n = len(items)
    high_n = sum(1 for it in items.values() if it.severity == threshold_data.get("risk_severity", "high"))
    recall_high = 1.0 - (auto_high_missed / high_n) if high_n else None
    return {
        "model_method": model_method,
        "thresholds": {"review_above": thresholds.review_above, "block_above": thresholds.block_above},
        "fail_open": fail_open,
        "n": n,
        "high_n": high_n,
        "actions": dict(actions),
        "automation_rate": auto_n / n if n else None,
        "review_rate": review_n / n if n else None,
        "block_rate": block_n / n if n else None,
        "recall_high_at_auto": recall_high,
        "auto_high_missed": auto_high_missed,
        "auto_error_rate": (auto_errors / auto_n) if auto_n else None,
        "outcomes": dict(outcomes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid 方式を評価する(ローカル計算)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--model-method", default="pplx_decider",
                        help="層2で使う判断モデルの生応答(rule→モデル→閾値)")
    parser.add_argument("--thresholds", default="config/thresholds.json")
    parser.add_argument("--rules", default="config/rules/support_classification.toml")
    parser.add_argument("--fail-open", action="store_true")
    parser.add_argument("--no-rule-commit", action="store_true",
                        help="層1のルール即確定を無効化(risk一致blockのみ残す)")
    args = parser.parse_args(argv)

    result = evaluate(
        dataset=Path(REPO_ROOT / args.dataset),
        raw_dir=Path(REPO_ROOT / args.raw_dir),
        model_method=args.model_method,
        thresholds_path=Path(REPO_ROOT / args.thresholds),
        fail_open=bool(args.fail_open),
        rules_path=Path(REPO_ROOT / args.rules),
        rule_commit=not args.no_rule_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
