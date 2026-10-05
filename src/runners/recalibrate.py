"""再較正: calib で学習した温度・isotonic を test に適用し、前後を並べる。

対象は「確率として公式に定義された値」だけ(mercury / pplx_decider / jev / embedding_lr)。
LLM の自己申告確率(stated_probability)は対象外。
閾値探索と同じく、較正の学習も calib のみで行う。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import REPO_ROOT, load_config
from common.dataset import load_jsonl
from evaluation.calibration import recalibration_report

PROBABILITY_METHODS = ("embedding_lr", "mercury", "pplx_decider", "jev")


def _load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["item_id"]] = rec
    return out


def _collect_pairs(
    raw: dict[str, dict],
    items: dict,
    risk_severity: str,
) -> tuple[list[float], list[int]]:
    probs: list[float] = []
    labels: list[int] = []
    for item_id, rec in raw.items():
        item = items.get(item_id)
        if item is None or rec.get("error_type") or rec.get("risk_score") is None:
            continue
        probs.append(float(rec["risk_score"]))
        labels.append(1 if item.severity == risk_severity else 0)
    return probs, labels


def collect(*, repo_root: Path = REPO_ROOT, methods: list[str] | None = None) -> dict:
    config = load_config(repo_root=repo_root)
    risk_severity = str(config.get("evaluation.risk_severity", "high"))
    n_bins = int(config.get("evaluation.calibration_bins", 10))
    methods = methods or list(PROBABILITY_METHODS)
    stem = "support_classification"
    calib_items = {it.id: it for it in load_jsonl(repo_root / "data" / "calib" / f"{stem}.jsonl")}
    test_items = {it.id: it for it in load_jsonl(repo_root / "data" / "test" / f"{stem}.jsonl")}

    report: dict = {
        "risk_severity": risk_severity,
        "target": "severity=high を正例とした risk 確率",
        "note": "学習は calib のみ。isotonic は 0/1 ちょうどを出し得るため log loss 用に微小量クリップする。",
        "methods": {},
    }
    for method in methods:
        calib_raw = _load(repo_root / "results" / "raw_calib" / f"{stem}__{method}.jsonl")
        test_raw = _load(repo_root / "results" / "raw" / f"{stem}__{method}.jsonl")
        if not calib_raw or not test_raw:
            report["methods"][method] = {"skipped": "raw が無い(calib / test のどちらか未実行)"}
            continue
        # 確率として定義されていない方式は較正の対象外
        value_types = {r.get("value_type") for r in list(calib_raw.values()) + list(test_raw.values())
                       if not r.get("error_type")}
        if value_types != {"probability"}:
            report["methods"][method] = {"skipped": f"value_type={sorted(value_types)} は較正対象外"}
            continue
        p_c, y_c = _collect_pairs(calib_raw, calib_items, risk_severity)
        p_t, y_t = _collect_pairs(test_raw, test_items, risk_severity)
        if not p_c or not p_t:
            report["methods"][method] = {"skipped": "有効なリスク確率が揃わない"}
            continue
        report["methods"][method] = recalibration_report(p_c, y_c, p_t, y_t, n_bins=n_bins)
    return report


def render_markdown(report: dict) -> str:
    lines = ["# 再較正の前後(test)", ""]
    lines.append(f"- 正例の定義: severity = `{report['risk_severity']}`")
    lines.append("- 学習は calib のみ。test は評価だけに使う。")
    lines.append("- 対象は value_type=probability の方式のみ(LLM の自己申告確率は対象外)。")
    lines.append("")
    lines.append("| method | 温度 | 指標 | 前 | 温度後 | isotonic 後 |")
    lines.append("|---|---|---|---|---|---|")
    for method, m in report["methods"].items():
        if "before" not in m:
            lines.append(f"| {method} | - | - | {m.get('skipped', m.get('note', '-'))} | - | - |")
            continue
        t = m["temperature"]
        for metric in ("ece", "brier", "log_loss"):
            lines.append(
                f"| {method} | {t:.3f} | {metric} | {_fmt(m['before'].get(metric))} | "
                f"{_fmt(m['after_temperature'].get(metric))} | {_fmt(m['after_isotonic'].get(metric))} |"
            )
    lines.append("")
    lines.append("温度・isotonic はどちらも calib で学習した写像。isotonic は単調なので順位(AUROC)は変わらない。")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="再較正の前後を test で比較する")
    parser.add_argument("--methods", default=",".join(PROBABILITY_METHODS))
    parser.add_argument("--out", default="results/recalibration.json")
    args = parser.parse_args(argv)

    report = collect(methods=[m.strip() for m in args.methods.split(",") if m.strip()])
    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
