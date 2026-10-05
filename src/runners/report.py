"""集計とレポート生成。

保存済みの生応答(results/raw/*.jsonl)から指標を再計算し、results/report.md を出力する。
数値はすべて生データから計算する(手計算値を載せない)。
確率として扱うのは value_type == "probability" の値のみ。LLM の自己申告確率は
較正の対象にせず、AUROC(誤り検出力)等で別に評価する。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common.config import REPO_ROOT, load_config
from common.dataset import Item, load_jsonl
from evaluation.bootstrap import bootstrap_ci
from evaluation.calibration import (
    auroc,
    brier_score,
    ece,
    log_loss,
    probability_extremity,
    reliability_bins,
)
from evaluation.classification import (
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    summaries_by_flag,
)
from evaluation.risk_coverage import fixed_recall_report, risk_coverage_curve

PROBABILITY_VALUE_TYPES = ("probability",)


def load_raw(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["item_id"]] = rec
    return out


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def evaluate_method(records: dict[str, dict], items: dict[str, Item], *, risk_severity: str = "high") -> dict:
    ok = [r for r in records.values() if not r.get("error_type")]
    errors = Counter(r.get("error_type") for r in records.values() if r.get("error_type"))
    gold = [items[r["item_id"]].label for r in ok if r["item_id"] in items]
    pred = [r.get("label") for r in ok if r["item_id"] in items]
    labels = sorted({*gold, *[p for p in pred if p]})
    result: dict = {
        "n_records": len(records),
        "n_ok": len(ok),
        "errors": dict(errors),
        "parse_failures": errors.get("parse", 0),
    }
    if gold:
        result["accuracy"] = float(np.mean([g == p for g, p in zip(gold, pred)]))
        result["macro_f1"] = macro_f1(gold, pred, labels)
        result["per_class"] = {
            k: {kk: vv for kk, vv in v.items() if kk in ("precision", "recall", "f1", "support")}
            for k, v in per_class_metrics(gold, pred, labels).items()
        }
        result["confusion_matrix"] = {
            "labels": labels,
            "matrix": confusion_matrix(gold, pred, labels).tolist(),
        }
        # 曖昧な文は通常評価とは別集計(ユーザー指示)
        flags = [items[r["item_id"]].ambiguous for r in ok if r["item_id"] in items]
        result["ambiguous_split"] = summaries_by_flag(gold, pred, flags)
        # gold_label と severity の組み合わせ: 誤分類が高リスク案件に当たっているか
        wrong = [(items[r["item_id"]].label, items[r["item_id"]].severity)
                 for r in ok if r["item_id"] in items and r.get("label") != items[r["item_id"]].label]
        result["error_severity"] = {
            "n_wrong": len(wrong),
            "wrong_high": sum(1 for _, sev in wrong if sev == risk_severity),
            "cross_tab_gold_severity": {
                f"{label}|{sev}": count
                for (label, sev), count in sorted(
                    Counter((items[r["item_id"]].label, items[r["item_id"]].severity)
                            for r in ok if r["item_id"] in items).items()
                )
            },
        }
    # 高リスク検出(severity=high を陽性とする)
    p_risk: list[float] = []
    y_risk: list[int] = []
    for r in ok:
        item = items.get(r["item_id"])
        if item is None or r.get("risk_score") is None:
            continue
        p_risk.append(float(r["risk_score"]))
        y_risk.append(1 if item.severity == risk_severity else 0)
    if p_risk and len(set(y_risk)) == 2:
        value_types = {r.get("value_type") for r in ok if r.get("risk_score") is not None}
        result["risk_detection"] = {
            "n": len(p_risk),
            "positives": int(sum(y_risk)),
            "auroc": auroc(p_risk, y_risk),
            "auroc_ci": bootstrap_ci(auroc, [np.array(p_risk), np.array(y_risk)]),
        }
        if value_types <= set(PROBABILITY_VALUE_TYPES):
            result["calibration"] = {
                "ece": ece(p_risk, y_risk, n_bins=10),
                "brier": brier_score(p_risk, y_risk),
                "log_loss": log_loss(p_risk, y_risk),
                "reliability_bins": reliability_bins(p_risk, y_risk, n_bins=10),
                "extremity": probability_extremity(p_risk),
            }
        result["threshold_0_5"] = risk_coverage_curve(p_risk, y_risk, thresholds=[0.5])[0]
    # レイテンシ・コスト
    latencies = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]
    result["latency_ms"] = {
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
        "mean": float(np.mean(latencies)) if latencies else None,
    }
    costs = [r.get("estimated_cost_usd") for r in ok if r.get("estimated_cost_usd") is not None]
    if costs:
        result["cost_per_1000_usd"] = float(np.mean(costs) * 1000)
    return result


def build_report(
    *,
    dataset_path: Path,
    raw_dir: Path,
    methods: list[str],
    config,
    calib_raw_dir: Path | None = None,
    calib_dataset_path: Path | None = None,
    hybrid_paths: list[Path] | None = None,
    throughput_path: Path | None = None,
    out_path: Path | None = None,
) -> dict:
    stem = dataset_path.stem
    items = {it.id: it for it in load_jsonl(dataset_path)}
    # calib の生応答は calib のラベル・severity で評価する(test の辞書で引くと全件 None になる)
    calib_items = {it.id: it for it in load_jsonl(calib_dataset_path)} if calib_dataset_path else items
    risk_severity = str(config.get("evaluation.risk_severity", "high"))
    target_recall = float(config.get("evaluation.recall_target", 0.95))

    results: dict[str, dict] = {}
    for method in methods:
        records = load_raw(raw_dir / f"{stem}__{method}.jsonl")
        if not records:
            continue
        results[method] = evaluate_method(records, items, risk_severity=risk_severity)
        # 固定 Recall(calib の閾値を test に適用)。calib が無い場合は同一集合を使い警告する
        calib_records = load_raw((calib_raw_dir or raw_dir) / f"{stem}__{method}.jsonl")
        p_c, y_c, p_t, y_t, lt, lp = [], [], [], [], [], []
        for iid, r in calib_records.items():
            item = calib_items.get(iid)
            if item is None or r.get("risk_score") is None or r.get("error_type"):
                continue
            p_c.append(float(r["risk_score"]))
            y_c.append(1 if item.severity == risk_severity else 0)
        for iid, r in records.items():
            item = items.get(iid)
            if item is None:
                continue
            if r.get("risk_score") is None or r.get("error_type"):
                continue
            p_t.append(float(r["risk_score"]))
            y_t.append(1 if item.severity == risk_severity else 0)
            lt.append(item.label)
            lp.append(r.get("label"))
        if p_c and p_t and len(set(y_c)) == 2:
            results[method]["fixed_recall"] = fixed_recall_report(
                p_c, y_c, p_t, y_t, target=target_recall, labels_true_test=lt, labels_pred_test=lp
            )
    aggregate = {
        "dataset": str(dataset_path),
        "n_items": len(items),
        "risk_severity": risk_severity,
        "recall_target": target_recall,
        "calib_source": "separate" if calib_raw_dir else "same_as_test (smoke only)",
        "methods": results,
        "hybrid": _load_hybrid(hybrid_paths or []),
        "throughput": _load_json_file(throughput_path),
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return aggregate


def render_markdown(aggregate: dict) -> str:
    lines: list[str] = ["# 評価レポート(集計)", ""]
    lines.append(f"- データ: `{aggregate['dataset']}`({aggregate['n_items']} 件)")
    lines.append(f"- 高リスクの定義: severity = `{aggregate['risk_severity']}`")
    lines.append(f"- 目標 Recall: {aggregate['recall_target']}")
    lines.append(f"- 閾値の出所: {aggregate['calib_source']}")
    lines.append("")
    lines.append("## 1. 分類品質")
    lines.append("")
    lines.append("| method | n | accuracy | macro F1 | urgent_claim P/R/F1 |")
    lines.append("|---|---|---|---|---|")
    for method, m in aggregate["methods"].items():
        uc = (m.get("per_class") or {}).get("urgent_claim", {})
        f1 = m.get("macro_f1")
        lines.append(
            f"| {method} | {m['n_ok']} | {_fmt(m.get('accuracy'))} | {_fmt(f1)} | "
            f"{_fmt(uc.get('precision'))}/{_fmt(uc.get('recall'))}/{_fmt(uc.get('f1'))} |"
        )
    lines.append("")
    lines.append("## 2. 確率の較正")
    lines.append("")
    lines.append("| method | ECE | Brier | log loss | AUROC |")
    lines.append("|---|---|---|---|---|")
    for method, m in aggregate["methods"].items():
        cal = m.get("calibration") or {}
        rd = m.get("risk_detection") or {}
        lines.append(
            f"| {method} | {_fmt(cal.get('ece'))} | {_fmt(cal.get('brier'))} | {_fmt(cal.get('log_loss'))} | "
            f"{_fmt(rd.get('auroc'))} |"
        )
    lines.append("")
    lines.append("(ECE/Brier/log loss は value_type=probability の方式のみ。LLM の自己申告確率は較正の対象外。AUROC は全方式)")
    lines.append("")
    lines.append("### 曖昧な文(ambiguous)の別集計と、誤りの重要度")
    lines.append("")
    lines.append("| method | overall acc | ambiguous acc (n) | 非ambiguous acc | 誤り件数 | うち severity=high |")
    lines.append("|---|---|---|---|---|---|")
    for method, m in aggregate["methods"].items():
        split = m.get("ambiguous_split") or {}
        flagged = split.get("flagged") or {}
        err = m.get("error_severity") or {}
        lines.append(
            f"| {method} | {_fmt((split.get('all') or {}).get('accuracy'))} | "
            f"{_fmt(flagged.get('accuracy'))} ({flagged.get('n', 0)}) | "
            f"{_fmt((split.get('unflagged') or {}).get('accuracy'))} | {err.get('n_wrong', '-')} | "
            f"{err.get('wrong_high', '-')} |"
        )
    lines.append("")
    lines.append("## 3. 判断効率(固定 Recall)")
    lines.append("")
    lines.append("| method | calib 閾値 | test recall | recall 95%CI下限 | test 自動化率 | 自動化率 95%CI下限 | 自動処理の誤り率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for method, m in aggregate["methods"].items():
        fr = m.get("fixed_recall")
        if not fr:
            continue
        t = fr["test"]
        rci = (t.get("recall_high_ci") or {})
        aci = (t.get("automation_rate_ci") or {})
        lines.append(
            f"| {method} | {_fmt(fr['threshold_from_calib'])} | {_fmt(t.get('recall_high'))} | "
            f"{_fmt(rci.get('lo'))} | {_fmt(t.get('automation_rate'))} | {_fmt(aci.get('lo'))} | "
            f"{_fmt(t.get('auto_error_rate'))} |"
        )
    lines.append("")
    lines.append("CI はブートストラップ 1000 回(パーセンタイル法)。閾値は calib の生応答から決め、test では探索しない。")
    lines.append("")
    hybrid = aggregate.get("hybrid") or {}
    if hybrid:
        lines.append("### Hybrid(ルール → 判断モデル → 閾値 → auto / review / block)")
        lines.append("")
        lines.append("閾値は calib 由来。recall_high は auto に回した分だけで測った高リスク recall(見逃し=auto に入った severity=high)。")
        lines.append("")
        lines.append("| 構成 | 自動化率 | 人間レビュー率 | block 率 | 高リスク recall(auto) | 見逃し | 自動処理の誤り率 |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, h in hybrid.items():
            lines.append(
                f"| {name} | {_fmt(h.get('automation_rate'))} | {_fmt(h.get('review_rate'))} | "
                f"{_fmt(h.get('block_rate'))} | {_fmt(h.get('recall_high_at_auto'))} | "
                f"{h.get('auto_high_missed', '-')} | {_fmt(h.get('auto_error_rate'))} |"
            )
        lines.append("")
    lines.append("## 4. 運用性能(レイテンシ・コスト)")
    lines.append("")
    lines.append("| method | p50 ms | p95 ms | 推定コスト / 1000件 |")
    lines.append("|---|---|---|---|")
    for method, m in aggregate["methods"].items():
        lat = m.get("latency_ms") or {}
        lines.append(
            f"| {method} | {_fmt(lat.get('p50'), 1)} | {_fmt(lat.get('p95'), 1)} | "
            f"{_fmt(m.get('cost_per_1000_usd'), 4)} |"
        )
    lines.append("")
    throughput = aggregate.get("throughput")
    if throughput:
        lines.append(f"### スループット(実測、並列 {throughput.get('concurrency')}、生応答から計算せず実時間で測定)")
        lines.append("")
        lines.append("| method | 成功 | エラー | 壁時計秒 | 件/秒 |")
        lines.append("|---|---|---|---|---|")
        for method, t in (throughput.get("methods") or {}).items():
            lines.append(
                f"| {method} | {t.get('ok')} | {t.get('errors')} | {_fmt(t.get('wall_seconds'), 2)} | "
                f"{_fmt(t.get('items_per_second'), 2)} |"
            )
        lines.append("")
        lines.append("p50/p95 は 1 件あたりのレイテンシ、スループットは同じ並列数で実際に流した件/秒。"
                     "レイテンシが同じでもスループットは並列数とレート制限に依存する。")
        lines.append("")
    lines.append("## 5. 信頼性(エラー率・パース失敗率)")
    lines.append("")
    lines.append("| method | 記録数 | 成功 | パース失敗 | エラー内訳 |")
    lines.append("|---|---|---|---|---|")
    for method, m in aggregate["methods"].items():
        lines.append(
            f"| {method} | {m['n_records']} | {m['n_ok']} | {m['parse_failures']} | "
            f"{json.dumps(m['errors'], ensure_ascii=False) if m['errors'] else '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_hybrid(paths: list[Path]) -> dict:
    """hybrid_eval が出力した JSON を読む。キーはファイル名から作る。"""
    out: dict = {}
    for path in paths:
        if not Path(path).exists():
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        name = Path(path).stem.replace("hybrid_", "").replace("support_classification", "").strip("_")
        out[name] = data
    return out


def _load_json_file(path: Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(value, nd: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生応答から指標を再計算しレポートを出力する")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--calib-raw-dir", default=None)
    parser.add_argument("--calib-dataset", default="data/calib/support_classification.jsonl",
                        help="calib の正解データ(calib 生応答の評価に使う)")
    parser.add_argument("--methods", default="rule,mercury,llm_prompt,llm_json_schema,pplx_decider,embedding_lr")
    parser.add_argument("--hybrid", default=None,
                        help="hybrid_eval の出力 JSON(カンマ区切り)。第3章に Hybrid 表を足す")
    parser.add_argument("--throughput", default=None,
                        help="throughput の出力 JSON。第4章にスループット表を足す")
    parser.add_argument("--out", default="results/report.md")
    args = parser.parse_args(argv)

    config = load_config(repo_root=REPO_ROOT)
    dataset_path = Path(args.dataset)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    aggregate = build_report(
        dataset_path=dataset_path,
        raw_dir=Path(args.raw_dir),
        methods=methods,
        config=config,
        calib_raw_dir=Path(args.calib_raw_dir) if args.calib_raw_dir else None,
        calib_dataset_path=Path(REPO_ROOT / args.calib_dataset) if args.calib_dataset else None,
        hybrid_paths=[Path(REPO_ROOT / p.strip()) for p in args.hybrid.split(",") if p.strip()] if args.hybrid else None,
        throughput_path=Path(REPO_ROOT / args.throughput) if args.throughput else None,
        out_path=REPO_ROOT / "results" / "aggregate" / f"{dataset_path.stem}.json",
    )
    markdown = render_markdown(aggregate)
    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    print(json.dumps({m: {"accuracy": d.get("accuracy"), "auroc": (d.get("risk_detection") or {}).get("auroc"),
                          "errors": d.get("errors")} for m, d in aggregate["methods"].items()},
                     ensure_ascii=False, indent=2))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
