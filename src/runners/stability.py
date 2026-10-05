"""安定性テスト: 同じ入力を N 回実行し、確率のばらつきと判定反転率を出す。

仕様どおり全件ではなくサブセット(既定 50 件 × 20 回)で行う。無料枠の消費を抑えるため、
calib/test の本番評価とは別に、小さなサブセットで実行する。

使い方:
  python -m runners.stability --methods rule,embedding_lr,llm_prompt,llm_json_schema,pplx_decider \
      --subset 50 --runs 20 --out results/stability.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import REPO_ROOT, load_config
from evaluation.stability import summarize
from runners.benchmark import run_benchmark


def _raw_path(raw_dir: Path, stem: str, method: str, run_index: int) -> Path:
    suffix = f"__run{run_index}" if run_index else ""
    return raw_dir / f"{stem}__{method}{suffix}.jsonl"


def collect_runs(
    *,
    dataset: Path,
    methods: list[str],
    subset: int,
    runs: int,
    repo_root: Path = REPO_ROOT,
    raw_dir: Path | None = None,
    concurrency: int | None = None,
    train_on: str | None = None,
) -> dict:
    config = load_config(repo_root=repo_root)
    raw_dir = raw_dir or repo_root / "results" / "raw_stability"
    stem = dataset.stem
    for run_index in range(runs):
        run_benchmark(
            dataset_path=dataset, adapter_names=methods, config=config, repo_root=repo_root,
            raw_dir=raw_dir, run_index=run_index, limit=subset, concurrency=concurrency,
            train_on=train_on,
        )
    # 集計
    thresholds = {}
    tpath = repo_root / "config" / "thresholds.json"
    if tpath.exists():
        thresholds = json.loads(tpath.read_text(encoding="utf-8")).get("methods", {})

    report: dict = {"dataset": str(dataset), "subset": subset, "runs": runs, "methods": {}}
    for method in methods:
        per_item: dict[str, list[dict]] = {}
        for run_index in range(runs):
            path = _raw_path(raw_dir, stem, method, run_index)
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                per_item.setdefault(rec["item_id"], []).append(rec)
        th = thresholds.get(method) or {}
        report["methods"][method] = summarize(per_item, thresholds=th)
    return report


def render_markdown(report: dict) -> str:
    lines = ["# 安定性テスト", ""]
    lines.append(f"- データ: `{report['dataset']}` の先頭 {report['subset']} 件")
    lines.append(f"- 反復回数: {report['runs']}")
    lines.append("")
    lines.append("| method | 項目数 | 確率の平均SD | 確率の平均IQR | 閾値反転(平均) | 反転した項目 | 常に不確実(≥0.3) | ラベル不一致率 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for method, m in report["methods"].items():
        prob = m.get("probability") or {}
        flip = m.get("threshold_flip") or {}
        lines.append(
            f"| {method} | {m.get('n_items', 0)} | {_fmt(prob.get('mean_std'))} | "
            f"{_fmt(prob.get('mean_iqr'))} | {_fmt(flip.get('mean_flip_fraction'))} | "
            f"{flip.get('items_ever_flipping', '-')} | {flip.get('items_always_uncertain', '-')} | "
            f"{_fmt(m.get('label_disagreement_rate'))} |"
        )
    lines.append("")
    lines.append("確率を返さない方式(rule)はラベル不一致率のみ。閾値は calib 由来(`config/thresholds.json`)。")
    return "\n".join(lines)


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def collect_variants(
    *,
    dataset: Path,
    methods: list[str],
    subset: int,
    repo_root: Path = REPO_ROOT,
    raw_dir: Path | None = None,
    concurrency: int | None = None,
    train_on: str | None = None,
    task_path: str | Path = "config/tasks/support_classification.toml",
) -> dict:
    """言い換え・選択肢順序の入れ替えに対する判定の変動を測る。

    同じアイテムを、指示文を言い換えた版・カテゴリの並びを回転した版でも分類し、
    変異版間でラベルがどこまで一致するかを見る。run_index を 100+ にして
    キャッシュキーを本番評価と分ける(キャッシュキーには run_index が入る)。
    """
    from tasks.base import load_task_variants

    config = load_config(repo_root=repo_root)
    raw_dir = raw_dir or repo_root / "results" / "raw_variants"
    stem = dataset.stem
    variants = load_task_variants(repo_root / task_path)

    for index, (_name, vtask) in enumerate(variants.items()):
        run_benchmark(
            dataset_path=dataset, adapter_names=methods, config=config, repo_root=repo_root,
            raw_dir=raw_dir, run_index=100 + index, limit=subset, concurrency=concurrency,
            train_on=train_on, task=vtask,
        )

    thresholds = {}
    tpath = repo_root / "config" / "thresholds.json"
    if tpath.exists():
        thresholds = json.loads(tpath.read_text(encoding="utf-8")).get("methods", {})

    report: dict = {"dataset": str(dataset), "subset": subset,
                    "variants": list(variants.keys()), "methods": {}}
    for method in methods:
        per_item: dict[str, list[dict]] = {}
        for index, name in enumerate(variants):
            path = _raw_path(raw_dir, stem, method, 100 + index)
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["variant"] = name
                per_item.setdefault(rec["item_id"], []).append(rec)
        th = thresholds.get(method) or {}
        summary = summarize(per_item, thresholds=th)
        # 変異版間のラベル一致(項目ごとに全変異版で同じラベルか)
        disagreeing = []
        for item_id, runs in per_item.items():
            labels = {r.get("label") for r in runs if r.get("label") is not None}
            if len(labels) > 1:
                disagreeing.append(item_id)
        summary["label_disagreement_items"] = len(disagreeing)
        summary["label_disagreement_rate"] = (
            len(disagreeing) / len(per_item) if per_item else None
        )
        report["methods"][method] = summary
    return report


def render_variants_markdown(report: dict) -> str:
    lines = ["# 安定性テスト(言い換え・選択肢順序)", ""]
    lines.append(f"- データ: `{report['dataset']}` の先頭 {report['subset']} 件")
    lines.append(f"- 変異版: {', '.join(report['variants'])}")
    lines.append("")
    lines.append("| method | 変異版間でラベル不一致 | 確率の平均SD | 閾値反転(平均) |")
    lines.append("|---|---|---|---|")
    for method, m in report["methods"].items():
        prob = m.get("probability") or {}
        flip = m.get("threshold_flip") or {}
        lines.append(
            f"| {method} | {m.get('label_disagreement_items', '-')} / {m.get('n_items', 0)} "
            f"({_fmt(m.get('label_disagreement_rate'))}) | {_fmt(prob.get('mean_std'))} | "
            f"{_fmt(flip.get('mean_flip_fraction'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="同じ入力を N 回実行して安定性を測る")
    parser.add_argument("--dataset", default="data/test/support_classification.jsonl")
    parser.add_argument("--methods", default="rule,embedding_lr,llm_prompt,llm_json_schema,pplx_decider")
    parser.add_argument("--subset", type=int, default=50)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--mode", choices=["runs", "variants", "both"], default="runs",
                        help="runs=同一入力の反復 / variants=言い換え・選択肢順序 / both")
    parser.add_argument("--raw-dir", default="results/raw_stability")
    parser.add_argument("--variant-raw-dir", default="results/raw_variants")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--train-on", default="data/calib/support_classification.jsonl",
                        help="embedding_lr の学習データ(既定 calib)")
    parser.add_argument("--out", default="results/stability.json")
    args = parser.parse_args(argv)

    dataset = Path(REPO_ROOT / args.dataset)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ("runs", "both"):
        report = collect_runs(
            dataset=dataset, methods=methods, subset=args.subset, runs=args.runs,
            raw_dir=Path(REPO_ROOT / args.raw_dir), concurrency=args.concurrency,
            train_on=args.train_on,
        )
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.mode in ("variants", "both"):
        vreport = collect_variants(
            dataset=dataset, methods=methods, subset=args.subset,
            raw_dir=Path(REPO_ROOT / args.variant_raw_dir), concurrency=args.concurrency,
            train_on=args.train_on,
        )
        vout = out.with_name(out.stem + "_variants.json")
        vout.write_text(json.dumps(vreport, ensure_ascii=False, indent=2), encoding="utf-8")
        vout.with_suffix(".md").write_text(render_variants_markdown(vreport), encoding="utf-8")
        print(json.dumps(vreport, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
