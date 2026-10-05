"""スループットの実測: 同じ並列数で N 件を流し、壁時計時間から件/秒を出す。

レイテンシ(p50/p95)は report が出すが、スループットは並列数に依存するため
「同一の並列数で実際に流した時間」を測る必要がある。キャッシュを避けるため
run_index を専用の値にして、必ず新規に呼び出す。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common.config import REPO_ROOT, load_config
from runners.benchmark import run_benchmark

FRESH_RUN_INDEX = 900  # 安定性(0..N)・変異版(100+i)と衝突しない専用の値


def measure(
    *,
    dataset: Path,
    methods: list[str],
    n_items: int,
    concurrency: int,
    repo_root: Path = REPO_ROOT,
    raw_dir: Path | None = None,
    train_on: str | None = None,
    allow_paid_models: bool = False,
    run_index: int | None = None,
) -> dict:
    config = load_config(repo_root=repo_root)
    raw_dir = raw_dir or repo_root / "results" / "raw_throughput"
    # キャッシュに当たると「速すぎる」数字が出る(実測ではなくなる)。
    # 既定は時刻から作る一意な run_index にして、必ず新規に呼び出す。
    if run_index is None:
        run_index = FRESH_RUN_INDEX + (int(time.time()) % 1_000_000)
    results: dict = {"dataset": str(dataset), "n_items": n_items, "concurrency": concurrency,
                     "run_index": run_index, "methods": {}}
    for method in methods:
        started = time.perf_counter()
        summary = run_benchmark(
            dataset_path=dataset, adapter_names=[method], config=config, repo_root=repo_root,
            raw_dir=raw_dir, run_index=run_index, limit=n_items,
            concurrency=concurrency, train_on=train_on,
            allow_paid_models=allow_paid_models,
        )
        wall = time.perf_counter() - started
        info = summary["methods"][method]
        ok = int(info.get("ok", 0))
        results["methods"][method] = {
            "ok": ok,
            "errors": int(info.get("errors", 0)),
            "wall_seconds": wall,
            "items_per_second": (ok / wall) if wall > 0 else None,
        }
    return results


def render_markdown(report: dict) -> str:
    lines = ["# スループット実測", ""]
    lines.append(f"- データ: `{report['dataset']}` の先頭 {report['n_items']} 件")
    lines.append(f"- 並列数: {report['concurrency']}(全方式で統一)")
    lines.append(f"- run_index: {report.get('run_index')}(キャッシュに当たらない一意な値)")
    lines.append("- キャッシュを避けるため専用の run_index で新規に呼び出した実測値。")
    lines.append("")
    lines.append("| method | 成功 | エラー | 壁時計秒 | 件/秒 |")
    lines.append("|---|---|---|---|---|")
    for method, m in report["methods"].items():
        lines.append(
            f"| {method} | {m['ok']} | {m['errors']} | {m['wall_seconds']:.2f} | "
            f"{_fmt(m['items_per_second'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{float(v):.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="同一並列数でのスループットを実測する")
    parser.add_argument("--dataset", default="data/test/support_classification.jsonl")
    parser.add_argument("--methods", default="llm_prompt,llm_json_schema,pplx_decider,jev")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--raw-dir", default="results/raw_throughput")
    parser.add_argument("--train-on", default="data/calib/support_classification.jsonl")
    parser.add_argument("--allow-paid-models", action="store_true", help="jev を含めるときに必要")
    parser.add_argument("--run-index", type=int, default=None,
                        help="キャッシュ回避用。既定は時刻から作る一意な値")
    parser.add_argument("--out", default="results/throughput.json")
    args = parser.parse_args(argv)

    report = measure(
        dataset=Path(REPO_ROOT / args.dataset),
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        n_items=args.n, concurrency=args.concurrency,
        raw_dir=Path(REPO_ROOT / args.raw_dir), train_on=args.train_on,
        allow_paid_models=args.allow_paid_models, run_index=args.run_index,
    )
    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
