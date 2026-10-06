"""Jev のドライラン: 実 API を呼ばず、リクエスト形と推定コストを提示する。

同スキーマの先行実測で得た入力トークン数を渡すと概算が締まる。
本番実行の前に、この出力を使ってユーザーの承認を取る。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters.jev import JevAdapter
from common.config import REPO_ROOT, load_config
from common.dataset import load_jsonl
from tasks.base import load_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jev アダプタのドライラン(API を呼ばない)")
    parser.add_argument("--dataset", default="data/synthetic/support_classification.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--token-estimate", type=int, default=None,
                        help="1リクエストあたりの入力トークン数の実測値(例: mercury の実測)")
    parser.add_argument("--out", default="results/jev_dry_run.json")
    args = parser.parse_args(argv)

    config = load_config(repo_root=REPO_ROOT)
    task = load_task(REPO_ROOT / "config" / "tasks" / "support_classification.toml")
    items = load_jsonl(REPO_ROOT / args.dataset)
    if args.limit:
        items = items[: args.limit]

    adapter = JevAdapter(task, config, key="(dry-run: not used)", http=None, dry_run=True,
                         token_estimate=args.token_estimate)
    report = adapter.dry_run_report(items)
    report["dataset"] = args.dataset
    report["task"] = task.id
    out_path = Path(REPO_ROOT / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
