"""キャッシュから生応答(JSONL)を復元する。

benchmark は生応答を `results/raw/<dataset_stem>__<method>.jsonl` に書くため、
calib と test が同じ stem だと後から実行した方が上書きしてしまう。
その場合でも結果キャッシュ(prompt fingerprint + dataset fingerprint をキーに持つ)には
全件が残っているので、ここから任意のデータセットぶんを復元できる。

使い方:
  python -m runners.recover_raw --dataset data/calib/support_classification.jsonl \
      --out results/raw_calib --methods rule,embedding_lr,mercury,llm_prompt,llm_json_schema,pplx_decider
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters.registry import build_adapters  # noqa: F401  (型・依存の明示のため)
from common.cache import ResultCache, cache_key, dataset_fingerprint, prompt_fingerprint
from common.config import REPO_ROOT, load_config
from common.dataset import load_jsonl
from tasks.base import load_task


def adapter_model_id(method: str, config) -> str:
    """cache_key に使われる model 文字列を方式ごとに再現する(benchmark と同じ規則)。"""
    if method == "rule":
        return "keyword-rules"
    if method == "embedding_lr":
        return str(config.get("embedding.model"))
    if method == "mercury":
        return str(config.get("models.mercury.model"))
    if method in ("llm_prompt", "llm_json_schema"):
        return str(config.get("models.llm.model"))
    if method == "pplx_decider":
        return str(config.get("models.pplx_decider.model"))
    if method == "jev":
        return str(config.get("models.jev.model"))
    raise KeyError(f"未知の方式: {method}")


def recover(
    *,
    dataset: Path,
    out_dir: Path,
    methods: list[str],
    repo_root: Path = REPO_ROOT,
    run_index: int = 0,
) -> dict:
    config = load_config(repo_root=repo_root)
    task = load_task(repo_root / "config" / "tasks" / "support_classification.toml")
    cache = ResultCache(repo_root / "results" / "cache" / "support_classification.jsonl")
    items = load_jsonl(dataset)
    variant = f"{prompt_fingerprint(task.instruction, task.choice_instruction)}-{dataset_fingerprint(items)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"dataset": str(dataset), "n_items": len(items), "methods": {}}
    for method in methods:
        model = adapter_model_id(method, config)
        records, missing = [], 0
        for item in items:
            rec = cache.get(cache_key(method, model, task.id, item.id, run_index, variant))
            if rec is None:
                missing += 1
            else:
                records.append(rec)
        path = out_dir / f"{dataset.stem}__{method}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ok = sum(1 for r in records if not r.get("error_type"))
        report["methods"][method] = {"recovered": len(records), "ok": ok, "missing": missing, "out": str(path)}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="キャッシュから生応答を復元する")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="results/raw_calib")
    parser.add_argument("--methods", default="rule,embedding_lr,mercury,llm_prompt,llm_json_schema,pplx_decider")
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args(argv)

    report = recover(
        dataset=Path(REPO_ROOT / args.dataset),
        out_dir=Path(REPO_ROOT / args.out),
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        run_index=args.run_index,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(v["missing"] == 0 for v in report["methods"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
