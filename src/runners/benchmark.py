"""benchmark runner: 同一データ・同一条件で全方式を実行し、生応答を JSONL に残す。

- 並列数・timeout・リトライは config で統一
- 結果は (method, model, task, item, run_index) でキャッシュし、再開できる
- 無料枠の日次上限に達した方式は pending にして、他の方式・作業を止めない
- 全リクエストを監査ログ(入力・確率・リクエストID・実行日時・モデルID)に残す
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters.registry import FREE_ADAPTERS, PAID_ADAPTERS, build_adapters, build_http
from common.cache import AuditLog, ResultCache, cache_key, dataset_fingerprint, prompt_fingerprint
from common.dataset import Item, external_send_allowed, load_jsonl, load_manifest
from common.env import load_keys
from common.ratelimit import DailyQuota, DailyQuotaExhausted


def _predict_one(adapter, item: Item, run_index: int, cache: ResultCache, variant: str = ""):
    key = cache_key(adapter.name if not hasattr(adapter, "method_name") else adapter.method_name,
                    adapter.model, adapter.task.id, item.id, run_index, variant)
    cached = cache.get(key)
    if cached is not None:
        record = dict(cached)
        record["cached"] = True
        return record
    prediction = adapter.predict(item, run_index=run_index)
    record = prediction.as_dict()
    record["method"] = getattr(adapter, "method_name", adapter.name)
    cache.put(key, record)
    return record


def run_benchmark(
    *,
    dataset_path: str | Path,
    adapter_names: list[str],
    config,
    repo_root: Path,
    out_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    run_index: int = 0,
    limit: int | None = None,
    concurrency: int | None = None,
    confirm_external: bool = False,
    allow_paid_models: bool = False,
    task=None,
    train_on: str | Path | None = None,
) -> dict:
    dataset_path = Path(dataset_path)
    out_root = Path(out_dir) if out_dir else repo_root / "results"
    # 生応答の出力先。calib と test はファイル名が衝突し得るため、
    # --raw-dir で分離できるようにする(results/raw と results/raw_calib)。
    raw_dir = Path(raw_dir) if raw_dir else out_root / "raw"
    cache_dir = out_root / "cache"
    audit_dir = out_root / "audit"
    pending_dir = out_root / "pending"
    for d in (raw_dir, cache_dir, audit_dir, pending_dir):
        d.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(dataset_path)
    if limit:
        items = items[:limit]
    manifest = load_manifest(dataset_path.parent)
    external_allowed = external_send_allowed(manifest, confirm_external=confirm_external)

    if task is None:
        from tasks.base import load_task

        task = load_task(repo_root / "config" / "tasks" / f"{manifest.get('task', 'support_classification')}.toml")

    http = build_http(config)
    quota = DailyQuota(
        daily_limit=int(config.get("free_tier.daily_limit", 1000)),
        stop_on_exhaustion=bool(config.get("free_tier.stop_on_exhaustion", False)),
    )
    keys = load_keys()
    # 無料枠はアカウント共有。実際の残量を OpenRouter から取得して上限に反映する(無駄な 429 を避ける)
    free_remaining = None
    if keys.get("OPENROUTER_API_KEY") and any(n in FREE_ADAPTERS for n in adapter_names):
        from common.ratelimit import fetch_free_model_remaining

        free_remaining = fetch_free_model_remaining(keys["OPENROUTER_API_KEY"])
        if free_remaining is not None and quota.daily_limit is not None:
            quota.daily_limit = min(quota.daily_limit, free_remaining)
    cache = ResultCache(cache_dir / f"{dataset_path.stem}.jsonl")
    audit = AuditLog(audit_dir / f"{dataset_path.stem}.jsonl")
    adapters = build_adapters(
        config, task, adapter_names, repo_root=repo_root, http=http, cache=cache, quota=quota,
        audit=audit, allow_paid_models=allow_paid_models, keys=keys,
    )

    workers = int(concurrency or config.get("run.concurrency", 4))
    # キャッシュキーには指示文とデータ内容のハッシュを含める(データ差し替え・プロンプト変更で無効化)
    data_fp = dataset_fingerprint(items)
    variant = f"{prompt_fingerprint(task.instruction, task.choice_instruction)}-{data_fp}"

    # ローカルの学習型アダプタ(embedding_lr)は指定データで学習する(実運用では calib を渡す)
    if train_on is not None:
        train_items = load_jsonl(train_on)
        # 学習データと評価データが同一(calib 自身)のときは out-of-fold 予測を使い、学習リークを避ける
        same_set = Path(train_on).resolve() == dataset_path.resolve()
        for adapter in adapters.values():
            if hasattr(adapter, "train"):
                adapter.train(train_items, oof=same_set)
        train_on_note = {"path": str(train_on), "oof": same_set}
    else:
        train_on_note = None
    summary: dict = {"dataset": str(dataset_path), "n_items": len(items), "prompt_fingerprint": variant,
                     "data_fingerprint": data_fp, "train_on": train_on_note,
                     "free_tier_remaining_at_start": free_remaining, "methods": {}}

    for name, adapter in adapters.items():
        external = adapter.tier in ("free", "paid")
        if external and not external_allowed:
            summary["methods"][name] = {"skipped": "external_not_confirmed", "tier": adapter.tier}
            continue
        method_name = getattr(adapter, "method_name", adapter.name)
        records: list[dict] = []
        pending: list[str] = []
        quota_exhausted = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for item in items:
                if quota_exhausted:
                    pending.append(item.id)
                    continue
                futures[pool.submit(_predict_one, adapter, item, run_index, cache, variant) ] = item.id
            for future, item_id in futures.items():
                try:
                    records.append(future.result())
                except DailyQuotaExhausted:
                    quota_exhausted = True
                    pending.append(item_id)
                except Exception as exc:  # noqa: BLE001 - 1 件の失敗で run を止めない
                    records.append(
                        {
                            "item_id": item_id,
                            "method": method_name,
                            "model": adapter.model,
                            "error_type": "exception",
                            "raw_response": {"exception": str(exc)[:200]},
                        }
                    )
        # 安定性テストは同じ (stem, method) を反復するため、run_index>0 はファイル名で分離する
        suffix = f"__run{run_index}" if run_index else ""
        out_path = raw_dir / f"{dataset_path.stem}__{method_name}{suffix}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if pending:
            with open(pending_dir / f"{dataset_path.stem}__{method_name}{suffix}.json", "w", encoding="utf-8") as fh:
                json.dump({"run_index": run_index, "pending_item_ids": pending}, fh, ensure_ascii=False)
        ok = [r for r in records if not r.get("error_type")]
        errors = [r for r in records if r.get("error_type")]
        summary["methods"][name] = {
            "tier": adapter.tier,
            "model": adapter.model,
            "n": len(records),
            "ok": len(ok),
            "errors": len(errors),
            "pending": len(pending),
            "quota_exhausted": quota_exhausted,
            "out": str(out_path),
        }
    summary["free_tier_used_today"] = quota.used_today
    summary["free_tier_remaining"] = quota.remaining
    with open(out_root / f"summary__{dataset_path.stem}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    from common.config import REPO_ROOT, load_config

    parser = argparse.ArgumentParser(description="同一データ・同一条件で方式を実行する")
    parser.add_argument("--dataset", required=True, help="評価データ(JSONL)")
    parser.add_argument(
        "--adapters",
        default="rule,mercury,llm_prompt,llm_json_schema,pplx_decider",
        help="カンマ区切りの方式名",
    )
    parser.add_argument("--limit", type=int, default=None, help="先頭 N 件のみ")
    parser.add_argument("--run-index", type=int, default=0, help="安定性テストの反復番号")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--confirm-external", action="store_true", help="real データの外部送信を許可")
    parser.add_argument("--allow-paid-models", action="store_true", help="Jev 実行を許可")
    parser.add_argument("--train-on", default=None, help="embedding_lr の学習データ(実運用では calib)")
    parser.add_argument("--raw-dir", default=None,
                        help="生応答の出力先(既定 results/raw)。calib は results/raw_calib を指定して分離する")
    parser.add_argument("--config", default=None, help="設定ファイルの上書き")
    args = parser.parse_args(argv)

    config = load_config(path=args.config, repo_root=REPO_ROOT)
    summary = run_benchmark(
        dataset_path=args.dataset,
        adapter_names=[a.strip() for a in args.adapters.split(",") if a.strip()],
        config=config,
        repo_root=REPO_ROOT,
        run_index=args.run_index,
        limit=args.limit,
        concurrency=args.concurrency,
        confirm_external=args.confirm_external,
        allow_paid_models=args.allow_paid_models,
        train_on=args.train_on,
        raw_dir=args.raw_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

