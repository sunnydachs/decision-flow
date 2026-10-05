"""設定からアダプタを組み立てる。"""
from __future__ import annotations

from pathlib import Path

from adapters.base import BaseAdapter
from adapters.llm import LlmAdapter
from adapters.mercury import MercuryAdapter
from adapters.pplx_decider import PerplexityDeciderAdapter
from adapters.rule import RuleAdapter, load_rules
from common.budget import BudgetGuard
from common.env import require_key
from common.http import HttpClient
from common.ratelimit import RateLimiter
from tasks.base import TaskDefinition

FREE_ADAPTERS = ("mercury", "llm_prompt", "llm_json_schema", "span")
PAID_ADAPTERS = ("pplx_decider", "jev")
LOCAL_ADAPTERS = ("rule", "embedding_lr")

ALL_ADAPTERS = FREE_ADAPTERS + PAID_ADAPTERS + LOCAL_ADAPTERS


def build_http(config) -> HttpClient:
    return HttpClient(
        timeout_seconds=float(config.get("run.timeout_seconds", 30)),
        max_retries=int(config.get("run.max_retries", 3)),
        backoff_base_seconds=float(config.get("run.backoff_base_seconds", 1.0)),
        backoff_max_seconds=float(config.get("run.backoff_max_seconds", 20.0)),
    )


def build_adapters(
    config,
    task: TaskDefinition,
    names: list[str],
    *,
    repo_root: Path,
    http: HttpClient | None = None,
    cache=None,
    quota=None,
    budgets: dict | None = None,
    audit=None,
    allow_paid_models: bool = False,
    keys: dict | None = None,
) -> dict[str, BaseAdapter]:
    http = http or build_http(config)
    budgets = budgets if budgets is not None else {}
    free_limiter = RateLimiter(per_minute_limit=int(config.get("free_tier.per_minute_limit", 20)))
    adapters: dict[str, BaseAdapter] = {}

    def need_key(name: str) -> str:
        if keys and keys.get(name):
            return keys[name]
        return require_key(name, repo_root=repo_root)

    for name in names:
        if name == "rule":
            rules = load_rules(repo_root / "config" / "rules" / f"{task.id}.toml")
            adapters[name] = RuleAdapter(task, rules)
        elif name == "mercury":
            adapters[name] = MercuryAdapter(
                task, config, key=need_key("OPENROUTER_API_KEY"), http=http, quota=quota,
                cache=cache, audit=audit, rate_limiter=free_limiter,
            )
        elif name in ("llm_prompt", "llm_json_schema"):
            mode = "prompt" if name == "llm_prompt" else "json_schema"
            adapter = LlmAdapter(
                task, config, mode=mode, key=need_key("OPENROUTER_API_KEY"), http=http, quota=quota,
                cache=cache, audit=audit, rate_limiter=free_limiter,
            )
            adapters[name] = adapter
        elif name == "pplx_decider":
            budget = budgets.setdefault(
                "perplexity", BudgetGuard(limit_usd=float(config.get("budget.perplexity_usd", 0.5)), name="perplexity")
            )
            adapters[name] = PerplexityDeciderAdapter(
                task, config, key=need_key("PERPLEXITY_API_KEY"), http=http, budget=budget,
                cache=cache, audit=audit,
            )
        elif name == "jev":
            if not allow_paid_models:
                raise SystemExit(
                    "Jev を実行するには --allow-paid-models が必要です(誤実行の防止)。"
                )
            from adapters.jev import JevAdapter

            budget = budgets.setdefault(
                "jev", BudgetGuard(limit_usd=float(config.get("budget.jev_usd", 1.0)), name="jev")
            )
            adapters[name] = JevAdapter(
                task, config, key=need_key("TYPESAFE_API_KEY"), http=http, budget=budget,
                cache=cache, audit=audit,
            )
        elif name == "embedding_lr":
            from adapters.embedding_lr import EmbeddingLrAdapter

            adapters[name] = EmbeddingLrAdapter(task, config, repo_root=repo_root)
        else:
            raise SystemExit(f"未知のアダプタ名: {name}")
    return adapters
