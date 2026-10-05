"""共通基盤(HTTP/レート/予算/キャッシュ)と rule アダプタのテスト。"""
import io
import json
import urllib.error
from pathlib import Path

import pytest

from adapters.base import Prediction, utc_now_iso
from adapters.rule import RuleAdapter, load_rules
from common.budget import BudgetExceeded, BudgetGuard, estimate_cost_usd
from common.cache import AuditLog, ResultCache, cache_key
from common.dataset import Item
from common.http import HttpClient, classify_status
from common.ratelimit import DailyQuota, DailyQuotaExhausted, RateLimiter
from tasks.base import load_task

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_item(text: str, id_: str = "t1", label: str = "other", severity: str = "normal") -> Item:
    return Item(id=id_, text=text, label=label, severity=severity, language="ja")


# --- rule adapter -----------------------------------------------------------
def test_rule_adapter_matches_urgent_and_risk_score():
    task = load_task(REPO_ROOT / "config" / "tasks" / "support_classification.toml")
    rules = load_rules(REPO_ROOT / "config" / "rules" / "support_classification.toml")
    adapter = RuleAdapter(task, rules)
    p = adapter.predict(make_item("サービスがダウンし、業務が止まっています。至急対応をお願いします。"))
    assert p.label == "urgent_claim"
    assert p.risk_score == 1.0
    assert p.risk_score_type == "binary_rule"
    assert p.probabilities is None
    assert p.value_type == "none"
    assert p.model == "keyword-rules"


def test_rule_adapter_fallback_and_non_risk():
    task = load_task(REPO_ROOT / "config" / "tasks" / "support_classification.toml")
    rules = load_rules(REPO_ROOT / "config" / "rules" / "support_classification.toml")
    adapter = RuleAdapter(task, rules)
    p = adapter.predict(make_item("今日の天気を教えてください。"))
    assert p.label == "other"
    assert p.risk_score == 0.0
    assert p.raw_response["matched_label"] is None


# --- Prediction -------------------------------------------------------------
def test_prediction_roundtrip():
    p = Prediction(item_id="x", method="rule", model="m", label="billing", probabilities={"a": 0.7})
    d = p.as_dict()
    assert d["timestamp_utc"] == p.timestamp_utc
    restored = Prediction.from_dict({**d, "unknown_key": 1})
    assert restored.label == "billing"
    assert restored.probabilities == {"a": 0.7}
    assert utc_now_iso().endswith("+00:00")


# --- http -------------------------------------------------------------------
def test_classify_status():
    assert classify_status(200) is None
    assert classify_status(429) == "http_429"
    assert classify_status(503) == "http_5xx"
    assert classify_status(400) == "http_4xx"


def test_http_retries_on_429_and_honors_retry_after(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []

    class FakeResp:
        status = 200

        def __init__(self):
            self.headers = {"Content-Type": "application/json"}

        def read(self):
            return json.dumps({"answers": {}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            hdrs = {"Retry-After": "3"}
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", hdrs, io.BytesIO(b"rate limited"))
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = HttpClient(timeout_seconds=5, max_retries=2, sleep=sleeps.append)
    result = client.post_json("https://example.invalid/x", {"a": 1})
    assert result.ok is True
    assert result.attempts == 2
    assert calls["n"] == 2
    assert sleeps == [3.0]  # Retry-After を尊重


def test_http_does_not_retry_on_400(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b"bad"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = HttpClient(max_retries=3, sleep=lambda _: None)
    result = client.post_json("https://example.invalid/x", {})
    assert result.ok is False
    assert result.error_type == "http_4xx"
    assert calls["n"] == 1  # リトライしない


def test_http_gives_up_after_max_retries(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, io.BytesIO(b"down"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = HttpClient(max_retries=2, backoff_base_seconds=0.0, sleep=lambda _: None)
    result = client.post_json("https://example.invalid/x", {})
    assert result.ok is False
    assert result.error_type == "http_5xx"
    assert calls["n"] == 3  # 初回 + 2 リトライ
    assert result.attempts == 3


# --- rate limit / quota / budget / cache ------------------------------------
def test_rate_limiter_waits_min_interval():
    clock = {"t": 0.0}
    sleeps: list[float] = []

    def fake_sleep(s):
        sleeps.append(s)
        clock["t"] += s

    limiter = RateLimiter(per_minute_limit=20, sleep=fake_sleep, clock=lambda: clock["t"])
    limiter.wait()  # 初回は待たない(経過時間が大きい扱い)
    limiter.wait()
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(3.0)  # 60/20


def test_daily_quota_shared_counter():
    quota = DailyQuota(daily_limit=3)
    quota.consume()
    quota.consume()
    assert quota.remaining == 1
    quota.consume()
    assert quota.remaining == 0
    with pytest.raises(DailyQuotaExhausted):
        quota.consume()


def test_budget_guard_stops():
    guard = BudgetGuard(limit_usd=0.5, name="pplx")
    guard.add(0.4)
    assert guard.remaining_usd == pytest.approx(0.1)
    guard.check(0.05)
    with pytest.raises(BudgetExceeded):
        guard.check(0.2)
    assert estimate_cost_usd(1000, 0.04) == pytest.approx(0.00004)


def test_result_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    cache = ResultCache(cache_path)
    key = cache_key("rule", "model", "task", "item-1", 0)
    assert cache.get(key) is None
    cache.put(key, {"label": "billing"})
    assert cache.get(key) == {"label": "billing"}
    reloaded = ResultCache(cache_path)
    assert reloaded.get(key) == {"label": "billing"}
    assert cache_key("rule", "model", "task", "item-1", 0) == key
    assert cache_key("rule", "model", "task", "item-1", 1) != key


def test_audit_log_writes_jsonl(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.write({"item_id": "x", "action": "auto"})
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["action"] == "auto"
