"""LLM 応答パース / hybrid 判断 / Jev dry-run のテスト。"""
import json
from pathlib import Path

import pytest

from adapters.jev import JevAdapter, estimate_tokens_rough
from adapters.llm import build_json_schema, parse_llm_content
from adapters.rule import load_rules
from adapters.base import Prediction
from common.config import load_config
from common.dataset import Item
from hybrid.pipeline import ACTION_AUTO, ACTION_BLOCK, ACTION_REVIEW, HybridPipeline, Thresholds, decide
from tasks.base import load_task

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["urgent_claim", "billing", "technical", "sales", "other"]


# --- llm parsing ------------------------------------------------------------
def test_parse_llm_content_json():
    label, probs, ok = parse_llm_content(
        '{"label": "billing", "probabilities": {"billing": 0.9, "other": 0.1}}', LABELS
    )
    assert ok is True
    assert label == "billing"
    assert probs["billing"] == pytest.approx(0.9)


def test_parse_llm_content_with_prose_around_json():
    text = 'Sure, here is the result:\n{"label": "technical", "probabilities": {"technical": 0.8}}\nDone.'
    label, probs, ok = parse_llm_content(text, LABELS)
    assert ok is True
    assert label == "technical"


def test_parse_llm_content_plain_label_fallback():
    label, probs, ok = parse_llm_content("The category is sales.", LABELS)
    assert ok is False  # JSON ではない = パース失敗として計上
    assert label == "sales"
    assert probs is None


def test_parse_llm_content_garbage():
    label, probs, ok = parse_llm_content("I cannot answer that.", LABELS)
    assert ok is False
    assert label is None


def test_build_json_schema_shape():
    schema = build_json_schema(LABELS)
    assert schema["required"] == ["label", "probabilities"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["probabilities"]["properties"]) == set(LABELS)


# --- hybrid -----------------------------------------------------------------
def _thresholds():
    return Thresholds(block_above=0.9, review_above=0.3)


def test_hybrid_rule_layer_blocks_on_risk_rule():
    d = decide(item_id="x", label=None, risk_score=None, probabilities=None, request_id="r",
               error_type=None, thresholds=_thresholds(), rule_label="urgent_claim")
    assert d.action == ACTION_BLOCK
    assert d.source == "rule"


def test_hybrid_rule_layer_auto_on_non_risk_rule():
    d = decide(item_id="x", label=None, risk_score=None, probabilities=None, request_id="r",
               error_type=None, thresholds=_thresholds(), rule_label="billing")
    assert d.action == ACTION_AUTO


def test_hybrid_thresholds():
    base = dict(item_id="x", label="billing", probabilities={"billing": 0.9}, request_id="r",
                error_type=None, thresholds=_thresholds(), rule_label=None)
    assert decide(risk_score=0.1, **base).action == ACTION_AUTO
    assert decide(risk_score=0.5, **base).action == ACTION_REVIEW
    assert decide(risk_score=0.95, **base).action == ACTION_BLOCK


def test_hybrid_fail_closed_vs_open():
    base = dict(item_id="x", label=None, risk_score=None, probabilities=None, request_id="r",
                error_type="http_429", thresholds=_thresholds(), rule_label=None)
    assert decide(fail_open=False, **base).action == ACTION_REVIEW
    assert decide(fail_open=True, **base).action == ACTION_AUTO


def test_hybrid_thresholds_validation():
    with pytest.raises(ValueError):
        Thresholds(block_above=0.3, review_above=0.9).validate()


def test_hybrid_pipeline_with_prediction():
    task = load_task(REPO_ROOT / "config" / "tasks" / "support_classification.toml")
    rules = load_rules(REPO_ROOT / "config" / "rules" / "support_classification.toml")
    pipeline = HybridPipeline(rules=rules, thresholds=_thresholds(), risk_label=task.risk_label)
    pred = Prediction(item_id="x", method="mercury", model="m", label="billing",
                      probabilities={"billing": 0.8, "urgent_claim": 0.1}, risk_score=0.1,
                      request_id="r", error_type=None)
    d = pipeline.run(pred, "請求書の宛名を変更したいです。")
    assert d.action == ACTION_AUTO
    assert d.source == "rule"  # ルールに明確一致するため層1で確定
    d2 = pipeline.run(pred, "今日の天気について教えてください。")
    assert d2.action == ACTION_AUTO
    assert d2.source == "model"


# --- jev dry run ------------------------------------------------------------
def test_jev_dry_run_builds_request_and_estimates_cost():
    config = load_config(repo_root=REPO_ROOT)
    task = load_task(REPO_ROOT / "config" / "tasks" / "support_classification.toml")
    item = Item(id="x", text="サービスが止まっています。至急対応してください。", label="urgent_claim",
                severity="high", language="ja")
    adapter = JevAdapter(task, config, key="not-used", http=None, dry_run=True, token_estimate=300)
    report = adapter.dry_run_report([item, item])
    assert report["n_requests"] == 2
    assert report["request_example"]["model"] == "jev-1.13.0"
    assert "questions" in report["request_example"]
    assert report["estimated_total_input_tokens"] == 600
    # 600 tokens * $0.042/M
    assert report["estimated_cost_usd"] == pytest.approx(600 / 1_000_000 * 0.042)
    with pytest.raises(RuntimeError):
        adapter.predict(item)


def test_estimate_tokens_rough_positive():
    assert estimate_tokens_rough("あ" * 30) >= 10
