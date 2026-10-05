"""設定・タスク定義・データローダのテスト。"""
from pathlib import Path

import pytest

from common.config import ConfigError, load_config, load_task_config
from common.dataset import (
    DatasetError,
    annotator_agreement,
    external_send_allowed,
    label_counts,
    load_jsonl,
    load_manifest,
    severity_counts,
)
from common.env import ENV_NAMES, key_status, load_keys
from tasks.base import load_task

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config():
    cfg = load_config(repo_root=REPO_ROOT)
    assert cfg.get("run.concurrency") == 4
    assert cfg.get("evaluation.risk_label") == "urgent_claim"
    assert cfg.get("evaluation.risk_severity") == "high"
    assert cfg.get("models.mercury.model") == "decision-model-v1"
    assert cfg.get("models.jev.model") == "jev-1.13.0"  # バージョン固定
    assert cfg.get("models.llm.model") == "llm-generic-20b"
    assert cfg.get("models.llm.modes") is None  # モードは adapter 名(llm_prompt / llm_json_schema)で区別
    assert cfg.get("models.llm.tier") == "external_free"
    assert cfg.get("budget.perplexity_usd") == 0.5
    assert cfg.get("budget.jev_usd") == 1.0


def test_local_config_overrides(tmp_path):
    local = tmp_path / "local.toml"
    local.write_text("[run]\nconcurrency = 8\n", encoding="utf-8")
    cfg = load_config(local_path=local, repo_root=REPO_ROOT)
    assert cfg.get("run.concurrency") == 8
    assert cfg.get("run.timeout_seconds") == 30  # 既定は維持


def test_missing_required_key_raises(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[run]\nconcurrency = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=bad, local_path=tmp_path / "none.toml", repo_root=REPO_ROOT)


def test_load_task_definition():
    task = load_task(REPO_ROOT / "config" / "tasks" / "support_classification.toml")
    assert task.id == "support_classification"
    assert task.risk_label == "urgent_claim"
    assert task.labels == ("urgent_claim", "billing", "technical", "sales", "other")
    question = task.choice_question()
    criteria = question["category"]["criteria"]
    assert set(criteria) == set(task.labels)
    assert task.is_risk_label("urgent_claim")
    assert not task.is_risk_label("billing")


def test_load_synthetic_dataset():
    items = load_jsonl(REPO_ROOT / "data" / "synthetic" / "support_classification.jsonl")
    assert len(items) == 30
    labels = label_counts(items)
    assert labels["urgent_claim"] == 9
    sev = severity_counts(items)
    assert sev["high"] == 12  # severity は label と独立(billing/technical/sales にも high がある)
    assert sum(1 for it in items if it.ambiguous) == 2


def test_manifest_and_external_gate():
    manifest = load_manifest(REPO_ROOT / "data" / "synthetic")
    assert manifest["data_class"] == "synthetic"
    assert external_send_allowed(manifest, confirm_external=False) is True
    real = {"external_ok": False, "data_class": "real"}
    assert external_send_allowed(real, confirm_external=False) is False
    assert external_send_allowed(real, confirm_external=True) is True


def test_severity_validation(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id": "x", "text": "t", "label": "billing", "severity": "critical"}\n', encoding="utf-8"
    )
    with pytest.raises(DatasetError):
        load_jsonl(path)


def test_duplicate_id_validation(tmp_path):
    path = tmp_path / "dup.jsonl"
    line = '{"id": "x", "text": "t", "label": "billing", "severity": "normal"}\n'
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(DatasetError):
        load_jsonl(path)


def test_annotator_agreement():
    items = load_jsonl(REPO_ROOT / "data" / "synthetic" / "support_classification.jsonl")
    out = annotator_agreement(items)
    assert out["n"] == 2
    assert out["unanimous"] == 0
    assert out["agreement"] == 0.0


def test_key_status_never_returns_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=secret-value-xyz\n", encoding="utf-8")
    status = key_status([env_file])
    assert set(status) == set(ENV_NAMES)
    assert status["OPENROUTER_API_KEY"] is True
    keys = load_keys([env_file])
    assert keys["OPENROUTER_API_KEY"] == "secret-value-xyz"
