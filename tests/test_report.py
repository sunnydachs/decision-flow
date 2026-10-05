"""集計ロジックのテスト(小さな固定データで検証する)。"""
import json
from pathlib import Path

import numpy as np

from common.config import load_config
from common.dataset import Item, write_jsonl
from runners.report import build_report, evaluate_method, render_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]


def _items():
    return {
        "a": Item(id="a", text="x", label="urgent_claim", severity="high", language="ja"),
        "b": Item(id="b", text="y", label="billing", severity="normal", language="ja"),
        "c": Item(id="c", text="z", label="billing", severity="normal", language="ja"),
    }


def test_evaluate_method_metrics():
    records = {
        "a": {"item_id": "a", "label": "urgent_claim", "risk_score": 0.9, "value_type": "probability",
              "latency_ms": 10, "estimated_cost_usd": 0.0},
        "b": {"item_id": "b", "label": "billing", "risk_score": 0.1, "value_type": "probability",
              "latency_ms": 20, "estimated_cost_usd": 0.0},
        "c": {"item_id": "c", "label": "billing", "risk_score": 0.2, "value_type": "probability",
              "latency_ms": 30, "estimated_cost_usd": 0.0},
    }
    out = evaluate_method(records, _items())
    assert out["n_ok"] == 3
    assert out["accuracy"] == 1.0
    assert out["risk_detection"]["positives"] == 1
    assert out["auroc"] if "auroc" in out else True
    assert out["risk_detection"]["auroc"] == 1.0
    assert out["calibration"]["ece"] >= 0.0
    assert out["latency_ms"]["p50"] == 20.0
    assert out["cost_per_1000_usd"] == 0.0


def test_evaluate_method_without_probability_excludes_calibration():
    records = {
        "a": {"item_id": "a", "label": "urgent_claim", "risk_score": 0.9, "value_type": "stated_probability",
              "latency_ms": 10},
        "b": {"item_id": "b", "label": "billing", "risk_score": 0.1, "value_type": "stated_probability",
              "latency_ms": 10},
    }
    out = evaluate_method(records, _items())
    assert "calibration" not in out  # LLM の自己申告確率は較正の対象外
    assert "auroc" in out["risk_detection"]


def test_evaluate_method_counts_errors():
    records = {
        "a": {"item_id": "a", "label": None, "error_type": "http_429", "latency_ms": 5},
    }
    out = evaluate_method(records, _items())
    assert out["n_ok"] == 0
    assert out["errors"] == {"http_429": 1}


def test_build_report_and_markdown(tmp_path):
    dataset = tmp_path / "d.jsonl"
    write_jsonl(list(_items().values()), dataset)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with open(raw_dir / "d__m.jsonl", "w", encoding="utf-8") as fh:
        for iid, label, risk in (("a", "urgent_claim", 0.9), ("b", "billing", 0.1), ("c", "billing", 0.2)):
            fh.write(json.dumps({"item_id": iid, "label": label, "risk_score": risk,
                                 "value_type": "probability", "latency_ms": 10}) + "\n")
    config = load_config(repo_root=REPO_ROOT)
    aggregate = build_report(dataset_path=dataset, raw_dir=raw_dir, methods=["m"], config=config)
    assert aggregate["methods"]["m"]["accuracy"] == 1.0
    md = render_markdown(aggregate)
    assert "## 1. 分類品質" in md
    assert "| m |" in md
