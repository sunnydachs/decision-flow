"""安定性指標のユニットテスト(ネットワーク不要)。"""
from evaluation.stability import dispersion, flip_rate, label_flip_rate, summarize


def test_dispersion_basic():
    d = dispersion([0.5, 0.5, 0.5])
    assert d["n"] == 3
    assert d["std"] == 0.0
    assert d["range"] == 0.0
    assert d["iqr"] == 0.0


def test_dispersion_spread():
    d = dispersion([0.0, 0.5, 1.0])
    assert d["range"] == 1.0
    assert d["mean"] == 0.5
    assert d["std"] > 0.4
    assert d["mean_abs_dev"] > 0.3


def test_flip_rate_is_zero_when_all_on_one_side():
    assert flip_rate([0.9, 0.8, 0.85], threshold=0.5) == 0.0
    assert flip_rate([0.1, 0.2], threshold=0.5) == 0.0


def test_flip_rate_is_half_when_balanced():
    assert flip_rate([0.9, 0.1], threshold=0.5) == 0.5


def test_flip_rate_none_without_threshold():
    assert flip_rate([0.9, 0.1], threshold=None) is None


def test_label_flip_rate():
    assert label_flip_rate(["a", "a", "a"]) == 0.0
    assert label_flip_rate(["a", "b"]) == 1.0
    assert label_flip_rate([None]) is None


def test_summarize_reports_flip_and_dispersion():
    per_item = {
        "i1": [  # 安定: 常に閾値上
            {"item_id": "i1", "risk_score": 0.9, "label": "urgent_claim"},
            {"item_id": "i1", "risk_score": 0.91, "label": "urgent_claim"},
        ],
        "i2": [  # 不安定: 閾値をまたぐ
            {"item_id": "i2", "risk_score": 0.9, "label": "urgent_claim"},
            {"item_id": "i2", "risk_score": 0.1, "label": "billing"},
        ],
    }
    out = summarize(per_item, thresholds={"review_above": 0.5})
    assert out["n_items"] == 2
    assert out["n_runs"] == 2
    assert out["probability"]["max_std"] > 0.3
    assert out["threshold_flip"]["items_ever_flipping"] == 1
    assert out["label_disagreement_rate"] == 0.5


def test_summarize_counts_items_with_errors():
    per_item = {
        "i1": [{"item_id": "i1", "risk_score": None, "label": None, "error_type": "timeout"}],
        "i2": [{"item_id": "i2", "risk_score": 0.2, "label": "billing"}],
    }
    out = summarize(per_item, thresholds={"review_above": 0.5})
    assert out["items_with_any_error"] == 1
    assert "probability" in out
