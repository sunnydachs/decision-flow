"""指標のユニットテスト(既知値で検証する)。"""
import math

import numpy as np
import pytest

from evaluation.bootstrap import bootstrap_ci
from evaluation.calibration import (
    apply_temperature,
    auroc,
    brier_score,
    ece,
    log_loss,
    probability_extremity,
    reliability_bins,
    temperature_scale_fit,
)
from evaluation.classification import (
    accuracy,
    confusion_matrix,
    error_rate,
    macro_f1,
    per_class_metrics,
)
from evaluation.risk_coverage import (
    automation_rate,
    evaluate_threshold,
    fixed_recall_report,
    fixed_recall_threshold,
    recall_at_threshold,
    selective_risk,
)


# --- classification ---------------------------------------------------------
def test_accuracy_and_error_rate():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert error_rate(y_true, y_pred) == pytest.approx(0.25)


def test_confusion_matrix_and_per_class():
    y_true = ["a", "b", "b"]
    y_pred = ["a", "a", "b"]
    cm = confusion_matrix(y_true, y_pred, ["a", "b"])
    assert cm.tolist() == [[1, 0], [1, 1]]
    metrics = per_class_metrics(y_true, y_pred, ["a", "b"])
    assert metrics["a"]["precision"] == pytest.approx(0.5)
    assert metrics["a"]["recall"] == pytest.approx(1.0)
    assert metrics["b"]["precision"] == pytest.approx(1.0)
    assert metrics["b"]["recall"] == pytest.approx(0.5)
    assert metrics["a"]["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert macro_f1(y_true, y_pred, ["a", "b"]) == pytest.approx(2 / 3)


# --- calibration ------------------------------------------------------------
def test_ece_perfect_is_zero():
    probs = [0.0, 0.0, 1.0, 1.0]
    y = [0, 0, 1, 1]
    assert ece(probs, y, n_bins=2) == pytest.approx(0.0)


def test_ece_known_value():
    probs = [0.1, 0.1, 0.9, 0.9]
    y = [0, 0, 0, 1]
    # bin1: (0.1,0.1) empirical 0 -> gap 0.1, weight .5 / bin2: empirical .5 -> gap .4, weight .5
    assert ece(probs, y, n_bins=2) == pytest.approx(0.25)


def test_reliability_bins_equal_frequency():
    probs = [0.1, 0.2, 0.3, 0.4]
    bins = reliability_bins(probs, [0, 0, 1, 1], n_bins=2)
    assert [b["n"] for b in bins] == [2, 2]
    assert bins[0]["mean_prob"] == pytest.approx(0.15)
    assert bins[1]["empirical"] == pytest.approx(1.0)


def test_brier_and_log_loss():
    assert brier_score([0.1, 0.9], [0, 1]) == pytest.approx(0.01)
    assert log_loss([0.5, 0.5], [0, 1]) == pytest.approx(math.log(2))


def test_auroc_perfect_and_ties():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert auroc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)


def test_auroc_known_value():
    # pos: 0.6, 0.8 / neg: 0.2, 0.7 -> pairs: 0.6>0.2,0.6<0.7, 0.8>0.2,0.8>0.7 => 3/4
    assert auroc([0.6, 0.8, 0.2, 0.7], [1, 1, 0, 0]) == pytest.approx(0.75)


def test_temperature_scaling_reduces_log_loss_on_overconfident():
    probs = [0.99, 0.99, 0.01, 0.01]
    y = [1, 0, 1, 0]  # 半分は自信過剰な誤り
    base = log_loss(probs, y)
    t = temperature_scale_fit(probs, y)
    assert t > 1.0
    assert log_loss(apply_temperature(probs, t), y) < base


def test_probability_extremity():
    out = probability_extremity([0.01, 0.5, 0.99])
    assert out["share_<0.02"] == pytest.approx(1 / 3)
    assert out["share_>0.98"] == pytest.approx(1 / 3)


# --- risk coverage ----------------------------------------------------------
def test_recall_and_automation_rates():
    p = [0.1, 0.4, 0.6, 0.9]
    y = [1, 1, 0, 1]
    assert recall_at_threshold(p, y, 0.4) == pytest.approx(2 / 3)  # positive 0.4,0.9 >= 0.4
    assert automation_rate(p, 0.5) == pytest.approx(0.5)  # {0.1,0.4} < 0.5
    # 自動処理した 2 件がどちらも high → この閾値では見逃しが残る例
    assert selective_risk(p, y, 0.5) == pytest.approx(1.0)


def test_fixed_recall_threshold_maximizes_automation():
    p = [0.1, 0.2, 0.4, 0.6, 0.9]
    y = [1, 1, 1, 1, 0]
    # positives sorted = [0.1,0.2,0.4,0.6], target 0.75 -> k=3 -> 3番目に大きい = 0.2
    t = fixed_recall_threshold(p, y, target=0.75)
    assert t == pytest.approx(0.2)
    assert recall_at_threshold(p, y, t) >= 0.75
    # これより大きい閾値では制約を満たさない
    assert recall_at_threshold(p, y, 0.2001) < 0.75
    # target 0.5 -> k=2 -> 0.4
    assert fixed_recall_threshold(p, y, target=0.5) == pytest.approx(0.4)


def test_fixed_recall_threshold_all_positive():
    p = [0.2, 0.5, 0.7]
    y = [1, 1, 1]
    assert fixed_recall_threshold(p, y, target=1.0) == pytest.approx(0.2)


def test_evaluate_threshold_auto_error_rate():
    p = [0.1, 0.2, 0.8, 0.9]
    y_risk = [0, 0, 1, 1]
    labels_true = ["billing", "sales", "urgent_claim", "urgent_claim"]
    labels_pred = ["billing", "other", "urgent_claim", "urgent_claim"]
    out = evaluate_threshold(p, y_risk, labels_true, labels_pred, threshold=0.5)
    assert out["automation_rate"] == pytest.approx(0.5)
    assert out["recall_high"] == pytest.approx(1.0)
    assert out["auto_error_rate"] == pytest.approx(0.5)  # 自動処理 2 件のうち 1 件が誤り


def test_fixed_recall_report_uses_calib_threshold_on_test():
    calib_p = [0.1, 0.2, 0.4, 0.6, 0.9]
    calib_y = [1, 1, 1, 1, 0]
    # test 側で閾値を探さない: calib の 0.2 がそのまま適用される
    test_p = [0.6, 0.5, 0.2, 0.1]
    test_y = [1, 1, 0, 0]
    report = fixed_recall_report(calib_p, calib_y, test_p, test_y, target=0.75)
    assert report["threshold_from_calib"] == pytest.approx(0.2)
    assert report["test"]["recall_high"] == pytest.approx(1.0)  # 0.6 と 0.5 が flagged
    assert report["test"]["automation_rate"] == pytest.approx(0.25)  # 0.1 のみ自動処理


# --- bootstrap --------------------------------------------------------------
def test_bootstrap_ci_is_deterministic_and_bounds_point():
    def mean_fn(x):
        return float(np.mean(x))

    data = np.array([0.1, 0.2, 0.9, 0.8, 0.5, 0.4])
    point, lo, hi = bootstrap_ci(mean_fn, [data], n_samples=200, seed=7)
    assert point == pytest.approx(float(np.mean(data)))
    assert lo <= point <= hi
    point2, lo2, hi2 = bootstrap_ci(mean_fn, [data], n_samples=200, seed=7)
    assert (lo, hi) == (lo2, hi2)
