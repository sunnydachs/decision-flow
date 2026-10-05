"""再較正(temperature / isotonic)のユニットテスト。"""
import numpy as np
import pytest

from evaluation.calibration import (
    IsotonicRecalibrator,
    apply_temperature,
    ece,
    recalibration_report,
    temperature_scale_fit,
)


def _overconfident(n=400, seed=0):
    """確率は極端(0/1 付近)だが実際の正例率は控えめ、という過信データを作る。"""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    probs = np.where(y == 1, rng.uniform(0.8, 1.0, n), rng.uniform(0.0, 0.2, n))
    return probs.tolist(), y.tolist()


def test_isotonic_is_monotone_and_bounded():
    probs, y = _overconfident()
    iso = IsotonicRecalibrator().fit(probs[:200], y[:200])
    out = iso.apply(probs[200:])
    assert out.min() >= 0.0 and out.max() <= 1.0
    # 入力が昇順なら出力も非減少
    order = np.argsort(np.array(probs[200:]))
    sorted_out = np.asarray(out)[order]
    assert np.all(np.diff(sorted_out) >= -1e-9)


def test_isotonic_requires_fit_first():
    with pytest.raises(RuntimeError):
        IsotonicRecalibrator().apply([0.5])


def test_isotonic_keeps_ranking():
    """単調写像なので順位は変わらない(AUROC が保たれる)。"""
    from evaluation.calibration import auroc

    probs, y = _overconfident()
    iso = IsotonicRecalibrator().fit(probs[:200], y[:200])
    before = auroc(probs[200:], y[200:])
    after = auroc(iso.apply(probs[200:]).tolist(), y[200:])
    assert after == pytest.approx(before, abs=1e-9)


def test_temperature_is_positive():
    probs, y = _overconfident()
    t = temperature_scale_fit(probs, y)
    assert t > 0
    out = apply_temperature(probs, t)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_recalibration_report_shapes_and_improvement():
    probs, y = _overconfident()
    report = recalibration_report(probs[:200], y[:200], probs[200:], y[200:])
    for key in ("before", "after_temperature", "after_isotonic"):
        assert set(report[key]) == {"ece", "brier", "log_loss"}
    assert report["n_calib"] == 200 and report["n_test"] == 200
    # 過信データなので、再較正後に ECE が悪化してはいけない(単調写像なので大きく改善するはず)
    assert min(report["after_temperature"]["ece"], report["after_isotonic"]["ece"]) <= report["before"]["ece"]


def test_recalibration_report_needs_both_classes():
    out = recalibration_report([0.1, 0.9], [0, 0], [0.1, 0.9], [0, 0])
    assert "note" in out

