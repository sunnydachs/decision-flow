"""運用上の判断効率: リスク-カバレッジと固定 Recall の評価。

P_high = 高リスク(risk_label)確率。閾値 t に対し
  - flagged(人間レビュー/即時対応) = P_high >= t
  - 自動処理                        = P_high <  t
  - 高リスク recall                 = P(P_high >= t | severity=high)
  - 自動化率                        = P(P_high < t)
固定 Recall 評価は「calib で recall>=目標 を満たす最大の t を選び、test では t を固定適用する」。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def recall_at_threshold(p_risk: Sequence[float], y_risk: Sequence[int], threshold: float) -> float:
    p_risk = np.asarray(p_risk, dtype=float)
    y_risk = np.asarray(y_risk).astype(int)
    positives = y_risk == 1
    if positives.sum() == 0:
        return float("nan")
    return float((p_risk[positives] >= threshold).mean())


def automation_rate(p_risk: Sequence[float], threshold: float) -> float:
    p_risk = np.asarray(p_risk, dtype=float)
    if len(p_risk) == 0:
        return float("nan")
    return float((p_risk < threshold).mean())


def review_rate(p_risk: Sequence[float], threshold: float) -> float:
    rate = automation_rate(p_risk, threshold)
    return float("nan") if np.isnan(rate) else 1.0 - rate


def selective_risk(p_risk: Sequence[float], y_risk: Sequence[int], threshold: float) -> float:
    """自動処理した分に含まれる高リスク案件の割合(重大誤りの混入率)。"""
    p_risk = np.asarray(p_risk, dtype=float)
    y_risk = np.asarray(y_risk).astype(int)
    auto = p_risk < threshold
    if auto.sum() == 0:
        return float("nan")
    return float(y_risk[auto].mean())


def fixed_recall_threshold(
    p_risk: Sequence[float], y_risk: Sequence[int], target: float = 0.95
) -> float:
    """calib で recall >= target を満たす最大の閾値(= 自動化率を最大化する閾値)。

    positives を昇順ソートし、k = ceil(target * n_pos) として k 番目に大きい確率を返す。
    P_high >= t と判定するため、この t で recall >= target が保証される。
    """
    p_risk = np.asarray(p_risk, dtype=float)
    y_risk = np.asarray(y_risk).astype(int)
    positives = np.sort(p_risk[y_risk == 1])
    n_pos = len(positives)
    if n_pos == 0:
        return float("nan")
    if target <= 0:
        return 0.0
    k = int(np.ceil(target * n_pos))
    k = max(1, min(k, n_pos))
    return float(positives[n_pos - k])


def evaluate_threshold(
    p_risk: Sequence[float],
    y_risk: Sequence[int],
    labels_true: Sequence[str] | None = None,
    labels_pred: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> dict:
    """閾値 1 点での運用指標(自動化率・レビュー率・recall・自動処理分の混入率・自動処理分の誤り率)。"""
    out = {
        "threshold": float(threshold),
        "recall_high": recall_at_threshold(p_risk, y_risk, threshold),
        "automation_rate": automation_rate(p_risk, threshold),
        "review_rate": review_rate(p_risk, threshold),
        "selective_risk": selective_risk(p_risk, y_risk, threshold),
        "n": int(len(np.asarray(p_risk))),
    }
    if labels_true is not None and labels_pred is not None:
        labels_true = np.asarray(labels_true)
        labels_pred = np.asarray(labels_pred)
        auto = np.asarray(p_risk, dtype=float) < threshold
        out["auto_n"] = int(auto.sum())
        out["auto_error_rate"] = (
            float((labels_true[auto] != labels_pred[auto]).mean()) if auto.sum() else None
        )
    return out


def risk_coverage_curve(
    p_risk: Sequence[float],
    y_risk: Sequence[int],
    thresholds: Sequence[float] | None = None,
    n_points: int = 50,
) -> list[dict]:
    """閾値を動かしたときの (自動化率, 高リスク recall, 混入率) の曲線。"""
    p_risk = np.asarray(p_risk, dtype=float)
    if thresholds is None:
        if len(p_risk) == 0:
            return []
        quantiles = np.linspace(0.0, 1.0, n_points)
        thresholds = np.unique(np.quantile(p_risk, quantiles))
    return [evaluate_threshold(p_risk, y_risk, threshold=float(t)) for t in thresholds]


def fixed_recall_report(
    p_risk_calib: Sequence[float],
    y_risk_calib: Sequence[int],
    p_risk_test: Sequence[float],
    y_risk_test: Sequence[int],
    target: float = 0.95,
    labels_true_test: Sequence[str] | None = None,
    labels_pred_test: Sequence[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 20261004,
) -> dict:
    """calib で閾値を決め、test に固定適用した結果を返す(閾値探索は calib のみ)。

    高リスク recall と自動化率にはブートストラップ信頼区間を付ける。
    test の高リスク件数が少ないと区間が広くなるため、点推定だけで判断しない。
    """
    threshold = fixed_recall_threshold(p_risk_calib, y_risk_calib, target=target)
    report = {
        "target_recall": float(target),
        "threshold_from_calib": threshold,
        "calib": evaluate_threshold(p_risk_calib, y_risk_calib, threshold=threshold),
    }
    test = evaluate_threshold(
        p_risk_test,
        y_risk_test,
        labels_true=labels_true_test,
        labels_pred=labels_pred_test,
        threshold=threshold,
    )
    test.update(_bootstrap_bounds(p_risk_test, y_risk_test, threshold, n_boot=n_boot, alpha=alpha, seed=seed))
    report["test"] = test
    return report


def _bootstrap_bounds(
    p_risk: Sequence[float],
    y_risk: Sequence[int],
    threshold: float,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 20261004,
) -> dict:
    """recall(高リスクのみ)と自動化率(全件)のブートストラップ信頼区間。"""
    from evaluation.bootstrap import bootstrap_ci

    p = np.asarray(p_risk, dtype=float)
    y = np.asarray(y_risk, dtype=int)
    out: dict = {}
    # recall は正例(高リスク)だけの集合でリサンプルする
    pos = p[y == 1]
    if pos.size:
        point, lo, hi = bootstrap_ci(
            lambda pr, yy: recall_at_threshold(pr, yy, threshold),
            [pos, np.ones(pos.size, dtype=int)],
            n_samples=n_boot, alpha=alpha, seed=seed,
        )
        out["recall_high_ci"] = {"point": point, "lo": lo, "hi": hi,
                                 "n_positives": int(pos.size)}
    if p.size:
        point, lo, hi = bootstrap_ci(
            lambda pr: automation_rate(pr, threshold),
            [p],
            n_samples=n_boot, alpha=alpha, seed=seed,
        )
        out["automation_rate_ci"] = {"point": point, "lo": lo, "hi": hi, "n": int(p.size)}
    return out
