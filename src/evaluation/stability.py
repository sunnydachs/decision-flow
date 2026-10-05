"""安定性(同じ入力に対する出力のばらつき)を測る。

方式ごとに、同じ入力を N 回(既定 20)実行したときの
- 確率のばらつき(項目ごとの標準偏差・IQR)
- 閾値をまたいで判定が反転する率

を計算する。確率を返さない方式は「判定(ラベル)の反転率」だけを見る。

注意: 無料枠は全 :free モデル共有なので、安定性テストは全件ではなくサブセット
(既定 50 件 × N 回)で行う(仕様どおり)。
"""
from __future__ import annotations

import math
from statistics import mean, pstdev


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def dispersion(values: list[float]) -> dict:
    """1 項目について、N 回の値のばらつき。"""
    if not values:
        return {"n": 0}
    m = mean(values)
    return {
        "n": len(values),
        "mean": m,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "iqr": _quantile(values, 0.75) - _quantile(values, 0.25),
        "mean_abs_dev": mean(abs(v - m) for v in values),
    }


def flip_rate(values: list[float], threshold: float | None) -> float | None:
    """N 回の値のうち、閾値の両側に分かれた比率(判定が反転し得る度合い)。

    0 = 全回同じ側(安定)、0.5 = 半々(最も不安定)。閾値が無い方式は None。
    """
    if not values or threshold is None:
        return None
    above = sum(1 for v in values if v >= threshold)
    p = above / len(values)
    return min(p, 1.0 - p)


def label_flip_rate(labels: list[str | None]) -> float | None:
    """N 回のラベルが完全一致しない項目の比率(ラベルを持つ方式)。"""
    present = [l for l in labels if l is not None]
    if not present:
        return None
    return 0.0 if len(set(present)) == 1 else 1.0


def summarize(
    per_item_runs: dict[str, list[dict]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """{item_id: [run0 の record, run1 の record, ...]} から方式全体の安定性を出す。

    record は {"risk_score": float|None, "label": str|None, "error_type": str|None} を含む想定。
    """
    thresholds = thresholds or {}
    prob_items, label_items, flip_items = [], [], []
    error_flags = []
    for _item_id, runs in per_item_runs.items():
        probs = [r.get("risk_score") for r in runs]
        probs = [float(p) for p in probs if p is not None]
        labels = [r.get("label") for r in runs]
        error_flags.append(any(r.get("error_type") for r in runs))
        if probs:
            prob_items.append(dispersion(probs))
            t = thresholds.get("review_above")
            fr = flip_rate(probs, t) if t is not None else None
            if fr is not None:
                flip_items.append(fr)
        lf = label_flip_rate(labels)
        if lf is not None:
            label_items.append(lf)

    n_items = len(per_item_runs)
    out: dict = {
        "n_items": n_items,
        "n_runs": max((len(v) for v in per_item_runs.values()), default=0),
        "items_with_any_error": sum(error_flags),
    }
    if prob_items:
        out["probability"] = {
            "mean_std": mean(d["std"] for d in prob_items),
            "max_std": max(d["std"] for d in prob_items),
            "mean_iqr": mean(d["iqr"] for d in prob_items),
            "mean_range": mean(d["range"] for d in prob_items),
            "mean_abs_dev": mean(d["mean_abs_dev"] for d in prob_items),
        }
    if flip_items:
        out["threshold_flip"] = {
            "threshold": thresholds.get("review_above"),
            "mean_flip_fraction": mean(flip_items),
            "items_ever_flipping": sum(1 for f in flip_items if f > 0.0),
            "items_always_uncertain": sum(1 for f in flip_items if f >= 0.3),
        }
    if label_items:
        out["label_disagreement_rate"] = mean(label_items)
    return out
