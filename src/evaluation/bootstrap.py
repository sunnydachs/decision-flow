"""ブートストラップ信頼区間(パーセンタイル法)。

指標関数 stat_fn(*arrays) を、配列と同じ長さでリサンプリングして分布を作る。
乱数 seed を固定できるため再現可能。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    stat_fn: Callable[..., float],
    arrays: Sequence[np.ndarray],
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 20261004,
) -> tuple[float, float, float]:
    """(point, lo, hi) を返す。arrays は同じ長さの 1 次元配列の列。"""
    arrs = [np.asarray(a) for a in arrays]
    if not arrs:
        raise ValueError("arrays が空です")
    n = len(arrs[0])
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat_fn(*arrs))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        stats[i] = stat_fn(*[a[idx] for a in arrs])
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)
