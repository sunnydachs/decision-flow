"""確率の較正の指標。

対象は「公式に確率として定義された値」のみ(noul / choice の確率、ロジスティック回帰の確率など)。
期待スコアや confidence は確率ではないため、ここでは扱わない。
すべて二値リスク(高リスクかどうか)の確率を前提とする。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_EPS = 1e-9


def reliability_bins(probs: Sequence[float], y: Sequence[int], n_bins: int = 10) -> list[dict]:
    """等頻度ビンの信頼性データ(ビンごとの平均確率と実測頻度)。"""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(probs) == 0:
        return []
    order = np.argsort(probs, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    out: list[dict] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        out.append(
            {
                "n": int(len(chunk)),
                "mean_prob": float(probs[chunk].mean()),
                "empirical": float(y[chunk].mean()),
                "gap": float(y[chunk].mean() - probs[chunk].mean()),
            }
        )
    return out


def ece(probs: Sequence[float], y: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error(等頻度ビン、サンプル数で重み付け)。"""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(probs) == 0:
        return float("nan")
    bins = reliability_bins(probs, y, n_bins)
    total = len(probs)
    return float(sum(b["n"] / total * abs(b["gap"]) for b in bins))


def brier_score(probs: Sequence[float], y: Sequence[int]) -> float:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(probs) == 0:
        return float("nan")
    return float(np.mean((probs - y) ** 2))


def log_loss(probs: Sequence[float], y: Sequence[int]) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0 - _EPS)
    y = np.asarray(y, dtype=float)
    if len(probs) == 0:
        return float("nan")
    return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))


def auroc(probs: Sequence[float], y: Sequence[int]) -> float:
    """AUROC(タイは平均順位で処理)。y は 0/1。"""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    sorted_p = probs[order]
    ranks = np.empty(len(probs), dtype=float)
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _logit(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def temperature_scale_fit(probs: Sequence[float], y: Sequence[int]) -> float:
    """NLL を最小化する温度 T を返す(T>1 で鈍らせる)。"""
    z = _logit(np.asarray(probs, dtype=float))
    y = np.asarray(y, dtype=float)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 240))
    losses = []
    for t in grid:
        p = _sigmoid(z / t)
        losses.append(float(-np.mean(y * np.log(p + _EPS) + (1.0 - y) * np.log(1.0 - p + _EPS))))
    return float(grid[int(np.argmin(losses))])


def apply_temperature(probs: Sequence[float], temperature: float) -> np.ndarray:
    return _sigmoid(_logit(np.asarray(probs, dtype=float)) / temperature)


def probability_extremity(probs: Sequence[float], edges: Sequence[float] = (0.02, 0.05, 0.95, 0.98)) -> dict:
    """確率が極端に偏っていないかの確認(指定しきい値未満/超の割合)。"""
    probs = np.asarray(probs, dtype=float)
    if len(probs) == 0:
        return {}
    out: dict[str, float] = {}
    for edge in edges:
        if edge < 0.5:
            out[f"share_<{edge}"] = float((probs < edge).mean())
        else:
            out[f"share_>{edge}"] = float((probs > edge).mean())
    out["mean"] = float(probs.mean())
    out["std"] = float(probs.std())
    return out
