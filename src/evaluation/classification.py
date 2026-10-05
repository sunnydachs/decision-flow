"""分類品質の指標(accuracy / 混同行列 / クラス別 precision・recall・F1)。"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def error_rate(y_true: Sequence, y_pred: Sequence) -> float:
    """1 - accuracy(自動処理した分の誤り率などに使う)。"""
    acc = accuracy(y_true, y_pred)
    return float("nan") if np.isnan(acc) else 1.0 - acc


def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence[str]) -> np.ndarray:
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in index and pred in index:
            matrix[index[true], index[pred]] += 1
    return matrix


def per_class_metrics(y_true: Sequence, y_pred: Sequence, labels: Sequence[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        support = int((y_true == label).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = float("nan") if (np.isnan(precision) or np.isnan(recall)) else 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return result


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence[str]) -> float:
    per_class = per_class_metrics(y_true, y_pred, labels)
    values = [m["f1"] for m in per_class.values() if not np.isnan(m["f1"])]
    return float(np.mean(values)) if values else float("nan")


def summaries_by_flag(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    flags: Sequence[bool],
) -> dict[str, dict]:
    """フラグ別(例: ambiguous)の正答率・件数。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    flags = np.asarray(flags)
    out: dict[str, dict] = {}
    for name, mask in (("all", np.ones(len(y_true), dtype=bool)), ("flagged", flags), ("unflagged", ~flags)):
        n = int(mask.sum())
        out[name] = {"n": n, "accuracy": accuracy(y_true[mask], y_pred[mask]) if n else None}
    return out
