"""目視確認サンプリング/一致率算出のテスト。"""
import csv
from pathlib import Path

import pytest

from common.dataset import Item
from runners.review_sample import score, stratified_sample, write_sample


def _items():
    out = []
    labels = ["urgent_claim", "billing", "technical", "sales", "other"]
    for i in range(50):
        label = labels[i % 5]
        out.append(Item(id=f"x-{i:03d}", text=f"text {i}", label=label,
                        severity="high" if label == "urgent_claim" else "normal", language="ja"))
    return out


def test_stratified_sample_covers_all_labels_and_size():
    sample = stratified_sample(_items(), 20, seed=1)
    assert len(sample) == 20
    assert {it.label for it in sample} == {"urgent_claim", "billing", "technical", "sales", "other"}
    assert len({it.id for it in sample}) == 20


def test_write_and_score_roundtrip(tmp_path):
    out = tmp_path / "review.csv"
    sample = stratified_sample(_items(), 10, seed=2)
    write_sample(sample, out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows[0]["reviewed_label"] == ""
    # 8件一致・2件不一致になるように記入する
    for i, row in enumerate(rows):
        row["reviewed_label"] = row["generated_label"] if i >= 2 else "billing"
        row["reviewed_severity"] = row["generated_severity"]
    out2 = tmp_path / "filled.csv"
    with open(out2, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = score(out2)
    assert result["reviewed"] == 10
    assert 0.0 <= result["label_agreement"] <= 1.0
    assert result["severity_agreement"] == 1.0
    assert isinstance(result["disagreements"], list)


def test_cohen_kappa_known_values():
    from runners.review_sample import cohen_kappa

    # 完全一致
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # 完全不一致 + 一様分布 -> κ は負(偶然より悪い)
    assert cohen_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)
    # 一致 6/10 で一様寄りの周辺分布 -> κ は負になる(生一致率 0.6 でもチャンス以下)
    a = ["x"] * 8 + ["y"] * 2
    b = ["x"] * 6 + ["y"] * 2 + ["x"] * 2
    assert cohen_kappa(a, b) == pytest.approx(-0.25)
    # 偏ったラベルでは生の一致率が κ を大きく上回ること(過大評価の実例)
    skewed_true = ["urgent"] * 9 + ["billing"]
    skewed_pred = ["urgent"] * 8 + ["billing"] * 2   # 生一致率 0.9
    raw = sum(1 for p, t in zip(skewed_pred, skewed_true) if p == t) / 10
    k = cohen_kappa(skewed_pred, skewed_true)
    assert k is not None and k < raw  # κ は一致率より厳しい


def test_cohen_kappa_none_cases():
    from runners.review_sample import cohen_kappa

    assert cohen_kappa([], []) is None
    assert cohen_kappa(["a"], ["a", "b"]) is None


def test_score_reports_kappa_and_anchors(tmp_path):
    from runners.review_sample import LITERATURE_ANCHORS

    sample = stratified_sample(_items(), 10, seed=3)
    out = tmp_path / "review.csv"
    write_sample(sample, out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    for i, row in enumerate(rows):
        row["reviewed_label"] = row["generated_label"] if i >= 1 else "other"
        row["reviewed_severity"] = row["generated_severity"]
    out2 = tmp_path / "filled.csv"
    with open(out2, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = score(out2)
    assert result["label_cohen_kappa"] is not None
    assert result["severity_cohen_kappa"] == 1.0
    # 文献アンカーが同封される(レポートで比較できるように)
    assert "trained_annotators_agreement" in result["literature_anchors"]
    assert "expert_kappa" in result["literature_anchors"]
    assert LITERATURE_ANCHORS["expert_kappa"]["value"] == 0.788
