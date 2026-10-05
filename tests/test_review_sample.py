"""目視確認サンプリング/一致率算出のテスト。"""
import csv
from pathlib import Path

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
