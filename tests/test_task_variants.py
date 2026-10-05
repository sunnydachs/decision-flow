"""言い換え・選択肢順序の変異版のテスト(ネットワーク不要)。"""
from pathlib import Path

from tasks.base import load_task, load_task_variants

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = REPO_ROOT / "config" / "tasks" / "support_classification.toml"


def test_variants_are_defined():
    variants = load_task_variants(TASK)
    assert "base" in variants
    assert "paraphrase_1" in variants and "paraphrase_2" in variants
    assert "order_rotated" in variants
    assert len(variants) >= 4


def test_all_variants_keep_the_same_label_set_and_risk_label():
    base = load_task(TASK)
    for name, task in load_task_variants(TASK).items():
        assert set(task.labels) == set(base.labels), name
        assert task.risk_label == base.risk_label, name
        assert task.id == base.id, name


def test_paraphrases_change_only_the_instruction():
    base = load_task(TASK)
    variants = load_task_variants(TASK)
    assert variants["paraphrase_1"].instruction != base.instruction
    assert variants["paraphrase_1"].categories == base.categories


def test_order_rotated_changes_order_but_not_set():
    base = load_task(TASK)
    rotated = load_task_variants(TASK)["order_rotated"]
    assert rotated.labels != base.labels           # 並び順が変わる
    assert set(rotated.labels) == set(base.labels)  # 集合は同じ
    assert rotated.labels[0] != base.labels[0]      # 先頭が変わる
