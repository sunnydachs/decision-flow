"""生成 runner のヘルパーのテスト(ネットワーク不要)。"""
from pathlib import Path

from common.config import load_config
from tasks.base import load_task
from runners.generate import assign_ids, build_breakdown, extract_json_array, normalize_items

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"urgent_claim", "billing", "technical", "sales", "other"}


def test_extract_json_array_plain():
    rows = extract_json_array('[{"text": "a", "label": "billing", "severity": "normal"}]')
    assert len(rows) == 1


def test_extract_json_array_with_code_fence_and_prose():
    content = 'どうぞ:\n```json\n[{"text": "a", "label": "billing", "severity": "normal"}]\n```'
    assert len(extract_json_array(content)) == 1
    assert len(extract_json_array('はい: [{"text":"a","label":"other","severity":"high"}] 以上')) == 1


def test_extract_json_array_from_object_wrapper():
    content = '{"items": [{"text": "a", "label": "sales", "severity": "normal"}]}'
    assert len(extract_json_array(content)) == 1


def test_extract_json_array_garbage():
    assert extract_json_array("生成できませんでした") == []
    assert extract_json_array("") == []


def test_normalize_items_filters_invalid_rows():
    rows = [
        {"text": "ok", "label": "billing", "severity": "normal", "bucket": "normal"},
        {"text": "", "label": "billing", "severity": "normal"},
        {"text": "x", "label": "unknown_label", "severity": "normal"},
        {"text": "y", "label": "sales", "severity": "critical"},
    ]
    items, problems = normalize_items(rows, LABELS, prefix="tst", start=1)
    assign_ids(items, "tst")
    assert len(items) == 1
    assert items[0].id == "tst-0001"
    assert items[0].label == "billing"
    assert items[0].meta == {"bucket": "normal"}
    assert len(problems) == 3


def test_assign_ids_is_unique_and_sequential_after_dedup():
    # 重複除去で件数がずれても id が衝突しないこと(過去の不具合の回帰テスト)
    rows = [{"text": f"t{i}", "label": "billing", "severity": "normal"} for i in range(3)]
    items, _ = normalize_items(rows, LABELS, prefix="p", start=1)
    assign_ids(items, "p")
    assert [it.id for it in items] == ["p-0001", "p-0002", "p-0003"]
    # 2 回目のバッチで 1 件落ちても、前のバッチの id と重複しない
    rows2 = [{"text": "t3", "label": "sales", "severity": "normal"},
             {"text": "", "label": "sales", "severity": "normal"}]
    items2, _ = normalize_items(rows2, LABELS, prefix="p", start=4)
    merged = items + items2
    assign_ids(merged, "p")
    assert len({it.id for it in merged}) == len(merged) == 4


def test_normalize_items_marks_ambiguous():
    rows = [{"text": "動きません。", "label": "technical", "severity": "normal", "ambiguous": True, "bucket": "ambiguous"}]
    items, _ = normalize_items(rows, LABELS, prefix="tst", start=1)
    assert items[0].ambiguous is True


def test_build_breakdown_counts_urgent():
    text = build_breakdown(20, 0.27)
    assert "urgent_claim(severity=high): 5 件" in text  # round(20*0.27)=5
    assert "その他" in text


def test_data_generation_config_uses_nim_and_separate_lineage():
    config = load_config(repo_root=REPO_ROOT)
    assert config.get("data_generation.provider") == "nim"
    assert config.get("data_generation.key_env") == "NVIDIA_API_KEY"
    model = str(config.get("data_generation.model"))
    # 評価対象モデルとは別系統であること
    for evaluated in ("mercury", "pplx-decider", "jev", "qwen3.8-27b"):
        assert evaluated not in model
    assert config.get("data_generation.extra_body") == {"thinking": False}


def test_task_definitions_load():
    task = load_task(REPO_ROOT / "config" / "tasks" / "support_classification.toml")
    assert task.choice_instruction is not None
    question = task.choice_question()
    assert "instructions" in question["category"]
