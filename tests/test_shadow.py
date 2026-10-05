"""shadow モード(既存判定と Decision Model の差分)のテスト。"""
import json
from pathlib import Path

from runners.shadow import load_jsonl
import runners.shadow as shadow


def _write(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def test_load_jsonl_accepts_id_or_item_id(tmp_path):
    path = _write(tmp_path / "a.jsonl", [{"id": "x", "label": "billing"}, {"item_id": "y", "label": "sales"}])
    out = load_jsonl(path)
    assert set(out) == {"x", "y"}


def test_shadow_reports_diffs_and_transitions(tmp_path, capsys):
    existing = _write(tmp_path / "existing.jsonl", [
        {"id": "a", "decision": "billing"},
        {"id": "b", "decision": "urgent_claim"},
        {"id": "c", "decision": "sales"},
    ])
    model = _write(tmp_path / "model.jsonl", [
        {"item_id": "a", "label": "billing", "risk_score": 0.01},          # 一致
        {"item_id": "b", "label": "technical", "risk_score": 0.2},        # 差分
        {"item_id": "c", "label": "sales", "risk_score": 0.0},            # 一致
    ])
    out_path = tmp_path / "diff.jsonl"
    code = shadow.main(["--existing", str(existing), "--model-raw", str(model), "--out", str(out_path)])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.split("diff ->")[0])
    assert summary["n_common"] == 3
    assert summary["n_diff"] == 1
    assert abs(summary["diff_rate"] - 1 / 3) < 1e-9
    assert summary["top_transitions"][0]["existing"] == "urgent_claim"
    # 差分の明細が JSONL で残る(監査のため)
    lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["id"] == "b" and lines[0]["decision_model"] == "technical"


def test_shadow_handles_disjoint_ids(tmp_path, capsys):
    existing = _write(tmp_path / "e.jsonl", [{"id": "a", "decision": "billing"}])
    model = _write(tmp_path / "m.jsonl", [{"item_id": "z", "label": "sales"}])
    out_path = tmp_path / "d.jsonl"
    shadow.main(["--existing", str(existing), "--model-raw", str(model), "--out", str(out_path)])
    summary = json.loads(capsys.readouterr().out.split("diff ->")[0])
    assert summary["n_common"] == 0
    assert summary["diff_rate"] is None
