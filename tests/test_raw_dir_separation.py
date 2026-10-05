"""calib と test の生応答が衝突しないこと(--raw-dir 分離)の回帰テスト。

background: benchmark は生応答を results/raw/<dataset_stem>__<method>.jsonl に書くため、
calib と test が同じ stem だと後から実行した方が上書きしてしまう事故が起きた。
"""
from pathlib import Path

import json

from common.config import load_config
from common.dataset import Item, write_jsonl
from runners.benchmark import run_benchmark
from runners.recover_raw import adapter_model_id

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dataset(tmp_path: Path, name: str, n: int = 3) -> Path:
    path = tmp_path / name / "support_classification.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [
        Item(id=f"{name}-{i:03d}", text=f"本文 {i} のクレーム", label="urgent_claim",
             severity="high", language="ja")
        for i in range(n)
    ]
    write_jsonl(items, path)
    (path.parent / "manifest.json").write_text(
        json.dumps({"source": name, "license": "internal", "data_class": "synthetic",
                    "external_ok": True, "task": "support_classification"}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_benchmark_writes_raw_to_given_raw_dir(tmp_path):
    config = load_config(repo_root=REPO_ROOT)
    dataset = _dataset(tmp_path, "calib")
    raw_dir = tmp_path / "raw_calib"
    summary = run_benchmark(
        dataset_path=dataset, adapter_names=["rule"], config=config,
        repo_root=REPO_ROOT, out_dir=tmp_path / "results", raw_dir=raw_dir,
    )
    assert summary["methods"]["rule"]["n"] == 3
    assert (raw_dir / "support_classification__rule.jsonl").exists()
    assert str(summary["methods"]["rule"]["out"]).startswith(str(raw_dir))


def test_two_datasets_with_same_stem_do_not_collide(tmp_path):
    """calib と test が同じ stem でも、raw-dir を分ければ別ファイルに残る。"""
    config = load_config(repo_root=REPO_ROOT)
    calib = _dataset(tmp_path, "calib", n=2)
    test = _dataset(tmp_path, "test", n=2)
    run_benchmark(dataset_path=calib, adapter_names=["rule"], config=config,
                  repo_root=REPO_ROOT, out_dir=tmp_path / "results", raw_dir=tmp_path / "raw_calib")
    run_benchmark(dataset_path=test, adapter_names=["rule"], config=config,
                  repo_root=REPO_ROOT, out_dir=tmp_path / "results", raw_dir=tmp_path / "raw")
    calib_lines = (tmp_path / "raw_calib" / "support_classification__rule.jsonl").read_text().splitlines()
    test_lines = (tmp_path / "raw" / "support_classification__rule.jsonl").read_text().splitlines()
    assert len(calib_lines) == 2 and len(test_lines) == 2
    assert '"calib-000"' in calib_lines[0] and '"test-000"' in test_lines[0]


def test_adapter_model_id_matches_cache_key_sources(tmp_path):
    config = load_config(repo_root=REPO_ROOT)
    assert adapter_model_id("rule", config) == "keyword-rules"
    assert adapter_model_id("embedding_lr", config) == config.get("embedding.model")
    assert adapter_model_id("llm_prompt", config) == config.get("models.llm.model")
    assert adapter_model_id("llm_json_schema", config) == config.get("models.llm.model")
    assert adapter_model_id("mercury", config) == config.get("models.mercury.model")
    assert adapter_model_id("pplx_decider", config) == config.get("models.pplx_decider.model")
