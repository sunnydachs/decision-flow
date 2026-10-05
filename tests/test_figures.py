"""記事用の図の回帰テスト。

画像は目視できないので機械的に検証する:
  - 期待したファイルが生成される(PNG/SVG)
  - SVG に凡例・軸ラベルの文字が <text> として出ている(欠落していない)
  - PNG が空でない(インク被覆率が下限以上)
  - 図に描かれる数値が成果物(raw/しきい値)から再現できる
"""
from pathlib import Path

import numpy as np
import pytest

from runners.figures import METHODS, _risk_arrays, _thresholds, build_all

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def figures(tmp_path_factory):
    out = tmp_path_factory.mktemp("figures")
    written = build_all(repo_root=REPO_ROOT, out_dir=out)
    return out, written


def test_all_expected_figures_are_written(figures):
    _out, written = figures
    assert set(written) >= {
        "fig1_calibration", "fig2_risk_coverage", "fig3_probability_shape",
        "fig4_tradeoff", "fig5_stability",
    }
    for paths in written.values():
        for p in paths:
            assert Path(p).exists() and Path(p).stat().st_size > 1000


def test_svg_keeps_text_labels(figures):
    out, written = figures
    svgs = {Path(p).stem: Path(p) for p in out.glob("*.svg")}
    assert len(svgs) >= 5
    for stem, path in svgs.items():
        text = path.read_text(encoding="utf-8")
        assert "<text" in text, f"{stem} に文字が描かれていない(パス化されている)"
        assert len(text) > 3000


def test_png_has_visible_content(figures):
    import matplotlib.image as mpimg

    out, _written = figures
    for png in out.glob("*.png"):
        img = mpimg.imread(png)
        ink = float((img[..., :3] < 0.98).any(axis=-1).mean())
        assert 0.02 < ink < 0.60, f"{png.name} のインク被覆率が不自然: {ink:.3f}"


def test_probability_shape_matches_the_reported_collapse():
    """fig3 が描く『0/1 への張り付き』が実データと一致すること。"""
    probs, _y = _risk_arrays(REPO_ROOT / "results", REPO_ROOT, "jev")
    arr = np.asarray(probs)
    assert (arr == 0).mean() > 0.5   # jev は過半がちょうど 0
    assert (arr == 1).mean() > 0.2   # かつ 2 割超がちょうど 1
    smooth, _ = _risk_arrays(REPO_ROOT / "results", REPO_ROOT, "pplx_decider")
    s = np.asarray(smooth)
    assert (s == 0).mean() == 0.0 and (s == 1).mean() == 0.0  # pplx は張り付かない


def test_risk_coverage_inputs_cover_every_method_with_a_threshold():
    thresholds = _thresholds(REPO_ROOT)
    covered = 0
    for method in METHODS:
        probs, ys = _risk_arrays(REPO_ROOT / "results", REPO_ROOT, method)
        if not probs:
            continue
        assert set(ys) <= {0, 1}
        assert (thresholds.get(method) or {}).get("review_above") is not None
        covered += 1
    assert covered >= 6  # rule/embedding/llm x2/pplx/jev
