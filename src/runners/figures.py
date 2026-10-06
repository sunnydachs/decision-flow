"""記事用の図を生成する(すべて英字ラベル。データは results/ の成果物から読むだけ)。

方針: accuracy は全方式 0.88-0.91 に収まって差が見えないので、**差が出る形**を描く。
  1. fig1_calibration      信頼性図(較正の形)。確率が張り付く方式と滑らかな方式の差。
  2. fig2_risk_coverage    リスク-カバレッジ曲線。固定Recall 95% の線と各方式の閾値。
  3. fig3_probability_shape リスク確率の分布(0/1 への張り付き具合)。
  4. fig4_tradeoff         レイテンシ × 自動化率 × コスト の散布図(ぱっと見の比較)。
  5. fig5_stability        同一入力の反転率と言い換え・選択肢順序の不一致率。

matplotlib は Agg(ヘッドレス)。日本語は使わない(フォント依存を避けるため英語)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 文字をパス化せず <text> として残す(記事用に編集しやすく、ラベルの機械検証もできる)
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.size"] = 10.5
matplotlib.rcParams["axes.titlesize"] = 11.5
matplotlib.rcParams["axes.labelsize"] = 10.5
matplotlib.rcParams["xtick.labelsize"] = 9.5
matplotlib.rcParams["ytick.labelsize"] = 9.5
matplotlib.rcParams["legend.fontsize"] = 9.5

from common.config import REPO_ROOT, load_config  # noqa: E402
from common.dataset import load_jsonl  # noqa: E402
from evaluation.calibration import reliability_bins  # noqa: E402
from evaluation.risk_coverage import automation_rate, recall_at_threshold  # noqa: E402

METHODS = ("rule", "embedding_lr", "llm_prompt", "llm_json_schema", "pplx_decider", "jev", "mercury")
PROB_METHODS = ("embedding_lr", "pplx_decider", "jev", "mercury")  # 公式に確率と定義された値のみ
COLORS = {
    "rule": "#6b7280",          # grey
    "embedding_lr": "#2563eb",  # blue
    "llm_prompt": "#16a34a",    # green
    "llm_json_schema": "#ea580c",  # orange (緑と紛らかわないよう別色相)
    "pplx_decider": "#db2777",  # pink
    "jev": "#7c3aed",           # violet
    "mercury": "#0891b2",       # cyan
}
MARKERS = {"rule": "s", "embedding_lr": "o", "llm_prompt": "^", "llm_json_schema": "v",
           "pplx_decider": "D", "jev": "P", "mercury": "X"}


def _load_raw(results: Path, method: str, split: str = "raw") -> list[dict]:
    path = results / split / f"support_classification__{method}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dataset(repo_root: Path, split: str) -> dict:
    items = load_jsonl(repo_root / "data" / split / "support_classification.jsonl")
    return {it.id: it for it in items}


def _risk_arrays(results: Path, repo_root: Path, method: str,
                 data_split: str = "test", raw_split: str = "raw"):
    """(risk_score のリスト, severity=high の 0/1 リスト)を返す。

    data_split は正解データ(test/calib)、raw_split は生応答のディレクトリ(raw/raw_calib)。
    """
    items = _dataset(repo_root, data_split)
    probs, ys = [], []
    for rec in _load_raw(results, method, raw_split):
        item = items.get(rec.get("item_id"))
        if item is None or rec.get("error_type") or rec.get("risk_score") is None:
            continue
        probs.append(float(rec["risk_score"]))
        ys.append(1 if item.severity == "high" else 0)
    return probs, ys


def _thresholds(repo_root: Path) -> dict:
    path = repo_root / "config" / "thresholds.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("methods", {})


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".svg"):
        path = out_dir / f"{name}{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


# --- 1. 信頼性図 -------------------------------------------------------------
def fig_calibration(results: Path, repo_root: Path, out_dir: Path, bins: int = 10):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1.2, label="perfect calibration")
    for method in PROB_METHODS:
        probs, ys = _risk_arrays(results, repo_root, method)
        if not probs:
            continue
        rb = reliability_bins(probs, ys, n_bins=bins)
        xs = [b["mean_prob"] for b in rb if b.get("n")]
        yv = [b["empirical"] for b in rb if b.get("n")]
        ax.plot(xs, yv, marker=MARKERS[method], color=COLORS[method], lw=1.6, ms=5, label=method)
    ax.set_xlabel("predicted probability of high risk")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (reliability diagram, test n=500)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out_dir, "fig1_calibration")


# --- 2. リスク-カバレッジ ----------------------------------------------------
def fig_risk_coverage(results: Path, repo_root: Path, out_dir: Path, target: float = 0.95):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    thresholds = _thresholds(repo_root)
    for method in METHODS:
        probs, ys = _risk_arrays(results, repo_root, method)
        if not probs:
            continue
        grid = np.unique(np.quantile(probs, np.linspace(0, 1, 120)))
        xs = [automation_rate(probs, float(t)) for t in grid]
        yv = [recall_at_threshold(probs, ys, float(t)) for t in grid]
        ax.plot(xs, yv, color=COLORS[method], lw=1.8, label=method)
        th = (thresholds.get(method) or {}).get("review_above")
        if th is not None:
            ax.plot([automation_rate(probs, float(th))], [recall_at_threshold(probs, ys, float(th))],
                    marker=MARKERS[method], color=COLORS[method], ms=9,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.axhline(target, color="#111827", ls=":", lw=1.4)
    ax.text(0.02, target + 0.006, f"recall target {target:.0%}", fontsize=8, color="#111827")
    ax.set_xlabel("automation rate (share auto-processed)")
    ax.set_ylabel("high-risk recall")
    ax.set_title("Risk–coverage: what can be automated at a fixed recall floor?\n"
                 "(markers = threshold chosen on calib, applied unchanged to test)")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.55, 1.02)
    ax.grid(alpha=0.25)
    # 7方式になると図内の空きが足りず曲線と重なるため、凡例は図の外(右)に置く
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    return _save(fig, out_dir, "fig2_risk_coverage")


# --- 3. 確率の形 -------------------------------------------------------------
def fig_probability_shape(results: Path, repo_root: Path, out_dir: Path):
    edges = [(-0.001, 0.001), (0.001, 0.02), (0.02, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 0.98),
             (0.98, 0.999), (0.999, 1.001)]
    labels = ["= 0", "0-0.02", "0.02-0.1", "0.1-0.5", "0.5-0.9", "0.9-0.98", "0.98-1", "= 1"]
    methods = [m for m in METHODS if _risk_arrays(results, repo_root, m)[0]]
    data = {m: np.zeros(len(edges)) for m in methods}
    for m in methods:
        probs, _ = _risk_arrays(results, repo_root, m)
        p = np.asarray(probs)
        for i, (lo, hi) in enumerate(edges):
            data[m][i] = float(((p > lo) & (p <= hi)).mean())
    x = np.arange(len(labels))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for i, m in enumerate(methods):
        ax.bar(x + i * width - 0.4 + width / 2, data[m], width,
               color=COLORS[m], label=m, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("risk-score band")
    ax.set_ylabel("share of items")
    ax.set_title("Where each method puts its probability mass (test n=500)\n"
                 "rule and jev collapse onto the extremes; pplx_decider and embedding_lr stay continuous")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    return _save(fig, out_dir, "fig3_probability_shape")


# --- 4. トレードオフ(表形式プロット) ----------------------------------------
def fig_tradeoff(results: Path, repo_root: Path, out_dir: Path, target: float = 0.95,
                 show_cost: bool = True):
    """レイテンシ × 自動化率(× コスト)。散布図+接続線は7方式になるとラベルが重なるため、
    行=方式の表形式プロットにする(調整ループを排除し、数値を読み取りやすくする)。
    show_cost=False で記事用のコスト列なし版(fig4_tradeoff_nocost)。"""
    thresholds = _thresholds(repo_root)
    rows = []
    for method in METHODS:
        probs, ys = _risk_arrays(results, repo_root, method)
        raw = _load_raw(results, method)
        if not probs or not raw:
            continue
        lat = [r["latency_ms"] for r in raw if r.get("latency_ms") is not None and not r.get("error_type")]
        cost = sum(r.get("estimated_cost_usd") or 0.0 for r in raw)
        th = (thresholds.get(method) or {}).get("review_above")
        if th is None:
            continue
        auto = automation_rate(probs, float(th))
        recall = recall_at_threshold(probs, ys, float(th))
        p50 = float(np.percentile(lat, 50)) if lat else 0.0
        per_1000 = cost / max(1, len(raw)) * 1000.0
        misses = sum(1 for p, y in zip(probs, ys) if y == 1 and p < th)
        rows.append((method, p50, auto, recall, per_1000, misses))
    # 自動化率の降順(= 運用上の魅力順)で並べる
    rows.sort(key=lambda r: (-r[2], r[1]))

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 0.62 * n + 1.7), sharex=False,
                             gridspec_kw={"hspace": 0.55})
    if n == 1:
        axes = [axes]
    bar_colors = [COLORS[r[0]] for r in rows]
    for ax, (method, p50, auto, recall, per_1000, misses) in zip(axes, rows):
        ax.barh([0], [auto], color=COLORS[method], height=0.62, zorder=3)
        ax.axvline(target, color="#111827", ls=":", lw=1.1, zorder=2)
        ax.set_xlim(0, 1.0)
        ax.set_yticks([])
        ax.set_ylim(-0.55, 0.75)
        ax.grid(axis="x", alpha=0.25, zorder=1)
        ax.text(-0.02, 0.02, method, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10, color=COLORS[method], fontweight="bold")
        note = (f"recall {recall:.3f}   misses {misses}   p50 {p50:.0f} ms")
        if show_cost:
            note += f"   ${per_1000:.4f}/1k"
        ax.text(1.005, 0.02, note, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8.6, color="#374151")
        if auto > 0.03:
            ax.text(auto + 0.015, 0, f"{auto:.1%}", ha="left", va="center",
                    fontsize=9, color=COLORS[method], zorder=4)
        else:
            ax.text(0.012, 0, "0% (recall floor cannot be met otherwise)",
                    ha="left", va="center", fontsize=8.2, color="#6b7280", zorder=4)
    axes[0].set_title(
        "Automation rate at the calib threshold (recall floor 95% dotted)\n"
        + ("methods sorted by automation; per-method recall / misses / latency / cost on the right"
           if show_cost else
           "methods sorted by automation; per-method recall / misses / latency on the right"),
        fontsize=11.5, loc="left")
    fig.subplots_adjust(left=0.17, right=0.80)
    return _save(fig, out_dir, "fig4_tradeoff" if show_cost else "fig4_tradeoff_nocost")


# --- 5. 安定性 ---------------------------------------------------------------
def fig_stability(repo_root: Path, out_dir: Path):
    runs_path = repo_root / "results" / "stability.json"
    var_path = repo_root / "results" / "stability_variants.json"
    if not runs_path.exists():
        return []
    runs = json.loads(runs_path.read_text(encoding="utf-8"))["methods"]
    variants = json.loads(var_path.read_text(encoding="utf-8"))["methods"] if var_path.exists() else {}
    methods = [m for m in METHODS if m in runs or m in variants]
    # 未測定の方式は図に注記する(空欄を「安定=0」と誤読させないため)
    unmeasured = [m for m in METHODS if m not in methods]
    same = [runs.get(m, {}).get("label_disagreement_rate") or 0.0 for m in methods]
    par = [variants.get(m, {}).get("label_disagreement_rate") or 0.0 for m in methods]
    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    b1 = ax.bar(x - width / 2, same, width, color="#0ea5e9", label="same input, 20 runs")
    b2 = ax.bar(x + width / 2, par, width, color="#f59e0b",
                label="paraphrased instruction / rotated option order")
    # 0% も明示する(棒が無いだけだと「データ欠損」に見えるため)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.0%}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9,
                        color="#374151")
    ax.set_xticks(x)
    # 長い名前も1行で(アンダースコア改行は「分割ラベル」に見えるため。代わりに図を広げる)
    ax.set_xticklabels(methods, fontsize=9.5)
    fig.set_size_inches(10.5, 4.8)
    ax.set_ylabel("share of items whose label changed")
    ax.set_title("Stability (50-item subset)\n"
                 "the local baseline and pplx_decider never flip; the LLM flips on every axis")
    ax.annotate("0% = never changed", xy=(0.01, 0.93), xycoords="axes fraction",
                fontsize=9, color="#374151")
    if unmeasured:
        ax.annotate(f"not measured: {', '.join(unmeasured)}",
                    xy=(0.99, 0.93), xycoords="axes fraction", ha="right", fontsize=9, color="#6b7280")
    ax.set_ylim(0, max(same + par + [0.01]) * 1.35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "fig5_stability")


def build_all(repo_root: Path = REPO_ROOT, out_dir: Path | None = None) -> dict[str, list[str]]:
    config = load_config(repo_root=repo_root)
    results = repo_root / "results"
    out_dir = out_dir or results / "figures"
    target = float(config.get("evaluation.recall_target", 0.95))
    bins = int(config.get("evaluation.calibration_bins", 10))
    written: dict[str, list[str]] = {}

    def record(name: str, paths: list[Path]) -> None:
        if paths:
            # relative to the repo root: absolute machine paths leak the personal
            # environment when index.json is committed
            written[name] = [str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT)
                             else str(p) for p in paths]

    record("fig1_calibration", fig_calibration(results, repo_root, out_dir, bins=bins))
    record("fig2_risk_coverage", fig_risk_coverage(results, repo_root, out_dir, target=target))
    record("fig3_probability_shape", fig_probability_shape(results, repo_root, out_dir))
    record("fig4_tradeoff", fig_tradeoff(results, repo_root, out_dir, target=target))
    record("fig4_tradeoff_nocost", fig_tradeoff(results, repo_root, out_dir, target=target,
                                                show_cost=False))
    record("fig5_stability", fig_stability(repo_root, out_dir))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="記事用の図を生成する(英字ラベル)")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--json", default="results/figures/index.json")
    args = parser.parse_args(argv)
    written = build_all(out_dir=Path(REPO_ROOT / args.out))
    index = Path(REPO_ROOT / args.json)
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, paths in written.items():
        print(f"{name}: {', '.join(Path(p).name for p in paths)}")
    print(f"index -> {index}")
    return 0 if len(written) >= 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
