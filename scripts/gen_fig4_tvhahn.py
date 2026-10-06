#!/usr/bin/env python3
"""A/B 比較用: fig4(7方式トレードオフ)を tvhahn 流儀で再作成する。

- tvhahn/matplotlib-skill P1 (horizontal bar) の Signature を適用:
  despined / bar-end value labels / hidden x-ticks / dimgrey annotations /
  insight annotation below axes / baseline axvline
- データは既存の artifacts(thresholds.json + raw JSONL)から派生(ハードコード禁止)
- 出力: results/figures_ab/fig4_tvhahn.png (+svg) — 現行 fig4_tradeoff_nocost と並べる用
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from evaluation.risk_coverage import automation_rate, recall_at_threshold  # noqa: E402

OUT = REPO / "results" / "figures_ab"
OUT.mkdir(parents=True, exist_ok=True)

# --- Data (derive from artifacts; never hardcode) ---
thresholds = json.loads((REPO / "config" / "thresholds.json").read_text())["methods"]
items = {}
for line in (REPO / "data" / "test" / "support_classification.jsonl").read_text().splitlines():
    if line.strip():
        d = json.loads(line)
        items[d["id"]] = d

LABELS = {  # 記事の表現と一致させる表示名
    "rule": "keyword rules",
    "embedding_lr": "embeddings + LR",
    "llm_prompt": "general LLM (prompt)",
    "llm_json_schema": "general LLM (schema)",
    "pplx_decider": "pplx-decider",
    "jev": "Jev",
    "mercury": "decision-model",
}
ACCENT = "#bd0c0c"  # tvhahn ACCENT_RED
GREY = "#9aa0a6"

rows = []
for method, label in LABELS.items():
    raw_p = REPO / "results" / "raw" / f"support_classification__{method}.jsonl"
    if not raw_p.exists():
        continue
    probs, ys = [], []
    for line in raw_p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        it = items.get(r.get("item_id"))
        if it is None or r.get("error_type") or r.get("risk_score") is None:
            continue
        probs.append(float(r["risk_score"]))
        ys.append(1 if it["severity"] == "high" else 0)
    th = (thresholds.get(method) or {}).get("review_above")
    if th is None or not probs:
        continue
    t = float(th)
    rows.append({
        "method": method, "label": label,
        "auto": sum(1 for p in probs if p < t) / len(probs),  # = automation_rate (P < review_above)
        "recall": recall_at_threshold(probs, ys, t),
        "misses": sum(1 for p, y in zip(probs, ys) if y == 1 and p < t),
    })
rows.sort(key=lambda r: r["auto"])  # 横棒は小さい順に積み上げ(上が最良)

# --- tvhahn style invariants ---
sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

n = len(rows)
fig_h = max(4, 1 + n * 0.7)
fig, ax = plt.subplots(figsize=(max(7, fig_h * 0.95), fig_h), dpi=150)

# neutral ranking: grey bars + ACCENT_RED on the standout (highest automation)
top = max(rows, key=lambda r: r["auto"])
colors = [ACCENT if r["method"] == top["method"] else GREY for r in rows]
ypos = np.arange(n)
ax.barh(ypos, [r["auto"] for r in rows], height=0.62, color=colors, zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels([r["label"] for r in rows], fontsize=10.5)

# bar-end value labels (the labels carry the numbers, not the axis)
for y, r in zip(ypos, rows):
    ax.text(r["auto"] + 0.012, y, f"{r['auto']:.1%}", ha="left", va="center",
            weight="semibold", size=10.5,
            color=ACCENT if r["method"] == top["method"] else "dimgrey")

# baseline + 95% reference line (kept: the safety floor is the chart's message)
ax.axvline(x=0, color="lightgrey", linewidth=0.8, zorder=0)
ax.axvline(x=0.95, color="dimgrey", linestyle=":", linewidth=1.1, zorder=1)
ax.text(0.955, n - 0.4, "95% floor", fontsize=8.5, color="dimgrey", va="top")

ax.set_xlim(0, 1.0)
ax.set_xticks([])  # labels carry the numbers
ax.grid(False)
ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
ax.set_ylabel("")
ax.set_xlabel("")
ax.set_title("Share of the inbox each method can auto-process\n"
             "while missing fewer than 5% of urgent claims (test n=500)",
             fontsize=13, loc="left", pad=7, color="dimgrey")
# insight annotation BELOW the axes (never inside where it can overlap rows)
best_feasible = [r for r in rows if r["recall"] >= 0.95]
if best_feasible:
    b = max(best_feasible, key=lambda r: r["auto"])
    fig.text(0.98, 0.01,
             f"Among methods meeting the 95% floor, {b['label']} automates the most "
             f"({b['auto']:.1%}) — {b['misses']} urgent claims missed in 500",
             ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic")
sns.despine(left=True, bottom=True)

fig.savefig(OUT / "fig4_tvhahn.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig4_tvhahn.svg", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT/'fig4_tvhahn.png'}")
print(f"rows: {n} (derive from artifacts)")
