#!/usr/bin/env python3
"""図を 外部エンドポイントの vision モデルに読ませて検証する(画像を目視できない環境向け)。

本モデルが vision 非対応でも、外部エンドポイントの vision モデルに data URL で画像を送れば読める。
実測(2026-10-06):
  reviewer-a … 速く(1-8s)正確。既定の第一選択
  reviewer-b             … 最も正確だが遅い(50-90s)。第2の独立検証に使う
  meta/llama-3.2-11b-vision-instruct          … 速いが細部を落とす(凡例を数え落とした)
  meta/llama-3.2-90b-vision-instruct          … 軸ラベルを「不明」と誤答(不採用)

使い方:
  python3 scripts/review_figures.py --glob "results/figures/fig*.png" \
      --models reviewer-a,reviewer-b \
      --out results/figures/review.json

終了コード: 1 件でも失敗があれば 1(検証を飛ばしたまま成功扱いにしない)。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.env import lookup_key  # noqa: E402
from common.http import HttpClient  # noqa: E402

ENDPOINT = "https://<external-endpoint>/v1/chat/completions"
DEFAULT_MODELS = [
    "reviewer-a",
    "reviewer-b",
]
QUESTION = (
    "Read this figure carefully and report factually:\n"
    "(A) the exact title,\n"
    "(B) every legend entry, in order,\n"
    "(C) every axis label and tick label you can read,\n"
    "(D) any legibility problem you can actually see (overlapping text, labels cut off at the edges, "
    "text too small, colors you cannot distinguish, blank areas where data seems missing). "
    "If there is none, say 'clean'.\n"
    "(E) one line: what does the figure claim?\n"
    "Only report what you can read. Say 'cannot read' rather than guessing."
)


def read_image(model: str, key: str, image: Path, *, timeout: float = 240.0,
               retries: int = 3, max_tokens: int = 1200) -> dict:
    data = base64.b64encode(image.read_bytes()).decode()
    last = ""
    for attempt in range(retries):
        # 空応答(reasoning が max_tokens を食う)を避けるため、再試行ごとに上限を上げる
        budget = max_tokens * (attempt + 1)
        payload = {
            "model": model,
            "max_tokens": budget,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                ],
            }],
        }
        client = HttpClient(timeout_seconds=timeout, max_retries=0)
        started = time.perf_counter()
        result = client.post_json(ENDPOINT, payload, headers={"Authorization": f"Bearer {key}"})
        elapsed = time.perf_counter() - started
        if result.ok:
            choice = result.body["choices"][0]
            content = (choice.get("message") or {}).get("content") or ""
            # HTTP 200 でも content が空のことがある(実測: deepseek が低い max_tokens で空を返す)。
            # これを成功として数えると「検証したつもりで読めていない」状態になる。
            if content.strip():
                return {
                    "model": model, "image": str(image), "ok": True,
                    "seconds": round(elapsed, 1), "max_tokens": budget,
                    "text": content,
                }
            last = f"empty content (max_tokens={budget}, finish={choice.get('finish_reason')})"
        else:
            last = f"{result.status} {result.error_type}"
        time.sleep(3)
    return {"model": model, "image": str(image), "ok": False, "seconds": None, "text": f"FAIL {last}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="外部エンドポイントの vision モデルで図を読んで検証する")
    parser.add_argument("--glob", default="results/figures/fig*.png")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--out", default=None, help="結果 JSON の保存先")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    images = sorted(repo_root.glob(args.glob))
    if not images:
        print(f"画像が見つかりません: {args.glob}", file=sys.stderr)
        return 1
    key = lookup_key("LLM_API_KEY", [repo_root / ".env", Path<env-file>])
    if not key:
        print("LLM_API_KEY が見つかりません", file=sys.stderr)
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    jobs = [(model, image) for model in models for image in images]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: read_image(j[0], key, j[1]), jobs))

    for res in results:
        status = f"{res['seconds']}s" if res["ok"] else "FAIL"
        print(f"\n===== {Path(res['image']).name} @ {res['model']} ({status}) =====")
        print(res["text"])

    if args.out:
        out = repo_root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreview -> {out}")

    failures = [r for r in results if not r["ok"]]
    if failures:
        print(f"\n{len(failures)}/{len(results)} 件が失敗(検証未完了)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
