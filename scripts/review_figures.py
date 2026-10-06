#!/usr/bin/env python3
"""図を vision モデルに読ませて検証する(画像を目視できない環境向け)。

vision 対応の OpenAI 互換エンドポイントに data URL で画像を送り、凡例・軸ラベル・
可読性の問題を機械的に報告させる。モデルID・エンドポイント・キー名は
環境ごとに --endpoint/--models/--key-env または config/local.toml で設定する。

使い方:
  python3 scripts/review_figures.py --glob "results/figures/fig*.png" \
      --models <visionモデルID> --endpoint <OpenAI 互換エンドポイント> \
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


def read_image(model: str, key: str, image: Path, *, endpoint: str, timeout: float = 240.0,
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
        # reasoning 系モデルは reasoning がトークン予算を使い切り、HTTP 200 + content 空になる。
        # 無効化フラグで回避できる(対応しないサーバは未知キーを無視する)。
        payload["reasoning_effort"] = "none"
        payload["thinking"] = False
        client = HttpClient(timeout_seconds=timeout, max_retries=0)
        started = time.perf_counter()
        result = client.post_json(endpoint, payload, headers={"Authorization": f"Bearer {key}"})
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
    parser = argparse.ArgumentParser(description="vision モデルで図を読んで検証する")
    parser.add_argument("--glob", default="results/figures/fig*.png")
    parser.add_argument("--models", required=True, help="vision モデルID(カンマ区切り)")
    parser.add_argument("--endpoint", required=True, help="OpenAI 互換 chat/completions エンドポイント")
    parser.add_argument("--key-env", default="VISION_API_KEY", help="キーの環境変数名")
    parser.add_argument("--out", default=None, help="結果 JSON の保存先")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    images = sorted(repo_root.glob(args.glob))
    if not images:
        print(f"画像が見つかりません: {args.glob}", file=sys.stderr)
        return 1
    key = lookup_key(args.key_env, [repo_root / ".env"])
    if not key:
        print(f"{args.key_env} が見つかりません", file=sys.stderr)
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    jobs = [(model, image) for model in models for image in images]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: read_image(j[0], key, j[1], endpoint=args.endpoint), jobs))

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
