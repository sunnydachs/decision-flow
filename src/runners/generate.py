"""評価データの生成(LLM による日本語問い合わせ文の合成)。

生成に使うモデルは評価対象とは別系統にする(config: data_generation.model)。
calib と test は**別の実行**で生成する(同じデータで較正と評価をしない)。

仕様(ユーザー指示):
  - 5カテゴリ + severity を指定。urgent_claim(severity=high)は全体の 25〜30%。
  - 難しいケース(丁寧な文面の緊急案件、怒りの表現がある通常質問、複数カテゴリにまたがる文、短文)と
    曖昧な文(ambiguous=true、通常評価とは別集計)を意図的に含める。
  - test の高リスクは 100件以上。
  - 生成後の目視確認はユーザーが行う(このスクリプトは生成と機械的な検証まで)。
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from adapters.base import utc_now_iso
from common.config import REPO_ROOT, load_config
from common.dataset import Item, write_jsonl
from common.env import require_key
from common.http import HttpClient
from tasks.base import load_task

ENDPOINT = "https://外部ルーティング.ai/api/v1/chat/completions"

BUCKETS = (
    "normal",           # 明確な問い合わせ
    "polite_urgent",    # 丁寧な文面だが緊急(見逃しやすい)
    "angry_normal",     # 怒りの表現があるが緊急ではない
    "multi_topic",      # 複数カテゴリにまたがる
    "very_short",       # 短文
    "ambiguous",        # 曖昧(ambiguous=true)
)

SYSTEM_PROMPT_TEMPLATE = """あなたはカスタマーサポートの評価データを作る日本語の専門家です。
ある企業(B2B SaaS・サブスクリプションサービス)のサポート窓口に届く問い合わせ文を、指定された内訳で作成してください。
出力は JSON 配列のみ。各要素は次のキーを持ちます:
  text (string, 日本語の問い合わせ文),
  label (string, 下記カテゴリ名のいずれか),
  severity (string, "high" | "normal"),
  bucket (string, 上で指定された種類),
  ambiguous (boolean)

カテゴリの定義(この定義に厳密に従うこと):
{categories}

規則:
- label と severity は独立。urgent_claim は原則 severity="high"。それ以外のラベルにも severity="high" を付けてよい
  (例: 締切間際の見積依頼、支払いが止まって業務が止まる、復旧が必要な不具合)。
- urgent_claim は「即時対応が必要な緊急のクレーム」に限る。単に商品の交換や配送遅延を求めるものは、
  billing / technical / other のいずれかに分類し、緊急性が高ければ severity="high" にする。
- 同じ文面の使い回しをしない。1件あたり 15〜120 文字程度。
- 出力は JSON 配列のみ。説明文やコードフェンスを付けない。"""

USER_TEMPLATE = """次の内訳で {n} 件の問い合わせ文を作成してください(合計 {n} 件)。
{breakdown}
今日の日付: {today}
seed: {seed}(文面のバリエーション用。同じ seed でも完全一致は避ける)"""


def build_breakdown(n: int, urgent_ratio: float) -> str:
    n_urgent = max(1, round(n * urgent_ratio))
    n_other = n - n_urgent
    lines = [
        f"- urgent_claim(severity=high): {n_urgent} 件",
        f"- その他(billing / technical / sales / other に分散、severity は主に normal): {n_other} 件",
        "- 内訳(種類): そのうち 丁寧な文面だが緊急(polite_urgent) を 1〜2 件、"
        "怒りの表現があるが緊急でない(angry_normal) を 1〜2 件、複数カテゴリ(multi_topic) を 1 件、"
        "短文(very_short) を 1 件、曖昧(ambiguous=true) を 1 件含める",
    ]
    return "\n".join(lines)


def extract_json_array(content: str) -> list[dict]:
    if not content:
        return []
    text = content.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = data.get("items") or data.get("data") or []
    return [d for d in data if isinstance(d, dict)]


def normalize_items(rows: list[dict], labels: set[str], prefix: str, start: int) -> tuple[list[Item], list[str]]:
    """行を検証して Item にする。id は呼び出し側で確定させる(重複除去後に連番を振る)。"""
    items: list[Item] = []
    problems: list[str] = []
    for i, row in enumerate(rows, start=start):
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip()
        severity = str(row.get("severity", "")).strip()
        if not text or label not in labels or severity not in ("high", "normal"):
            problems.append(f"row {i}: invalid (label={label!r}, severity={severity!r}, len={len(text)})")
            continue
        items.append(
            Item(
                id="",  # 重複除去のあとに連番を振る
                text=text,
                label=label,
                severity=severity,
                language="ja",
                ambiguous=bool(row.get("ambiguous", False)),
                meta={"bucket": str(row.get("bucket", ""))} if row.get("bucket") else {},
            )
        )
    return items, problems


def assign_ids(items: list[Item], prefix: str) -> None:
    """重複除去後の確定した順序で一意な連番 id を振る。"""
    for idx, item in enumerate(items, 1):
        item.id = f"{prefix}-{idx:04d}"


def generate(
    *,
    task,
    config,
    n: int,
    batch_size: int,
    prefix: str,
    seed: int,
    out_path: Path,
    key: str,
    max_batches: int | None = None,
) -> dict:
    endpoint = str(config.get("data_generation.endpoint"))
    models = [str(config.get("data_generation.model"))] + [
        str(m) for m in (config.get("data_generation.fallback_models") or [])
    ]
    max_tokens = int(config.get("data_generation.max_tokens", 4096))
    temperature = float(config.get("data_generation.temperature", 1.0))
    timeout = float(config.get("data_generation.timeout_seconds", 300))
    http = HttpClient(timeout_seconds=timeout, max_retries=2, backoff_base_seconds=2.0)
    extra_body = dict(config.get("data_generation.extra_body") or {})
    urgent_ratio = float(config.get("data_generation.urgent_ratio", 0.27))
    labels = set(task.labels)
    rng = random.Random(seed)
    collected: list[Item] = []
    seen: set[str] = set()
    problems: list[str] = []
    batches = 0
    calls = 0
    consecutive_failures = 0
    model_index = 0
    while len(collected) < n and (max_batches is None or batches < max_batches):
        batches += 1
        want = min(batch_size, n - len(collected))
        user = USER_TEMPLATE.format(n=want, breakdown=build_breakdown(want, urgent_ratio),
                                    today=utc_now_iso()[:10], seed=rng.randint(1, 10**9))
        payload = {
            "model": models[model_index],
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_TEMPLATE.format(
                        categories="\n".join(f"- {name}: {desc}" for name, desc in task.descriptions().items())
                    ),
                },
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra_body,
        }
        calls += 1
        result = http.post_json(endpoint, payload, headers={"Authorization": f"Bearer {key}"})
        if not result.ok:
            problems.append(f"batch {batches} [{models[model_index]}]: http {result.status} {result.error_type}")
            consecutive_failures += 1
            if consecutive_failures >= 2 and model_index + 1 < len(models):
                model_index += 1
                consecutive_failures = 0
                problems.append(f"-> モデルを {models[model_index]} に降格")
            continue
        consecutive_failures = 0
        try:
            content = result.body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            problems.append(f"batch {batches}: malformed response")
            continue
        rows = extract_json_array(content or "")
        items, probs = normalize_items(rows, labels, prefix, start=len(collected) + 1)
        problems.extend(f"batch {batches}: {p}" for p in probs)
        for item in items:
            key_text = re.sub(r"\s+", "", item.text)
            if key_text in seen:
                continue
            seen.add(key_text)
            collected.append(item)
        assign_ids(collected, prefix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(collected, out_path)
        print(
            f"[batch {batches}] model={models[model_index]} +{len(items)} -> total {len(collected)}/{n} "
            f"({result.latency_ms/1000:.0f}s, in={((result.body or {}).get('usage') or {}).get('prompt_tokens')} "
            f"out={((result.body or {}).get('usage') or {}).get('completion_tokens')})",
            flush=True,
        )
    # 検証
    from collections import Counter

    label_counts = Counter(it.label for it in collected)
    sev_counts = Counter(it.severity for it in collected)
    urgent = label_counts.get(task.risk_label, 0)
    return {
        "provider": config.get("data_generation.provider"),
        "model": models[model_index],
        "n": len(collected),
        "requested": n,
        "batches": batches,
        "calls": calls,
        "label_counts": dict(label_counts),
        "severity_counts": dict(sev_counts),
        "urgent_ratio": urgent / len(collected) if collected else 0.0,
        "ambiguous": sum(1 for it in collected if it.ambiguous),
        "high_severity": sev_counts.get("high", 0),
        "problems": problems[:50],
        "problem_count": len(problems),
        "out": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="評価データ(日本語問い合わせ)を LLM で生成する")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--prefix", default="gen")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(repo_root=REPO_ROOT)
    if config.get("data_generation.model") is None:
        raise SystemExit("config に data_generation.model を設定してください")
    task = load_task(REPO_ROOT / "config" / "tasks" / "support_classification.toml")
    key = require_key(str(config.get("data_generation.key_env", "OPENROUTER_API_KEY")), repo_root=REPO_ROOT)
    summary = generate(
        task=task, config=config, n=args.n, batch_size=args.batch_size, prefix=args.prefix,
        seed=args.seed, out_path=Path(REPO_ROOT / args.out), key=key, max_batches=args.max_batches,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
