"""結果キャッシュ(途中再開)と監査ログ。

キャッシュキーは (method, model, task_id, item_id, run_index) のハッシュ。
同じキーの結果が既にあれば API を呼ばずに再利用できる(日次上限をまたいだ再開)。
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


def cache_key(method: str, model: str, task_id: str, item_id: str, run_index: int = 0, variant: str = "") -> str:
    raw = f"{method}|{model}|{task_id}|{item_id}|{run_index}|{variant}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def prompt_fingerprint(*texts: str | None) -> str:
    """指示文・質問文の変更でキャッシュが無効化されるようにする短いハッシュ。"""
    payload = "\u0000".join(t or "" for t in texts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def dataset_fingerprint(items) -> str:
    """データセットの内容(本文・ラベル・severity)のハッシュ。データを差し替えたらキャッシュを無効化する。"""
    parts = sorted(f"{it.id}\u0001{it.text}\u0001{it.label}\u0001{it.severity}" for it in items)
    return hashlib.sha256("\u0002".join(parts).encode("utf-8")).hexdigest()[:12]


class ResultCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = record.get("key")
                    if key:
                        self._store[key] = record.get("value", {})

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._store.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = value
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._store)


class AuditLog:
    """全リクエストの入力・確率・閾値・最終アクション・リクエストID を JSONL で残す。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
