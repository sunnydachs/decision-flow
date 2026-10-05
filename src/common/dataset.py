"""データのスキーマ・ローダ・manifest 検証。

入力フォーマット(JSONL 1 行 = 1 件):
  id              : str   一意な ID
  text            : str   問い合わせ本文
  label           : str   正解カテゴリ(label と severity は独立)
  severity        : str   "high" | "normal"(高リスク候補を中心に付与)
  language        : str   "ja" など
  ambiguous       : bool  曖昧な文(通常評価とは別集計)
  annotator_labels: list[str]  任意。複数人のラベル(一致率の算出用)

manifest.json (データセットディレクトリ直下):
  source          : str   来歴(例 "llm-generated-v1", "jev-ja-eval", "MASSIVE")
  license         : str
  data_class      : str   "synthetic" | "public" | "real"
  external_ok     : bool  外部 API に送ってよいか(real は false)
  provenance      : str   生成方法・モデル名などの説明
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SEVERITIES = ("high", "normal")


class DatasetError(ValueError):
    pass


@dataclass
class Item:
    id: str
    text: str
    label: str
    severity: str
    language: str
    ambiguous: bool = False
    annotator_labels: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "severity": self.severity,
            "language": self.language,
            "ambiguous": self.ambiguous,
        }
        if self.annotator_labels:
            out["annotator_labels"] = list(self.annotator_labels)
        if self.meta:
            out["meta"] = dict(self.meta)
        return out


def _require(record: dict, key: str, path: Path, lineno: int) -> object:
    if key not in record:
        raise DatasetError(f"{path}:{lineno} に必須キー {key} がありません")
    return record[key]


def parse_item(record: dict, path: Path, lineno: int) -> Item:
    severity = str(_require(record, "severity", path, lineno))
    if severity not in SEVERITIES:
        raise DatasetError(
            f"{path}:{lineno} の severity={severity!r} は {SEVERITIES} のいずれかにしてください"
        )
    labels = record.get("annotator_labels") or []
    if not isinstance(labels, list):
        raise DatasetError(f"{path}:{lineno} の annotator_labels は配列にしてください")
    return Item(
        id=str(_require(record, "id", path, lineno)),
        text=str(_require(record, "text", path, lineno)),
        label=str(_require(record, "label", path, lineno)),
        severity=severity,
        language=str(record.get("language", "ja")),
        ambiguous=bool(record.get("ambiguous", False)),
        annotator_labels=[str(x) for x in labels],
        meta=dict(record.get("meta") or {}),
    )


def load_jsonl(path: str | Path) -> list[Item]:
    path = Path(path)
    items: list[Item] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno} が JSON として読めません: {exc}") from exc
            item = parse_item(record, path, lineno)
            if item.id in seen:
                raise DatasetError(f"{path}:{lineno} で id={item.id} が重複しています")
            seen.add(item.id)
            items.append(item)
    return items


def write_jsonl(items: list[Item], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")


def load_manifest(dataset_dir: str | Path) -> dict:
    path = Path(dataset_dir) / "manifest.json"
    if not path.exists():
        raise DatasetError(f"manifest.json がありません: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def external_send_allowed(manifest: dict, confirm_external: bool) -> bool:
    """外部 API に送信してよいか。real データは --confirm-external が必須。"""
    if manifest.get("external_ok", False):
        return True
    return bool(confirm_external)


def annotator_agreement(items: list[Item]) -> dict:
    """複数ラベルがある項目の一致率(単純一致: 全員一致かどうか)。"""
    multi = [it for it in items if len(it.annotator_labels) >= 2]
    if not multi:
        return {"n": 0, "unanimous": 0, "agreement": None}
    unanimous = sum(1 for it in multi if len(set(it.annotator_labels)) == 1)
    return {"n": len(multi), "unanimous": unanimous, "agreement": unanimous / len(multi)}


def severity_counts(items: list[Item]) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it.severity] = out.get(it.severity, 0) + 1
    return out


def label_counts(items: list[Item]) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it.label] = out.get(it.label, 0) + 1
    return out
