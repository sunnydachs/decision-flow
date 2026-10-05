"""キーワードルール方式(ベースライン)。確率は返さない(None)。"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from adapters.base import BaseAdapter, Prediction
from common.dataset import Item
from tasks.base import TaskDefinition


@dataclass
class RuleSet:
    ordered: list[tuple[str, tuple[str, ...]]]
    fallback_label: str

    def match(self, text: str) -> str:
        for label, patterns in self.ordered:
            if any(p in text for p in patterns):
                return label
        return self.fallback_label

    def matched_label(self, text: str) -> str | None:
        """フォールバックを除いた「明確な一致」を返す(hybrid 層1 用)。"""
        for label, patterns in self.ordered:
            if label == self.fallback_label:
                continue
            if any(p in text for p in patterns):
                return label
        return None


def load_rules(path: str | Path) -> RuleSet:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    ordered = [
        (str(rule["label"]), tuple(str(p) for p in rule.get("patterns", [])))
        for rule in data.get("rules", [])
    ]
    return RuleSet(ordered=ordered, fallback_label=str(data.get("fallback_label", "other")))


class RuleAdapter(BaseAdapter):
    name = "rule"
    tier = "local"

    def __init__(self, task: TaskDefinition, rules: RuleSet):
        super().__init__(task, model="keyword-rules")
        self.rules = rules

    def predict(self, item: Item, run_index: int = 0) -> Prediction:
        label = self.rules.match(item.text)
        # ルールは確率を返せないため None。リスク判定は「risk_label に一致したか」の 0/1。
        risk = 1.0 if label == self.task.risk_label else 0.0
        prediction = self._prediction(
            item,
            run_index=run_index,
            label=label,
            probabilities=None,
            value_type="none",
            risk_score=risk,
            risk_score_type="binary_rule",
            raw_response={"matched_label": self.rules.matched_label(item.text), "fallback": label},
            latency_ms=0.0,
        )
        self._audit(prediction, text=item.text)
        return prediction
