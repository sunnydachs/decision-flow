"""hybrid: ルール → 判断モデル → 閾値 → 自動/人間レビュー/ブロック。

層1: ルール(明確なパターンは即確定)
層2: 判断モデルの確率(risk_score)
層3: 閾値で auto / review / block に分岐。閾値は calib で決め、test で評価する。
判断不能(タイムアウト/パース失敗/429)のときの既定動作は config で fail-open / fail-closed を選べる。
"""
from __future__ import annotations

from dataclasses import dataclass

from adapters.rule import RuleSet

ACTION_AUTO = "auto"
ACTION_REVIEW = "review"
ACTION_BLOCK = "block"


@dataclass
class Thresholds:
    block_above: float  # risk_score がこれ以上なら block(人間に即時エスカレーション)
    review_above: float  # risk_score がこれ以上なら review、未満は auto

    def validate(self) -> None:
        if not (0.0 <= self.review_above <= self.block_above <= 1.0):
            raise ValueError("閾値は 0 <= review_above <= block_above <= 1 の順にしてください")


@dataclass
class HybridDecision:
    item_id: str
    action: str
    source: str  # "rule" | "model" | "fallback"
    risk_score: float | None
    threshold: float | None
    request_id: str | None
    label: str | None
    probabilities: dict[str, float] | None
    error_type: str | None


def decide(
    *,
    item_id: str,
    label: str | None,
    risk_score: float | None,
    probabilities: dict[str, float] | None,
    request_id: str | None,
    error_type: str | None,
    thresholds: Thresholds,
    rule_label: str | None = None,
    risk_label: str = "urgent_claim",
    fail_open: bool = False,
) -> HybridDecision:
    # 層1: ルールの明確一致(ただし高リスクでの誤確定を避けるため、risk ラベル一致のみ即確定)
    if rule_label is not None:
        action = ACTION_BLOCK if rule_label == risk_label else ACTION_AUTO
        return HybridDecision(item_id, action, "rule", 1.0 if rule_label == risk_label else 0.0,
                              None, request_id, rule_label, probabilities, error_type)
    # 判断不能
    if error_type is not None or risk_score is None:
        action = ACTION_AUTO if fail_open else ACTION_REVIEW
        return HybridDecision(item_id, action, "fallback", risk_score, thresholds.review_above,
                              request_id, label, probabilities, error_type)
    # 層2/3: 確率と閾値
    if risk_score >= thresholds.block_above:
        action = ACTION_BLOCK
    elif risk_score >= thresholds.review_above:
        action = ACTION_REVIEW
    else:
        action = ACTION_AUTO
    return HybridDecision(item_id, action, "model", risk_score, thresholds.review_above,
                          request_id, label, probabilities, error_type)


class HybridPipeline:
    def __init__(self, *, rules: RuleSet, thresholds: Thresholds, risk_label: str, fail_open: bool = False,
                 rule_commit: bool = True):
        thresholds.validate()
        self.rules = rules
        self.thresholds = thresholds
        self.risk_label = risk_label
        self.fail_open = fail_open
        # rule_commit=False: 層1の即確定を無効化し、ルール一致はモデル確率と併用する
        # (high 案件が非 risk ラベルのルール一致で auto 化する見逃しを防ぐ)
        self.rule_commit = rule_commit

    def run(self, prediction, text: str) -> HybridDecision:
        rule_label = self.rules.matched_label(text)
        if rule_label is not None and self.rule_commit:
            action = ACTION_BLOCK if rule_label == self.risk_label else ACTION_AUTO
            return HybridDecision(item_id=prediction.item_id, action=action, source="rule",
                                  risk_score=1.0 if rule_label == self.risk_label else 0.0,
                                  threshold=None, request_id=prediction.request_id, label=rule_label,
                                  probabilities=prediction.probabilities, error_type=prediction.error_type)
        if rule_label is not None and rule_label == self.risk_label:
            # rule_commit に関わらず、risk ラベルへのルール一致は即 block
            return HybridDecision(item_id=prediction.item_id, action=ACTION_BLOCK, source="rule",
                                  risk_score=1.0, threshold=self.thresholds.review_above,
                                  request_id=prediction.request_id, label=rule_label,
                                  probabilities=prediction.probabilities, error_type=prediction.error_type)
        # 判断不能
        if prediction.error_type is not None or prediction.risk_score is None:
            action = ACTION_AUTO if self.fail_open else ACTION_REVIEW
            return HybridDecision(item_id=prediction.item_id, action=action, source="fallback",
                                  risk_score=prediction.risk_score,
                                  threshold=self.thresholds.review_above, request_id=prediction.request_id,
                                  label=prediction.label, probabilities=prediction.probabilities,
                                  error_type=prediction.error_type)
        # 層2/3: 確率と閾値
        if prediction.risk_score >= self.thresholds.block_above:
            action = ACTION_BLOCK
        elif prediction.risk_score >= self.thresholds.review_above:
            action = ACTION_REVIEW
        else:
            action = ACTION_AUTO
        source = "model" if rule_label is None else "rule_and_model"
        return HybridDecision(item_id=prediction.item_id, action=action, source=source,
                              risk_score=prediction.risk_score, threshold=self.thresholds.review_above,
                              request_id=prediction.request_id, label=prediction.label,
                              probabilities=prediction.probabilities, error_type=prediction.error_type)
