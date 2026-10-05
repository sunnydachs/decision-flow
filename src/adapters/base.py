"""アダプタの共通インターフェース。

入力: テキスト + タスク定義(カテゴリと説明)
出力: 予測ラベル / カテゴリ別確率(取得できる場合のみ。取れない方式は None)/ 値の種別 /
      生レスポンス / レイテンシ(ms)/ 入出力トークン数 / 推定コスト / エラー種別 / リクエストID
全リクエストに実行日時(UTC)とモデルIDを記録する。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

from common.dataset import Item
from tasks.base import TaskDefinition

# probabilities の値の種別
VALUE_TYPES = (
    "probability",          # 公式に確率として定義された値(noul / choice 確率など)
    "stated_probability",   # LLM が自己申告した数値(確率として扱わない)
    "expected_score",       # score の期待値(確率ではない)
    "behavior_probability",  # 振る舞いの存在確率(span-01-lite)
    "confidence",           # 分布から導出された統計量(確率ではない)
    "binary_rule",          # ルールの一致(0/1)
    "none",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Prediction:
    item_id: str
    method: str
    model: str
    timestamp_utc: str = field(default_factory=utc_now_iso)
    run_index: int = 0
    label: str | None = None
    probabilities: dict[str, float] | None = None
    value_type: str | None = None
    risk_score: float | None = None
    risk_score_type: str | None = None
    raw_response: dict | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    error_type: str | None = None
    request_id: str | None = None
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class BaseAdapter:
    name = "base"
    tier = "local"  # "free" | "paid" | "local"

    def __init__(
        self,
        task: TaskDefinition,
        model: str = "n/a",
        *,
        http=None,
        cache=None,
        quota=None,
        budget=None,
        audit=None,
        rate_limiter=None,
    ):
        self.task = task
        self.model = model
        self.http = http
        self.cache = cache
        self.quota = quota
        self.budget = budget
        self.audit = audit
        self.rate_limiter = rate_limiter

    def _wait_rate_limit(self) -> None:
        if self.rate_limiter is not None:
            self.rate_limiter.wait()

    def predict(self, item: Item, run_index: int = 0) -> Prediction:  # pragma: no cover - 抽象
        raise NotImplementedError

    # --- 補助 ---
    def _prediction(self, item: Item, run_index: int = 0, **kwargs: Any) -> Prediction:
        return Prediction(
            item_id=item.id,
            method=self.name,
            model=self.model,
            run_index=run_index,
            **kwargs,
        )

    def _audit(self, prediction: Prediction, *, text: str | None = None, extra: dict | None = None) -> None:
        if self.audit is None:
            return
        record = prediction.as_dict()
        if text is not None:
            record["input"] = text
        if extra:
            record.update(extra)
        self.audit.write(record)

    @staticmethod
    def default_label(probabilities: dict[str, float] | None) -> str | None:
        if not probabilities:
            return None
        return max(probabilities.items(), key=lambda kv: kv[1])[0]
