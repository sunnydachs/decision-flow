"""Mercury Decide(評価対象の Decision Model)。

契約(評価対象モデルの公式ドキュメントと先行プロジェクトの実測で確認。エンドポイントは
config: models.mercury.endpoint から渡す):
  body {"model", "state", "questions": {<key>: {"type":"choice","instructions","criteria"}}}
  resp {"answers": {<key>: {"type":"choice","choice","probabilities","confidence"}},
        "id", "model", "provider", "usage": {"cost","input_tokens","output_tokens"}}
choice の probabilities は公式に「calibrated probability taken directly from the model」と記載されている。
出典: 評価対象モデルの公式ドキュメント(config/local.toml のコメントに記録)
"""
from __future__ import annotations

from adapters.base import BaseAdapter, Prediction
from adapters.choice_common import parse_choice, usage_tokens
from common.dataset import Item
from tasks.base import TaskDefinition


class MercuryAdapter(BaseAdapter):
    name = "mercury"
    tier = "free"
    question_id = "category"

    def __init__(
        self,
        task: TaskDefinition,
        config,
        *,
        key: str,
        http,
        quota=None,
        cache=None,
        audit=None,
        endpoint: str | None = None,
        model: str | None = None,
        rate_limiter=None,
    ):
        super().__init__(task, model=model or str(config.get("models.mercury.model")), http=http,
                         cache=cache, quota=quota, audit=audit, rate_limiter=rate_limiter)
        self.endpoint = endpoint or str(config.get("models.mercury.endpoint"))
        self.key = key

    def build_payload(self, text: str) -> dict:
        return {
            "model": self.model,
            "state": text,
            "questions": self.task.choice_question(self.question_id),
        }

    def parse_response(self, body: dict) -> tuple[str | None, dict[str, float] | None, str | None]:
        answer = ((body or {}).get("answers") or {}).get(self.question_id)
        label, probs = parse_choice(answer)
        request_id = (body or {}).get("id")
        return label, probs, str(request_id) if request_id else None

    def predict(self, item: Item, run_index: int = 0) -> Prediction:
        if self.quota is not None:
            self.quota.consume()
        self._wait_rate_limit()
        payload = self.build_payload(item.text)
        result = self.http.post_json(
            self.endpoint, payload, headers={"Authorization": f"Bearer {self.key}"}
        )
        input_tokens, output_tokens = usage_tokens(result.body)
        cost = ((result.body or {}).get("usage") or {}).get("cost")
        if not result.ok:
            prediction = self._prediction(
                item,
                run_index=run_index,
                latency_ms=result.latency_ms,
                error_type=result.error_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response={"status": result.status, "error": result.text},
            )
            self._audit(prediction, text=item.text)
            return prediction
        label, probs, request_id = self.parse_response(result.body or {})
        risk = probs.get(self.task.risk_label) if probs else (1.0 if label == self.task.risk_label else 0.0)
        prediction = self._prediction(
            item,
            run_index=run_index,
            label=label,
            probabilities=probs,
            value_type="probability" if probs else "none",
            risk_score=float(risk) if risk is not None else None,
            risk_score_type="probability" if probs else "binary_rule",
            raw_response=result.body,
            latency_ms=result.latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            error_type=None if label else "parse",
            request_id=request_id,
        )
        self._audit(prediction, text=item.text)
        return prediction
