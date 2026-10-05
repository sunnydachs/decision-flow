"""Perplexity Decisions API(有料・安価)。

契約(公式ドキュメントで確認):
  POST https://api.perplexity.ai/v1/decisions
  Authorization: Bearer $PERPLEXITY_API_KEY
  body {"model": "pplx-decider-v1-27b", "state", "questions": {<name>:{type,instructions,criteria}}}
  resp {"model","answers":{...},"usage":{"input_tokens","output_tokens"}}
  - noul = yes の確率 0..1 / choice = 各選択肢の確率 + 最尤 + confidence / score = 期待値 + legend + 確率
  - 課金: 入力 $0.04 / 100万トークン、出力無料、リクエスト課金なし
  - レート: 10 req/s per organization、429 は Retry-After
出典: https://docs.perplexity.ai/api-reference/decisions-post
      https://docs.perplexity.ai/docs/decisions/quickstart
予算上限(config: budget.perplexity_usd)で停止する。--allow-paid-models は不要(Jev のみ必須)。
"""
from __future__ import annotations

from adapters.base import BaseAdapter, Prediction
from adapters.choice_common import parse_choice, usage_tokens
from common.budget import estimate_cost_usd
from common.dataset import Item
from tasks.base import TaskDefinition


class PerplexityDeciderAdapter(BaseAdapter):
    name = "pplx_decider"
    tier = "paid"
    question_id = "category"

    def __init__(
        self,
        task: TaskDefinition,
        config,
        *,
        key: str,
        http,
        budget=None,
        cache=None,
        audit=None,
        endpoint: str | None = None,
        model: str | None = None,
        rate_limiter=None,
    ):
        super().__init__(task, model=model or str(config.get("models.pplx_decider.model")), http=http,
                         cache=cache, audit=audit, rate_limiter=rate_limiter)
        self.endpoint = endpoint or str(config.get("models.pplx_decider.endpoint"))
        self.key = key
        self.budget = budget
        self.price_input_per_mtok = float(config.get("models.pplx_decider.price_input_usd_per_mtok") or 0.0)

    def build_payload(self, text: str) -> dict:
        return {
            "model": self.model,
            "state": text,
            "questions": self.task.choice_question(self.question_id),
        }

    def predict(self, item: Item, run_index: int = 0) -> Prediction:
        if self.budget is not None:
            self.budget.check(0.0)  # 次に進めると上限を超える場合のみ例外
        self._wait_rate_limit()
        payload = self.build_payload(item.text)
        result = self.http.post_json(
            self.endpoint, payload, headers={"Authorization": f"Bearer {self.key}"}
        )
        input_tokens, output_tokens = usage_tokens(result.body)
        cost = estimate_cost_usd(input_tokens, self.price_input_per_mtok)
        if self.budget is not None:
            self.budget.add(cost)
        request_id = result.headers.get("x-request-id") or (result.body or {}).get("id")
        if not result.ok:
            prediction = self._prediction(
                item,
                run_index=run_index,
                latency_ms=result.latency_ms,
                error_type=result.error_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                request_id=str(request_id) if request_id else None,
                raw_response={"status": result.status, "error": result.text},
            )
            self._audit(prediction, text=item.text)
            return prediction
        answer = ((result.body or {}).get("answers") or {}).get(self.question_id)
        label, probs = parse_choice(answer)
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
            estimated_cost_usd=cost,
            error_type=None if label else "parse",
            request_id=str(request_id) if request_id else None,
        )
        self._audit(prediction, text=item.text)
        return prediction
