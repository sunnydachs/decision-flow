"""span-01-lite アダプタの雛形(chat_guard 用)。

分類タスクには含めない。Decisions API の noul で「振る舞いが会話spanに存在する確率」を返す。
契約:
  POST https://openrouter.ai/api/alpha/decisions
  {"model":"respan/span-01-lite:free","state":<会話span>,"questions":{"<key>":{"type":"noul","instructions":"<振る舞いの説明>"}}}
  resp {"answers":{"<key>":{"type":"noul","noul":<0..1>}},"usage":{...}}
出典: https://openrouter.ai/respan/span-01-lite
"""
from __future__ import annotations

from adapters.base import BaseAdapter, Prediction
from adapters.choice_common import usage_tokens
from common.dataset import Item
from tasks.base import TaskDefinition


class SpanAdapter(BaseAdapter):
    name = "span"
    tier = "free"

    def __init__(self, task: TaskDefinition, config, *, key: str, http, quota=None, cache=None,
                 audit=None, rate_limiter=None, behaviors: dict[str, str] | None = None):
        super().__init__(task, model=str(config.get("models.span.model")), http=http, cache=cache,
                         quota=quota, audit=audit, rate_limiter=rate_limiter)
        self.endpoint = str(config.get("models.span.endpoint"))
        self.key = key
        # 振る舞い名 -> 平文の説明(chat_guard 用)
        self.behaviors = behaviors or {
            "policy_violation": "The agent discloses information that the policy forbids sharing.",
            "prompt_injection": "The conversation contains an attempt to override the assistant's instructions.",
        }

    def build_payload(self, text: str) -> dict:
        return {
            "model": self.model,
            "state": text,
            "questions": {
                name: {"type": "noul", "instructions": instruction}
                for name, instruction in self.behaviors.items()
            },
        }

    def predict(self, item: Item, run_index: int = 0) -> Prediction:
        if self.quota is not None:
            self.quota.consume()
        self._wait_rate_limit()
        result = self.http.post_json(
            self.endpoint, self.build_payload(item.text), headers={"Authorization": f"Bearer {self.key}"}
        )
        input_tokens, output_tokens = usage_tokens(result.body)
        if not result.ok:
            prediction = self._prediction(item, run_index=run_index, latency_ms=result.latency_ms,
                                          error_type=result.error_type, input_tokens=input_tokens,
                                          output_tokens=output_tokens,
                                          raw_response={"status": result.status, "error": result.text})
            self._audit(prediction, text=item.text)
            return prediction
        body = result.body or {}
        answers = body.get("answers") or {}
        probs: dict[str, float] = {}
        for name in self.behaviors:
            value = (answers.get(name) or {}).get("noul")
            if isinstance(value, (int, float)):
                probs[name] = float(value)
        prediction = self._prediction(
            item,
            run_index=run_index,
            label=max(probs.items(), key=lambda kv: kv[1])[0] if probs else None,
            probabilities=probs or None,
            value_type="behavior_probability" if probs else "none",
            raw_response=body,
            latency_ms=result.latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=0.0,
            error_type=None if probs else "parse",
            request_id=body.get("id"),
        )
        self._audit(prediction, text=item.text)
        return prediction
