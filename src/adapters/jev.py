"""TypeSafe Jev アダプタ(公式エンドポイント + TYPESAFE_API_KEY)。

契約(公式ドキュメントで確認):
  POST https://api.typesafe.ai/v1/systemone
  Authorization: Bearer $TYPESAFE_API_KEY
  {"state": <text|object|array>, "model": "jev-1.13.0", "questions": {<name>:{type,instructions,criteria}}}
  resp {"model": "jev-1.13.0", "answers": {<name>:{type, choice, probabilities, confidence}},
        "usage": {"input_tokens","output_tokens"}}
  - 価格: 入力 $42/Btok = $0.042 / 100万トークン、出力無料
  - バージョンは固定(jev-latest は使わない)
出典: https://docs.typesafe.ai/api , https://docs.typesafe.ai/models
      https://docs.typesafe.ai/model-jaggedness/jev-1.13

dry_run=True では API を呼ばず、リクエスト形と推定トークン・推定コストだけを返す。
実際の実行は registry 側で --allow-paid-models を必須にしている(誤実行の防止)。
"""
from __future__ import annotations

import math

from adapters.base import BaseAdapter, Prediction
from adapters.choice_common import parse_choice, usage_tokens
from common.budget import estimate_cost_usd
from common.dataset import Item
from tasks.base import TaskDefinition


def estimate_tokens_rough(text: str) -> int:
    """トークナイザ情報が公式に無いため、文字数からの粗い概算(あくまで目安)。"""
    return max(1, int(math.ceil(len(text) / 1.5)))


class JevAdapter(BaseAdapter):
    name = "jev"
    tier = "paid"
    question_id = "category"

    def __init__(self, task: TaskDefinition, config, *, key: str, http, budget=None, cache=None,
                 audit=None, endpoint: str | None = None, model: str | None = None, dry_run: bool = False,
                 token_estimate: int | None = None):
        super().__init__(task, model=model or str(config.get("models.jev.model")), http=http,
                         cache=cache, budget=budget, audit=audit)
        self.endpoint = endpoint or str(config.get("models.jev.endpoint"))
        self.key = key
        self.budget = budget
        self.dry_run = dry_run
        self.price_input_per_mtok = float(config.get("models.jev.price_input_usd_per_mtok") or 0.0)
        self.token_estimate = token_estimate  # mercury 等の実測値を渡すと概算が締まる

    def build_payload(self, text: str) -> dict:
        return {"state": text, "model": self.model, "questions": self.task.choice_question(self.question_id)}

    def dry_run_report(self, items: list[Item]) -> dict:
        """実 API を呼ばずに、リクエスト形・推定トークン・推定コストを出す。"""
        payloads = [self.build_payload(item.text) for item in items]
        if self.token_estimate is not None:
            per_request = int(self.token_estimate)
            basis = "mercury-decide 実測の入力トークン数(同スキーマの概算)"
        else:
            per_request = max(estimate_tokens_rough(it.text) + 200 for it in items) if items else 0
            basis = "文字数からの粗い概算(トークナイザ未確認)"
        total_tokens = per_request * len(items)
        cost = estimate_cost_usd(total_tokens, self.price_input_per_mtok)
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "n_requests": len(items),
            "request_example": payloads[0] if payloads else None,
            "estimated_input_tokens_per_request": per_request,
            "estimated_total_input_tokens": total_tokens,
            "token_estimate_basis": basis,
            "estimated_cost_usd": cost,
            "note": "推定コストはトークン数×公式単価からの概算であり、実際の請求額ではない。",
        }

    def predict(self, item: Item, run_index: int = 0) -> Prediction:
        if self.dry_run:
            raise RuntimeError("dry_run=True では predict() を呼べません(実 API を呼ばない)")
        if self.budget is not None:
            self.budget.check(0.0)
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
            prediction = self._prediction(item, run_index=run_index, latency_ms=result.latency_ms,
                                          error_type=result.error_type, input_tokens=input_tokens,
                                          output_tokens=output_tokens, estimated_cost_usd=cost,
                                          request_id=str(request_id) if request_id else None,
                                          raw_response={"status": result.status, "error": result.text})
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
