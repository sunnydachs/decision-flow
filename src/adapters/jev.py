"""TypeSafe Jev アダプタ(公式 SDK 経由。キーは TYPESAFE_API_KEY)。

契約(公式ドキュメント + SDK 実地確認 2026-10-05):
  POST https://api.typesafe.ai/v1/systemone
  Authorization: Bearer $TYPESAFE_API_KEY   Content-Type: application/json
  {"state": <text|object|array>, "model": "jev-1.13.0", "questions": {<id>:{type,instructions,criteria}}}
  resp {"model": "...", "answers": {<id>:{type, choice, probabilities, confidence}},
        "usage": {"input_tokens","output_tokens"}}

  - モデル ID は固定の `jev-1.13.0`(`jev-latest` は使わない。alias は将来動くため)
  - choice の回答は `probabilities`(合計 1 の確率)を返す → 較正の対象にできる値
  - `confidence` は確率分布から導かれた値で、確率そのものではない → 較正の対象にしない(別指標)
  - noul は「yes の確率」、score は確率加重値(今回の分類タスクでは choice を使う)
  - 価格: 入力 $42/Btok = $0.042 / 100万トークン、出力無料
  - レート制限: 100K tokens/s, 80 req/s。429 と 529(Overloaded)は SDK が backoff で再試行
  - データ: 「Jev is not trained on customer requests or responses」= 学習に使われない
  - 言語: 英語が最良。CJK は handled but not equally well → 日本語タスクでは注意(レポートに明記)
出典: https://docs.typesafe.ai/api , https://docs.typesafe.ai/models , https://docs.typesafe.ai/sdk

SDK の実地確認(ネットワーク不要の introspection):
  TypeSafeClient(*, api_key, model, retry, timeout, base_url)
  client.system_one(state, questions, *, model, retry, timeout) -> SystemOneResponse
  SystemOneResponse(model, usage, answers) / ChoiceAnswer(type, choice, confidence, probabilities)
  Usage(input_tokens, output_tokens)
  例外: TypeSafeAuthenticationError / TypeSafeRateLimitError / TypeSafeUnprocessableEntityError /
        TypeSafeAPITimeoutError / TypeSafeAPIConnectionError / TypeSafeAPIError

dry_run=True では API を呼ばず、リクエスト形と推定トークン・推定コストだけを返す。
実際の実行は registry 側で --allow-paid-models を必須にしている(誤実行の防止)。
"""
from __future__ import annotations

import math
import time

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
                 token_estimate: int | None = None, use_sdk: bool | None = None, sdk_client=None):
        super().__init__(task, model=model or str(config.get("models.jev.model")), http=http,
                         cache=cache, budget=budget, audit=audit)
        self.endpoint = endpoint or str(config.get("models.jev.endpoint"))
        self.key = key
        self.budget = budget
        self.dry_run = dry_run
        self.price_input_per_mtok = float(config.get("models.jev.price_input_usd_per_mtok") or 0.0)
        self.token_estimate = token_estimate  # mercury 等の実測値を渡すと概算が締まる
        self.timeout_seconds = float(config.get("models.jev.timeout_seconds") or 120)
        # 既定は SDK 経由(公式 SDK を使う)。SDK が無い環境だけ HTTP にフォールバックする。
        self._sdk_client = sdk_client
        self.use_sdk = self._sdk_available() if use_sdk is None else bool(use_sdk)

    def _sdk_available(self) -> bool:
        if self._sdk_client is not None:
            return True
        try:
            import typesafe_sdk  # noqa: F401
        except Exception:  # noqa: BLE001 - SDK 未導入なら HTTP にフォールバック
            return False
        return True

    def _client(self):
        if self._sdk_client is not None:
            return self._sdk_client
        from typesafe_sdk import TypeSafeClient

        self._sdk_client = TypeSafeClient(api_key=self.key, model=self.model,
                                          timeout=self.timeout_seconds)
        return self._sdk_client

    def build_questions(self):
        """タスク定義から SDK の Choice 質問を組み立てる(質問文は config 側が単一の出所)。"""
        from typesafe_sdk import Choice

        spec = self.task.choice_question(self.question_id)[self.question_id]
        return {self.question_id: Choice(instructions=spec["instructions"], criteria=spec["criteria"])}

    def build_payload(self, text: str) -> dict:
        return {"state": text, "model": self.model, "questions": self.task.choice_question(self.question_id)}

    def dry_run_report(self, items: list[Item]) -> dict:
        """実 API を呼ばずに、リクエスト形・推定トークン・推定コストを出す。"""
        payloads = [self.build_payload(item.text) for item in items]
        if self.token_estimate is not None:
            per_request = int(self.token_estimate)
            basis = "先行実測の入力トークン数(同スキーマの概算)"
        else:
            per_request = max(estimate_tokens_rough(it.text) + 200 for it in items) if items else 0
            basis = "文字数からの粗い概算(トークナイザ未確認)"
        total_tokens = per_request * len(items)
        cost = estimate_cost_usd(total_tokens, self.price_input_per_mtok)
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "transport": "typesafe-sdk" if self.use_sdk else "http",
            "n_requests": len(items),
            "request_example": payloads[0] if payloads else None,
            "estimated_input_tokens_per_request": per_request,
            "estimated_total_input_tokens": total_tokens,
            "token_estimate_basis": basis,
            "estimated_cost_usd": cost,
            "note": "推定コストはトークン数×公式単価からの概算であり、実際の請求額ではない。",
        }

    # --- 実行 ---------------------------------------------------------------

    def predict(self, item: Item, run_index: int = 0) -> Prediction:
        if self.dry_run:
            raise RuntimeError("dry_run=True では predict() を呼べません(実 API を呼ばない)")
        if self.budget is not None:
            self.budget.check(0.0)
        prediction = self._predict_sdk(item, run_index=run_index) if self.use_sdk \
            else self._predict_http(item, run_index=run_index)
        self._audit(prediction, text=item.text)
        return prediction

    def _predict_sdk(self, item: Item, *, run_index: int) -> Prediction:
        started = time.perf_counter()
        try:
            response = self._client().system_one(
                state=item.text, questions=self.build_questions(), model=self.model
            )
        except Exception as exc:  # noqa: BLE001 - SDK の例外を分類して記録する
            latency = (time.perf_counter() - started) * 1000.0
            return self._prediction(item, run_index=run_index, latency_ms=latency,
                                    error_type=_classify_sdk_error(exc),
                                    raw_response={"exception": f"{type(exc).__name__}: {str(exc)[:300]}"})
        latency = (time.perf_counter() - started) * 1000.0
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
        cost = estimate_cost_usd(input_tokens or 0, self.price_input_per_mtok)
        if self.budget is not None:
            self.budget.add(cost)
        answer = (getattr(response, "answers", None) or {}).get(self.question_id)
        label, probs = parse_choice(_answer_to_dict(answer))
        risk = probs.get(self.task.risk_label) if probs else (1.0 if label == self.task.risk_label else 0.0)
        return self._prediction(
            item,
            run_index=run_index,
            label=label,
            probabilities=probs,
            value_type="probability" if probs else "none",
            risk_score=float(risk) if risk is not None else None,
            risk_score_type="probability" if probs else "binary_rule",
            raw_response=_answer_to_raw(response),
            latency_ms=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            error_type=None if label else "parse",
            request_id=None,  # SDK のレスポンスに request id は無い(モデル ID は raw に残す)
        )

    def _predict_http(self, item: Item, *, run_index: int) -> Prediction:
        payload = self.build_payload(item.text)
        result = self.http.post_json(
            self.endpoint, payload, headers={"Authorization": f"Bearer {self.key}"}
        )
        input_tokens, output_tokens = usage_tokens(result.body)
        cost = estimate_cost_usd(input_tokens or 0, self.price_input_per_mtok)
        if self.budget is not None:
            self.budget.add(cost)
        request_id = result.headers.get("x-request-id") or (result.body or {}).get("id")
        if not result.ok:
            return self._prediction(item, run_index=run_index, latency_ms=result.latency_ms,
                                    error_type=result.error_type, input_tokens=input_tokens,
                                    output_tokens=output_tokens, estimated_cost_usd=cost,
                                    request_id=str(request_id) if request_id else None,
                                    raw_response={"status": result.status, "error": result.text})
        answer = ((result.body or {}).get("answers") or {}).get(self.question_id)
        label, probs = parse_choice(answer)
        risk = probs.get(self.task.risk_label) if probs else (1.0 if label == self.task.risk_label else 0.0)
        return self._prediction(
            item, run_index=run_index, label=label, probabilities=probs,
            value_type="probability" if probs else "none",
            risk_score=float(risk) if risk is not None else None,
            risk_score_type="probability" if probs else "binary_rule",
            raw_response=result.body, latency_ms=result.latency_ms,
            input_tokens=input_tokens, output_tokens=output_tokens, estimated_cost_usd=cost,
            error_type=None if label else "parse",
            request_id=str(request_id) if request_id else None,
        )


def _answer_to_dict(answer) -> dict | None:
    """SDK の Answer オブジェクトを parse_choice が読める dict にする。"""
    if answer is None:
        return None
    if isinstance(answer, dict):
        return answer
    if hasattr(answer, "model_dump"):
        return answer.model_dump()
    return {
        "type": getattr(answer, "type", None),
        "choice": getattr(answer, "choice", None),
        "probabilities": getattr(answer, "probabilities", None),
        "confidence": getattr(answer, "confidence", None),
    }


def _answer_to_raw(response) -> dict:
    """監査ログ用に、レスポンスを JSON 化可能な形へ(confidence は確率ではないので別項目)。"""
    raw: dict = {"model": getattr(response, "model", None)}
    usage = getattr(response, "usage", None)
    if usage is not None:
        raw["usage"] = {"input_tokens": getattr(usage, "input_tokens", None),
                        "output_tokens": getattr(usage, "output_tokens", None)}
    answers = getattr(response, "answers", None) or {}
    raw["answers"] = {k: _answer_to_dict(v) for k, v in answers.items()}
    return raw


def _classify_sdk_error(exc: Exception) -> str:
    """SDK の例外を、他の方式と同じ error_type 語彙に寄せる。"""
    mapping = {
        "TypeSafeRateLimitError": "http_429",
        "TypeSafeAuthenticationError": "http_401",
        "TypeSafeUnprocessableEntityError": "http_422",
        "TypeSafeNotFoundError": "http_404",
        "TypeSafeAPITimeoutError": "timeout",
        "TypeSafeAPIConnectionError": "connection",
        "TypeSafeInternalServerError": "http_5xx",
        "TypeSafeAPIError": "http_5xx",
    }
    return mapping.get(type(exc).__name__, "exception")
