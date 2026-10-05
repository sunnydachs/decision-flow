"""比較用の汎用 LLM(外部ルーティング chat completions)を 2 モードで呼ぶ。

  - mode="prompt"      : プロンプトで JSON を頼むだけ(response_format なし)
  - mode="json_schema" : response_format={"type":"json_schema","json_schema":{...,"strict":true}} を強制

構造化出力の公式仕様:
  https://外部ルーティング.ai/docs/guides/features/structured-outputs
出力の確率は LLM が自己申告した数値であり、公式に確率として定義された値ではない
(value_type="stated_probability"。較正の検証・再較正の対象にはしない)。
"""
from __future__ import annotations

import json
import re

from adapters.base import BaseAdapter, Prediction
from adapters.choice_common import usage_tokens
from common.dataset import Item
from tasks.base import TaskDefinition

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def build_json_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": labels, "description": "最も適切なカテゴリ名"},
            "probabilities": {
                "type": "object",
                "properties": {label: {"type": "number"} for label in labels},
                "required": list(labels),
                "additionalProperties": False,
            },
        },
        "required": ["label", "probabilities"],
        "additionalProperties": False,
    }


def parse_llm_content(content: str, labels: list[str]) -> tuple[str | None, dict[str, float] | None, bool]:
    """(label, probabilities, parsed_ok) を返す。JSON でなければカテゴリ名の走査にフォールバック。"""
    if not content:
        return None, None, False
    text = content.strip()
    match = _JSON_BLOCK.search(text)
    candidates = [text, match.group(0) if match else None]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        label = data.get("label") or data.get("category")
        label = str(label).strip() if label is not None else None
        if label is not None and label not in labels:
            # 前後の空白や引用を許容して再探索
            for known in labels:
                if label.lower() == known.lower():
                    label = known
                    break
        probs = data.get("probabilities")
        if isinstance(probs, dict):
            try:
                probs = {str(k): float(v) for k, v in probs.items()}
            except (TypeError, ValueError):
                probs = None
        else:
            probs = None
        if label or probs:
            return label, probs, True
    for known in labels:
        if known in text:
            return known, None, False
    return None, None, False


class LlmAdapter(BaseAdapter):
    name = "llm"
    tier = "free"
    endpoint = "https://外部ルーティング.ai/api/v1/chat/completions"

    def __init__(
        self,
        task: TaskDefinition,
        config,
        *,
        mode: str,
        key: str,
        http,
        quota=None,
        cache=None,
        audit=None,
        model: str | None = None,
        rate_limiter=None,
    ):
        if mode not in ("prompt", "json_schema"):
            raise ValueError("mode は 'prompt' か 'json_schema'")
        super().__init__(task, model=model or str(config.get("models.llm.primary")), http=http,
                         cache=cache, quota=quota, audit=audit, rate_limiter=rate_limiter)
        self.mode = mode
        self.key = key
        self.endpoint = str(config.get("models.llm.endpoint") or self.endpoint)

    @property
    def method_name(self) -> str:
        return f"llm_{self.mode}"

    def build_payload(self, text: str) -> dict:
        labels = list(self.task.labels)
        if self.mode == "json_schema":
            user = f"問い合わせ文:\n{text}"
        else:
            user = (
                f"問い合わせ文:\n{text}\n\n"
                f"次の JSON だけを出力してください: "
                '{"label": "<カテゴリ名>", "probabilities": {' + ", ".join(f'"{l}": <0-1の数値>' for l in labels) + "}}"
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.task.instruction},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if self.mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "support_classification",
                    "strict": True,
                    "schema": build_json_schema(labels),
                },
            }
        return payload

    def predict(self, item: Item, run_index: int = 0) -> Prediction:
        if self.quota is not None:
            self.quota.consume()
        self._wait_rate_limit()
        payload = self.build_payload(item.text)
        result = self.http.post_json(
            self.endpoint, payload, headers={"Authorization": f"Bearer {self.key}"}
        )
        input_tokens, output_tokens = usage_tokens(result.body)
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
        body = result.body or {}
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = ""
        labels = list(self.task.labels)
        label, probs, parsed_ok = parse_llm_content(content or "", labels)
        risk = None
        if probs and self.task.risk_label in probs:
            risk = probs[self.task.risk_label]
        elif label is not None:
            risk = 1.0 if label == self.task.risk_label else 0.0
        prediction = self._prediction(
            item,
            run_index=run_index,
            label=label,
            probabilities=probs,
            value_type="stated_probability" if probs else ("none" if not label else "stated_probability"),
            risk_score=float(risk) if risk is not None else None,
            risk_score_type="stated_probability" if probs else "binary_rule",
            raw_response=body,
            latency_ms=result.latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=None,  # 無料枠(:free)は課金なし
            error_type=None if parsed_ok else "parse",
            request_id=body.get("id"),
        )
        self._audit(prediction, text=item.text)
        return prediction
