"""Decision Model 系アダプタの共通処理(choice 応答の解釈)。"""
from __future__ import annotations


def parse_choice(answer: dict | None) -> tuple[str | None, dict[str, float] | None]:
    """answers.<key> から (choice, probabilities) を取り出す。"""
    if not isinstance(answer, dict):
        return None, None
    label = answer.get("choice")
    label = str(label) if label is not None else None
    raw_probs = answer.get("probabilities")
    probs: dict[str, float] | None = None
    if isinstance(raw_probs, dict):
        try:
            probs = {str(k): float(v) for k, v in raw_probs.items()}
        except (TypeError, ValueError):
            probs = None
    return label, probs


def usage_tokens(body: dict | None) -> tuple[int | None, int | None]:
    usage = (body or {}).get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    return (
        int(input_tokens) if isinstance(input_tokens, (int, float)) else None,
        int(output_tokens) if isinstance(output_tokens, (int, float)) else None,
    )
