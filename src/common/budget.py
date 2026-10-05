"""予算ガード(推定コストの累積上限)。

コストは API が返す usage のトークン数 × 公式単価からの**推定値**であり、実際の請求額ではない。
上限を超えそうになったら停止する。
"""
from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetGuard:
    limit_usd: float
    name: str
    spent_usd: float = 0.0
    requests: int = 0
    _stopped: bool = field(default=False, init=False)

    def would_exceed(self, cost_usd: float) -> bool:
        return (self.spent_usd + max(0.0, cost_usd)) > self.limit_usd + 1e-12

    def check(self, cost_usd: float = 0.0) -> None:
        """送信前に呼ぶ。上限を超える見込みなら BudgetExceeded。"""
        if self.would_exceed(cost_usd):
            self._stopped = True
            raise BudgetExceeded(
                f"{self.name}: 予算上限 ${self.limit_usd:.4f} を超える見込みです "
                f"(累積 ${self.spent_usd:.6f} + 今回 ${max(0.0, cost_usd):.6f})"
            )

    def add(self, cost_usd: float | None) -> None:
        self.requests += 1
        if cost_usd:
            self.spent_usd += float(cost_usd)
        if self.would_exceed(0.0):
            self._stopped = True

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    @property
    def stopped(self) -> bool:
        return self._stopped


def estimate_cost_usd(input_tokens: int | None, price_input_per_mtok: float) -> float | None:
    """入力トークン数 × 単価(100万トークンあたり)。出力無料のモデルを想定。"""
    if input_tokens is None:
        return None
    return input_tokens / 1_000_000.0 * price_input_per_mtok
