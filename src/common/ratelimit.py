"""レート制限(最小送信間隔)と日次上限の管理。

無料枠(:free)はモデルをまたいで共有されるため、方式をまたいで共有できる DailyQuota を用意する。
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone


class RateLimiter:
    """直列化された最小送信間隔(per-minute 制限を守るための保守的な実装)。"""

    def __init__(self, per_minute_limit: int | None = None, min_interval_seconds: float | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        if min_interval_seconds is not None:
            self.min_interval = min_interval_seconds
        elif per_minute_limit:
            self.min_interval = 60.0 / float(per_minute_limit)
        else:
            self.min_interval = 0.0
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last: float | None = None

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = self._clock()
            if self._last is not None:
                elapsed = now - self._last
                if elapsed < self.min_interval:
                    self._sleep(self.min_interval - elapsed)
                    now = self._clock()
            self._last = now


class DailyQuotaExhausted(RuntimeError):
    pass


def fetch_free_model_remaining(api_key: str, timeout: float = 20.0) -> int | None:
    """外部ルーティング の無料枠の実残量を取得する(GET /api/v1/key。消費しない)。"""
    import json
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            "https://外部ルーティング.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    remaining = ((data or {}).get("data") or {}).get("free_model_daily_requests", {}).get("remaining")
    return int(remaining) if isinstance(remaining, (int, float)) else None


class DailyQuota:
    """UTC 日ごとの送信数を数える共有カウンタ(無料枠の全モデル共有を表現)。"""

    def __init__(self, daily_limit: int | None = None, stop_on_exhaustion: bool = False):
        self.daily_limit = daily_limit
        self.stop_on_exhaustion = stop_on_exhaustion
        self._lock = threading.Lock()
        self._day = self._today()
        self._used = 0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _rollover(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._used = 0

    @property
    def used_today(self) -> int:
        with self._lock:
            self._rollover()
            return self._used

    @property
    def remaining(self) -> int | None:
        if self.daily_limit is None:
            return None
        return max(0, self.daily_limit - self.used_today)

    def consume(self, n: int = 1) -> None:
        """1 送信分を消費する。上限到達かつ stop_on_exhaustion のときだけ例外。"""
        with self._lock:
            self._rollover()
            if self.daily_limit is not None and self._used + n > self.daily_limit:
                raise DailyQuotaExhausted(
                    f"無料枠の日次上限に達しました(used={self._used}, limit={self.daily_limit}, day={self._day} UTC)"
                )
            self._used += n
