"""共通 HTTP クライアント。

タイムアウト・指数バックオフ・Retry-After の尊重・エラー分類を 1 か所にまとめ、
全方式が同じ実行条件(timeout/リトライ方針)を使うようにする。
実際の処理は urllib(標準ライブラリ)で行う。sleep はテスト用に差し替え可能。
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class HttpResult:
    ok: bool
    status: int | None
    body: dict | None
    text: str | None
    headers: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    error_type: str | None = None  # "http_429" | "http_5xx" | "http_4xx" | "timeout" | "connection" | "parse"
    attempts: int = 1
    retry_after: float | None = None


def classify_status(status: int) -> str | None:
    if 200 <= status < 300:
        return None
    if status == 429:
        return "http_429"
    if status >= 500:
        return "http_5xx"
    return "http_4xx"


class HttpClient:
    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self._sleep = sleep

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return max(0.0, min(retry_after, self.backoff_max_seconds))
        return min(self.backoff_base_seconds * (2**attempt), self.backoff_max_seconds)

    def post_json(
        self,
        url: str,
        payload: dict,
        headers: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> HttpResult:
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        data = json.dumps(payload).encode("utf-8")
        timeout = timeout_seconds or self.timeout_seconds
        last = HttpResult(ok=False, status=None, body=None, text=None)
        for attempt in range(self.max_retries + 1):
            last.attempts = attempt + 1
            start = time.perf_counter()
            try:
                req = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (固定の公式エンドポイント)
                    raw = resp.read().decode("utf-8", errors="replace")
                    latency = (time.perf_counter() - start) * 1000.0
                    response_headers = {k.lower(): v for k, v in resp.headers.items()}
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    return HttpResult(
                        ok=False,
                        status=resp.status,
                        body=None,
                        text=raw[:2000],
                        headers=response_headers,
                        latency_ms=latency,
                        error_type="parse",
                        attempts=attempt + 1,
                    )
                return HttpResult(
                    ok=True,
                    status=resp.status,
                    body=body,
                    text=None,
                    headers=response_headers,
                    latency_ms=latency,
                    attempts=attempt + 1,
                )
            except urllib.error.HTTPError as exc:  # 4xx/5xx
                latency = (time.perf_counter() - start) * 1000.0
                text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                headers_lower = {k.lower(): v for k, v in (exc.headers or {}).items()}
                retry_after = None
                if "retry-after" in headers_lower:
                    try:
                        retry_after = float(headers_lower["retry-after"])
                    except ValueError:
                        retry_after = None
                error_type = classify_status(exc.code)
                last = HttpResult(
                    ok=False,
                    status=exc.code,
                    body=None,
                    text=text[:2000],
                    headers=headers_lower,
                    latency_ms=latency,
                    error_type=error_type,
                    attempts=attempt + 1,
                    retry_after=retry_after,
                )
                if exc.code not in RETRYABLE_STATUS or attempt >= self.max_retries:
                    return last
            except (socket.timeout, TimeoutError) as exc:
                latency = (time.perf_counter() - start) * 1000.0
                last = HttpResult(
                    ok=False,
                    status=None,
                    body=None,
                    text=str(exc)[:200],
                    latency_ms=latency,
                    error_type="timeout",
                    attempts=attempt + 1,
                )
            except urllib.error.URLError as exc:
                latency = (time.perf_counter() - start) * 1000.0
                last = HttpResult(
                    ok=False,
                    status=None,
                    body=None,
                    text=str(exc.reason)[:200],
                    latency_ms=latency,
                    error_type="connection",
                    attempts=attempt + 1,
                )
            if attempt < self.max_retries:
                self._sleep(self._backoff(attempt, last.retry_after))
        return last
