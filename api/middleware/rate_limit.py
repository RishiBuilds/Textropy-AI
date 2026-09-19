import os
import time
import threading
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("textropy.ratelimit")

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "30"))          
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60")) 

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int, window: int):
        super().__init__(app)
        self.limit = limit
        self.window = window
        self._lock = threading.Lock()
        self._counters: dict = {}  

    def _client_key(self, request: Request) -> str:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        key = self._client_key(request)
        now = time.time()

        with self._lock:
            window_start, count = self._counters.get(key, (now, 0))
            if now - window_start >= self.window:
                window_start, count = now, 0
            count += 1
            self._counters[key] = (window_start, count)

            if len(self._counters) > 10000:
                cutoff = now - self.window
                self._counters = {
                    k: v for k, v in self._counters.items() if v[0] >= cutoff
                }

        if count > self.limit:
            retry_after = int(self.window - (now - window_start)) or 1
            logger.warning(f"[Textropy AI] Rate limit exceeded for {key}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Slow down."},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

def setup_rate_limit(app: FastAPI) -> None:
    app.add_middleware(RateLimitMiddleware, limit=RATE_LIMIT, window=RATE_LIMIT_WINDOW)
    logger.info(
        f"[Textropy AI] Rate limiting enabled: {RATE_LIMIT} req / "
        f"{RATE_LIMIT_WINDOW}s per client."
    )
