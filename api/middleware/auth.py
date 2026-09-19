import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("textropy.auth")

EXEMPT_PATHS = ("/health", "/", "/docs", "/redoc", "/openapi.json")

class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or path in EXEMPT_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != self.api_key:
            logger.warning(
                f"[Textropy AI] Unauthorized request to {path} from "
                f"{request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header."},
            )
        return await call_next(request)

def setup_auth(app: FastAPI) -> None:
    api_key = os.getenv("TEXTROPY_API_KEY", "").strip()
    if not api_key:
        logger.info(
            "[Textropy AI] Auth disabled (set TEXTROPY_API_KEY in .env to enable)."
        )
        return
    app.add_middleware(APIKeyMiddleware, api_key=api_key)
    logger.info("[Textropy AI] API-key authentication enabled.")
