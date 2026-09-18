import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("textropy.cors")


def get_allowed_origins() -> list[str]:
    env_origins = os.getenv("ALLOWED_ORIGINS", "")
    is_production = os.getenv("PRODUCTION", "false").lower() == "true"

    if is_production and env_origins:
        origins = [o.strip() for o in env_origins.split(",") if o.strip()]
        logger.info(f"[Textropy AI] CORS production mode - allowed origins: {origins}")
        return origins

    default_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8501",
    ]

    if env_origins:
        extra = [o.strip() for o in env_origins.split(",") if o.strip()]
        default_origins.extend(extra)

    origins = list(dict.fromkeys(default_origins))
    logger.info(f"[Textropy AI] CORS dev mode - allowed origins: {origins}")
    return origins


class CORSRejectionLogger(BaseHTTPMiddleware):
    def __init__(self, app, allowed_origins: list[str], is_production: bool):
        super().__init__(app)
        self.allowed_origins = set(allowed_origins)
        self.is_production = is_production

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin", "")

        if self.is_production and origin:
            is_allowed = (
                origin in self.allowed_origins
                or origin.startswith("chrome-extension://")
            )

            if not is_allowed:
                logger.warning(
                    f"[Textropy AI] CORS REJECTED - origin='{origin}' "
                    f"path='{request.url.path}' method='{request.method}'"
                )

        response = await call_next(request)
        return response


def setup_cors(app: FastAPI) -> None:
    allowed_origins = get_allowed_origins()
    is_production = os.getenv("PRODUCTION", "false").lower() == "true"

    app.add_middleware(
        CORSRejectionLogger,
        allowed_origins=allowed_origins,
        is_production=is_production,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins if is_production else ["*"],
        allow_origin_regex=r"^chrome-extension://.*$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Processing-Time-Ms", "X-Client-Version"],
    )

    mode = "production" if is_production else "dev"
    logger.info(f"[Textropy AI] CORS middleware mounted ({mode} mode)")
