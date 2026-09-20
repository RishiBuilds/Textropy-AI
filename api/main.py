import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

from api.middleware.cors import setup_cors
from api.middleware.auth import setup_auth
from api.middleware.rate_limit import setup_rate_limit
from api.routers.ocr import router as ocr_router
from api.models.schemas import ChatRequest, ChatResponse
from core.chat_engine import chat_with_document, QUICK_ACTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("textropy.api")

_startup_time: float = 0.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()

    nvidia_key = os.getenv("NVIDIA_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    logger.info("=" * 60)
    logger.info("  Textropy AI OCR API starting up...")
    logger.info(f"   Project root: {PROJECT_ROOT}")
    logger.info(f"   NVIDIA API Key: {'Found' if nvidia_key else 'Missing'}")
    logger.info(f"   OpenRouter API Key: {'Found' if openrouter_key else 'Missing'}")
    logger.info(f"   Port: {os.getenv('API_PORT', '8000')}")
    logger.info("=" * 60)

    if not nvidia_key and not openrouter_key:
        logger.warning(
            "No API keys found! At least one of NVIDIA_API_KEY or "
            "OPENROUTER_API_KEY must be set in .env for OCR to work."
        )

    yield

    logger.info("Textropy AI OCR API shutting down...")

app = FastAPI(
    title="Textropy AI OCR API",
    description=(
        "FastAPI bridge for Textropy AI. "
        "Provides REST endpoints for image/PDF OCR processing "
        "using NVIDIA Nemotron and OpenRouter models."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

setup_auth(app)
setup_rate_limit(app)
setup_cors(app)
app.include_router(ocr_router)

@app.get("/health", tags=["System"])
async def health_check():
    uptime = round(time.time() - _startup_time, 2) if _startup_time else 0.0
    return JSONResponse(
        content={
            "status": "ok",
            "service": "textropy-ai",
            "version": "2.0.0",
            "uptime_seconds": uptime,
        }
    )

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Textropy AI OCR API is running",
        "docs": "/docs",
        "health": "/health",
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.document_text.strip():
        raise HTTPException(status_code=400, detail="document_text cannot be empty.")

    try:
        answer = await chat_with_document(req.document_text, req.question, req.history)
        return ChatResponse(answer=answer, model="meta-llama/llama-4-maverick:free")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/actions")
async def chat_actions():
    return {"actions": QUICK_ACTIONS}

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )