<p align="center">
  <img src="assets/textropy_ai-banner.png" width="420" alt="Textropy AI"/>
</p>

<h1 align="center">Textropy AI</h1>

<p align="center">
  <b>Full-stack OCR platform that turns handwritten and printed notes into LaTeX-perfect text, then lets you chat with the results.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-2F4BFF.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%2B-16213A.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.64-2F4BFF.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-0.141-1E8E5A.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Chrome_MV3-extension-FFE66D.svg" alt="Chrome Extension">
  <img src="https://img.shields.io/github/stars/RishiBuilds/Textropy-AI?style=social" alt="Stars">
</p>

<p align="center">
  <a href="#architecture">Architecture</a> · 
  <a href="#features">Features</a> · 
  <a href="#quick-start">Quick Start</a> · 
  <a href="#api-reference">API</a> · 
  <a href="#chrome-extension">Extension</a> · 
  <a href="#docker">Docker</a> · 
  <a href="#tech-stack">Tech Stack</a>
</p>

---

## Overview

Textropy AI is not a wrapper around a generic OCR API. It is a three-surface platform (Streamlit dashboard, FastAPI backend, Chrome extension) built on a shared `core/` engine that pipelines image enhancement, subject-aware prompt engineering, and vision-language model inference to produce structured, LaTeX-ready output from math-heavy documents.

The system uses **Qwen 2.5 VL 72B**, **NVIDIA Nemotron**, and other VLMs via OpenRouter and NVIDIA NIM, with prompts specifically engineered for each academic discipline (Physics, Calculus, Linear Algebra, Chemistry, Statistics, Computer Science). This means integrals, matrices, chemical formulas, and Greek notation are transcribed with high fidelity rather than being destroyed by general-purpose OCR.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                               CLIENT LAYER                               │
│                                                                          │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐  │
│  │    Streamlit     │     │    Chrome MV3    │     │     Any HTTP     │  │
│  │    Dashboard     │     │    Extension     │     │      Client      │  │
│  │    (app.py)      │     │   (extension/)   │     │   (curl, SDK)    │  │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘  │
│           │                        │                        │            │
│           │ (direct Python import) │ REST (JSON)            │ REST (JSON)│
└───────────┼────────────────────────┼────────────────────────┼────────────┘
            │                        │                        │             
            ▼                        ▼                        ▼             
┌──────────────────────────────────────────────────────────────────────────┐
│                             API LAYER (api/)                             │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  FastAPI Application (api/main.py)                               │   │
│   │  - Lifespan Manager: Startup/shutdown logging, API key check     │   │
│   │  - GET  /health          -> HealthResponse                       │   │
│   │  - POST /ocr             -> OCRResponse (file upload, full page) │   │
│   │  - POST /ocr/spot        -> SpotResponse (bounding box JSON)     │   │
│   │  - POST /chat            -> ChatResponse (grounded doc chat)     │   │
│   │  - GET  /chat/actions    -> Quick action presets list            │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌────────────────────── Middleware Pipeline ───────────────────────┐   │
│   │                                                                  │   │
│   │  Request -> [Auth] -> [Rate Limit] -> [CORS] -> Router -> Exec   │   │
│   │                                                                  │   │
│   │  - Auth:       X-API-Key header validation (opt-in via env)      │   │
│   │  - Rate Limit: 30 req/60s sliding window (IP / API key bucketed) │   │
│   │  - CORS:       Origin allow-list + chrome-extension:// origins   │   │
│   └────────────────────────────────┬─────────────────────────────────┘   │
│                                    │                                     │
└────────────────────────────────────┼─────────────────────────────────────┘
                                     │                                      
                                     ▼                                      
┌──────────────────────────────────────────────────────────────────────────┐
│                           CORE ENGINE (core/)                            │
│                                                                          │
│   ┌───────────────────┐    ┌────────────────────┐   ┌──────────────────┐ │
│   │   ocr_engine.py   │    │   chat_engine.py   │   │image_enhancer.py │ │
│   │                   │    │                    │   │                  │ │
│   │ - Subject prompts │    │ - chat_with_doc    │   │ - CLAHE local    │ │
│   │   (LaTeX syntax)  │    │   (grounded async  │   │   contrast       │ │
│   │ - Model routing   │    │   document Q&A)    │   │ - Gaussian       │ │
│   │   (OR / NIM)      │    │ - QUICK_ACTIONS    │   │   unsharp mask   │ │
│   │ - LaTeX cleaner   │    │   (6 presets)      │   │ - Hough deskew   │ │
│   │ - parse_json      │    │ - AsyncOpenAI      │   │   (conservative, │ │
│   │ - normalize_boxes │    │   client wrapper   │   │   opt-in)        │ │
│   └─────────┬─────────┘    └─────────┬──────────┘   └──────────────────┘ │
│             │                        │                                   │
└─────────────┼────────────────────────┼───────────────────────────────────┘
              │                        │                                    
              ▼                        ▼                                    
┌──────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SERVICES & MODELS                        │
│                                                                          │
│   ┌──────────────────────────────┐    ┌──────────────────────────────┐   │
│   │  OpenRouter API              │    │  NVIDIA NIM                  │   │
│   │  - Qwen 2.5 VL 72B (default) │    │  - Nemotron OCR v1           │   │
│   │  - Nemotron Nano 12B VL      │    │    (direct REST, base64)     │   │
│   │  - Baidu Qianfan OCR         │    │                              │   │
│   │  - Llama 4 Maverick (chat)   │    │                              │   │
│   └──────────────────────────────┘    └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Shared `core/` package**: Both `app.py` (Streamlit) and `api/` (FastAPI) import from `core/`. There is zero duplicated engine code. Root-level shims (`ocr_engine.py`, `chat_engine.py`, `image_enhancer.py`) re-export for backwards compatibility.
- **Subject-aware prompt engineering**: Instead of a single generic OCR prompt, `get_ocr_prompt()` generates discipline-specific system prompts with strict LaTeX transcription rules (fraction orientation, bracket types, sign preservation, self-verification steps).
- **Dual model routing**: OCR requests route through OpenRouter (OpenAI-compatible client) by default. NVIDIA Nemotron OCR v1 uses a direct REST call with base64 image payload and bounded timeout.
- **Async chat, sync OCR**: The chat engine uses `AsyncOpenAI` for non-blocking document Q&A. OCR inference is synchronous (VLM latency dominates) and parallelized across PDF pages using `ThreadPoolExecutor(max_workers=3)`.

---

## Features

### OCR Pipeline

| Capability | Details |
|---|---|
| **Full-page extraction** | Single-shot extraction of all text and math from an image or PDF page. Output is valid LaTeX with `$$` delimiters. |
| **Text spotting** | Returns JSON array of `{bbox_2d, text_content}` with coordinates normalized to 1000×1000. Bounding boxes are block-level (paragraphs, equations) not character-level. |
| **PDF batch processing** | Up to 30 pages processed in parallel via `ThreadPoolExecutor`. Per-page results stitched with page separators. |
| **Image enhancement** | OpenCV pipeline: CLAHE local contrast normalization → Gaussian unsharp masking → optional Hough-line deskew (conservative, ±0.5°–15° range). |

### Document Intelligence

| Capability | Details |
|---|---|
| **Grounded chat** | Llama 4 Maverick answers questions strictly from the extracted document text. Supports up to 24K characters of context (configurable via `CHAT_CONTEXT_CHARS`). |
| **Quick actions** | Six one-click presets: Summarize, Explain equations, Simplify, Translate to English, Translate to Hindi, Key formulas. |
| **Correction training** | Users can diff-edit OCR output. Each correction is appended to `corrections.jsonl` as a training example for downstream fine-tuning. |

### Subject-Aware Prompt Engineering

Prompts are not one-size-fits-all. `get_ocr_prompt()` selects from specialized system prompts and extraction protocols:

| Subject | Prompt Focus |
|---|---|
| Physics | Vector notation, subscripts, force diagram labels, SI units, Greek symbols |
| Calculus | Integral bounds, limit notation, derivative operators, `d/dx` formatting |
| Linear Algebra | Matrix notation, transpose symbols, determinants, eigenvalue notation |
| Chemistry | Molecular formulas, oxidation states, reaction arrows, stoichiometric coefficients |
| Statistics | Probability notation `P(A\|B)`, summation symbols, distribution notation |
| Computer Science | Big-O notation, pseudocode formatting, logical operators |
| Auto-detect | Identifies the subject first, then applies appropriate rules |

---

## Supported Models

| Model | Provider | Type | Use Case |
|---|---|---|---|
| Qwen 2.5 VL 72B | OpenRouter (paid) | Vision-Language | Highest math OCR accuracy |
| Nemotron Nano 12B VL | OpenRouter (free) | Vision-Language | General text + math |
| Baidu Qianfan OCR Fast | OpenRouter (free) | OCR | Quick text extraction |
| Nemotron OCR v1 | NVIDIA NIM | OCR | NVIDIA-hosted, direct REST |
| Llama 4 Maverick | OpenRouter (free) | LLM | Document chat (async) |

---

## Quick Start

### Prerequisites

- Python **3.10+**
- An [OpenRouter](https://openrouter.ai) API key (free models available)
- *(Optional)* An [NVIDIA](https://build.nvidia.com) API key for Nemotron OCR v1

### 1. Clone and Install

```bash
git clone https://github.com/RishiBuilds/Textropy-AI.git
cd Textropy-AI
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Required — at least one of these must be set
OPENROUTER_API_KEY=sk-or-v1-your-key-here
NVIDIA_API_KEY=nvapi-your-key-here          # optional

# API security (optional — omit to disable auth)
TEXTROPY_API_KEY=your-secret-key

# Tuning (optional)
CHAT_CONTEXT_CHARS=24000                    # max chars forwarded to chat model
RATE_LIMIT=30                               # requests per window
RATE_LIMIT_WINDOW=60                        # window in seconds
NVIDIA_REQUEST_TIMEOUT=60                   # seconds before NVIDIA call times out
```

### 3. Run

**Dashboard only** (Streamlit, direct model calls):
```bash
streamlit run app.py
```

**API only** (FastAPI, for the extension or external clients):
```bash
python -m uvicorn api.main:app --reload --port 8000
```

**Both services** (helper scripts):
```bash
# Windows
.\dev.ps1 all

# macOS / Linux
./dev.sh all
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |
| Health Check | http://localhost:8000/health |

---

## API Reference

### `POST /ocr` — Full-Page Extraction

```bash
curl -X POST http://localhost:8000/ocr \
  -H "X-API-Key: your-secret-key" \
  -F "file=@notes.png" \
  -F "subject=Calculus" \
  -F "model=auto" \
  -F "enhance_img=true"
```

**Response:**
```json
{
  "status": "success",
  "text": "$$\\int_{0}^{\\pi} \\sin x \\, dx = 2$$",
  "model_used": "qwen/qwen-2.5-vl-72b-instruct",
  "pages": 1,
  "subject": "Calculus"
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | JPG, PNG, WEBP, or PDF (max 20 MB) |
| `subject` | string | `Auto-detect` | One of: Auto-detect, Physics, Calculus, Linear Algebra, Chemistry, Statistics, Computer Science, Other |
| `model` | string | `auto` | `auto` routes via `auto_select_model()`, or pass any OpenRouter/NVIDIA model ID |
| `enhance_img` | bool | `false` | Run OpenCV enhancement pipeline before OCR (images only) |
| `page` | int | all | PDF only: extract a single page, or omit for batch (up to 30 pages) |

### `POST /ocr/spot` — Text Spotting

```bash
curl -X POST http://localhost:8000/ocr/spot \
  -H "X-API-Key: your-secret-key" \
  -F "file=@page.png" \
  -F "subject=Physics"
```

Returns `{boxes: "<JSON array>", model_used: "..."}` where each box contains `bbox_2d` (normalized to 1000) and `text_content` (LaTeX).

### `POST /chat` — Document Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "document_text": "The derivative of x^2 is 2x...",
    "question": "Explain the power rule used here",
    "history": []
  }'
```

### `GET /health` — Health Check

```json
{"status": "ok", "service": "textropy-ai", "version": "2.0.0", "uptime_seconds": 42.0}
```

### `GET /chat/actions` — Available Quick Actions

```json
{"actions": {"summarize": "...", "explain_equations": "...", "simplify": "...", ...}}
```

> Interactive API playground with try-it-out: **http://localhost:8000/docs**

---

## Chrome Extension

The Chrome extension (Manifest V3) lets you extract text from any webpage without leaving your tab.

### Install

1. Start the API server: `python -m uvicorn api.main:app --port 8000`
2. Open `chrome://extensions` → enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` folder

### Usage

1. Click the Textropy icon in your toolbar
2. Select a subject (or leave on Auto-detect)
3. Click **Capture & Extract** → drag to select a screen region
4. The extracted LaTeX text is ready to copy

Configure a remote API endpoint from the extension's **Settings** page (works with Docker deployments or a VPS).

---

## Docker

Run the complete stack with Docker Compose:

```bash
docker compose up --build
```

| Service | Port | Container Health Check |
|---|---|---|
| API (FastAPI + Uvicorn) | 8000 | `curl http://localhost:8000/health` every 30s |
| Dashboard (Streamlit) | 8501 | `curl http://localhost:8501/_stcore/health` |

Both services read from `.env`, restart automatically (`unless-stopped`), and the dashboard depends on the API service.

---

## Project Structure

```
Textropy-AI/
├── app.py                      # Streamlit dashboard — upload, extract, spot, chat, export
│
├── core/                       # Shared engine (single source of truth)
│   ├── __init__.py             # Unified re-exports for core package
│   ├── ocr_engine.py           # Prompt generation, model routing, OpenRouter/NVIDIA clients
│   ├── chat_engine.py          # Async document chat with Llama 4 Maverick, quick actions
│   └── image_enhancer.py       # CLAHE, unsharp masking, Hough-line deskew
│
├── api/                        # FastAPI REST backend
│   ├── main.py                 # App factory, lifespan, /chat and /health endpoints
│   ├── routers/
│   │   └── ocr.py              # /ocr and /ocr/spot — file upload, PDF batch, enhancement
│   ├── middleware/
│   │   ├── auth.py             # X-API-Key header validation (opt-in)
│   │   ├── cors.py             # Env-based origin allow-list, rejection logging
│   │   └── rate_limit.py       # Sliding window rate limiter (IP/key bucketed)
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   ├── requirements.txt        # API-specific pinned dependencies
│   ├── Dockerfile              # API-only container
│   ├── start.sh                # Bash startup script
│   └── start.bat               # Windows startup script
│
├── extension/                  # Chrome Manifest V3 extension
│   ├── manifest.json           # Permissions: activeTab, tabs, storage
│   ├── background.js           # Service worker
│   ├── popup.html / popup.js   # Main UI — capture, extract, copy
│   └── options.html / options.js  # Settings — API endpoint, API key
│
├── ocr_engine.py               # Backwards-compatibility shim → core.ocr_engine
├── chat_engine.py              # Backwards-compatibility shim → core.chat_engine
├── image_enhancer.py           # Backwards-compatibility shim → core.image_enhancer
│
├── Dockerfile                  # Full-stack container (dashboard + API)
├── docker-compose.yml          # Two-service orchestration (api + dashboard)
├── dev.ps1 / dev.sh            # Cross-platform dev launcher scripts
├── requirements.txt            # Root pinned dependencies
├── .streamlit/config.toml      # Streamlit theme configuration
└── assets/                     # Brand assets (banner)
```

---

## Tech Stack

| Layer | Technology | Details |
|---|---|---|
| **Frontend** | Streamlit 1.64 | Custom CSS design system (Bricolage Grotesque + Instrument Sans, cobalt brand palette) |
| **Backend** | FastAPI 0.141 + Uvicorn | Async ASGI with lifespan management, structured logging |
| **Middleware** | Custom (Starlette) | API-key auth, sliding window rate limiter, CORS with rejection logging |
| **OCR Models** | OpenRouter + NVIDIA NIM | Qwen 2.5 VL 72B, Nemotron Nano 12B, Qianfan OCR, Nemotron OCR v1 |
| **Chat Model** | OpenRouter (async) | Llama 4 Maverick with 24K char context window |
| **Image Processing** | OpenCV + Pillow + PyMuPDF | CLAHE, unsharp masking, Hough deskew, PDF rasterization at 200 DPI |
| **Extension** | Chrome MV3 | Vanilla JS, zero dependencies, service worker architecture |
| **Containerization** | Docker + Compose | Multi-service with health checks and auto-restart |
| **Validation** | Pydantic v2 | Typed request/response schemas with field constraints |

---

## Contributing

Contributions are welcome. Some areas with room for improvement:

- Additional subject hints (Biology, Economics, Music theory)
- Export formats (DOCX, Notion, Anki cards)
- Fine-tuning pipeline that consumes the `corrections.jsonl` dataset
- WebSocket streaming for real-time OCR progress

---

## License

[MIT](LICENSE)

---

<p align="center">
  <i>Built for students, researchers, and anyone whose notes are smarter than their OCR.</i>
</p>
