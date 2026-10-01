import ast
import asyncio
import base64
import concurrent.futures
import datetime
import difflib
import hashlib
import html
import io
import json
import os
import re
import tempfile
import uuid
import pymupdf as fitz
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageOps

from core.ocr_engine import (
    inference_with_api, auto_select_model, get_ocr_prompt,
    preprocess_latex, parse_json, normalize_boxes
)
from core.chat_engine import chat_with_document, QUICK_ACTIONS
from core.image_enhancer import enhance_image

__version__ = "0.2.0"

load_dotenv(override=True)

st.set_page_config(
    page_title="Textropy AI",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODE_FULL = "Full page"
MODE_SPOT = "Text spotting (bounding boxes)"
MAX_PAGES_LIMIT = 30
CORRECTIONS_FILE = "corrections.jsonl"

QUICK_LABELS = [
    ("summarize", "Summarize"),
    ("explain_equations", "Explain equations"),
    ("simplify", "Simplify"),
    ("translate_english", "Translate to English"),
    ("translate_hindi", "Translate to Hindi"),
    ("key_formulas", "Key formulas"),
]

CONTENT_TYPES = ["Auto-detect", "Prose", "Poetry", "Math", "Code", "Table"]

CONTENT_TYPE_HINTS = {
    "Prose": "The image contains ordinary prose. Preserve paragraph structure.",
    "Poetry": "The image contains a poem. Preserve EVERY line break, stanza break, and leading indentation EXACTLY as written. Never merge lines into a paragraph.",
    "Math": "The image contains mathematics. Transcribe all expressions as LaTeX wrapped in $$ ... $$.",
    "Code": "The image contains source code. Preserve indentation and wrap it in a fenced code block.",
    "Table": "The image contains a table. Output it as a Markdown table.",
}

SUBJECT_TO_CONTENT_TYPE = {
    "Auto-detect": None,
    "Physics": "Math",
    "Calculus": "Math",
    "Linear Algebra": "Math",
    "Chemistry": "Math",
    "Statistics": "Math",
    "Computer Science": "Code",
    "Other": None,
}

MATH_SUBJECTS = {"Physics", "Calculus", "Linear Algebra", "Chemistry", "Statistics"}

CHAT_SUGGESTIONS = {
    "Poetry": [
        ("Fix spelling", "Fix any spelling or transcription mistakes in this text. "
         "Return ONLY the full corrected text, preserving every line break, stanza break, and indentation."),
        ("Explain this stanza", "Explain the meaning of each stanza of this poem."),
        ("Convert to modern English", "Rewrite this text in modern English while keeping its line structure."),
    ],
    "Math": [
        ("Fix errors", "Check the transcription of these equations for mistakes (flipped fractions, wrong signs, "
         "wrong exponents). Return ONLY the full corrected text with LaTeX intact."),
        ("Explain step by step", "Explain the equations in this document step by step."),
        ("Simplify", "Simplify or solve the main expressions in this document, showing each step."),
    ],
    "Code": [
        ("Fix errors", "Check this code transcription for mistakes and return ONLY the full corrected code."),
        ("Explain this code", "Explain what this code does, section by section."),
        ("Add comments", "Return the code with helpful comments added."),
    ],
    "Table": [
        ("Fix errors", "Check this table transcription for mistakes and return ONLY the full corrected table."),
        ("Summarize", "Summarize the key facts in this table."),
    ],
    "_default": [
        ("Fix spelling", "Fix any spelling or transcription mistakes in this text. Return ONLY the full corrected text."),
        ("Summarize", "Summarize this document in a clear, concise format with key points."),
        ("Explain", "Explain the main ideas in this document."),
    ],
}

CONTENT_TYPE_PATTERNS = {
    "Poetry": re.compile(r"(?:^|\n)\s*(?:poetry|stanza|verse\b|poem\b)", re.IGNORECASE),
    "Code": re.compile(r"(?:^|\n)\s*(?:code\b|program\b|script\b|pseudocode\b)", re.IGNORECASE),
    "Table": re.compile(r"(?:^|\n)\s*table\b", re.IGNORECASE),
    "Math": re.compile(r"(?:^|\n)\s*(?:math(?:ematics|ematical)?|equation|calculus|algebra)\b", re.IGNORECASE),
    "Prose": re.compile(r"(?:^|\n)\s*(?:prose|text\b|paragraph|article|essay|letter)\b", re.IGNORECASE),
}

MODEL_MAP = {
    "Nemotron Nano 12B VL (Free Vision Model)": "nvidia/nemotron-nano-12b-v2-vl:free",
    "Baidu Qianfan OCR Fast (Free OCR Model)": "baidu/qianfan-ocr-fast:free",
    "Qwen 2.5 VL 72B (Paid, Ultimate Math OCR)": "qwen/qwen-2.5-vl-72b-instruct",
    "Nemotron OCR v1 (Nvidia API)": "nvidia/nemotron-ocr-v1",
}

SUBJECT_HINTS = {
    "Auto-detect": "First identify the subject area in one word, then extract all text and mathematical expressions precisely.",
    "Physics": "Pay special attention to vector notation, subscripts, force diagram labels, and SI units. Preserve all Greek symbols.",
    "Calculus": "Preserve integral bounds, limit notation, derivative operators (d/dx), and all superscripts/subscripts exactly.",
    "Linear Algebra": "Preserve matrix notation, transpose symbols, determinants, eigenvalue notation, and vector arrows.",
    "Chemistry": "Preserve molecular formulas, oxidation states, reaction arrows, and stoichiometric coefficients exactly.",
    "Statistics": "Preserve probability notation P(A|B), summation symbols, Greek letters, and distribution notation.",
    "Computer Science": "Preserve Big-O notation, pseudocode formatting, logical operators, and code-like expressions.",
    "Other": "",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --paper: #F8FAFC;
    --panel: #FFFFFF;
    --ink: #0F172A;
    --ink-secondary: #334155;
    --muted: #64748B;
    --muted-light: #94A3B8;
    --line: #E2E8F0;
    --line-subtle: #F1F5F9;
    --cobalt: #4F46E5;
    --cobalt-dark: #3730A3;
    --cobalt-light: #6366F1;
    --cobalt-soft: rgba(79, 70, 229, 0.08);
    --cobalt-glow: rgba(79, 70, 229, 0.22);
    --hl: #FEF08A;
    --ok: #10B981;
    --ok-soft: rgba(16, 185, 129, 0.1);
    --rose: #F43F5E;
    --rose-soft: rgba(244, 63, 94, 0.08);
    --amber: #F59E0B;
    --amber-soft: rgba(245, 158, 11, 0.1);
    --shadow-sm: 0 1px 3px 0 rgba(15, 23, 42, 0.06), 0 1px 2px -1px rgba(15, 23, 42, 0.04);
    --shadow-md: 0 4px 16px -2px rgba(15, 23, 42, 0.07), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
    --shadow-lg: 0 12px 32px -4px rgba(15, 23, 42, 0.09), 0 4px 12px -2px rgba(15, 23, 42, 0.04);
    --radius-sm: 8px;
    --radius: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --display: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    --body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'JetBrains Mono', monospace;
}

::selection { background: #E0E7FF; color: var(--cobalt-dark); }

.stApp {
    background-color: var(--paper);
    background-image:
        radial-gradient(ellipse 90% 45% at 50% -10%, rgba(99, 102, 241, 0.07), transparent 70%),
        radial-gradient(circle 350px at 95% 15%, rgba(59, 130, 246, 0.04), transparent),
        linear-gradient(rgba(15, 23, 42, 0.022) 1px, transparent 1px),
        linear-gradient(90deg, rgba(15, 23, 42, 0.022) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
    color: var(--ink);
}
.stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stCaptionContainer"], .stApp .stMarkdown {
    font-family: var(--body);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
    font-family: var(--display);
    color: var(--ink);
    letter-spacing: -0.02em;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.4rem; max-width: 1440px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
.side-brand {
    display: flex; align-items: center; gap: 0.75rem;
    margin: 0 0 1.2rem 0;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--line);
}
.side-brand-icon {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
    display: flex; align-items: center; justify-content: center;
    color: #fff; box-shadow: 0 3px 10px rgba(79, 70, 229, 0.25);
    flex-shrink: 0;
}
.side-brand-icon svg { width: 19px; height: 19px; fill: none; stroke: currentColor; stroke-width: 2.2; }
.side-brand-text { display: flex; flex-direction: column; }
.side-brand-name {
    font-family: var(--display); font-weight: 800; font-size: 1.15rem;
    letter-spacing: -0.02em; color: var(--ink); line-height: 1.15;
}
.side-brand-desc {
    font-size: 0.68rem; font-weight: 600; color: var(--muted);
    letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px;
}
.side-label {
    font-family: var(--display);
    font-weight: 700;
    font-size: 0.95rem;
    margin: 0.4rem 0 0.35rem 0;
    color: var(--ink);
}
.side-note { color: var(--muted); font-size: 0.82rem; line-height: 1.5; margin: 0.25rem 0 0.5rem 0; }
.sidebar-divider { height: 1px; background: var(--line); margin: 1.1rem 0; }
.side-section-title {
    font-family: var(--display);
    font-weight: 700;
    font-size: 0.84rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin: 0.9rem 0 0.5rem 0;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.side-section-title::after {
    content: ""; flex: 1; height: 1px; background: var(--line);
}

/* Topbar Redesign */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1.25rem; flex-wrap: wrap; margin-bottom: 1.6rem;
    padding: 0.85rem 1.35rem;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    animation: fadeUp .4s ease both;
}
.brand-group {
    display: flex; align-items: center; gap: 0.75rem;
}
.brand-icon {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
    display: inline-flex; align-items: center; justify-content: center;
    color: #fff; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.28);
}
.brand-icon svg { width: 19px; height: 19px; display: block; fill: none; stroke: currentColor; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }
.wordmark {
    font-family: var(--display);
    font-weight: 800;
    font-size: 1.32rem;
    letter-spacing: -0.025em;
    color: var(--ink);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0;
    background: none;
}
.version-pill {
    font-family: var(--body);
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--cobalt);
    background: var(--cobalt-soft);
    border: 1px solid rgba(79, 70, 229, 0.2);
    border-radius: 999px;
    padding: 0.12rem 0.5rem;
    letter-spacing: 0.04em;
}
.steps { list-style: none; display: flex; align-items: center; margin: 0; padding: 0; }
.steps li { display: flex; align-items: center; gap: 0.5rem; color: var(--muted); font-size: 0.88rem; font-weight: 500; }
.steps li + li::before {
    content: ""; width: 2rem; height: 2px; background: var(--line); margin: 0 0.55rem; border-radius: 2px;
    transition: background .3s ease;
}
.steps li.done + li::before { background: var(--ok); }
.steps .n {
    width: 1.65rem; height: 1.65rem; border-radius: 50%;
    border: 1.5px solid var(--line); background: var(--panel);
    display: inline-grid; place-items: center; font-size: 0.78rem; font-weight: 700;
    transition: all .25s cubic-bezier(.4,0,.2,1);
    color: var(--muted);
}
.steps li.current { color: var(--ink); font-weight: 700; }
.steps li.current .n {
    background: linear-gradient(135deg, var(--cobalt) 0%, var(--cobalt-light) 100%);
    border-color: var(--cobalt); color: #fff;
    box-shadow: 0 0 0 3.5px rgba(79, 70, 229, 0.18);
}
.steps li.done { color: var(--ink); font-weight: 600; }
.steps li.done .n {
    background: var(--ok); border-color: var(--ok); color: #fff;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15);
}
.topbar-badge {
    display: flex; align-items: center; gap: 0.45rem;
    background: #F8FAFC;
    border: 1px solid var(--line);
    padding: 0.32rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--ink-secondary);
}
.topbar-badge .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--ok);
    box-shadow: 0 0 0 2.5px rgba(16, 185, 129, 0.22);
}

/* Hero Section Redesign */
.hero { animation: fadeUp .5s ease both; margin-bottom: 1.5rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.55rem;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(79, 70, 229, 0.24);
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.08);
    color: var(--cobalt); border-radius: 999px;
    padding: 0.32rem 0.95rem; font-size: 0.82rem; font-weight: 600;
    letter-spacing: 0.01em; margin-bottom: 0.85rem;
    backdrop-filter: blur(8px);
}
.hero-badge .live-dot {
    position: relative; width: 8px; height: 8px; display: inline-flex;
}
.hero-badge .live-dot .ping {
    position: absolute; inset: 0; border-radius: 50%; background: var(--ok);
    animation: ping 1.8s cubic-bezier(0, 0, 0.2, 1) infinite; opacity: 0.75;
}
.hero-badge .live-dot .dot {
    position: relative; width: 8px; height: 8px; border-radius: 50%; background: var(--ok);
}
@keyframes ping {
    75%, 100% { transform: scale(2.2); opacity: 0; }
}
.hero-badge .badge-separator { color: var(--muted-light); }
.hero-badge .badge-accent { font-weight: 700; color: var(--cobalt-dark); }

.hero h1 {
    font-size: clamp(2.3rem, 4.4vw, 3.4rem);
    line-height: 1.08;
    font-weight: 800;
    margin: 0.4rem 0 1.1rem 0;
    letter-spacing: -0.03em;
    color: var(--ink);
}
.hero-gradient {
    background: linear-gradient(135deg, #3730A3 0%, #4F46E5 35%, #2563EB 70%, #7C3AED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: 1.08rem; line-height: 1.62; color: var(--muted); max-width: 54ch; margin: 0 0 0.9rem 0;
}
.hero .fine { font-size: 0.88rem; color: #94A3B8; margin-top: -0.2rem; margin-bottom: 1.1rem; }

.hero-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.1rem 0 1.4rem 0; }
.hero-chip {
    display: inline-flex; align-items: center; gap: 0.45rem;
    border: 1px solid var(--line); background: var(--panel);
    border-radius: 999px; padding: 0.35rem 0.85rem; font-size: 0.82rem; font-weight: 600;
    color: var(--ink-secondary); box-shadow: var(--shadow-sm);
    transition: all .2s ease;
}
.hero-chip:hover {
    border-color: var(--cobalt-light); color: var(--cobalt);
    transform: translateY(-1px); box-shadow: var(--shadow-md);
}
.hero-chip svg { width: 14px; height: 14px; stroke: var(--cobalt); fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

.hero-stats {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem;
    margin-top: 1.4rem; padding-top: 1.2rem;
    border-top: 1px solid var(--line);
}
.hero-stat-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 0.85rem 1rem;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
    transition: all .2s ease;
}
.hero-stat-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--cobalt), var(--cobalt-light));
    opacity: 0.85;
}
.hero-stat-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: rgba(79, 70, 229, 0.3);
}
.hero-stat-card .num {
    font-family: var(--display); font-weight: 800; font-size: 1.55rem; line-height: 1.1;
    background: linear-gradient(135deg, var(--ink) 0%, var(--cobalt) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-stat-card .lbl {
    font-size: 0.74rem; font-weight: 600; color: var(--muted); margin-top: 0.25rem;
    letter-spacing: 0.01em;
}

/* Preview Window Mockup */
.preview-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
    margin-top: 0.6rem;
}
.preview-card-header {
    background: #F8FAFC;
    border-bottom: 1px solid var(--line);
    padding: 0.65rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.mac-dots { display: flex; gap: 6px; }
.mac-dot { width: 10px; height: 10px; border-radius: 50%; }
.mac-dot.red { background: #FF5F56; }
.mac-dot.yellow { background: #FFBD2E; }
.mac-dot.green { background: #27C93F; }
.preview-title { font-size: 0.76rem; font-weight: 600; color: var(--muted); font-family: var(--mono); }

.panel-title { font-family: var(--display); font-weight: 700; font-size: 1.08rem; margin: 0 0 0.45rem 0; color: var(--ink); }
.chips { margin: 0.2rem 0 1rem 0; }
.chip {
    display: inline-block; border: 1px solid var(--line); background: var(--panel);
    border-radius: 999px; padding: 0.2rem 0.8rem; font-size: 0.82rem; font-weight: 500; color: var(--ink-secondary);
    margin: 0 0.4rem 0.3rem 0; box-shadow: var(--shadow-sm);
    transition: all .15s ease;
}
.chip:hover { border-color: var(--cobalt-light); color: var(--cobalt); }
.empty {
    border: 1.5px dashed #CBD5E1; border-radius: var(--radius); padding: 2.2rem 1.4rem;
    background: rgba(255,255,255,0.85); color: var(--muted); text-align: center; line-height: 1.55;
    animation: fadeUp .4s ease both; box-shadow: var(--shadow-sm);
}
.empty b { color: var(--ink); }
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--radius); }
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {
    border-color: var(--line);
    box-shadow: var(--shadow-sm);
    background: var(--panel);
}

/* Buttons */
.stButton button, .stDownloadButton button {
    border-radius: 10px; font-weight: 600; box-shadow: var(--shadow-sm);
    border: 1px solid var(--line); background: var(--panel); color: var(--ink);
    transition: all .2s cubic-bezier(.4,0,.2,1);
    position: relative; overflow: hidden;
}
.stButton button p, .stDownloadButton button p { color: inherit; font-weight: 600; }
.stButton button:hover, .stDownloadButton button:hover {
    border-color: var(--cobalt-light); color: var(--cobalt); background: #FAF5FF;
    transform: translateY(-2px); box-shadow: var(--shadow-md);
}
.stButton button:active, .stDownloadButton button:active { transform: translateY(0); box-shadow: none; }
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
    border-color: transparent !important; color: #fff !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.32) !important;
}
.stButton button[kind="primary"] p, .stButton button[data-testid="stBaseButton-primary"] p {
    color: #fff !important;
}
.stButton button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #4338CA 0%, #2563EB 100%) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
    transform: translateY(-2px);
}
.stButton button[kind="primary"]:active {
    transform: translateY(0); box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25) !important;
}
button:focus-visible, textarea:focus-visible, input:focus-visible {
    outline: 2px solid var(--cobalt) !important; outline-offset: 2px;
}

.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background: transparent; padding: 0.55rem 0; color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--cobalt); height: 3px; border-radius: 3px 3px 0 0; }

[data-testid="stFileUploaderDropzone"] {
    background: var(--panel); border: 2px dashed #CBD5E1; border-radius: var(--radius); padding: 1.6rem;
    transition: all .2s ease; box-shadow: var(--shadow-sm);
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--cobalt); background: #F8FAFC;
    transform: translateY(-1px); box-shadow: var(--shadow-md);
}
.stTextArea textarea {
    font-family: var(--mono); font-size: 0.86rem; line-height: 1.55;
    background: var(--panel); color: var(--ink); border-radius: 10px;
    border-color: var(--line);
}
[data-testid="stChatMessage"] {
    background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}
[data-testid="stExpander"] details {
    background: var(--panel); border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important; box-shadow: var(--shadow-sm);
}
[data-testid="stAlert"] { border-radius: 10px; box-shadow: var(--shadow-sm); }

div[role="radiogroup"] { gap: 0.5rem; }
div[role="radiogroup"] label {
    border: 1px solid var(--line); background: var(--panel); border-radius: 999px;
    padding: 0.35rem 1.05rem; cursor: pointer; box-shadow: var(--shadow-sm);
    transition: all .15s ease;
}
div[role="radiogroup"] label:hover { border-color: var(--cobalt); transform: translateY(-1px); }
div[role="radiogroup"] label:has(input:checked) {
    background: var(--cobalt-soft); border-color: var(--cobalt);
}
div[role="radiogroup"] label:has(input:checked) p { color: var(--cobalt); font-weight: 700; }
div[role="radiogroup"] label > div:first-child { display: none; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cobalt-light); }

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] .side-note,
section[data-testid="stSidebar"] small { color: var(--muted) !important; }
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    position: sticky; top: 2.6rem; z-index: 999;
    background: var(--paper); padding-top: 0.3rem;
}

mark.sus { background: #FEF3C7; color: #92400E; border-radius: 3px; padding: 0 0.15em; font-weight: 600; }
mark.hl-match { background: var(--hl); color: var(--ink); border-radius: 3px; padding: 0 0.15em; }
mark.sus.hl-match { background: linear-gradient(var(--hl), #FEF3C7); }

.prog-steps { display: flex; gap: 0.45rem; align-items: center; margin: 0.5rem 0; }
.prog-step {
    flex: 1; display: flex; align-items: center; justify-content: center; gap: 0.45rem;
    border: 1px solid var(--line); background: var(--panel); border-radius: 999px;
    padding: 0.35rem 0.75rem; font-size: 0.82rem; font-weight: 600; color: var(--muted);
    transition: all .2s ease; box-shadow: var(--shadow-sm);
}
.prog-step.active {
    border-color: var(--cobalt); color: var(--cobalt); background: var(--cobalt-soft);
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
}
.prog-step.done { border-color: var(--ok); color: var(--ok); background: var(--ok-soft); }

.word-btn button { padding: 0.1rem 0.55rem !important; min-height: 0 !important; height: auto !important; font-size: 0.8rem !important; }

.stDownloadButton button {
    background: linear-gradient(135deg, var(--cobalt) 0%, var(--cobalt-light) 100%) !important;
    border-color: transparent !important; color: #fff !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.32) !important;
}
.stDownloadButton button:hover {
    background: linear-gradient(135deg, var(--cobalt-dark) 0%, var(--cobalt) 100%) !important;
    border-color: transparent !important; color: #fff !important;
    transform: translateY(-2px);
}

.chip-select [data-testid="stSelectbox"] > div > div {
    border-radius: 999px; min-height: 2rem; font-size: 0.82rem;
    background: var(--panel); border-color: var(--line); box-shadow: var(--shadow-sm);
}
.chip-select [data-testid="stSelectbox"] { max-width: 13rem; }

.app-footer {
    text-align: center; padding: 2rem 0 1.2rem; margin-top: 2.5rem;
    border-top: 1px solid var(--line); color: var(--muted); font-size: 0.8rem;
}
.app-footer a { color: var(--cobalt); text-decoration: none; font-weight: 600; }
.app-footer a:hover { text-decoration: underline; }

@media (max-width: 640px) {
    .steps li:not(.current) .t { display: none; }
    .steps li + li::before { width: 0.8rem; margin: 0 0.3rem; }
    .hero-stats { grid-template-columns: 1fr; gap: 0.6rem; }
    .topbar { padding: 0.75rem 1rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

_DEFAULTS = {
    "session_id": None,
    "extracted_text": "",
    "correction_editor": "",
    "spot_response": "",
    "spot_image": None,
    "subject": "Auto-detect",
    "chat_history": [],
    "doc_id": None,
    "chat_histories": {},
    "content_type": None,
    "content_type_override": None,
    "last_model_id": None,
    "extract_error": None,
    "zoom": 100,
    "rotate": 0,
    "highlight_word": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = uuid.uuid4().hex if _k == "session_id" else _v


def reset_extraction_state():
    st.session_state["extracted_text"] = ""
    st.session_state["correction_editor"] = ""
    st.session_state["spot_response"] = ""
    st.session_state["spot_image"] = None
    st.session_state["chat_history"] = []
    st.session_state["content_type"] = None
    st.session_state["content_type_override"] = None
    st.session_state["extract_error"] = None
    st.session_state["highlight_word"] = ""


def on_new_source():
    if st.session_state.get("doc_id"):
        st.session_state["chat_histories"][st.session_state["doc_id"]] = st.session_state.get("chat_history", [])
    st.session_state["doc_id"] = None
    st.session_state["extracted_text"] = ""
    st.session_state["correction_editor"] = ""
    st.session_state["spot_response"] = ""
    st.session_state["spot_image"] = None
    st.session_state["content_type"] = None
    st.session_state["content_type_override"] = None
    st.session_state["extract_error"] = None
    st.session_state["highlight_word"] = ""
    st.session_state.pop("quick_source", None)


def set_extraction(text: str):
    if st.session_state.get("doc_id"):
        st.session_state["chat_histories"][st.session_state["doc_id"]] = []
    st.session_state["extracted_text"] = text
    st.session_state["correction_editor"] = text
    st.session_state["chat_history"] = []


def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)


def stretch_button(label, **kwargs):
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def detect_content_type(text: str, subject: str):
    mapped = SUBJECT_TO_CONTENT_TYPE.get(subject)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return mapped or "Prose", ""
    first = lines[0]
    m = re.match(r"^\**\s*(?:type\s*[:\-–]\s*)?([A-Za-z][A-Za-z ]{1,24}?)\s*[:\-–]?\s*\**\s*$", first)
    if m:
        cand = m.group(1).strip()
        for ctype, pat in CONTENT_TYPE_PATTERNS.items():
            if pat.search(cand):
                return ctype, "\n".join(lines[1:]).strip()
    return mapped or "Prose", ""


def effective_content_type(subject: str) -> str:
    override = st.session_state.get("content_type_override")
    if override:
        return override
    detected = st.session_state.get("content_type")
    if detected:
        return detected
    return SUBJECT_TO_CONTENT_TYPE.get(subject) or "Auto-detect"


def is_math_type(subject: str, ctype: str) -> bool:
    if ctype == "Math":
        return True
    if ctype == "Auto-detect":
        return subject in MATH_SUBJECTS
    return False


def count_suspicious_words(text: str):
    pattern = re.compile(
        r"\b(?:[A-Za-z]{2,}'[A-Za-z]{2,}|(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{2,})\b|[\x00-\x08\x0b\x0c\x0e-\x1f]"
    )
    return pattern.findall(text or "")


def docx_bytes(text: str) -> bytes:
    paras = []
    for line in text.split("\n"):
        esc = html.escape(line, quote=False)
        content = (
            '<w:r><w:t xml:space="preserve">' + esc + "</w:t></w:r>" if esc else ""
        )
        paras.append("<w:p>" + content + "</w:p>")
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>" + "".join(paras) + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    buf = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)
    buf.seek(0)
    return buf.read()


def progress_steps(placeholder, active: int):
    labels = ["Enhancing image", "Reading", "Formatting"]
    parts = []
    for i, label in enumerate(labels, start=1):
        state = "done" if i < active else ("active" if i == active else "")
        mark = "&#10003;" if state == "done" else str(i)
        parts.append(f'<div class="prog-step {state}"><span>{mark}</span><span>{label}</span></div>')
    placeholder.markdown(f'<div class="prog-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def zoomed_viewer(image_path: str, img_height: int, key: str):
    try:
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
    except OSError:
        show_image(st.session_state.get("spot_image") or Image.new("RGB", (8, 8), "white"))
        return
    frame_h = min(760, max(380, int(img_height * 0.85) + 90))
    components.html(
        f"""
        <style>
          html,body{{margin:0;padding:0;background:#fff;font-family:'Instrument Sans',system-ui,sans-serif;}}
          #bar{{display:flex;gap:6px;align-items:center;padding:6px 8px;border-bottom:1px solid #D8DEE8;
               position:sticky;top:0;background:#fff;z-index:5;}}
          #bar button{{border:1px solid #D8DEE8;background:#fff;color:#16213A;border-radius:8px;
               padding:4px 12px;font:600 13px 'Instrument Sans',system-ui,sans-serif;cursor:pointer;}}
          #bar button:hover{{border-color:#2F4BFF;color:#2F4BFF;}}
          #zl{{font:600 12px 'JetBrains Mono',monospace;color:#454F62;min-width:44px;text-align:center;}}
          #wrap{{height:{frame_h - 46}px;overflow:auto;cursor:grab;background:
               repeating-conic-gradient(#F4F6FA 0% 25%, #fff 0% 50%) 0 0/24px 24px;}}
          #wrap.grabbing{{cursor:grabbing;}}
          #img{{display:block;margin:16px auto;transform-origin:top center;transition:transform .12s ease;
               max-width:none;box-shadow:0 4px 24px rgba(22,33,58,0.14);}}
        </style>
        <div id="bar">
          <button id="zi" title="Zoom in">Zoom +</button>
          <button id="zo" title="Zoom out">Zoom &minus;</button>
          <button id="rr" title="Rotate right">&#8635;</button>
          <button id="rl" title="Rotate left">&#8634;</button>
          <button id="rs" title="Reset">Reset</button>
          <span id="zl">100%</span>
        </div>
        <div id="wrap"><img id="img" src="data:image/jpeg;base64,{b64}"></div>
        <script>
          let zoom=1, rot=0;
          const img=document.getElementById('img'), zl=document.getElementById('zl'),
                wrap=document.getElementById('wrap');
          function apply(){{ img.style.transform=`scale(${{zoom}}) rotate(${{rot}}deg)`;
            zl.textContent=Math.round(zoom*100)+'%'; }}
          document.getElementById('zi').onclick=()=>{{zoom=Math.min(6,zoom*1.25);apply();}};
          document.getElementById('zo').onclick=()=>{{zoom=Math.max(0.1,zoom/1.25);apply();}};
          document.getElementById('rr').onclick=()=>{{rot=(rot+90)%360;apply();}};
          document.getElementById('rl').onclick=()=>{{rot=(rot+270)%360;apply();}};
          document.getElementById('rs').onclick=()=>{{zoom=1;rot=0;apply();}};
          wrap.addEventListener('wheel',e=>{{ if(e.ctrlKey){{ e.preventDefault();
            zoom=Math.min(6,Math.max(0.1, zoom*(e.deltaY<0?1.15:0.87))); apply(); }} }},{{passive:false}});
          let drag=null;
          wrap.addEventListener('mousedown',e=>{{ drag={{x:e.clientX,y:e.clientY,
            l:wrap.scrollLeft,t:wrap.scrollTop}}; wrap.classList.add('grabbing'); e.preventDefault(); }});
          window.addEventListener('mousemove',e=>{{ if(drag){{ wrap.scrollLeft=drag.l-(e.clientX-drag.x);
            wrap.scrollTop=drag.t-(e.clientY-drag.y); }} }});
          window.addEventListener('mouseup',()=>{{drag=null;wrap.classList.remove('grabbing');}});
        </script>
        """,
        height=frame_h,
    )


def render_structured(text: str, ctype: str):
    sus_set = set(count_suspicious_words(text))
    hl = st.session_state.get("highlight_word", "")

    def mark_up(raw: str) -> str:
        esc = html.escape(raw)
        for w in sorted(sus_set, key=len, reverse=True):
            if not w:
                continue
            cls = "sus hl-match" if hl and w == hl else "sus"
            esc = esc.replace(html.escape(w), f'<mark class="{cls}">{html.escape(w)}</mark>')
        if hl and hl not in sus_set:
            esc = esc.replace(html.escape(hl), f'<mark class="hl-match">{html.escape(hl)}</mark>')
        return esc

    if ctype == "Math":
        render_latex_safely(preprocess_latex(text))
        return
    if ctype == "Code":
        m = re.search(r"```(?:[a-zA-Z]*)?\n(.*?)```", text, re.DOTALL)
        st.code(m.group(1) if m else text)
        return
    if ctype == "Poetry":
        html_lines = []
        for ln in text.split("\n"):
            stripped = ln.strip()
            if not stripped:
                html_lines.append("<br>")
                continue
            indent_px = min(60, (len(ln) - len(ln.lstrip())) * 8)
            html_lines.append(f'<div style="padding-left:{indent_px}px">{mark_up(stripped)}</div>')
        st.markdown(
            '<div style="white-space:normal;line-height:1.75;font-size:1.02rem">'
            + "".join(html_lines) + "</div>",
            unsafe_allow_html=True,
        )
        return
    paragraphs = [mark_up(p.replace("\n", " ")) for p in re.split(r"\n\s*\n", text) if p.strip()]
    st.markdown(
        '<div style="line-height:1.7;font-size:1.02rem">'
        + "".join(f"<p>{p}</p>" for p in paragraphs) + "</div>",
        unsafe_allow_html=True,
    )


def apply_edit_to_document(answer: str):
    extracted = st.session_state.get("extracted_text", "")
    if not extracted:
        return
    cleaned = answer.strip().strip("`").strip()
    if not cleaned or cleaned == extracted.strip():
        return
    ratio = difflib.SequenceMatcher(None, cleaned, extracted.strip()).ratio()
    if ratio < 0.4:
        return
    st.info("This reply looks like a corrected full version of the document.")
    with st.expander("Preview changes"):
        diff = "\n".join(difflib.unified_diff(
            extracted.splitlines(), cleaned.splitlines(), lineterm="", n=2
        ))
        st.code(diff or "(no visible diff)", language="diff")
    if st.button("Apply this edit to the extracted text", key=f"apply_edit_{abs(hash(answer)) % 10**8}"):
        set_extraction(cleaned)
        st.success("Extracted text updated from the chat reply.")
        st.rerun()


def render_latex_safely(text: str):
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        try:
            st.markdown(block)
        except Exception:
            st.code(block, language="latex")


def render_topbar(current: int):
    labels = ["Add image", "Extract", "Review"]
    items = []
    for i, label in enumerate(labels, start=1):
        state = "done" if i < current else ("current" if i == current else "")
        mark = "&#10003;" if state == "done" else str(i)
        items.append(f'<li class="{state}"><span class="n">{mark}</span><span class="t">{label}</span></li>')
    st.markdown(
        f'<div class="topbar">'
        f'<div class="brand-group">'
        f'<span class="brand-icon"><svg viewBox="0 0 24 24"><path d="M4 7V4h16v3M9 20h6M12 4v16"/></svg></span>'
        f'<span class="wordmark">Textropy AI <span class="version-pill">v{__version__}</span></span>'
        f'</div>'
        f'<ol class="steps">{"".join(items)}</ol>'
        f'<div class="topbar-badge"><span class="dot"></span><span>Vision OCR Active</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def copy_button(text: str):
    payload = html.escape(json.dumps(text), quote=True)
    components.html(
        f"""
        <style>
          html,body{{background:transparent!important;margin:0;padding:0;}}
          button{{width:100%;height:40px;border-radius:10px;border:1px solid #E2E8F0;background:#fff;
                 color:#0F172A;font:600 13px 'Plus Jakarta Sans',system-ui,sans-serif;cursor:pointer;
                 transition:all .2s ease;box-shadow:0 1px 2px rgba(15,23,42,0.04);}}
          button:hover{{border-color:#6366F1;color:#4F46E5;background:#FAF5FF;transform:translateY(-1px);
                 box-shadow:0 4px 12px rgba(79,70,229,0.15);}}
        </style>
        <button onclick="navigator.clipboard.writeText({payload}).then(()=>{{this.textContent='✓ Copied';
          setTimeout(()=>this.textContent='Copy to clipboard',1500)}}).catch(()=>{{}})">Copy to clipboard</button>
        """,
        height=46,
    )


def export_bar(text: str, key: str):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.download_button("Download .txt", data=text, file_name="extracted.txt",
                           mime="text/plain", key=f"txt_{key}")
    with col_b:
        tex = ("\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n"
               f"\\begin{{document}}\n\n{text}\n\n\\end{{document}}")
        st.download_button("Download .tex", data=tex, file_name="extracted.tex",
                           mime="text/plain", key=f"tex_{key}")
    with col_c:
        st.download_button(
            "Download .docx", data=docx_bytes(text), file_name="extracted.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"docx_{key}",
        )
    with col_d:
        copy_button(text)


def parse_boxes(raw: str):
    clean_json = parse_json(raw)
    fixed = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', clean_json)
    try:
        boxes = ast.literal_eval(fixed)
    except Exception:
        try:
            boxes = json.loads(fixed, strict=False)
        except Exception:
            boxes = []
            for block in re.findall(r'\{[^{}]*\}', clean_json):
                bbox_match = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', block)
                text_match = re.search(r'"text(?:_content)?"\s*:\s*"([^"]+)"', block)
                if bbox_match and text_match:
                    try:
                        coords = [int(float(x.strip())) for x in bbox_match.group(1).split(',')]
                        txt = text_match.group(1).replace('\\\\', '\\').replace('\\"', '"')
                        boxes.append({"bbox_2d": coords, "text_content": txt})
                    except Exception:
                        pass
            if not boxes:
                raise ValueError("The response could not be read as bounding boxes.")
    return normalize_boxes(boxes), clean_json


def plot_text_bounding_boxes(image_path, raw_response):
    img = Image.open(image_path)
    width, height = img.size

    try:
        boxes, _ = parse_boxes(raw_response)
    except Exception as e:
        st.error(f"Could not read the boxes: {e}\n\nRaw output:\n\n{raw_response}")
        return img

    size = max(16, int(height * 0.025))
    font = ImageFont.load_default()
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, size=size)
            break
        except Exception:
            continue

    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        bbox = box.get("bbox_2d", box.get("box_2d"))
        if not bbox or len(bbox) < 4:
            continue

        x1 = int(bbox[0] / 1000.0 * width)
        y1 = int(bbox[1] / 1000.0 * height)
        x2 = int(bbox[2] / 1000.0 * width)
        y2 = int(bbox[3] / 1000.0 * height)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        overlay_draw.rectangle(
            ((x1, y1), (x2, y2)),
            outline=(47, 75, 255, 255),
            width=max(2, int(height * 0.003)),
            fill=(47, 75, 255, 36),
        )

        if box.get("text_content", box.get("text", "")):
            badge = str(i + 1)
            try:
                left, top, right, bottom = text_draw.textbbox((0, 0), badge, font=font)
            except Exception:
                left, top, right, bottom = 0, 0, 10, 10
            bw, bh = right - left + 12, bottom - top + 8
            bx, by = max(0, x1), max(0, y1 - bh - 4)
            overlay_draw.rectangle(((bx, by), (bx + bw, by + bh)), fill=(255, 230, 109, 245))
            text_draw.text((bx + 6, by + 2), badge, fill=(22, 33, 58), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


def clean_box_text(box) -> str:
    txt = str(box.get("text_content", box.get("text", ""))).strip().replace("`", "")
    if txt.startswith("$$"):
        txt = txt[2:].strip()
    elif txt.startswith("$"):
        txt = txt[1:].strip()
    if txt.endswith("$$"):
        txt = txt[:-2].strip()
    elif txt.endswith("$"):
        txt = txt[:-1].strip()

    txt = re.sub(r'\\sqrt\[([^\]]+)\]\[([^\]]+)\]', r'\\sqrt[\1]{\2}', txt)

    math_keywords = ['\\frac', '\\min', '\\max', '\\left', '\\right', '\\cos', '\\sin',
                     '\\log', '\\sum', '^', '_', '\\Big', '\\sqrt']
    if any(m in txt for m in math_keywords):
        txt = f"$${txt}$$"
    return txt


def ask_document(question: str):
    history = st.session_state["chat_history"]
    history.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        try:
            answer = asyncio.run(chat_with_document(
                st.session_state["extracted_text"], question, history[:-1]
            ))
        except Exception as e:
            answer = f"Error: {e}"
    history.append({"role": "assistant", "content": answer})
    doc_id = st.session_state.get("doc_id")
    if doc_id:
        st.session_state["chat_histories"][doc_id] = history


def run_batch_extraction(pdf, n, prompt, sys_prompt, model_id) -> str:
    sid = st.session_state["session_id"]
    paths, parts = {}, {}
    progress = st.progress(0.0, text="Preparing pages...")
    try:
        for i in range(n):
            pix = pdf.load_page(i).get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            path = os.path.join(tempfile.gettempdir(), f"ocr_temp_page_{sid}_{i}.jpg")
            img.save(path)
            paths[i] = path

        def work(i):
            try:
                return i, inference_with_api(paths[i], prompt, sys_prompt=sys_prompt, model_id=model_id)
            except Exception as e:
                return i, f"[Page {i + 1} extraction failed: {e}]"

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(work, i) for i in range(n)]
            for done, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                i, result = fut.result()
                parts[i] = result
                progress.progress(done / n, text=f"Extracted {done} of {n} pages")
    finally:
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)
        progress.empty()

    return "".join(
        f"\n\n---\n### Page {i + 1}\n---\n\n{parts.get(i, '[Missing]')}" for i in range(n)
    )


def load_source(camera_image, quick_source, uploaded_file):
    src = {"image": None, "label": "", "is_pdf": False, "pdf": None, "num_pages": 0, "doc_id": None}
    try:
        if camera_image is not None:
            img, label = Image.open(camera_image), "Camera photo"
            doc_id = "cam_" + hashlib.md5(camera_image.getvalue()).hexdigest()
        elif quick_source:
            img, label = Image.open(io.BytesIO(quick_source["data"])), quick_source["name"]
            doc_id = "q_" + hashlib.md5(quick_source["data"]).hexdigest()
        elif uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            if ext == "pdf":
                pdf = fitz.open(stream=file_bytes, filetype="pdf")
                n = len(pdf)
                page_idx = 0
                if n > 1:
                    page_idx = int(st.sidebar.number_input(
                        "PDF page", min_value=1, max_value=n, value=1,
                        on_change=reset_extraction_state)) - 1
                pix = pdf.load_page(page_idx).get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                label = f"{uploaded_file.name}, page {page_idx + 1} of {n}"
                src.update(is_pdf=True, pdf=pdf, num_pages=n)
                doc_id = "pdf_" + hashlib.md5(file_bytes).hexdigest() + f"_p{page_idx}"
            else:
                img, label = Image.open(uploaded_file), uploaded_file.name
                doc_id = "up_" + hashlib.md5(file_bytes).hexdigest()
        else:
            return src

        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        src.update(image=img, label=label, doc_id=doc_id)
    except Exception as e:
        st.sidebar.error(f"Could not open this file: {e}")
        src.update(image=None, is_pdf=False, pdf=None, num_pages=0, doc_id=None)
    return src


def generate_sample_image() -> bytes:
    img = Image.new("RGB", (900, 640), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for name in ("georgia.ttf", "times.ttf", "arial.ttf", "DejaVuSerif.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, size=40)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    lines = [
        "The woods are lovely, dark and deep,",
        "But I have promises to keep,",
        "    And miles to go before I sleep,",
        "    And miles to go before I sleep.",
    ]
    y = 90
    for ln in lines:
        draw.text((90, y), ln, fill=(22, 33, 58), font=font)
        y += 96
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def quick_upload_widget():
    quick_file = st.file_uploader(
        "Drop an image of your notes",
        type=["jpg", "jpeg", "png", "webp"],
        key="quick_upload",
        label_visibility="collapsed",
    )
    if quick_file is not None:
        reset_extraction_state()
        st.session_state["quick_source"] = {"name": quick_file.name, "data": quick_file.getvalue()}
        st.rerun()


sb = st.sidebar

sb.markdown(
    '<div class="side-brand">'
    '<div class="side-brand-icon">'
    '<svg viewBox="0 0 24 24"><path d="M4 7V4h16v3M9 20h6M12 4v16"/></svg>'
    '</div>'
    '<div class="side-brand-text">'
    '<span class="side-brand-name">Textropy AI</span>'
    '<span class="side-brand-desc">Vision OCR Engine</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

sb.markdown('<div class="side-section-title">Image</div>', unsafe_allow_html=True)
uploaded_file = sb.file_uploader(
    "Upload an image or PDF",
    type=["jpg", "jpeg", "png", "webp", "pdf"],
    on_change=on_new_source,
    label_visibility="collapsed",
)
use_camera = sb.toggle("Use camera", value=False)
camera_image = sb.camera_input("Take a photo", on_change=on_new_source) if use_camera else None

quick_source = st.session_state.get("quick_source")
src = load_source(camera_image, quick_source, uploaded_file)
image = src["image"]

if src["doc_id"] and src["doc_id"] != st.session_state.get("doc_id"):
    new_id = src["doc_id"]
    st.session_state["chat_history"] = list(st.session_state["chat_histories"].get(new_id, []))
    st.session_state["doc_id"] = new_id
    st.session_state["zoom"] = 100
    st.session_state["rotate"] = 0

if image is not None:
    sb.caption(src["label"])
    if quick_source and sb.button("Remove image", key="remove_quick"):
        on_new_source()
        st.rerun()

sb.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
sb.markdown('<div class="side-section-title">Settings</div>', unsafe_allow_html=True)
model_choice = sb.selectbox(
    "Model",
    ["Auto (Smart Routing)", *MODEL_MAP.keys()],
)
subject = sb.selectbox(
    "Subject",
    ["Auto-detect", "Physics", "Calculus", "Linear Algebra", "Chemistry",
     "Statistics", "Computer Science", "Other"],
)
subject_hint = SUBJECT_HINTS[subject]
sb.markdown(
    f'<div class="side-note">{html.escape(subject_hint) if subject_hint else "No subject hint is added."}</div>',
    unsafe_allow_html=True,
)

subject_changed = st.session_state.get("subject") != subject
st.session_state["subject"] = subject

smart_routing_on = model_choice == "Auto (Smart Routing)"
if smart_routing_on:
    selected_model_id = auto_select_model(subject)
    routed_name = next((k for k, v in MODEL_MAP.items() if v == selected_model_id), selected_model_id)
    sb.caption(f"Smart Routing will use: {routed_name}")
else:
    selected_model_id = MODEL_MAP[model_choice]

sb.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
with sb.expander("Advanced settings"):
    enhance = st.toggle("Auto-enhance image", value=True)
    add_type_prompt = st.toggle(
        "Ask the model to label the document type", value=True,
        help="Adds a one-line 'type:' label (Poetry, Prose, Math, ...) used to drive the layout.",
    )

temp_image_path = os.path.join(tempfile.gettempdir(), f"ocr_temp_{st.session_state['session_id']}.jpg")
if image is not None:
    to_save = image
    if enhance and not src["is_pdf"]:
        enhanced_image = enhance_image(image)
        to_save = enhanced_image
        with sb.expander("Enhancement preview"):
            p1, p2 = st.columns(2)
            with p1:
                show_image(image, caption="Original")
            with p2:
                show_image(enhanced_image, caption="Enhanced")
    to_save.save(temp_image_path)

if os.path.exists(CORRECTIONS_FILE):
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        correction_lines = f.readlines()
    with sb.expander(f"Saved corrections ({len(correction_lines)})"):
        st.caption("Each correction you save becomes a training example.")
        st.download_button(
            "Export dataset",
            data="".join(correction_lines),
            file_name="corrections.jsonl",
            mime="application/json",
        )

has_result = bool(st.session_state["extracted_text"]) or bool(st.session_state["spot_response"])
current_step = 1 if image is None else (3 if has_result else 2)
render_topbar(current_step)

if image is None:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-badge">
                    <span class="live-dot"><span class="ping"></span><span class="dot"></span></span>
                    <span>Next-Gen Vision AI</span>
                    <span class="badge-separator">•</span>
                    <span class="badge-accent">LaTeX 2.0</span>
                </div>
                <h1>Turn math &amp; notes into flawless <span class="hero-gradient">LaTeX in seconds</span></h1>
                <p>Upload handwritten or printed equations, research formulas, chemistry symbols, or multi-page lecture notes. Textropy synthesizes clean LaTeX, detects document types, and answers questions about your page.</p>
                <p class="fine">Supports high-res JPG, PNG, WEBP and batch PDF parsing up to 30 pages.</p>
                <div class="hero-chips">
                    <span class="hero-chip">
                        <svg viewBox="0 0 24 24"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>
                        Handwritten &amp; Printed Math
                    </span>
                    <span class="hero-chip">
                        <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                        PDF Batching (30 Pages)
                    </span>
                    <span class="hero-chip">
                        <svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                        Interactive Chat &amp; Refinement
                    </span>
                    <span class="hero-chip">
                        <svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>
                        One-Click Word &amp; TeX Export
                    </span>
                </div>
                <div class="hero-stats">
                    <div class="hero-stat-card">
                        <div class="num">30+</div>
                        <div class="lbl">Pages Per PDF</div>
                    </div>
                    <div class="hero-stat-card">
                        <div class="num">5</div>
                        <div class="lbl">Vision Models</div>
                    </div>
                    <div class="hero-stat-card">
                        <div class="num">99%</div>
                        <div class="lbl">LaTeX Fidelity</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div style="background:var(--panel); border:1px solid var(--line); border-radius:var(--radius-lg); padding:1.2rem; box-shadow:var(--shadow-sm); margin-bottom:1rem;">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem;">
                    <div style="font-family:var(--display); font-weight:700; font-size:1.02rem; color:var(--ink);">Quick Document Ingestion</div>
                    <span style="font-size:0.74rem; font-weight:600; color:var(--cobalt); background:var(--cobalt-soft); padding:0.18rem 0.55rem; border-radius:999px;">Drag &amp; Drop Ready</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        quick_upload_widget()
        st.caption("Drop a photo, scan, or PDF above to begin instant extraction.")
        st.markdown(
            """
                <div style="height:1px; background:var(--line); margin:0.9rem 0 0.8rem 0;"></div>
                <div style="font-size:0.8rem; font-weight:600; color:var(--muted); margin-bottom:0.4rem;">Or explore with instant sample data:</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load sample document", key="sample", use_container_width=True):
            reset_extraction_state()
            st.session_state["quick_source"] = {
                "name": "Sample poem (generated)",
                "data": generate_sample_image(),
            }
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="preview-card">
            <div class="preview-card-header">
                <div class="mac-dots">
                    <span class="mac-dot red"></span>
                    <span class="mac-dot yellow"></span>
                    <span class="mac-dot green"></span>
                </div>
                <span class="preview-title">preview_sample.tex &mdash; Textropy Neural Engine</span>
                <span style="font-size:0.72rem; font-weight:700; color:var(--ok); background:var(--ok-soft); padding:0.15rem 0.5rem; border-radius:999px;">Clean Math Output</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        ex_a, ex_b = st.columns([1, 1], gap="large")
        with ex_a:
            st.caption("INPUT NOTE / EQUATION")
            st.markdown(
                '<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:1.1rem; color:#475569; font-style:italic;">'
                '"Integral of sin x from 0 to pi equals 2"'
                '</div>',
                unsafe_allow_html=True,
            )
        with ex_b:
            st.caption("SYNTHESIZED LATEX")
            st.latex(r"\int_0^{\pi} \sin x \, dx = 2")
            st.code(r"\int_0^{\pi} \sin x \, dx = 2", language="latex")

else:
    mode = st.radio("Mode", [MODE_FULL, MODE_SPOT], horizontal=True, label_visibility="collapsed")
    st.markdown(
        f'<div class="chips"><span class="chip">{html.escape(src["label"])}</span>'
        f'<span class="chip">Subject: {html.escape(subject)}</span></div>',
        unsafe_allow_html=True,
    )

    if mode == MODE_FULL:
        col_src, col_out = st.columns(2, gap="large")

        with col_src:
            st.markdown('<div class="panel-title">Source</div>', unsafe_allow_html=True)
            with st.container(border=True):
                zoomed_viewer(temp_image_path, image.height, "src_viewer")

        with col_out:
            st.markdown('<div class="panel-title">Extracted content</div>', unsafe_allow_html=True)
            if subject_changed and st.session_state["extracted_text"]:
                st.warning("The subject changed. Extract again for the best result.")

            ctype_for_button = effective_content_type(subject)
            extract_label = (
                "Extract text and math" if is_math_type(subject, ctype_for_button) else "Extract text"
            )
            if st.session_state["extracted_text"]:
                extract_label = "Re-extract"

            prog = st.empty()
            if stretch_button(extract_label, type="primary", key="extract"):
                st.session_state["extract_error"] = None
                progress_steps(prog, 1)
                try:
                    sys_prompt, prompt = get_ocr_prompt(subject, mode="full_page")
                    if add_type_prompt:
                        prompt += (
                            "\n\nFIRST, output one line in the format `type: Poetry` (one of: Poetry, "
                            "Prose, Math, Code, Table). Then output the extracted content."
                        )
                    hint = CONTENT_TYPE_HINTS.get(ctype_for_button)
                    if hint:
                        prompt += f"\n\nAdditional context: {hint}"
                    progress_steps(prog, 2)
                    response = inference_with_api(
                        temp_image_path, prompt, sys_prompt=sys_prompt, model_id=selected_model_id
                    )
                    progress_steps(prog, 3)
                    detected, body = detect_content_type(response, subject)
                    st.session_state["content_type"] = detected
                    set_extraction(body if body else response)
                    st.session_state["last_model_id"] = selected_model_id
                    st.toast("Text extracted")
                    prog.empty()
                except Exception as e:
                    st.session_state["extract_error"] = str(e)
                    prog.empty()

            if st.session_state["extract_error"]:
                st.error(f"Extraction failed: {st.session_state['extract_error']}")
                if st.button("Retry extraction", key="retry_extract"):
                    st.session_state["extract_error"] = None
                    st.rerun()

            if src["is_pdf"] and src["num_pages"] > 1:
                n = src["num_pages"]
                with st.expander(f"Extract all {n} pages"):
                    if n > MAX_PAGES_LIMIT:
                        st.caption(f"Batch extraction is limited to {MAX_PAGES_LIMIT} pages. This PDF has {n}.")
                    else:
                        st.caption(f"Reads every page in one go. This makes {n} model calls.")
                        if st.button("Extract all pages", key="batch"):
                            b_sys, b_prompt = get_ocr_prompt(subject, mode="full_page")
                            doc = run_batch_extraction(src["pdf"], n, b_prompt, b_sys, selected_model_id)
                            set_extraction(doc)
                            st.session_state["content_type"] = SUBJECT_TO_CONTENT_TYPE.get(subject) or "Prose"
                            st.session_state["last_model_id"] = selected_model_id
                            st.rerun()

            if not st.session_state["extracted_text"]:
                if not st.session_state["extract_error"]:
                    st.markdown(
                        f'<div class="empty">Nothing extracted yet.<br>'
                        f'Press <b>{extract_label}</b> to read this page.</div>',
                        unsafe_allow_html=True,
                    )
            else:
                extracted = st.session_state["extracted_text"]
                if "correction_editor" not in st.session_state:
                    st.session_state["correction_editor"] = extracted

                ctype = effective_content_type(subject)
                math_mode = is_math_type(subject, ctype)
                suspicious = count_suspicious_words(extracted)

                chip_c1, chip_c2 = st.columns([1, 1.6])
                with chip_c1:
                    st.markdown('<div class="chip-select">', unsafe_allow_html=True)
                    sel_idx = CONTENT_TYPES.index(ctype) if ctype in CONTENT_TYPES else 0
                    chosen_type = st.selectbox(
                        "Detected type (tap to correct)",
                        CONTENT_TYPES,
                        index=sel_idx,
                        key="content_type_select",
                        label_visibility="collapsed",
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                with chip_c2:
                    trust = []
                    if suspicious:
                        trust.append(f"{len(suspicious)} low-confidence word{'s' if len(suspicious) != 1 else ''}")
                    if smart_routing_on and st.session_state.get("last_model_id"):
                        used = st.session_state["last_model_id"]
                        used_name = next((k for k, v in MODEL_MAP.items() if v == used), used)
                        trust.append(f"Smart Routing used: {used_name}")
                    if trust:
                        st.markdown(
                            f'<div class="chips" style="margin-top:0.35rem;">'
                            + "".join(f'<span class="chip">{html.escape(t)}</span>' for t in trust)
                            + "</div>",
                            unsafe_allow_html=True,
                        )

                if chosen_type != ctype:
                    if st.session_state.get("content_type") == chosen_type:
                        pass
                    else:
                        st.session_state["content_type_override"] = None if chosen_type == "Auto-detect" else chosen_type
                        if chosen_type != "Auto-detect":
                            st.warning(f'Type changed to "{chosen_type}". Press Re-extract to apply it.')
                else:
                    st.session_state["content_type_override"] = None

                tab_labels = ["Rendered", "LaTeX source", "Chat"] if math_mode else ["Rendered", "Edit text", "Chat"]
                tabs = st.tabs(tab_labels)

                with tabs[0]:
                    with st.container(height=470, border=False):
                        render_structured(extracted, ctype if ctype != "Auto-detect" else "Prose")
                    if math_mode:
                        st.caption("Complex equations may render better in the LaTeX source tab.")

                with tabs[1]:
                    corrected = st.text_area(
                        "Edit the text to fix any mistakes",
                        height=370,
                        key="correction_editor",
                    )
                    st.caption("Ctrl+Enter inside the box applies your corrections.")
                    if st.button("Apply corrections", key="save_correction"):
                        if corrected.strip() == extracted.strip():
                            st.info("No changes to save.")
                        else:
                            try:
                                with open(temp_image_path, "rb") as fh:
                                    img_hash = hashlib.md5(fh.read()).hexdigest()
                            except FileNotFoundError:
                                img_hash = "unknown_hash"

                            record = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "image_hash": img_hash,
                                "subject": st.session_state.get("subject", "Auto-detect"),
                                "content_type": ctype,
                                "original": extracted,
                                "corrected": corrected,
                            }
                            with open(CORRECTIONS_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record) + "\n")

                            st.session_state["extracted_text"] = corrected
                            st.success("Corrections applied. The rendered view and downloads now use them.")

                            diff = list(difflib.unified_diff(
                                extracted.splitlines(), corrected.splitlines(), lineterm="", n=2
                            ))
                            if diff:
                                with st.expander("View changes"):
                                    st.code("\n".join(diff), language="diff")

                    if suspicious:
                        st.markdown(
                            '<div class="side-note" style="margin-top:0.6rem"><b>Low-confidence words</b> '
                            '(highlighted in amber in the Rendered tab). Fix them above, or re-read a single line '
                            'without redoing the whole page.</div>',
                            unsafe_allow_html=True,
                        )
                        hl = st.session_state.get("highlight_word", "")
                        wcols = st.columns(min(6, max(1, len(suspicious))))
                        for i, w in enumerate(suspicious[:12]):
                            with wcols[i % len(wcols)]:
                                st.markdown('<div class="word-btn">', unsafe_allow_html=True)
                                if st.button(w, key=f"sus_{i}"):
                                    st.session_state["highlight_word"] = "" if hl == w else w
                                    st.rerun()
                                st.markdown('</div>', unsafe_allow_html=True)
                        rr_col1, rr_col2 = st.columns([3, 1])
                        with rr_col1:
                            reread_line = st.text_input(
                                "Paste a line or phrase to re-read", key="reread_input",
                                placeholder='e.g. the line containing "dar\'test"',
                            )
                        with rr_col2:
                            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
                            do_reread = st.button("Re-read this region", key="reread_btn")
                        if do_reread and reread_line.strip():
                            with st.spinner("Re-reading that region..."):
                                try:
                                    rr_sys, rr_prompt = get_ocr_prompt(subject, mode="full_page")
                                    rr_prompt += (
                                        f"\n\nFocus on the part of the image that says something like "
                                        f"\"{reread_line.strip()}\". Transcribe ONLY that word/line precisely. "
                                        "Output only the corrected text for that region, nothing else."
                                    )
                                    fixed = inference_with_api(
                                        temp_image_path, rr_prompt, sys_prompt=rr_sys, model_id=selected_model_id
                                    ).strip()
                                    new_text = extracted.replace(reread_line.strip(), fixed)
                                    if new_text == extracted and suspicious:
                                        new_text = extracted.replace(suspicious[0], fixed, 1)
                                    if new_text != extracted:
                                        st.session_state["extracted_text"] = new_text
                                        st.session_state["correction_editor"] = new_text
                                        st.success("Region re-read and patched in place.")
                                        st.rerun()
                                    else:
                                        st.info(f'The model read: "{fixed}". Apply it manually in the editor above.')
                                except Exception as e:
                                    st.error(f"Re-read failed: {e}")

                with tabs[2]:
                    suggestions = CHAT_SUGGESTIONS.get(ctype, CHAT_SUGGESTIONS["_default"])
                    scols = st.columns(3)
                    for col, (label, q) in zip(scols, suggestions):
                        with col:
                            if stretch_button(label, key=f"sg_{label.replace(' ', '_')}"):
                                ask_document(q)
                                st.rerun()
                    with st.expander("More quick actions"):
                        for key, label in QUICK_LABELS:
                            if stretch_button(label, key=f"qa_{key}"):
                                ask_document(QUICK_ACTIONS[key])
                                st.rerun()

                    with st.container(height=300, border=False):
                        if not st.session_state["chat_history"]:
                            st.caption("Ask anything about the extracted text, or start with a suggested action above.")
                        for i, msg in enumerate(st.session_state["chat_history"]):
                            with st.chat_message(msg["role"]):
                                if msg["role"] == "user":
                                    st.markdown(msg["content"])
                                else:
                                    render_structured(msg["content"], ctype if ctype != "Auto-detect" else "Prose")
                                    if i == len(st.session_state["chat_history"]) - 1:
                                        apply_edit_to_document(msg["content"])

                    user_input = st.chat_input("Ask about this document")
                    if user_input:
                        ask_document(user_input)
                        st.rerun()

                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                export_bar(st.session_state["extracted_text"], "full")
                st.caption("Shortcuts: press Enter in the chat box to send, Ctrl+C to copy selected text.")

    else:
        if stretch_button("Find text regions", type="primary", key="spot"):
            with st.spinner("Finding text regions..."):
                sys_prompt, prompt = get_ocr_prompt(subject, mode="text_spotting")
                if subject_hint:
                    prompt += f"\n\nAdditional context: {subject_hint}"
                try:
                    response = inference_with_api(
                        temp_image_path, prompt, sys_prompt=sys_prompt, model_id=selected_model_id
                    )
                    st.session_state["spot_response"] = response
                    st.session_state["spot_image"] = plot_text_bounding_boxes(temp_image_path, response)
                except Exception as e:
                    st.error(f"Text spotting failed: {e}")
                    st.session_state["extract_error"] = None
                    if st.button("Retry", key="retry_spot"):
                        st.rerun()

        col_img, col_txt = st.columns(2, gap="large")

        with col_img:
            st.markdown('<div class="panel-title">Detected regions</div>', unsafe_allow_html=True)
            if st.session_state["spot_image"] is not None:
                spot_view = os.path.join(tempfile.gettempdir(), f"ocr_spot_{st.session_state['session_id']}.jpg")
                st.session_state["spot_image"].save(spot_view)
                with st.container(border=True):
                    zoomed_viewer(spot_view, st.session_state["spot_image"].height, "spot_viewer")
            else:
                with st.container(border=True):
                    zoomed_viewer(temp_image_path, image.height, "src_viewer_spot")

        with col_txt:
            st.markdown('<div class="panel-title">Text in each box</div>', unsafe_allow_html=True)
            if subject_changed and st.session_state["spot_response"]:
                st.warning("The subject changed. Run text spotting again for the best result.")

            if not st.session_state["spot_response"]:
                st.markdown(
                    '<div class="empty">No regions found yet.<br>'
                    'Press <b>Find text regions</b> to outline each piece of text.</div>',
                    unsafe_allow_html=True,
                )
            else:
                response = st.session_state["spot_response"]
                try:
                    boxes, clean_json = parse_boxes(response)
                    full_text = " \n\n".join(
                        f"**Box {i}**\n\n{clean_box_text(b)}" for i, b in enumerate(boxes, start=1)
                    )

                    spot_tabs = st.tabs(["Rendered", "LaTeX source"])
                    with spot_tabs[0]:
                        with st.container(height=470, border=False):
                            render_latex_safely(preprocess_latex(full_text))
                    with spot_tabs[1]:
                        st.code(full_text, language="markdown")

                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                    export_bar(full_text, "spot")

                    with st.expander("Raw box data (JSON)"):
                        st.code(clean_json, language="json")
                except Exception:
                    st.warning("The boxes could not be read. Showing the raw response instead.")
                    st.code(response, language="json")

if image is None and os.path.exists(temp_image_path):
    try:
        os.remove(temp_image_path)
    except OSError:
        pass

st.markdown(
    '<div class="app-footer">'
    f'Textropy AI v{__version__} &mdash; Built with <a href="https://github.com" target="_blank">open source</a> models.'
    '</div>',
    unsafe_allow_html=True,
)