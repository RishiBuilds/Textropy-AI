import ast
import asyncio
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

load_dotenv(override=True)

st.set_page_config(
    page_title="Textropy AI",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODE_FULL = "Full page (text and math)"
MODE_SPOT = "Text spotting (bounding boxes)"
MAX_PAGES_LIMIT = 30
PANEL_H = 640
CORRECTIONS_FILE = "corrections.jsonl"

QUICK_LABELS = [
    ("summarize", "Summarize"),
    ("explain_equations", "Explain equations"),
    ("simplify", "Simplify"),
    ("translate_english", "Translate to English"),
    ("translate_hindi", "Translate to Hindi"),
    ("key_formulas", "Key formulas"),
]

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
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --paper: #EEF1F5;
    --panel: #FFFFFF;
    --ink: #16213A;
    --muted: #5C6678;
    --line: #D8DEE8;
    --cobalt: #2F4BFF;
    --cobalt-dark: #2039D6;
    --cobalt-light: #5B74FF;
    --cobalt-soft: rgba(47, 75, 255, 0.08);
    --hl: #FFE66D;
    --ok: #1E8E5A;
    --shadow-sm: 0 1px 3px rgba(22, 33, 58, 0.07), 0 4px 14px rgba(22, 33, 58, 0.05);
    --shadow-md: 0 4px 12px rgba(22, 33, 58, 0.09), 0 14px 34px rgba(22, 33, 58, 0.08);
    --radius: 12px;
    --display: 'Bricolage Grotesque', 'Segoe UI', sans-serif;
    --body: 'Instrument Sans', 'Segoe UI', system-ui, sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace;
}

::selection { background: var(--hl); color: var(--ink); }

.stApp {
    background-color: var(--paper);
    background-image:
        linear-gradient(rgba(22,33,58,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(22,33,58,0.05) 1px, transparent 1px);
    background-size: 32px 32px;
    color: var(--ink);
}
.stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stCaptionContainer"], .stApp .stMarkdown {
    font-family: var(--body);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
    font-family: var(--display);
    color: var(--ink);
    letter-spacing: -0.01em;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; max-width: 1440px; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FBFCFE 0%, #F2F5FA 100%);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
.side-logo { margin: 0 0 1rem 0.1rem; }
.side-wordmark {
    font-family: var(--display); font-weight: 800; font-size: 1.15rem;
    letter-spacing: -0.02em; color: var(--ink); padding: 0 0.18em;
    background: linear-gradient(transparent 60%, var(--hl) 60%, var(--hl) 92%, transparent 92%);
}
.side-tagline {
    display: block; margin-top: 0.3rem; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted);
}
.side-label {
    font-family: var(--display);
    font-weight: 700;
    font-size: 0.98rem;
    margin: 0.4rem 0 0.35rem 0;
    color: var(--ink);
}
.side-note { color: var(--muted); font-size: 0.82rem; line-height: 1.45; margin: 0.25rem 0 0.5rem 0; }

.topbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; flex-wrap: wrap; margin-bottom: 1.4rem;
}
.wordmark {
    font-family: var(--display);
    font-weight: 800;
    font-size: 1.45rem;
    letter-spacing: -0.02em;
    color: var(--ink);
    padding: 0 0.2em;
    background: linear-gradient(transparent 60%, var(--hl) 60%, var(--hl) 92%, transparent 92%);
}
.steps { list-style: none; display: flex; align-items: center; margin: 0; padding: 0; }
.steps li { display: flex; align-items: center; gap: 0.5rem; color: var(--muted); font-size: 0.92rem; }
.steps li + li::before {
    content: ""; width: 1.6rem; height: 1.5px; background: var(--line); margin: 0 0.6rem;
}
.steps .n {
    width: 1.55rem; height: 1.55rem; border-radius: 50%;
    border: 1.5px solid var(--line); background: var(--panel);
    display: inline-grid; place-items: center; font-size: 0.78rem; font-weight: 600;
}
.steps li.current { color: var(--ink); font-weight: 600; }
.steps li.current .n {
    background: var(--cobalt); border-color: var(--cobalt); color: #fff;
    box-shadow: 0 0 0 4px rgba(47, 75, 255, 0.16);
}
.steps li.done .n { background: var(--ok); border-color: var(--ok); color: #fff; }

.hero-badge {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: var(--cobalt-soft); border: 1px solid rgba(47, 75, 255, 0.22);
    color: var(--cobalt-dark); border-radius: 999px;
    padding: 0.28rem 0.85rem; font-size: 0.8rem; font-weight: 600;
    letter-spacing: 0.02em; margin-bottom: 0.6rem;
}
.hero-badge .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--ok);
    box-shadow: 0 0 0 3px rgba(30, 142, 90, 0.18);
}
.hero h1 {
    font-size: clamp(2.1rem, 4.4vw, 3.3rem);
    line-height: 1.06;
    font-weight: 800;
    margin: 0.4rem 0 1rem 0;
    max-width: 15ch;
}
.hero h1 mark {
    background: linear-gradient(transparent 60%, var(--hl) 60%, var(--hl) 92%, transparent 92%);
    color: inherit; padding: 0 0.06em;
}
.hero p { font-size: 1.06rem; line-height: 1.6; color: var(--muted); max-width: 52ch; margin: 0 0 0.8rem 0; }
.hero .fine { font-size: 0.9rem; }
.hero .hero-icon { display: inline-flex; vertical-align: -3px; margin: 0 0.35rem 0 0; color: var(--cobalt); }
.hero .hero-icon svg { width: 15px; height: 15px; display: block; }

.panel-title { font-family: var(--display); font-weight: 700; font-size: 1.08rem; margin: 0 0 0.45rem 0; }
.chips { margin: 0.2rem 0 1rem 0; }
.chip {
    display: inline-block; border: 1px solid var(--line); background: var(--panel);
    border-radius: 999px; padding: 0.15rem 0.75rem; font-size: 0.82rem; color: var(--muted);
    margin: 0 0.4rem 0.3rem 0;
}
.empty {
    border: 1.5px dashed #AEB8CB; border-radius: var(--radius); padding: 2.2rem 1.4rem;
    background: rgba(255,255,255,0.75); color: var(--muted); text-align: center; line-height: 1.55;
    animation: fadeUp .4s ease both;
}
.empty b { color: var(--ink); }
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--radius); }
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {
    border-color: var(--line);
    box-shadow: var(--shadow-sm);
}

.stButton button, .stDownloadButton button {
    border-radius: 8px; font-weight: 600; box-shadow: none;
    border: 1px solid var(--line); background: var(--panel); color: var(--ink);
    transition: border-color .15s ease, color .15s ease, background .15s ease,
                transform .15s ease, box-shadow .15s ease;
}
.stButton button p, .stDownloadButton button p { color: inherit; }
.stButton button:hover, .stDownloadButton button:hover {
    border-color: var(--cobalt); color: var(--cobalt); background: var(--panel);
    transform: translateY(-1px); box-shadow: var(--shadow-sm);
}
.stButton button:active, .stDownloadButton button:active { transform: translateY(0); box-shadow: none; }
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--cobalt) 0%, var(--cobalt-light) 100%);
    border-color: var(--cobalt); color: #fff;
    box-shadow: 0 4px 14px rgba(47, 75, 255, 0.32);
}
.stButton button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, var(--cobalt-dark) 0%, var(--cobalt) 100%);
    border-color: var(--cobalt-dark); color: #fff;
    box-shadow: 0 6px 20px rgba(47, 75, 255, 0.42);
}
button:focus-visible, textarea:focus-visible, input:focus-visible {
    outline: 2px solid var(--cobalt) !important; outline-offset: 2px;
}

.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background: transparent; padding: 0.5rem 0; color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--cobalt); height: 3px; }

[data-testid="stFileUploaderDropzone"] {
    background: var(--panel); border: 1.5px dashed #9AA6BD; border-radius: 10px; padding: 1.5rem;
    transition: border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cobalt); background: #F7F9FF; }
.stTextArea textarea {
    font-family: var(--mono); font-size: 0.86rem; line-height: 1.55;
    background: var(--panel); color: var(--ink); border-radius: 8px;
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

/* Segmented-pill style for the Mode radio */
div[role="radiogroup"] { gap: 0.5rem; }
div[role="radiogroup"] label {
    border: 1px solid var(--line); background: var(--panel); border-radius: 999px;
    padding: 0.35rem 1.05rem; cursor: pointer; box-shadow: var(--shadow-sm);
    transition: border-color .15s ease, background .15s ease, color .15s ease, transform .15s ease;
}
div[role="radiogroup"] label:hover { border-color: var(--cobalt); transform: translateY(-1px); }
div[role="radiogroup"] label:has(input:checked) {
    background: var(--cobalt-soft); border-color: var(--cobalt);
}
div[role="radiogroup"] label:has(input:checked) p { color: var(--cobalt-dark); font-weight: 600; }
div[role="radiogroup"] label > div:first-child { display: none; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #B7C0D1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cobalt-light); }

@media (max-width: 640px) {
    .steps li:not(.current) .t { display: none; }
    .steps li + li::before { width: 0.8rem; margin: 0 0.3rem; }
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


def on_new_source():
    reset_extraction_state()
    st.session_state.pop("quick_source", None)


def set_extraction(text: str):
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
        f'<div class="topbar"><span class="wordmark">Textropy AI</span>'
        f'<ol class="steps">{"".join(items)}</ol></div>',
        unsafe_allow_html=True,
    )


def copy_button(text: str):
    payload = html.escape(json.dumps(text), quote=True)
    components.html(
        f"""
        <style>
          html,body{{background:transparent!important;margin:0;padding:0;}}
          button{{width:100%;height:40px;border-radius:8px;border:1px solid #D8DEE8;background:#fff;
                 color:#16213A;font:600 14px 'Instrument Sans',system-ui,sans-serif;cursor:pointer;}}
          button:hover{{border-color:#2F4BFF;color:#2F4BFF;}}
        </style>
        <button onclick="navigator.clipboard.writeText({payload}).then(()=>{{this.textContent='Copied';
          setTimeout(()=>this.textContent='Copy to clipboard',1500)}}).catch(()=>{{}})">Copy to clipboard</button>
        """,
        height=46,
    )


def export_bar(text: str, key: str):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button("Download .md", data=text, file_name="extracted.md",
                           mime="text/markdown", key=f"md_{key}")
    with col_b:
        tex = ("\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n"
               f"\\begin{{document}}\n\n{text}\n\n\\end{{document}}")
        st.download_button("Download .tex", data=tex, file_name="extracted.tex",
                           mime="text/plain", key=f"tex_{key}")
    with col_c:
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
    src = {"image": None, "label": "", "is_pdf": False, "pdf": None, "num_pages": 0}
    try:
        if camera_image is not None:
            img, label = Image.open(camera_image), "Camera photo"
        elif quick_source:
            img, label = Image.open(io.BytesIO(quick_source["data"])), quick_source["name"]
        elif uploaded_file is not None:
            ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            if ext == "pdf":
                pdf = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
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
            else:
                img, label = Image.open(uploaded_file), uploaded_file.name
        else:
            return src

        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        src.update(image=img, label=label)
    except Exception as e:
        st.sidebar.error(f"Could not open this file: {e}")
        src.update(image=None, is_pdf=False, pdf=None, num_pages=0)
    return src


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
    '<div class="side-logo"><span class="side-wordmark">Textropy AI</span>'
    '<span class="side-tagline">Intelligent OCR engine</span></div>',
    unsafe_allow_html=True,
)

sb.markdown('<div class="side-label">Image</div>', unsafe_allow_html=True)
uploaded_file = sb.file_uploader(
    "Upload an image or PDF",
    type=["jpg", "jpeg", "png", "pdf"],
    on_change=on_new_source,
    label_visibility="collapsed",
)
use_camera = sb.toggle("Use camera", value=False)
camera_image = sb.camera_input("Take a photo", on_change=on_new_source) if use_camera else None

quick_source = st.session_state.get("quick_source")
src = load_source(camera_image, quick_source, uploaded_file)
image = src["image"]

if image is not None:
    sb.caption(src["label"])
    if quick_source and sb.button("Remove image", key="remove_quick"):
        on_new_source()
        st.rerun()

sb.markdown('<div class="side-label">Reading settings</div>', unsafe_allow_html=True)
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

if model_choice == "Auto (Smart Routing)":
    selected_model_id = auto_select_model(subject)
    sb.caption("Auto-routed to Qwen 2.5 VL, best for text and math.")
else:
    selected_model_id = MODEL_MAP[model_choice]

enhance = sb.toggle("Auto-enhance image", value=True)

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
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero">
                <span class="hero-badge"><span class="dot"></span>Powered by vision-language AI</span>
                <h1>Get clean <mark>LaTeX</mark> from a photo of your notes.</h1>
                <p>Upload a page of handwritten or printed maths, physics or chemistry.
                Textropy reads the text and equations, lets you fix any mistakes,
                and answers questions about the page.</p>
                <p class="fine">Accepts JPG, PNG, WEBP and PDF. PDFs up to 30 pages can be read in one batch.</p>
                <div class="chips">
                    <span class="chip"><span class="hero-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg></span>Handwritten &amp; printed math</span>
                    <span class="chip"><span class="hero-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></span>PDF batch up to 30 pages</span>
                    <span class="chip"><span class="hero-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></span>Chat with your document</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="empty" style="margin-top:1.2rem;">'
            '<div style="color:#2F4BFF;margin-bottom:0.5rem;">'
            '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
            '<polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg></div>'
            '<b>Upload from the sidebar</b><br>'
            'Use the file uploader or camera in the sidebar to add an image or PDF.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel-title">What you get back</div>', unsafe_allow_html=True)
    with st.container(border=True):
        ex_a, ex_b = st.columns([1, 1], gap="large")
        with ex_a:
            st.caption("On the page")
            st.markdown("*Integral of sin x from 0 to pi equals 2*")
        with ex_b:
            st.caption("In Textropy")
            st.latex(r"\int_0^{\pi} \sin x \, dx = 2")
            st.code(r"\int_0^{\pi} \sin x \, dx = 2", language="latex")

else:
    mode = st.radio("Mode", [MODE_FULL, MODE_SPOT], horizontal=True, label_visibility="collapsed")
    st.markdown(
        f'<div class="chips"><span class="chip">{html.escape(src["label"])}</span>'
        f'<span class="chip">Model: {html.escape(selected_model_id)}</span>'
        f'<span class="chip">Subject: {html.escape(subject)}</span></div>',
        unsafe_allow_html=True,
    )

    if mode == MODE_FULL:
        col_src, col_out = st.columns(2, gap="large")

        with col_src:
            st.markdown('<div class="panel-title">Source</div>', unsafe_allow_html=True)
            with st.container(height=PANEL_H, border=True):
                show_image(image)

        with col_out:
            st.markdown('<div class="panel-title">Extracted content</div>', unsafe_allow_html=True)
            if subject_changed and st.session_state["extracted_text"]:
                st.warning("The subject changed. Extract again for the best result.")

            sys_prompt, prompt = get_ocr_prompt(subject, mode="full_page")

            if stretch_button("Extract text and math", type="primary", key="extract"):
                with st.spinner("Reading the page..."):
                    try:
                        response = inference_with_api(
                            temp_image_path, prompt, sys_prompt=sys_prompt, model_id=selected_model_id
                        )
                        set_extraction(response)
                        st.toast("Text extracted")
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

            if src["is_pdf"] and src["num_pages"] > 1:
                n = src["num_pages"]
                with st.expander(f"Extract all {n} pages"):
                    if n > MAX_PAGES_LIMIT:
                        st.caption(f"Batch extraction is limited to {MAX_PAGES_LIMIT} pages. This PDF has {n}.")
                    else:
                        st.caption(f"Reads every page in one go. This makes {n} model calls.")
                        if st.button("Extract all pages", key="batch"):
                            doc = run_batch_extraction(src["pdf"], n, prompt, sys_prompt, selected_model_id)
                            set_extraction(doc)
                            st.rerun()

            if not st.session_state["extracted_text"]:
                st.markdown(
                    '<div class="empty">Nothing extracted yet.<br>'
                    'Press <b>Extract text and math</b> to read this page.</div>',
                    unsafe_allow_html=True,
                )
            else:
                extracted = st.session_state["extracted_text"]
                if "correction_editor" not in st.session_state:
                    st.session_state["correction_editor"] = extracted

                tab_rendered, tab_raw, tab_chat = st.tabs(["Rendered", "LaTeX source", "Chat"])

                with tab_rendered:
                    with st.container(height=470, border=False):
                        render_latex_safely(preprocess_latex(extracted))
                    st.caption("Complex equations may render better in the LaTeX source tab.")

                with tab_raw:
                    corrected = st.text_area(
                        "Edit the text to fix any mistakes",
                        height=370,
                        key="correction_editor",
                    )
                    if st.button("Save correction", key="save_correction"):
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
                                "original": extracted,
                                "corrected": corrected,
                            }
                            with open(CORRECTIONS_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record) + "\n")

                            st.session_state["extracted_text"] = corrected
                            st.success("Correction saved. The rendered view and downloads now use it.")

                            diff = list(difflib.unified_diff(
                                extracted.splitlines(), corrected.splitlines(), lineterm="", n=2
                            ))
                            if diff:
                                with st.expander("View changes"):
                                    st.code("\n".join(diff), language="diff")

                with tab_chat:
                    for row_start in (0, 3):
                        cols = st.columns(3)
                        for col, (key, label) in zip(cols, QUICK_LABELS[row_start:row_start + 3]):
                            with col:
                                if stretch_button(label, key=f"qa_{key}"):
                                    ask_document(QUICK_ACTIONS[key])
                                    st.rerun()

                    with st.container(height=300, border=False):
                        if not st.session_state["chat_history"]:
                            st.caption("Ask anything about the extracted text, or start with one of the actions above.")
                        for msg in st.session_state["chat_history"]:
                            with st.chat_message(msg["role"]):
                                if msg["role"] == "user":
                                    st.markdown(msg["content"])
                                else:
                                    render_latex_safely(preprocess_latex(msg["content"]))

                    user_input = st.chat_input("Ask about this document")
                    if user_input:
                        ask_document(user_input)
                        st.rerun()

                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                export_bar(st.session_state["extracted_text"], "full")

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

        col_img, col_txt = st.columns(2, gap="large")

        with col_img:
            st.markdown('<div class="panel-title">Detected regions</div>', unsafe_allow_html=True)
            with st.container(height=PANEL_H, border=True):
                show_image(st.session_state["spot_image"] if st.session_state["spot_image"] is not None else image)

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

                    tab_r, tab_l = st.tabs(["Rendered", "LaTeX source"])
                    with tab_r:
                        with st.container(height=470, border=False):
                            render_latex_safely(preprocess_latex(full_text))
                    with tab_l:
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