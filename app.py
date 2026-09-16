import streamlit as st
import streamlit.components.v1 as components
import json
import ast
import uuid
from PIL import Image, ImageDraw, ImageFont
import base64
import os
import io
import fitz
import re
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
from image_enhancer import enhance_image

load_dotenv(override=True)

st.set_page_config(page_title="Math OCR & Text Spotting", layout="wide")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""
if "correction_editor" not in st.session_state:
    st.session_state["correction_editor"] = ""
if "spot_response" not in st.session_state:
    st.session_state["spot_response"] = ""
if "spot_image" not in st.session_state:
    st.session_state["spot_image"] = None
if "subject" not in st.session_state:
    st.session_state["subject"] = "Auto-detect"

def reset_extraction_state():
    st.session_state["extracted_text"] = ""
    st.session_state["correction_editor"] = ""
    st.session_state["spot_response"] = ""
    st.session_state["spot_image"] = None

st.sidebar.title("Textropy AI")
st.sidebar.markdown("Powered by **OpenRouter Vision Models**")

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Nemotron Nano 12B VL (Free Vision Model)",
        "Gemma 4 31B (Free Vision Model)",
        "InclusionAI Ling 3.0 Flash VL (Free Vision Model)",
        "Qwen 2.5 VL 72B (Paid, Ultimate Math OCR)"
    ]
)

model_map = {
    "Nemotron Nano 12B VL (Free Vision Model)": "nvidia/nemotron-nano-12b-v2-vl:free",
    "Gemma 4 31B (Free Vision Model)": "google/gemma-4-31b-it:free",
    "InclusionAI Ling 3.0 Flash VL (Free Vision Model)": "inclusionai/ling-3.0-flash-vl:free",
    "Qwen 2.5 VL 72B (Paid, Ultimate Math OCR)": "qwen/qwen-2.5-vl-72b-instruct"
}
selected_model_id = model_map[model_choice]

subject = st.sidebar.selectbox(
    "Subject / Context",
    ["Auto-detect", "Physics", "Calculus", "Linear Algebra", "Chemistry", "Statistics", "Computer Science", "Other"]
)

subject_hints = {
    "Auto-detect": "First identify the subject area in one word, then extract all text and mathematical expressions precisely.",
    "Physics": "Pay special attention to vector notation, subscripts, force diagram labels, and SI units. Preserve all Greek symbols.",
    "Calculus": "Preserve integral bounds, limit notation, derivative operators (d/dx), and all superscripts/subscripts exactly.",
    "Linear Algebra": "Preserve matrix notation, transpose symbols, determinants, eigenvalue notation, and vector arrows.",
    "Chemistry": "Preserve molecular formulas, oxidation states, reaction arrows, and stoichiometric coefficients exactly.",
    "Statistics": "Preserve probability notation P(A|B), summation symbols, Greek letters, and distribution notation.",
    "Computer Science": "Preserve Big-O notation, pseudocode formatting, logical operators, and code-like expressions.",
    "Other": ""
}
subject_hint = subject_hints[subject]
st.sidebar.caption(f"Active hint: {subject_hint}" if subject_hint else "Active hint: None")

subject_changed = False
if st.session_state.get("subject") != subject:
    subject_changed = True
st.session_state["subject"] = subject

uploaded_file = st.sidebar.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"], on_change=reset_extraction_state)

st.sidebar.markdown("---")
st.sidebar.markdown("**Paste via keyboard shortcut:**")
st.sidebar.info(
    "Tip: Take a screenshot, save it, and upload above. "
    "Or use Windows Snipping Tool / Mac Screenshot and save as PNG."
)

if os.path.exists("corrections.jsonl"):
    with open("corrections.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    count = len(lines)
    st.sidebar.metric("Total corrections saved", count)
    st.sidebar.download_button(
        "Export Corrections Dataset",
        data="".join(lines),
        file_name="corrections.jsonl",
        mime="application/json"
    )

use_camera = st.sidebar.toggle("Use camera instead", value=False)
enhance = st.sidebar.toggle("Auto-enhance image", value=True)
camera_image = None
if use_camera:
    camera_image = st.sidebar.camera_input("Take a photo of your document", on_change=reset_extraction_state)

image = None

try:
    if camera_image is not None and uploaded_file is not None:
        st.sidebar.info("Using camera image.")

    if camera_image is not None:
        image = Image.open(camera_image)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        st.sidebar.success("Camera image captured.")

    elif uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext == 'pdf':
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            num_pages = len(pdf_document)

            if num_pages > 1:
                page_num = st.sidebar.number_input("Select PDF Page", min_value=1, max_value=num_pages, value=1, on_change=reset_extraction_state) - 1
            else:
                page_num = 0

            page = pdf_document.load_page(page_num)

            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            st.sidebar.image(image, caption=f"PDF Page {page_num+1} of {num_pages}", use_container_width=True)
        else:
            image = Image.open(uploaded_file)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)

    temp_image_path = os.path.join(tempfile.gettempdir(), f"ocr_temp_{st.session_state['session_id']}.jpg")
    if image is not None:
        image.save(temp_image_path)
except Exception as e:
    st.sidebar.error(f"Failed to load image/PDF: {str(e)}")
    image = None

if image is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower() if uploaded_file else ''
    if enhance and file_ext != 'pdf':
        original_image = image.copy()
        enhanced_image = enhance_image(image)
        enhanced_image.save(temp_image_path)

        with st.sidebar.expander("Enhancement Preview"):
            prev1, prev2 = st.columns(2)
            with prev1:
                st.image(original_image, caption="Original", use_container_width=True)
            with prev2:
                st.image(enhanced_image, caption="Enhanced", use_container_width=True)

        if st.sidebar.button("Use Original Instead"):
            original_image.save(temp_image_path)
            st.sidebar.caption("Using original image.")
