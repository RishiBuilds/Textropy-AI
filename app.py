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
import hashlib
import datetime
import difflib
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

def preprocess_latex(text: str) -> str:
    text = text.strip()

    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    text = text.strip()

    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    text = text.replace(r"\(", "$").replace(r"\)", "$")

    text = text.replace(r"\begin{array}", "$$\n\\begin{array}")
    text = text.replace(r"\end{array}", "\\end{array}\n$$")

    text = re.sub(r'\$\$\s*\$\$', '$$', text)

    if ('\\frac' in text or '\\min' in text or '\\max' in text or '^' in text or '\\left' in text) and '$$' not in text and '$' not in text:
        text = f"$$\n{text}\n$$"

    return text

def render_latex_safely(text: str):
    blocks = text.split('\n\n')
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        try:
            st.markdown(block)
        except:
            st.code(block, language="latex")

def parse_json(json_output):
    match_array = re.search(r'\[.*\]', json_output, re.DOTALL)
    match_obj = re.search(r'\{.*\}', json_output, re.DOTALL)

    if match_array and match_obj:
        if json_output.find('[') < json_output.find('{'):
            return match_array.group(0)
        return match_obj.group(0)
    elif match_array:
        return match_array.group(0)
    elif match_obj:
        return match_obj.group(0)
    return json_output.strip()

def normalize_boxes(boxes):
    if isinstance(boxes, dict):
        if "bbox_2d" in boxes and "text_content" in boxes:
            bboxes = boxes["bbox_2d"]
            texts = boxes["text_content"]
            if isinstance(bboxes, list) and isinstance(texts, list):
                return [{"bbox_2d": b, "text_content": t} for b, t in zip(bboxes, texts)]
        return [boxes]
    return boxes

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def inference_with_api(image_path, prompt, sys_prompt="You are a precise document extraction assistant.", model_id="qwen/qwen-2.5-vl-72b-instruct"):
    if not os.getenv('OPENROUTER_API_KEY'):
        raise ValueError("Please provide an OpenRouter API Key in your .env file or sidebar.")

    try:
        base64_image = encode_image(image_path)
        client = OpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=2000,
            temperature=0.1
        )
        content = completion.choices[0].message.content
        if content is None:
            raise Exception("The API returned an empty response (NoneType). This usually happens if the model is overloaded, or the image was blocked by safety filters.")
        return content
    except Exception as e:
        raise Exception(f"API Inference failed: {str(e)}")

def plot_text_bounding_boxes(image_path, bounding_boxes):
    img = Image.open(image_path)
    width, height = img.size

    draw = ImageDraw.Draw(img)

    bounding_boxes_json = parse_json(bounding_boxes)

    dynamic_size = max(16, int(height * 0.025))
    try:
        font = ImageFont.truetype("arial.ttf", size=dynamic_size)
    except:
        font = ImageFont.load_default()

    try:
        fixed_json = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', bounding_boxes_json)
        boxes = ast.literal_eval(fixed_json)
    except Exception:
        try:
            boxes = json.loads(fixed_json, strict=False)
        except Exception:
            boxes = []
            blocks = re.findall(r'\{[^{}]*\}', bounding_boxes_json)
            for block in blocks:
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
                st.error(f"Failed to parse JSON response completely.\n\nRaw Output: {bounding_boxes_json}")
                return img

    boxes = normalize_boxes(boxes)

    img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_draw = ImageDraw.Draw(img)

    for i, bounding_box in enumerate(boxes):
        bbox_2d = bounding_box.get("bbox_2d", bounding_box.get("box_2d", None))

        if bbox_2d and len(bbox_2d) >= 4:
            abs_x1 = int(bbox_2d[0] / 1000.0 * width)
            abs_y1 = int(bbox_2d[1] / 1000.0 * height)
            abs_x2 = int(bbox_2d[2] / 1000.0 * width)
            abs_y2 = int(bbox_2d[3] / 1000.0 * height)

            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            overlay_draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)),
                outline=(0, 255, 0, 255),
                width=max(2, int(height * 0.003)),
                fill=(0, 255, 0, 40)
            )

            text_content = bounding_box.get("text_content", bounding_box.get("text", ""))
            if text_content:
                badge_text = str(i + 1)

                try:
                    left, top, right, bottom = text_draw.textbbox((0, 0), badge_text, font=font)
                    tw, th = right - left, bottom - top
                except:
                    tw, th = dynamic_size, dynamic_size

                pad = int(dynamic_size * 0.3)
                badge_w = tw + (pad * 2)
                badge_h = th + (pad * 2)

                text_draw.rectangle(
                    ((abs_x1, max(0, abs_y1 - badge_h)), (abs_x1 + badge_w, max(0, abs_y1))),
                    fill="black"
                )
                text_draw.text((abs_x1 + pad, max(0, abs_y1 - badge_h + pad//2)), badge_text, fill="lime", font=font)

    final_img = Image.alpha_composite(img, overlay)
    return final_img.convert('RGB')

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

st.title("Advanced Math & Text OCR")

mode = st.radio("Select Mode", ["Full Page OCR (Text + Math)", "Text Spotting (Bounding Boxes)"], on_change=reset_extraction_state)

if image is not None:
    if mode == "Full Page OCR (Text + Math)":
        st.header("Full Page OCR")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Source")
            with st.container(height=600):
                st.image(image, use_container_width=True)

        with col2:
            st.subheader("Extracted Content")
            if subject_changed and st.session_state.get("extracted_text"):
                st.warning("Subject changed - re-extract for best results.")
            with st.container(height=600):
                if subject in ["Auto-detect", "Other"]:
                    sys_prompt = "You are a highly precise, versatile OCR engine. Your purpose is pixel-perfect transcription of any image content."
                    prompt = """Extract all text, numbers, and any other content from this image exactly as it appears.
1. Preserve all languages (e.g. English, Hindi, etc.) precisely.
2. Preserve capitalization and punctuation.
3. If there are tables or lists, format them using Markdown.
Output ONLY the extracted content without any commentary or conversational filler."""
                else:
                    sys_prompt = """You are an elite mathematical OCR engine with PhD-level precision.
Your sole purpose is pixel-perfect LaTeX transcription.

ABSOLUTE RULES:
1. NEVER paraphrase, simplify, or interpret - transcribe EXACTLY what you see
2. NEVER skip terms, even if the expression looks repetitive
3. NEVER guess - if a symbol is ambiguous, use the most mathematically consistent reading
4. Preserve ALL nested structures: parentheses depth, bracket types, operator order
5. Fraction rule: numerator is ALWAYS top, denominator is ALWAYS bottom - never swap
6. Floor brackets: use \\lfloor \\rfloor - never approximate as | or [
7. Ceiling brackets: use \\lceil \\rceil
8. Absolute value: use \\left| \\right|
9. Large brackets: use \\left( \\right), \\left[ \\right], \\left\\{ \\right\\} with correct \\bigg sizing
10. Exponents with complex expressions: use full {} grouping e^{\\frac{a}{b}(cx-d)}"""

                    prompt = """Perform pixel-perfect LaTeX extraction of ALL mathematical content in this image.

EXTRACTION PROTOCOL:
1. SCAN the entire image top-to-bottom, left-to-right - miss nothing
2. For each mathematical expression:
   - Identify ALL terms including signs (+ or -)
   - Check fraction orientation: top=numerator, bottom=denominator
   - Verify bracket matching: every \\left( must have \\right)
   - Count nested levels carefully

3. CRITICAL CHECKS before outputting:
   - Are all fractions correctly oriented? (not flipped)
   - Are floor/ceiling brackets \\lfloor \\rfloor vs \\lceil \\rceil correctly identified?
   - Are subscripts and superscripts on the correct symbol?
   - Are negative signs preserved on every term?
   - Are all \\min \\max \\cos \\sin \\log arguments complete?

4. FORMAT rules:
   - Wrap ALL math in $$ ... $$ for block equations
   - Use \\begin{array}{l} for multi-line expressions
   - Use \\\\ for line breaks within arrays
   - Use \\quad for alignment spacing
   - Non-math text: output as plain text above/below the math block
   - If you see diagrams/flowcharts/state machines: convert to Mermaid.js code block
   - If you see tables: convert to Markdown table format

5. SELF-CHECK: After extracting, mentally verify the first and last term
   of each major expression match the image exactly.

Output ONLY the extracted content - no explanations, no commentary."""

                if subject_hint:
                    prompt += f"\n\nAdditional context: {subject_hint}"

                if st.button("Extract Text & Math (Current Page)"):
                    with st.spinner("Extracting content via OpenRouter API..."):
                        try:
                            response = inference_with_api(temp_image_path, prompt, sys_prompt=sys_prompt, model_id=selected_model_id)
                            st.session_state["extracted_text"] = response
                            st.session_state["correction_editor"] = response
                        except Exception as e:
                            st.error(f"Error during inference: {str(e)}")

                if uploaded_file is not None and uploaded_file.name.split('.')[-1].lower() == 'pdf':
                    MAX_PAGES_LIMIT = 30
                    if num_pages > MAX_PAGES_LIMIT:
                        st.info(f"Batch mode is disabled because this PDF has {num_pages} pages (Safety Limit: {MAX_PAGES_LIMIT} pages). Please extract pages individually or upload a smaller PDF to prevent browser crashes.")
                    else:
                        process_all = st.checkbox("Process all pages (batch mode)")
                        if process_all:
                            n = num_pages
                            st.info(f"This will process all {n} pages with {n} API calls. This consumes more OpenRouter credits.")

                            if st.button("Process Entire Document"):
                                import concurrent.futures

                                full_document_parts = {}
                                progress_bar = st.progress(0)
                                status_text = st.empty()

                                current_session_id = st.session_state['session_id']

                                def process_page(page_idx):
                                    pg = pdf_document.load_page(page_idx)
                                    pix = pg.get_pixmap(dpi=200)
                                    img_bytes = pix.tobytes("png")
                                    img = Image.open(io.BytesIO(img_bytes))
                                    if img.mode == 'RGBA':
                                        img = img.convert('RGB')
                                    temp_path = os.path.join(tempfile.gettempdir(), f"ocr_temp_page_{current_session_id}_{page_idx}.jpg")
                                    img.save(temp_path)
                                    try:
                                        result = inference_with_api(temp_path, prompt, sys_prompt=sys_prompt, model_id=selected_model_id)
                                    except Exception as e:
                                        result = f"[Page {page_idx+1} extraction failed: {str(e)}]"
                                    finally:
                                        if os.path.exists(temp_path):
                                            os.remove(temp_path)
                                    return page_idx, result

                                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                                    futures = {executor.submit(process_page, i): i for i in range(n)}
                                    completed = 0
                                    for future in concurrent.futures.as_completed(futures):
                                        page_idx, result = future.result()
                                        full_document_parts[page_idx] = result
                                        completed += 1
                                        progress_bar.progress(completed / n)
                                        status_text.text(f"Processing... {completed} of {n} pages done")

                                progress_bar.empty()
                                status_text.empty()

                                full_document = ""
                                for i in range(n):
                                    full_document += f"\n\n---\n### Page {i+1}\n---\n\n{full_document_parts.get(i, '[Missing]')}"
                                st.session_state["extracted_text"] = full_document
                                st.session_state["correction_editor"] = full_document
                                st.success(f"All {n} pages processed.")
                                st.rerun()

                if st.session_state["extracted_text"]:
                    response = st.session_state["extracted_text"]
                    st.markdown("### Extracted Text:")
                    tab1, tab2 = st.tabs(["Rendered", "Raw LaTeX"])
                    with tab1:
                        formatted_response = preprocess_latex(response)
                        render_latex_safely(formatted_response)
                        st.caption("Complex equations may render better in the Raw LaTeX tab - copy and paste into Overleaf for full rendering.")
                    with tab2:
                        corrected = st.text_area(
                            "Edit extracted text (fix any errors below):",
                            value=st.session_state.get("extracted_text", ""),
                            height=400,
                            key="correction_editor"
                        )

                        if st.button("Save Correction"):
                            original = st.session_state.get("extracted_text", "")
                            if corrected.strip() == original.strip():
                                st.info("No changes detected.")
                            else:
                                try:
                                    img_hash = hashlib.md5(open(temp_image_path, "rb").read()).hexdigest()
                                except FileNotFoundError:
                                    img_hash = "unknown_hash"

                                record = {
                                    "timestamp": datetime.datetime.now().isoformat(),
                                    "image_hash": img_hash,
                                    "subject": st.session_state.get("subject", "Auto-detect"),
                                    "original": original,
                                    "corrected": corrected
                                }
                                with open("corrections.jsonl", "a", encoding="utf-8") as f:
                                    f.write(json.dumps(record) + "\n")

                                st.session_state["extracted_text"] = corrected
                                st.success("Correction saved.")

                                diff = list(difflib.unified_diff(
                                    original.splitlines(), corrected.splitlines(),
                                    lineterm='', n=2
                                ))
                                if diff:
                                    with st.expander("View changes"):
                                        st.code("\n".join(diff), language="diff")

                    st.markdown("---")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.download_button("Download .md", data=response, file_name="extracted.md", mime="text/markdown")
                    with col_b:
                        tex_content = f"\\documentclass{{article}}\n\\usepackage{{amsmath}}\n\\begin{{document}}\n\n{response}\n\n\\end{{document}}"
                        st.download_button("Download .tex", data=tex_content, file_name="extracted.tex", mime="text/plain")

                    response_escaped = response.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$').replace('\n', '\\n')
                    html_code = f"""
                    <button onclick="navigator.clipboard.writeText(`{response_escaped}`)" style="padding: 0.5rem 1rem; border-radius: 4px; border: 1px solid #ccc; background-color: #212529; color: white; cursor: pointer; width: 100%;">
                        Copy to Clipboard
                    </button>
                    """
                    components.html(html_code, height=50)

    elif mode == "Text Spotting (Bounding Boxes)":
        st.header("Text Spotting")

        if st.button("Spot Text"):
            with st.spinner("Spotting text via OpenRouter API..."):
                sys_prompt = """You are an elite mathematical OCR engine with PhD-level precision.
Your sole purpose is pixel-perfect LaTeX transcription.

ABSOLUTE RULES:
1. NEVER paraphrase, simplify, or interpret - transcribe EXACTLY what you see
2. NEVER skip terms, even if the expression looks repetitive
3. Fraction rule: numerator is ALWAYS top, denominator is ALWAYS bottom - never swap
4. Floor brackets: use \\lfloor \\rfloor - never approximate as | or [
5. Ceiling brackets: use \\lceil \\rceil
6. Absolute value: use \\left| \\right|
7. Large brackets: use \\left( \\right) with correct \\bigg sizing
8. Exponents with complex expressions: use full {} grouping"""

                prompt = """Perform precise text spotting with bounding boxes on this image.

For each detected region output JSON with:
- "bbox_2d": [xmin, ymin, xmax, ymax] normalized to 1000
- "text_content": exact LaTeX/text content

EXTRACTION RULES:
1. Math expressions: wrap in $$ ... $$
2. BLOCK LEVEL ONLY: ONLY output bounding boxes for logical BLOCKS of text/math (e.g. a full paragraph, a full equation). DO NOT output sub-boxes for individual symbols, terms, or fractions.
3. Fractions: ALWAYS top=numerator, bottom=denominator - NEVER flip
4. Floor brackets: \\lfloor \\rfloor (NOT | or [)
5. Preserve ALL negative signs - check every term
6. Complex exponents: full grouping: e^{\\frac{5}{256}\\left(\\frac{20x}{a}-139\\right)}
7. Nested brackets: match depth carefully, use \\bigg \\Big sizing
8. Multi-term \\min \\max: include ALL comma-separated arguments
9. Tables: put entire Markdown table in text_content
10. Diagrams/state machines: put entire Mermaid.js code in text_content
11. CRITICAL JSON REQUIREMENT: You MUST double-escape ALL LaTeX backslashes in your JSON strings. Output \\\\frac instead of \\frac, \\\\left instead of \\left, etc.

SELF-VERIFY each bbox:
- Fraction orientation correct?
- All signs preserved?
- Bracket depth balanced?

Output ONLY a valid JSON array. No markdown fences. No commentary."""

                if subject_hint:
                    prompt += f"\n\nAdditional context: {subject_hint}"
                try:
                    response = inference_with_api(temp_image_path, prompt, sys_prompt=sys_prompt, model_id=selected_model_id)
                    result_image = plot_text_bounding_boxes(temp_image_path, response)

                    st.session_state["spot_response"] = response
                    st.session_state["spot_image"] = result_image
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(height=600):
                if st.session_state["spot_image"] is not None:
                    st.image(st.session_state["spot_image"], caption="Text Spotting Result", use_container_width=True)
                else:
                    st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            if subject_changed and st.session_state.get("spot_response"):
                st.warning("Subject changed - re-extract for best results.")
            with st.container(height=600):
                if st.session_state["spot_response"]:
                    response = st.session_state["spot_response"]
                    st.markdown("### Detected Text:")

                    try:
                        clean_json = parse_json(response)
                        fixed_json = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', clean_json)
                        try:
                            boxes = ast.literal_eval(fixed_json)
                        except Exception:
                            try:
                                boxes = json.loads(fixed_json, strict=False)
                            except Exception:
                                boxes = []
                                blocks = re.findall(r'\{[^{}]*\}', clean_json)
                                for block in blocks:
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
                                    raise Exception("Regex parser failed.")

                        boxes = normalize_boxes(boxes)

                        extracted_lines = []
                        for idx, box in enumerate(boxes):
                            txt = box.get("text_content", box.get("text", "")).strip()

                            txt = txt.replace('`', '')

                            if txt.startswith('$$'): txt = txt[2:].strip()
                            elif txt.startswith('$'): txt = txt[1:].strip()
                            if txt.endswith('$$'): txt = txt[:-2].strip()
                            elif txt.endswith('$'): txt = txt[:-1].strip()

                            txt = re.sub(r'\\sqrt\[([^\]]+)\]\[([^\]]+)\]', r'\\sqrt[\1]{\2}', txt)

                            math_keywords = ['\\frac', '\\min', '\\max', '\\left', '\\right', '\\cos', '\\sin', '\\log', '\\sum', '^', '_', '\\Big', '\\sqrt']
                            if any(m in txt for m in math_keywords):
                                txt = f"$${txt}$$"

                            box_label = f"**Box {idx + 1}:**\n\n"
                            extracted_lines.append(box_label + txt)

                        full_text = " \n\n".join(extracted_lines)

                        formatted_text = preprocess_latex(full_text)

                        tab1, tab2 = st.tabs(["Rendered", "Raw LaTeX"])
                        with tab1:
                            render_latex_safely(formatted_text)
                            st.caption("Complex equations may render better in the Raw LaTeX tab - copy and paste into Overleaf for full rendering.")
                        with tab2:
                            st.code(full_text, language="markdown")

                        st.markdown("---")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.download_button("Download .md", data=full_text, file_name="extracted.md", mime="text/markdown", key="md_spot")
                        with col_b:
                            tex_content = f"\\documentclass{{article}}\n\\usepackage{{amsmath}}\n\\begin{{document}}\n\n{full_text}\n\n\\end{{document}}"
                            st.download_button("Download .tex", data=tex_content, file_name="extracted.tex", mime="text/plain", key="tex_spot")

                        response_escaped = full_text.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$').replace('\n', '\\n')
                        html_code = f"""
                        <button onclick="navigator.clipboard.writeText(`{response_escaped}`)" style="padding: 0.5rem 1rem; border-radius: 4px; border: 1px solid #ccc; background-color: #212529; color: white; cursor: pointer; width: 100%;">
                            Copy to Clipboard
                        </button>
                        """
                        components.html(html_code, height=50)

                        st.markdown(" ")
                        with st.expander("View Raw JSON Box Data"):
                            st.code(clean_json, language="json")
                    except Exception as e:
                        st.code(response, language="json")
else:
    st.info("Please upload or capture an image to begin.")

if "session_id" in st.session_state:
    temp_path = os.path.join(tempfile.gettempdir(), f"ocr_temp_{st.session_state['session_id']}.jpg")
    if os.path.exists(temp_path):
        try:
            if image is None:
                os.remove(temp_path)
        except:
            pass
