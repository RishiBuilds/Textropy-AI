import os
import io
import re
import json
import ast
import base64
import tempfile
import requests
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

REQUEST_TIMEOUT = float(os.getenv("OPENROUTER_REQUEST_TIMEOUT", "60"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def auto_select_model(subject):
    return "qwen/qwen-2.5-vl-72b-instruct"

def get_ocr_prompt(subject, mode="full_page"):
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
    subject_hint = subject_hints.get(subject, "")

    if mode == "text_spotting":
        sys_prompt = r"""You are an elite mathematical OCR engine with PhD-level precision.
Your sole purpose is pixel-perfect LaTeX transcription.

ABSOLUTE RULES:
1. NEVER paraphrase, simplify, or interpret - transcribe EXACTLY what you see
2. NEVER skip terms, even if the expression looks repetitive
3. Fraction rule: numerator is ALWAYS top, denominator is ALWAYS bottom - never swap
4. Floor brackets: use \lfloor \rfloor - never approximate as | or [
5. Ceiling brackets: use \lceil \rceil
6. Absolute value: use \left| \right|
7. Large brackets: use \left( \right) with correct \bigg sizing
8. Exponents with complex expressions: use full {} grouping"""

        prompt = r"""Perform precise text spotting with bounding boxes on this image.

For each detected region output JSON with:
- "bbox_2d": [xmin, ymin, xmax, ymax] normalized to 1000
- "text_content": exact LaTeX/text content

EXTRACTION RULES:
1. Math expressions: wrap in $$ ... $$
2. BLOCK LEVEL ONLY: ONLY output bounding boxes for logical BLOCKS of text/math (e.g. a full paragraph, a full equation). DO NOT output sub-boxes for individual symbols, terms, or fractions.
3. Fractions: ALWAYS top=numerator, bottom=denominator - NEVER flip
4. Floor brackets: \lfloor \rfloor (NOT | or [)
5. Preserve ALL negative signs - check every term
6. Complex exponents: full grouping: e^{\frac{5}{256}\left(\frac{20x}{a}-139\right)}
7. Nested brackets: match depth carefully, use \bigg \Big sizing
8. Multi-term \min \max: include ALL comma-separated arguments
9. Tables: put entire Markdown table in text_content
10. Diagrams/state machines: put entire Mermaid.js code in text_content
11. CRITICAL JSON REQUIREMENT: You MUST double-escape ALL LaTeX backslashes in your JSON strings. Output \\frac instead of \frac, \\left instead of \left, etc.

SELF-VERIFY each bbox:
- Fraction orientation correct?
- All signs preserved?
- Bracket depth balanced?

Output ONLY a valid JSON array. No markdown fences. No commentary."""

    elif subject in ["Auto-detect", "Other"]:
        sys_prompt = r"""You are an elite OCR engine with PhD-level precision in both text and mathematics.
Your purpose is pixel-perfect transcription of any image content.

CRITICAL RULES FOR MATHEMATICAL CONTENT:
1. ALL superscripts must use LaTeX caret notation: x^{2}, a^{3}, e^{n}
2. ALL subscripts must use LaTeX underscore notation: x_{1}, a_{n}
3. ALL fractions must use \frac{numerator}{denominator}
4. ALL Greek letters must use LaTeX commands: \alpha, \beta, \pi, \theta
5. Square roots: \sqrt{x}, cube roots: \sqrt[3]{x}
6. Wrap each equation or formula line in $$ ... $$ delimiters
7. NEVER output superscripts as plain text (WRONG: a2, x3. RIGHT: a^{2}, x^{3})
8. NEVER lose exponents, subscripts, or special notation"""

        prompt = r"""Extract ALL content from this image with pixel-perfect accuracy.

PROTOCOL:
1. SCAN the entire image top-to-bottom, left-to-right - miss nothing
2. Plain text: output as-is, preserving all languages, capitalization, and punctuation
3. Mathematical expressions and formulas:
   - Convert ALL math to proper LaTeX notation
   - Superscripts: x^{2} NOT x2, a^{3} NOT a3
   - Subscripts: x_{1} NOT x1
   - Fractions: \frac{a}{b}
   - Greek: \alpha, \beta, \pi, \sum, \int
   - Wrap each equation line in $$ ... $$ delimiters
   - Use \begin{array}{l} for multi-line expressions
4. Tables: format using Markdown table syntax
5. Lists: format using Markdown list syntax
6. Diagrams/flowcharts: convert to Mermaid.js code block

SELF-CHECK: verify every superscript and subscript is correctly marked with ^ and _ notation.

Output ONLY the extracted content - no explanations, no commentary."""
    else:
        sys_prompt = r"""You are an elite mathematical OCR engine with PhD-level precision.
Your sole purpose is pixel-perfect LaTeX transcription.

ABSOLUTE RULES:
1. NEVER paraphrase, simplify, or interpret - transcribe EXACTLY what you see
2. NEVER skip terms, even if the expression looks repetitive
3. NEVER guess - if a symbol is ambiguous, use the most mathematically consistent reading
4. Preserve ALL nested structures: parentheses depth, bracket types, operator order
5. Fraction rule: numerator is ALWAYS top, denominator is ALWAYS bottom - never swap
6. Floor brackets: use \lfloor \rfloor - never approximate as | or [
7. Ceiling brackets: use \lceil \rceil
8. Absolute value: use \left| \right|
9. Large brackets: use \left( \right), \left[ \right], \left\{ \right\} with correct \bigg sizing
10. Exponents with complex expressions: use full {} grouping e^{\frac{a}{b}(cx-d)}"""

        prompt = r"""Perform pixel-perfect LaTeX extraction of ALL mathematical content in this image.

EXTRACTION PROTOCOL:
1. SCAN the entire image top-to-bottom, left-to-right - miss nothing
2. For each mathematical expression:
   - Identify ALL terms including signs (+ or -)
   - Check fraction orientation: top=numerator, bottom=denominator
   - Verify bracket matching: every \left( must have \right)
   - Count nested levels carefully

3. CRITICAL CHECKS before outputting:
   - Are all fractions correctly oriented? (not flipped)
   - Are floor/ceiling brackets \lfloor \rfloor vs \lceil \rceil correctly identified?
   - Are subscripts and superscripts on the correct symbol?
   - Are negative signs preserved on every term?
   - Are all \min \max \cos \sin \log arguments complete?

4. FORMAT rules:
   - Wrap ALL math in $$ ... $$ for block equations
   - Use \begin{array}{l} for multi-line expressions
   - Use \\ for line breaks within arrays
   - Use \quad for alignment spacing
   - Non-math text: output as plain text above/below the math block
   - If you see diagrams/flowcharts/state machines: convert to Mermaid.js code block
   - If you see tables: convert to Markdown table format

5. SELF-CHECK: After extracting, mentally verify the first and last term
   of each major expression match the image exactly.

Output ONLY the extracted content - no explanations, no commentary."""

    if subject_hint:
        prompt += f"\n\nAdditional context: {subject_hint}"

    return sys_prompt, prompt

def inference_with_api(image_path, prompt, sys_prompt="You are a precise document extraction assistant.", model_id="qwen/qwen-2.5-vl-72b-instruct"):
    try:
        if model_id == "nvidia/nemotron-ocr-v1":
            if not os.getenv('NVIDIA_API_KEY'):
                raise ValueError("Please provide an NVIDIA_API_KEY in your .env file.")

            invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"
            headers = {
                "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
                "Accept": "application/json"
            }

            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            quality = 95
            while True:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=quality)
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                if len(base64_image) < 180000 or quality <= 10:
                    break
                quality -= 15
                if quality < 30:
                    img = img.resize((int(img.width * 0.8), int(img.height * 0.8)))

            payload = {
                "input": [
                    {
                        "type": "image_url",
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
            response = requests.post(
                invoke_url,
                headers=headers,
                json=payload,
                timeout=int(os.getenv("NVIDIA_REQUEST_TIMEOUT", "60")),
            )
            if response.status_code != 200:
                raise Exception(f"Nvidia API returned {response.status_code}: {response.text}")

            res_json = response.json()
            if "bbox_2d" in prompt:
                boxes = []
                if "data" in res_json and isinstance(res_json["data"], list):
                    for item in res_json["data"]:
                        if "text_detections" in item and isinstance(item["text_detections"], list):
                            for det in item["text_detections"]:
                                txt = det.get("text_prediction", {}).get("text", "")
                                pts = det.get("bounding_box", {}).get("points", [])
                                if pts and len(pts) >= 4:
                                    xs = [p.get("x", 0) for p in pts]
                                    ys = [p.get("y", 0) for p in pts]
                                    xmin, xmax = min(xs) * 1000, max(xs) * 1000
                                    ymin, ymax = min(ys) * 1000, max(ys) * 1000
                                    boxes.append({
                                        "bbox_2d": [int(xmin), int(ymin), int(xmax), int(ymax)],
                                        "text_content": txt
                                    })
                return json.dumps(boxes)
            else:
                extracted_texts = []
                if "data" in res_json and isinstance(res_json["data"], list):
                    for item in res_json["data"]:
                        if "text_detections" in item and isinstance(item["text_detections"], list):
                            for detection in item["text_detections"]:
                                if "text_prediction" in detection and "text" in detection["text_prediction"]:
                                    extracted_texts.append(detection["text_prediction"]["text"])
                if extracted_texts:
                    return "\n".join(extracted_texts)

            return json.dumps(res_json)

        if not os.getenv('OPENROUTER_API_KEY'):
            raise ValueError("Please provide an OpenRouter API Key in your .env file or sidebar.")

        base64_image = encode_image(image_path)
        client = OpenAI(
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
            timeout=REQUEST_TIMEOUT,
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
