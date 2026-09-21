import os
import io
import uuid
import asyncio
import tempfile

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from PIL import Image
import pymupdf as fitz

from core.ocr_engine import inference_with_api, auto_select_model, get_ocr_prompt
from core.image_enhancer import enhance_image
from api.models.schemas import OCRResponse, SpotResponse

router = APIRouter(tags=["OCR"])

MAX_FILE_SIZE = 20 * 1024 * 1024
PDF_PAGE_CONCURRENCY = 3

_pdf_semaphore = asyncio.Semaphore(PDF_PAGE_CONCURRENCY)

async def _read_upload(file: UploadFile) -> bytes:
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")

    contents = await file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")
    return contents

def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def _save_temp_image(img: Image.Image, temp_dir: str, file_id: str, suffix: str) -> str:
    temp_path = os.path.join(temp_dir, f"api_{suffix}_{file_id}.jpg")
    _to_rgb(img).save(temp_path, format="JPEG")
    return temp_path

def _ocr_single_image(img: Image.Image, prompt: str, sys_prompt: str, model_id: str,
                      temp_dir: str, file_id: str, suffix: str = "ocr") -> str:
    temp_path = _save_temp_image(img, temp_dir, file_id, suffix)
    try:
        return inference_with_api(temp_path, prompt, sys_prompt=sys_prompt, model_id=model_id)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def _ocr_pdf_page(contents: bytes, page_idx: int, prompt: str, sys_prompt: str,
                  model_id: str, temp_dir: str, file_id: str):
    try:
        with fitz.open(stream=contents, filetype="pdf") as doc:
            pg = doc.load_page(page_idx)
            pix = pg.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
        img = _to_rgb(Image.open(io.BytesIO(img_bytes)))
        result = _ocr_single_image(img, prompt, sys_prompt, model_id,
                                   temp_dir, f"{file_id}_{page_idx}", "ocr")
        return page_idx, result
    except Exception as e:
        return page_idx, f"[Page {page_idx + 1} extraction failed: {e}]"

def _pdf_page_count(contents: bytes) -> int:
    with fitz.open(stream=contents, filetype="pdf") as pdf_document:
        return len(pdf_document)

async def _ocr_pdf_page_async(contents: bytes, page_idx: int, prompt: str, sys_prompt: str,
                              model_id: str, temp_dir: str, file_id: str):
    async with _pdf_semaphore:
        return await asyncio.to_thread(
            _ocr_pdf_page, contents, page_idx, prompt, sys_prompt, model_id, temp_dir, file_id
        )

def _load_image(contents: bytes) -> Image.Image:
    return _to_rgb(Image.open(io.BytesIO(contents)))

@router.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    file: UploadFile = File(...),
    subject: str = Form("Auto-detect"),
    model: str = Form("auto"),
    enhance_img: bool = Form(False),
    page: Optional[int] = Form(None),
):
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")

    allowed_types = ["image/jpeg", "image/png", "image/webp", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    contents = await _read_upload(file)
    file_id = uuid.uuid4().hex
    temp_dir = tempfile.gettempdir()

    try:
        model_id = auto_select_model(subject) if model == "auto" else model
        sys_prompt, prompt = get_ocr_prompt(subject, mode="full_page")

        if file.content_type == "application/pdf":
            num_pages = await asyncio.to_thread(_pdf_page_count, contents)
            if page is not None:
                if page < 1 or page > num_pages:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Page {page} out of range (1-{num_pages}).",
                    )
                pages_to_process = [page - 1]
            else:
                pages_to_process = list(range(min(num_pages, 30)))

            if len(pages_to_process) == 1:
                _, result = await asyncio.to_thread(
                    _ocr_pdf_page, contents, pages_to_process[0], prompt,
                    sys_prompt, model_id, temp_dir, file_id,
                )
                return OCRResponse(text=result, model_used=model_id, pages=1, subject=subject)

            results: dict = {}
            page_results = await asyncio.gather(*(
                _ocr_pdf_page_async(contents, idx, prompt, sys_prompt, model_id, temp_dir, file_id)
                for idx in pages_to_process
            ))
            for idx, result in page_results:
                results[idx] = result

            all_text = [
                f"\n---\n### Page {idx + 1}\n---\n\n{results[idx]}"
                for idx in pages_to_process
            ]
            return OCRResponse(
                text="\n".join(all_text),
                model_used=model_id,
                pages=len(pages_to_process),
                subject=subject,
            )

        img = await asyncio.to_thread(_load_image, contents)

        if enhance_img:
            img = await asyncio.to_thread(enhance_image, img)

        result = await asyncio.to_thread(
            _ocr_single_image, img, prompt, sys_prompt, model_id, temp_dir, file_id, "ocr"
        )
        return OCRResponse(text=result, model_used=model_id, pages=1, subject=subject)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ocr/spot", response_model=SpotResponse)
async def spot_endpoint(
    file: UploadFile = File(...),
    subject: str = Form("Auto-detect"),
    model: str = Form("auto"),
    enhance_img: bool = Form(False),
):
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")

    contents = await _read_upload(file)
    file_id = uuid.uuid4().hex
    temp_dir = tempfile.gettempdir()

    try:
        model_id = auto_select_model(subject) if model == "auto" else model
        sys_prompt, prompt = get_ocr_prompt(subject, mode="text_spotting")

        img = await asyncio.to_thread(_load_image, contents)

        if enhance_img:
            img = await asyncio.to_thread(enhance_image, img)

        result = await asyncio.to_thread(
            _ocr_single_image, img, prompt, sys_prompt, model_id, temp_dir, file_id, "spot"
        )
        return SpotResponse(boxes=result, model_used=model_id)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
