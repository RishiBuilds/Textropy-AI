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
