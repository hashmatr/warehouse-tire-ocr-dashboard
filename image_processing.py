import streamlit as st
import base64
import json
import re
import time
import os
import html
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()
api_key = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

GEMINI_MODEL = "gemini-2.5-flash"

# App setup
st.set_page_config(page_title="Industrial Tire OCR Dashboard", layout="wide")

st.title("Tyre OCR Tracking Dashboard")
st.markdown("Automated batch sidewall character extraction pipeline.")

# Sidebar
st.sidebar.header("Model Settings")
temperature = 0.0
max_tokens = 1500


# Engine prompts and patterns
SYSTEM_PROMPT = (
    "You are a strict JSON-only OCR extraction engine. "
    "You must NEVER output any natural language, explanations, or markdown. "
    "Your entire response must be a single valid JSON object and nothing else."
)

OCR_PROMPT = (
    "Extract the tyre size and DOT number molded into the rubber of this tyre sidewall.\n\n"
    "Look for these two specific markings:\n"
    "1. TYRE SIZE: The main molded sizing code (e.g., 285/75R18, 35x12.50R20, P245/75R17).\n"
    "2. DOT NUMBER: The molded serial code located near the rim. "
    "It ALWAYS starts with 'DOT ' and is followed by alphanumeric characters, ending with a 4-digit date code (WWYY). "
    "Scan the entire circumference carefully. Do not stop reading early; extract the full sequence, including the 4-digit date code at the end.\n\n"
    "Respond ONLY with a single JSON object in the following format, no other text or markdown:\n"
    '{"extracted_size": "<SIZE_FROM_IMAGE>", "extracted_dot": "<DOT_FROM_IMAGE>"}\n'
    "If a value is not found or is unreadable, set it to null."
)


# Size patterns for extraction fallback
TYRE_SIZE_PATTERNS = [
    # Flotation size (e.g. 35x12.50R18)
    re.compile(r"\d{2,3}[/x]\d{1,2}\.\d{1,2}[RBD]?\d{0,2}", re.IGNORECASE),
    # Standard metric (e.g. 285/75R18)
    re.compile(r"(?:LT|P|ST)?\d{2,3}[/x]\d{2,3}[RBD]?\d{2,3}(?:\.\d)?", re.IGNORECASE),
    # Generic fallback (e.g. 35X12.50)
    re.compile(r"\d{2,3}[/x]\d{2,3}(?:\.\d{1,2})?", re.IGNORECASE),
]

# DOT number patterns for extraction fallback
DOT_NUMBER_PATTERNS = [
    # Full DOT: DOT XXXX XXXX XXXX WWYY
    re.compile(
        r"DOT\s*[A-Z0-9]{2,4}\s*[A-Z0-9]{2,4}\s*[A-Z0-9]{2,4}\s*\d{4}", re.IGNORECASE
    ),
    # DOT with varying segment lengths
    re.compile(r"DOT\s*[A-Z0-9\s]{4,16}\s*\d{4}", re.IGNORECASE),
    # Just DOT followed by content
    re.compile(r"DOT\s+[A-Z0-9\s]{6,}", re.IGNORECASE),
]


def decode_dot_date(dot_string):
    """Decode manufacturing date from DOT number's last 4 digits (WWYY format)."""
    if not dot_string:
        return None
    clean = dot_string.replace(" ", "")
    digits = re.search(r"(\d{4})$", clean)
    if digits:
        ww = int(digits.group(1)[:2])
        yy = int(digits.group(1)[2:])
        year = 2000 + yy
        if 1 <= ww <= 53:
            return f"Week {ww}, {year}"
    return None


def preprocess_image(image):
    """Enhance image for better OCR: upscale, sharpen, boost contrast."""
    # Upscale small images so the model can read fine text
    w, h = image.size
    min_dim = min(w, h)
    if min_dim < 1500:
        scale = 1500 / min_dim
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    # Boost contrast to make molded rubber text stand out
    image = ImageEnhance.Contrast(image).enhance(1.6)

    # Sharpen to improve edge clarity of embossed characters
    image = ImageEnhance.Sharpness(image).enhance(2.0)

    # Slight brightness boost to reveal dark-on-dark text
    image = ImageEnhance.Brightness(image).enhance(1.1)

    return image


def encode_image_to_base64(uploaded_file, enhance=True):
    # Convert image file to base64 JPEG with optional preprocessing
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if enhance:
        image = preprocess_image(image)
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def format_error_message(e):
    # Map raw API errors to user friendly messages
    error_str = str(e)
    if (
        "429" in error_str
        or "RESOURCE_EXHAUSTED" in error_str
        or "quota" in error_str.lower()
    ):
        return (
            "**Quota Exceeded** — Your Gemini API free-tier limit has been reached. "
            "Please wait a minute and try again, or upgrade to a paid plan at "
            "https://ai.google.dev/pricing"
        )
    elif (
        "401" in error_str
        or "403" in error_str
        or "UNAUTHENTICATED" in error_str
        or "PERMISSION_DENIED" in error_str
    ):
        return (
            "**Invalid API Key** — Your Gemini API key is invalid or expired. "
            "Please check the GEMINI_API_KEY in your .env file. "
            "Get a valid key at https://aistudio.google.com/apikey"
        )
    elif "404" in error_str or "not found" in error_str.lower():
        return (
            f"**Model Not Found** — The model `{GEMINI_MODEL}` was not found. "
            "Please check the model name or your endpoint settings."
        )
    elif "timeout" in error_str.lower() or "connection" in error_str.lower():
        return (
            "**Connection Error** — Could not reach the Gemini API. "
            "Please check your internet connection and try again."
        )
    else:
        return f"**Extraction Failed** — {error_str[:200]}"


def _try_parse_json(json_str):
    """Try to parse JSON and extract size + DOT fields."""
    try:
        data = json.loads(json_str)
        return data.get("extracted_size"), data.get("extracted_dot"), True
    except (json.JSONDecodeError, AttributeError):
        return None, None, False


def _rescue_truncated_json(json_str):
    """Attempt to fix and parse truncated JSON."""
    rescued = json_str
    if rescued.count('"') % 2 != 0:
        rescued += '"'
    if not rescued.endswith("}"):
        rescued += "}"
    return _try_parse_json(rescued)


def parse_extraction_result(content):
    # Parse LLM response using json or pattern match
    # Returns (size_result, dot_result, warning)
    if not content or not content.strip():
        return None, None, "Empty response from model"

    content = content.strip()

    # Fix truncated responses
    if content.startswith("{") and not content.endswith("}"):
        size, dot, ok = _rescue_truncated_json(content)
        if ok and (size or dot):
            return size, dot, f"Rescued truncated JSON (raw: {content})"

    # JSON parse
    size, dot, ok = _try_parse_json(content)
    if ok:
        return size, dot, None

    # Markdown code blocks
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
            size, dot, ok = _rescue_truncated_json(json_str)
            if ok:
                return size, dot, None
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
            size, dot, ok = _rescue_truncated_json(json_str)
            if ok:
                return size, dot, None
    except (json.JSONDecodeError, IndexError):
        pass

    # Extract JSON string regex
    try:
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            size, dot, ok = _try_parse_json(json_match.group())
            if ok:
                return size, dot, None
    except json.JSONDecodeError:
        pass

    # Fallback to regex patterns
    size_result = None
    dot_result = None

    for pattern in TYRE_SIZE_PATTERNS:
        match = pattern.search(content)
        if match:
            size_result = match.group()
            break

    for pattern in DOT_NUMBER_PATTERNS:
        match = pattern.search(content)
        if match:
            dot_result = match.group()
            break

    if size_result or dot_result:
        return size_result, dot_result, f"Extracted via pattern match (raw: {content})"

    return None, None, f"Could not parse response: {repr(content)}"


def _call_gemini(prompt, image_bytes, temp, tokens):
    """Single Gemini API call helper using Native JSON constraint."""
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL, system_instruction=SYSTEM_PROMPT
    )
    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            },
        ],
        generation_config={
            "temperature": temp,
            "max_output_tokens": 1500,
            "response_mime_type": "application/json",
        },
    )
    return response.text if hasattr(response, "text") else ""


def run_ocr_extraction(b64_string, temp, tokens):
    """Perform single-pass OCR extraction for tyre size and DOT number."""
    image_bytes = base64.b64decode(b64_string)

    content = _call_gemini(OCR_PROMPT, image_bytes, temp, tokens)
    size_result, dot_result, warning = parse_extraction_result(content)

    return size_result, dot_result, warning, content


def render_scrollable_value(label, value):
    """Render a value with a horizontal scroller if the text is long."""
    st.markdown(f"**{label}**")
    if value and len(value) > 25:
        escaped_value = html.escape(value)
        st.markdown(
            f'<div style="overflow-x:auto; white-space:nowrap; background:#0e1117; '
            f"border:1px solid #333; border-radius:6px; padding:8px 12px; "
            f'font-family:monospace; font-size:14px; color:#fafafa; max-width:100%;">'
            f"{escaped_value}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.code(value if value else "N/A", language=None)


# Select scope
run_mode = st.radio(
    "Select Processing Scope",
    ["Batch Processing (Multiple Images)", "Single Image Testing"],
    horizontal=True,
)

# Batch mode
if run_mode == "Batch Processing (Multiple Images)":
    st.subheader("Batch Image Upload")
    uploaded_files = st.file_uploader(
        "Drag and drop a batch of tyre images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"Staged **{len(uploaded_files)}** files for processing.")

        # Preview thumbnails
        with st.expander("Preview uploaded images", expanded=False):
            preview_cols = st.columns(min(len(uploaded_files), 6))
            for i, file in enumerate(uploaded_files):
                with preview_cols[i % 6]:
                    st.image(file, caption=file.name, width="stretch")

        if st.button("Launch Batch Pipeline"):
            if not api_key:
                st.warning(
                    "Please provide a valid Gemini API Key in the .env file (GEMINI_API_KEY) or Streamlit secrets."
                )
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                batch_results = []
                total_start = time.time()

                for idx, file in enumerate(uploaded_files):
                    status_text.text(
                        f"Processing {idx + 1}/{len(uploaded_files)}: {file.name}"
                    )

                    item_start = time.time()
                    try:
                        b64_string = encode_image_to_base64(file)
                        size_result, dot_result, warning, raw_content = (
                            run_ocr_extraction(b64_string, temperature, max_tokens)
                        )
                        elapsed = round(time.time() - item_start, 2)

                        dot_date = decode_dot_date(dot_result) if dot_result else None
                        status = "OK"
                        batch_results.append(
                            {
                                "Filename": file.name,
                                "Extracted Size": (
                                    str(size_result).upper() if size_result else "N/A"
                                ),
                                "DOT Number": (
                                    str(dot_result).upper() if dot_result else "N/A"
                                ),
                                "Mfg Date": dot_date if dot_date else "N/A",
                                "Status": status,
                                "Time (s)": elapsed,
                            }
                        )

                    except Exception as e:
                        elapsed = round(time.time() - item_start, 2)
                        batch_results.append(
                            {
                                "Filename": file.name,
                                "Extracted Size": "N/A",
                                "DOT Number": "N/A",
                                "Mfg Date": "N/A",
                                "Status": f"FAILED: {format_error_message(e)[:80]}",
                                "Time (s)": elapsed,
                            }
                        )

                    time.sleep(1.5)
                    progress_bar.progress((idx + 1) / len(uploaded_files))

                total_elapsed = round(time.time() - total_start, 2)
                status_text.text(f"Batch complete. Total time: {total_elapsed}s")

                # Summary metrics
                success_count = sum(1 for r in batch_results if r["Status"] == "OK")
                fail_count = len(batch_results) - success_count

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Processed", len(batch_results))
                col_m2.metric("Successful", success_count)
                col_m3.metric("Failed", fail_count)

                # Results table
                st.subheader("Results")
                df_results = pd.DataFrame(batch_results)
                st.dataframe(df_results, width="stretch")

                # CSV export
                csv_buffer = BytesIO()
                df_results.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="Export Results to CSV",
                    data=csv_data,
                    file_name="tyre_ocr_results.csv",
                    mime="text/csv",
                )

# Single testing mode
else:
    st.subheader("Single Image Testing")
    uploaded_file = st.file_uploader(
        "Upload a single tyre image...",
        type=["jpg", "jpeg", "png"],
        key="single_uploader",
    )

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width="stretch")
        with col2:
            if st.button("Run Extraction"):
                if not api_key:
                    st.warning(
                        "Please provide a valid Gemini API Key in the .env file (GEMINI_API_KEY) or Streamlit secrets."
                    )
                else:
                    with st.spinner("Running OCR extraction..."):
                        try:
                            b64_string = encode_image_to_base64(uploaded_file)

                            start_time = time.time()
                            size_result, dot_result, warning, raw_content = (
                                run_ocr_extraction(
                                    b64_string,
                                    temperature,
                                    max_tokens,
                                )
                            )
                            elapsed = round(time.time() - start_time, 2)

                            dot_date = (
                                decode_dot_date(dot_result) if dot_result else None
                            )

                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                render_scrollable_value(
                                    "Extracted Tyre Size",
                                    str(size_result).upper() if size_result else "N/A",
                                )
                                st.metric("Response Time", f"{elapsed}s")
                            with res_col2:
                                render_scrollable_value(
                                    "DOT Number",
                                    str(dot_result).upper() if dot_result else "N/A",
                                )
                                st.metric(
                                    "Manufacturing Date",
                                    dot_date if dot_date else "N/A",
                                )

                        except Exception as e:
                            st.error(format_error_message(e))
