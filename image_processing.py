import streamlit as st
import base64
import json
import re
import time
import os
import pandas as pd
from PIL import Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

QWEN_MODEL = "qwen/qwen-2.5-vl-72b-instruct"
QWEN_BASE_URL = "https://openrouter.ai/api/v1"

# App setup
st.set_page_config(page_title="Industrial Tire OCR Dashboard", layout="wide")

st.title("Tyre OCR Tracking Dashboard")
st.markdown("Automated batch sidewall character extraction pipeline.")

# Sidebar
st.sidebar.header("Model Settings")
st.sidebar.info(f"Using **Qwen** model: `{QWEN_MODEL}`")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
max_tokens = st.sidebar.number_input(
    "Max Tokens", min_value=20, max_value=500, value=300
)

# Engine prompts and patterns
SYSTEM_PROMPT = (
    "You are a strict JSON-only OCR extraction engine. "
    "You must NEVER output any natural language, explanations, or markdown. "
    "Your entire response must be a single valid JSON object and nothing else."
)

UPGRADED_PROMPT = (
    "Perform precise industrial OCR tracking on this tyre sidewall rubber. "
    "Locate the primary MOLDED raised rubber alphanumeric sizing sequence code. "
    "CRITICAL: Completely ignore any high-contrast white factory ink stamps, laser prints, or paint labels. "
    "Look strictly for the permanent structural text molded into the dark black rubber texture. "
    "Respond with ONLY this exact JSON format, no other text: "
    '{"extracted_size": "THE_SIZE_STRING_HERE"} '
    "If you cannot read the size, respond with: "
    '{"extracted_size": null}'
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


def encode_image_to_base64(uploaded_file):
    # Convert image file to base64 JPEG
    image = Image.open(uploaded_file)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
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
            f"**Model Not Found** — The model `{QWEN_MODEL}` was not found. "
            "Please check the model name or your endpoint settings."
        )
    elif "timeout" in error_str.lower() or "connection" in error_str.lower():
        return (
            "**Connection Error** — Could not reach the Gemini API. "
            "Please check your internet connection and try again."
        )
    else:
        return f"**Extraction Failed** — {error_str[:200]}"


def parse_extraction_result(content):
    # Parse LLM response using json or pattern match
    if not content or not content.strip():
        return None, "Empty response from model"

    content = content.strip()

    # Fix truncated responses
    if content.startswith("{") and not content.endswith("}"):
        rescued_content = content
        # If it ends inside a string key/value, close the quote
        if rescued_content.count('"') % 2 != 0:
            rescued_content += '"'
        # Close the curly brace
        rescued_content += "}"
        try:
            val = json.loads(rescued_content).get("extracted_size")
            if val:
                return val, f"Rescued truncated JSON (raw: {content})"
        except json.JSONDecodeError:
            pass

    # JSON parse
    try:
        return json.loads(content).get("extracted_size"), None
    except json.JSONDecodeError:
        pass

    # Markdown code blocks
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
            # If the markdown block is truncated, try to close quotes/braces
            if json_str.count('"') % 2 != 0:
                json_str += '"'
            if not json_str.endswith("}"):
                json_str += "}"
            return json.loads(json_str).get("extracted_size"), None
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
            if json_str.count('"') % 2 != 0:
                json_str += '"'
            if not json_str.endswith("}"):
                json_str += "}"
            return json.loads(json_str).get("extracted_size"), None
    except (json.JSONDecodeError, IndexError):
        pass

    # Extract JSON string regex
    try:
        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            return json.loads(json_match.group()).get("extracted_size"), None
    except json.JSONDecodeError:
        pass

    # Fallback to regex patterns
    for pattern in TYRE_SIZE_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group(), f"Extracted via pattern match (raw: {content})"

    return None, f"Could not parse response: {repr(content)}"


def run_ocr_extraction(client, b64_string, model, temp, tokens):
    # API call to extraction model
    response = client.chat.completions.create(
        model=model,
        temperature=temp,
        max_tokens=tokens,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": UPGRADED_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_string}"},
                    },
                ],
            },
        ],
    )
    content = response.choices[0].message.content
    result, warning = parse_extraction_result(content)
    return result, warning, content


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
                    "Please provide a valid OpenRouter API Key in the .env file (OPENROUTER_API_KEY)."
                )
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                batch_results = []
                client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
                total_start = time.time()

                for idx, file in enumerate(uploaded_files):
                    status_text.text(
                        f"Processing {idx + 1}/{len(uploaded_files)}: {file.name}"
                    )

                    item_start = time.time()
                    try:
                        b64_string = encode_image_to_base64(file)
                        result, warning, raw_content = run_ocr_extraction(
                            client, b64_string, QWEN_MODEL, temperature, max_tokens
                        )
                        elapsed = round(time.time() - item_start, 2)

                        status = "OK" if not warning else f"WARNING: {warning[:60]}"
                        batch_results.append(
                            {
                                "Filename": file.name,
                                "Extracted Size": str(result).upper()
                                if result
                                else "N/A",
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
                        "Please provide a valid OpenRouter API Key in the .env file (OPENROUTER_API_KEY)."
                    )
                else:
                    with st.spinner("Running OCR extraction..."):
                        try:
                            client = OpenAI(
                                api_key=api_key,
                                base_url=QWEN_BASE_URL,
                            )
                            b64_string = encode_image_to_base64(uploaded_file)

                            start_time = time.time()
                            result, warning, raw_content = run_ocr_extraction(
                                client,
                                b64_string,
                                QWEN_MODEL,
                                temperature,
                                max_tokens,
                            )
                            elapsed = round(time.time() - start_time, 2)

                            st.metric(
                                "Extracted Tyre Size",
                                str(result).upper() if result else "N/A",
                            )
                            st.metric("Response Time", f"{elapsed}s")
                            if warning:
                                st.warning(f"Parse note: {warning}")

                            with st.expander(
                                "Debug: Show raw model response", expanded=True
                            ):
                                st.code(
                                    raw_content
                                    if raw_content
                                    else "[No content returned]"
                                )

                        except Exception as e:
                            st.error(format_error_message(e))
