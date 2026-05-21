import os
import base64
from dotenv import load_dotenv
from groq import Groq
from pdf2image import convert_from_path
from io import BytesIO
import streamlit as st
import tempfile

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

import platform
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin" if platform.system() == "Windows" else None

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="MediSumm", page_icon="🏥", layout="centered")

st.title("🏥 MediSumm")
st.subheader("AI-Powered Medical Report Summarizer")
st.write("Upload a medical report PDF and get a simple, plain English summary instantly.")

# ── File uploader ─────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])

if uploaded_file:
    st.info("Report uploaded successfully. Click below to analyze.")

    if st.button("Analyze Report"):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Convert PDF to images
        with st.spinner("Reading your report..."):
            pages = convert_from_path(tmp_path, poppler_path=POPPLER_PATH)
            pages = pages[:3]  # First 3 pages

        # Convert to base64
        def image_to_base64(img):
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Build message
        image_contents = []
        for page in pages:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(page)}"
                }
            })

        image_contents.append({
            "type": "text",
            "text": """You are a medical assistant helping a patient understand their lab report.
            Analyze this medical report and provide:
            1. A brief overall summary in simple English
            2. List of normal results
            3. List of abnormal results with what they might indicate
            4. A disclaimer to consult a doctor
            
            Use simple language a non-medical person can understand."""
        })

        # Call LLM
        with st.spinner("Analyzing with AI..."):
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": image_contents}]
            )

        summary = response.choices[0].message.content

        # Display results
        st.success("Analysis complete!")
        st.markdown("---")
        st.markdown("### 📋 Report Summary")
        st.markdown(summary)
        st.markdown("---")
        st.caption("⚠️ This is an AI-generated summary for informational purposes only. Always consult a qualified doctor for medical advice.")

        # Clean up temp file
        os.unlink(tmp_path)