# paste the entire app code here
import os
import base64
import tempfile
from io import BytesIO

from groq import Groq
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from dotenv import load_dotenv

# ── API Setup ─────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

POPPLER_PATH = None  # None works on Linux/Colab/Streamlit Cloud

# ── Helper: Check if PDF is digital or scanned ───────────
def is_digital_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = " ".join([p.page_content for p in pages]).strip()
    return len(text) > 100  # If meaningful text found → digital

# ── Helper: Convert image to base64 ──────────────────────
def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ── RAG Pipeline for digital PDFs ────────────────────────
def build_rag_pipeline(pdf_path):
    # Load and chunk the document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    # Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ── Query RAG for follow-up questions ────────────────────
def ask_rag(vectorstore, question):
    # Retrieve relevant chunks
    relevant_chunks = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a medical assistant helping a patient understand their lab report. Answer only based on the provided context. Use simple language."
            },
            {
                "role": "user",
                "content": f"Context from medical report:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content

# ── Summarize digital PDF via RAG ────────────────────────
def summarize_digital(vectorstore):
    return ask_rag(vectorstore, 
        """Analyze this medical report and provide:
        1. A brief overall summary in simple English
        2. List of normal results
        3. List of abnormal results with what they might indicate
        4. A disclaimer to consult a doctor"""
    )

# ── Summarize scanned PDF via Vision LLM ─────────────────
def summarize_scanned(pdf_path):
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    pages = pages[:3]
    
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
    
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": image_contents}]
    )
    return response.choices[0].message.content

# ── Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="MediSumm", page_icon="🏥", layout="centered")
st.title("🏥 MediSumm")
st.subheader("AI-Powered Medical Report Summarizer")
st.write("Upload a medical report PDF and get a plain English summary with follow-up Q&A.")

uploaded_file = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])

if uploaded_file:
    st.info("Report uploaded. Click below to analyze.")

    if st.button("Analyze Report"):
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Detecting PDF type..."):
            digital = is_digital_pdf(tmp_path)

        if digital:
            st.success("Digital PDF detected — using RAG pipeline")
            with st.spinner("Building RAG pipeline..."):
                vectorstore = build_rag_pipeline(tmp_path)
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_type = "digital"

            with st.spinner("Generating summary..."):
                summary = summarize_digital(vectorstore)

        else:
            st.success("Scanned PDF detected — using Vision LLM")
            with st.spinner("Analyzing with Vision LLM..."):
                summary = summarize_scanned(tmp_path)
                st.session_state.pdf_type = "scanned"
                st.session_state.summary = summary

        st.session_state.summary = summary
        os.unlink(tmp_path)

# ── Display Summary ───────────────────────────────────────
if "summary" in st.session_state:
    st.markdown("---")
    st.markdown("### 📋 Report Summary")
    st.markdown(st.session_state.summary)
    st.markdown("---")
    st.caption("⚠️ AI-generated summary. Always consult a qualified doctor.")

    # ── Follow-up Q&A ─────────────────────────────────────
    st.markdown("### 💬 Ask a Follow-up Question")
    question = st.text_input("Ask anything about your report...")

    if question:
        if st.session_state.pdf_type == "digital" and "vectorstore" in st.session_state:
            with st.spinner("Finding answer..."):
                answer = ask_rag(st.session_state.vectorstore, question)
        else:
            # For scanned PDFs use summary as context
            with st.spinner("Finding answer..."):
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical assistant. Answer based on this report summary only."
                        },
                        {
                            "role": "user",
                            "content": f"Report summary:\n{st.session_state.summary}\n\nQuestion: {question}"
                        }
                    ]
                )
                answer = response.choices[0].message.content

        st.markdown(f"**Answer:** {answer}")
