import os
from dotenv import load_dotenv

load_dotenv()

import os

try:
    import streamlit as st

    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- LLM Config ---
GROQ_MODEL = "llama3-groq-8b-8192-tool-use-preview"
GROQ_MODEL_SMART = "llama-3.3-70b-versatile"
# --- Embeddings Config ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector Store Config ---
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Agent Config ---
MAX_RETRIES = 2
