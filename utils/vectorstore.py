import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from config import EMBED_MODEL, FAISS_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def get_embeddings():
    """Load HuggingFace embeddings (runs locally, no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(pdf_paths: list[str] = None, urls: list[str] = None) -> list:
    """Load documents from PDFs and/or URLs."""
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if pdf_paths:
        for path in pdf_paths:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                pages = loader.load()
                docs.extend(splitter.split_documents(pages))

    if urls:
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                pages = loader.load()
                docs.extend(splitter.split_documents(pages))
            except Exception as e:
                print(f"[Warning] Could not load {url}: {e}")

    return docs


def build_vectorstore(docs: list) -> FAISS:
    """Build and save a FAISS vector store from documents."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore


def load_vectorstore() -> FAISS | None:
    """Load existing FAISS index from disk."""
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = get_embeddings()
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


def get_retriever(vectorstore: FAISS, k: int = 4):
    """Return a retriever from the vector store."""
    return vectorstore.as_retriever(search_kwargs={"k": k})
