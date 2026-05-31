"""
embeddings.py

Handles:
1. Loading embedding model
2. Building FAISS vector store
3. Creating retriever
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = None


def get_embedding_model():
    """
    Load embedding model once and reuse it.
    """

    global EMBEDDING_MODEL

    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    return EMBEDDING_MODEL


def build_vector_store(documents):
    """
    Create FAISS vector store from document chunks.
    """

    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store


def get_retriever(vector_store, k=4):
    """
    Return top-k relevant chunks.
    """

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )