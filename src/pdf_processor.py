"""
pdf_processor.py

Handles:
1. PDF text extraction
2. Text chunking for RAG pipeline
"""

import PyPDF2

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        file_path (str): Path to PDF

    Returns:
        str: Complete extracted text
    """

    text = ""

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def chunk_text(text: str):
    """
    Split text into chunks.

    Args:
        text (str): Extracted PDF text

    Returns:
        List[Document]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]

    return documents