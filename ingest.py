# ingest.py
import os
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
import ollama

from config import (
    BOOKS_DIR,
    DB_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def embed_texts(texts):
    """Create embeddings using Ollama."""
    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
    )
    return response["embeddings"]


def ingest_pdf(pdf_path: Path, collection):
    """Extract PDF text page-by-page, chunk it, and store it with page metadata."""
    book_name = pdf_path.name
    doc = fitz.open(pdf_path)

    ids = []
    documents = []
    metadatas = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text").strip()

        if not text:
            continue

        page_number = page_index + 1
        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            documents.append(chunk)
            metadatas.append(
                {
                    "book": book_name,
                    "source": str(pdf_path),
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )

    if documents:
        embeddings = embed_texts(documents)
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"Ingested {book_name}: {len(documents)} chunks")


def main():
    books_dir = Path(BOOKS_DIR)
    books_dir.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    pdfs = sorted(books_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found. Add your PDFs to the '{BOOKS_DIR}' folder and run again.")
        return

    for pdf_path in pdfs:
        ingest_pdf(pdf_path, collection)

    print("Done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
