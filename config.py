# config.py

BOOKS_DIR = "books"
DB_DIR = "chroma_db"
COLLECTION_NAME = "book_pages"

# You said you already have llama3.2 installed.
LLM_MODEL = "llama3.2:latest"

# Install this with:
#   ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 6
