# PageWise
## Local Book PDF Q&A with Ollama

This is a starter Python application for asking questions about your PDF books locally.

It uses:

- Ollama for the local LLM
- `llama3.2` for answer generation
- `nomic-embed-text` for embeddings
- ChromaDB for local vector search
- PyMuPDF for PDF text extraction
- Streamlit for a simple UI

## 1. Install Ollama models

You said `llama3.2` is already installed.

You still need an embedding model:

```bash
ollama pull nomic-embed-text
```

Confirm your models:

```bash
ollama list
```

## 2. Install Python dependencies

From this project folder:

```bash
pip install -r requirements.txt
```

## 3. Add your PDFs

Put your PDF books inside the `books/` folder.

Example:

```text
books/
  my-book.pdf
  another-book.pdf
```

## 4. Ingest the PDFs

```bash
python ingest.py
```

This extracts text page-by-page, chunks the content, creates embeddings, and stores them in `chroma_db/`.

## 5. Run the app

```bash
streamlit run app.py
```

## 6. Ask questions

The app will retrieve relevant PDF chunks and ask `llama3.2` to answer using only those chunks.

Answers should include citations like:

```text
[Book: my-book.pdf, page 42]
```

## Notes

- For scanned PDFs, this starter app may not work because scanned PDFs need OCR.
- For better quality, increase retrieved chunks in `config.py` by changing `TOP_K`.
- If your Ollama model name is shown as `llama3.2:latest`, update `LLM_MODEL` in `config.py`.
