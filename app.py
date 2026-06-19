# app.py
import os
from pathlib import Path

import streamlit as st

from config import BOOKS_DIR, LLM_MODEL, EMBED_MODEL
from query import ask


st.set_page_config(page_title="Ask Your Books", layout="wide")

st.title("Ask Your Books")
st.caption(f"Local PDF Q&A using Ollama. LLM: {LLM_MODEL} | Embeddings: {EMBED_MODEL}")

books_dir = Path(BOOKS_DIR)
books_dir.mkdir(exist_ok=True)

with st.sidebar:
    st.header("Books folder")
    st.write(f"Put PDFs in: `{BOOKS_DIR}/`")

    pdfs = sorted([p.name for p in books_dir.glob("*.pdf")])

    if pdfs:
        st.write("Detected PDFs:")
        for pdf in pdfs:
            st.write(f"- {pdf}")
    else:
        st.warning("No PDFs found yet.")

    st.markdown("---")
    st.write("After adding PDFs, run:")
    st.code("python ingest.py", language="bash")


question = st.text_input("Ask a question about your PDFs")

if st.button("Ask") and question.strip():
    with st.spinner("Searching your PDFs and asking Ollama..."):
        try:
            answer, contexts = ask(question.strip())

            st.subheader("Answer")
            st.markdown(answer)

            with st.expander("Retrieved source chunks"):
                for i, item in enumerate(contexts, start=1):
                    st.markdown(
                        f"**Source {i}: {item['book']}, page {item['page_number']}**"
                    )
                    st.write(item["text"])
                    st.markdown("---")

        except Exception as exc:
            st.error(str(exc))
            st.info(
                "Make sure Ollama is running, your embedding model is installed, "
                "and you have already run `python ingest.py`."
            )
