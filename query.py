# query.py
import chromadb
import ollama

from config import DB_DIR, COLLECTION_NAME, LLM_MODEL, EMBED_MODEL, TOP_K


def embed_query(question: str):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=question,
    )
    return response["embeddings"][0]


def search_context(question: str, top_k: int = TOP_K):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    contexts = []

    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        contexts.append(
            {
                "text": doc,
                "book": metadata["book"],
                "source": metadata["source"],
                "page_number": metadata["page_number"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return contexts


def build_prompt(question: str, contexts):
    source_blocks = []

    for i, item in enumerate(contexts, start=1):
        source_blocks.append(
            f"""[Source {i}]
Book: {item["book"]}
Page: {item["page_number"]}
Chunk: {item["chunk_index"]}
Text:
{item["text"]}
"""
        )

    context_text = "\n\n".join(source_blocks)

    return f"""
You are a careful research assistant answering questions from the user's PDF book library.

Use ONLY the provided sources.
If the answer is not supported by the sources, say:
"I could not find this in the provided PDFs."

Citation rules:
- Cite page numbers inline.
- Use this citation format: [Book: filename, page X]
- At the end, include a "References" section listing the sources used.
- Do not invent page numbers or references.

Question:
{question}

Sources:
{context_text}

Answer:
""".strip()


def ask(question: str):
    contexts = search_context(question)
    prompt = build_prompt(question, contexts)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    answer = response["message"]["content"]

    return answer, contexts


if __name__ == "__main__":
    user_question = input("Ask a question: ")
    answer, _ = ask(user_question)
    print(answer)
