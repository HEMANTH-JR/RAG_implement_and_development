"""Minimal RAG pipeline: load -> chunk -> embed/store (Chroma) -> retrieve -> answer (Claude)."""
import hashlib
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DB_DIR = "chroma_db"
COLLECTION = "docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-5-5"
CHUNK_WORDS = 250
OVERLAP_WORDS = 40

_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
_db = chromadb.PersistentClient(path=DB_DIR)
_collection = _db.get_or_create_collection(COLLECTION, embedding_function=_embed_fn)


def load_documents(folder: str):
    """Yield (source_label, text) for every .txt, .md and .pdf file in folder."""
    for path in sorted(Path(folder).rglob("*")):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            yield path.name, path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
                yield f"{path.name} (p.{page_no})", page.extract_text() or ""


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS):
    """Split text into overlapping word windows."""
    words = text.split()
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + size])
        if piece.strip():
            yield piece


def index_folder(folder: str) -> int:
    """(Re)index all documents in folder. Safe to run repeatedly (upserts by chunk id)."""
    ids, docs, metas = [], [], []
    for source, text in load_documents(folder):
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(hashlib.md5(f"{source}-{i}-{chunk[:50]}".encode()).hexdigest())
            docs.append(chunk)
            metas.append({"source": source})
    for start in range(0, len(ids), 500):  # batch to stay under Chroma limits
        _collection.upsert(
            ids=ids[start:start + 500],
            documents=docs[start:start + 500],
            metadatas=metas[start:start + 500],
        )
    return len(ids)


def retrieve(question: str, k: int = 4):
    res = _collection.query(query_texts=[question], n_results=k)
    return list(zip(res["documents"][0], [m["source"] for m in res["metadatas"][0]]))


SYSTEM = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite passages like [1], [2]. If the context does not contain the answer, "
    "say you don't know based on the provided documents. Do not invent facts."
)


def answer(question: str, k: int = 4):
    """Return (answer_text, sources) where sources is a list of (source, snippet)."""
    hits = retrieve(question, k)
    if not hits:
        return "No documents are indexed yet.", []
    context = "\n\n".join(f"[{i}] ({src})\n{text}" for i, (text, src) in enumerate(hits, 1))
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    sources = [(src, text[:200] + "...") for text, src in hits]
    return msg.content[0].text, sources
