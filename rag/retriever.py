"""
Recuperador RAG — LangChain
---------------------------------
Usa langchain_chroma.Chroma para busca por similaridade semântica.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import chromadb
from langchain_chroma import Chroma

from rag.embeddings import get_embeddings  # modelo local gerenciado centralmente

CHROMA_HOST     = os.getenv("CHROMA_HOST",     "localhost")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my_collection")


@dataclass
class Chunk:
    """Um único chunk de contexto recuperado."""
    id:       str
    text:     str
    score:    float          # similaridade 0–1 (maior = mais relevante)
    source:   str = ""
    page:     int | None = None
    section:  str = ""
    keywords: str = ""


# ── Vectorstore LangChain ─────────────────────────────────────────────────────

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """Retorna o vectorstore LangChain (singleton — criado apenas uma vez)."""
    global _vectorstore
    if _vectorstore is None:
        client       = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            # Distância cosine é a correta para sentence-transformers (vetores normalizados)
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vectorstore


# ── API pública ───────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    collection=None,   # mantido para compatibilidade (ignorado)
    n_results: int = 5,
) -> list[Chunk]:
    """
    Busca os chunks mais relevantes usando LangChain similarity search.
    O vectorstore é criado internamente a partir das variáveis de ambiente.

    Retorna lista vazia se a coleção estiver vazia.
    Levanta EmbeddingMismatchError se detectar embedding incompatível.
    """
    vectorstore = get_vectorstore()

    try:
        total = vectorstore._collection.count()
        if total == 0:
            return []

        safe_k  = min(n_results, total)
        # similarity_search_with_score retorna distância bruta (sem validação)
        raw_results = vectorstore.similarity_search_with_score(query, k=safe_k)
    except Exception:
        return []

    # Com distância cosine: score = cosine_distance (0=idêntico, 2=oposto)
    # Similaridade = 1 - cosine_distance (resultado em [-1, 1], mas na prática [0, 1])
    scores_converted = [max(0.0, 1.0 - float(s)) for _, s in raw_results]

    # Detectar mismatch real: scores todos zerados é sinal de embeddings incompatíveis
    # (com cosine correto, pelo menos 1 resultado deve ter score > 0.01)
    if raw_results and len(raw_results) >= 3 and all(s < 0.01 for s in scores_converted):
        raise ValueError(
            "EMBEDDING_MISMATCH: os documentos foram indexados com uma função de "
            "embedding diferente da atual. Re-ingira os PDFs para corrigir."
        )

    chunks: list[Chunk] = []
    for (doc, raw_score), score in zip(raw_results, scores_converted):
        meta = doc.metadata or {}
        chunks.append(Chunk(
            id=meta.get("doc_id", ""),
            text=doc.page_content,
            score=round(score, 4),
            source=meta.get("source", ""),
            page=meta.get("page"),
            section=meta.get("section", ""),
            keywords=meta.get("keywords", ""),
        ))

    return chunks


def format_context(chunks: list[Chunk]) -> str:
    """
    Renderiza os chunks como bloco de contexto para injeção no prompt.
    Cada chunk é numerado e inclui fonte, página e seção quando disponíveis.
    """
    if not chunks:
        return "(Nenhum contexto relevante encontrado.)"

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[{i}]"
        if chunk.source:
            label += f" {chunk.source}"
            if chunk.page is not None:
                label += f", pág. {chunk.page}"
        if chunk.section:
            label += f" — {chunk.section}"
        parts.append(f"{label}\n{chunk.text}")

    return "\n\n---\n\n".join(parts)
