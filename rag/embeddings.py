"""
Gestão de embeddings — Ollama local (bge-m3)
-------------------------------------------------
Usa OllamaEmbeddings via llm_client.embed() para gerar vetores de 1024 dims.
Modelo configurável via EMBEDDING_MODEL no .env (padrão: bge-m3).
"""

from __future__ import annotations

import os

# ── Singleton ─────────────────────────────────────────────────────────────────
_embeddings = None


def get_embeddings():
    """Retorna OllamaEmbeddings como singleton (bge-m3 ou modelo configurado em EMBEDDING_MODEL)."""
    global _embeddings
    if _embeddings is None:
        from rag.ollama_embeddings import OllamaEmbeddings
        model_name  = os.getenv("EMBEDDING_MODEL") or "bge-m3"
        _embeddings = OllamaEmbeddings(model=model_name)
        print(f"[embeddings] Ollama local: {model_name} (1024 dims)")
    return _embeddings
