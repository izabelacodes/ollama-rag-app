"""
OllamaEmbeddings — wrapper LangChain para o servidor Ollama local
-----------------------------------------------------------------------------
Implementa a interface langchain_core.embeddings.Embeddings usando
llm_client.embed() e llm_client.embed_query().

Modelo padrão:
    bge-m3  — 1024 dims, multilíngue, estado-da-arte
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")


class OllamaEmbeddings(Embeddings):
    """
    LangChain Embeddings via Ollama local.

    Delega para llm_client.embed() / embed_query() — que por sua vez
    chamam POST {OLLAMA_BASE_URL}/api/embed.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or (os.getenv("EMBEDDING_MODEL") or "bge-m3")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para múltiplos textos (usado na ingestão)."""
        from llm_client import embed
        return embed(texts, model=self.model)

    def embed_query(self, text: str) -> List[float]:
        """Gera embedding para uma única query (usado na busca semântica)."""
        from llm_client import embed_query
        return embed_query(text, model=self.model)
