"""
Cadeia RAG
----------
Usa LangChain PromptTemplate para montar o prompt e chama o Ollama local
diretamente via llm_client.invoke() — sem LCEL intermediário.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.prompts import PromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_client import invoke, invoke_stream
from rag.retriever import Chunk, format_context, retrieve

# ── Padrões (sobrescritíveis via variáveis de ambiente) ───────────────────────
DEFAULT_MODEL      = os.getenv("LLM_MODEL",       "gemma4")
DEFAULT_N_RESULTS  = int(os.getenv("RAG_N_RESULTS", "8"))

# Limites de tamanho do prompt — sem isso, um n_results alto (a UI permite até 30)
# ou uma conversa longa (o histórico crescia sem limite a cada turno) podiam
# facilmente estourar a janela de contexto do Ollama (num_ctx), fazendo o
# modelo terminar com done_reason="length" sem gerar nenhum token de resposta.
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
MAX_HISTORY_TURNS = int(os.getenv("RAG_MAX_HISTORY_TURNS", "6"))

# ── Template de prompt ────────────────────────────────────────────────────────
# _PROMPT_TEMPLATE = PromptTemplate.from_template(
#     "Você é um assistente prestativo. Responda à pergunta do usuário exclusivamente "
#     "com base nas passagens de contexto fornecidas abaixo. Se o contexto não contiver "
#     "informações suficientes, diga isso claramente — não invente fatos.\n\n"
#     "Ao referenciar informações, mencione o rótulo da fonte "
#     "(ex.: \"[1] arquivo.pdf, pág. 3\") para que o usuário saiba de onde vieram.\n\n"
#     "Contexto:\n{context}\n\n"
#     "{history}"
#     "Pergunta: {question}\n"
#     "Resposta:"
# )

from langchain_core.prompts import PromptTemplate

_PROMPT_TEMPLATE = PromptTemplate.from_template(
"""
Você é um assistente especializado em responder perguntas utilizando exclusivamente as informações recuperadas de uma base documental indexada no ChromaDB.

Instruções:

- Utilize SOMENTE as informações presentes no CONTEXTO abaixo.
- Não utilize conhecimento próprio, memória ou informações externas.
- Se a resposta não estiver explicitamente presente no contexto, responda:
  "Não encontrei informações suficientes na base documental para responder a essa pergunta."
- Caso existam informações parcialmente relacionadas, informe isso claramente e indique suas limitações.
- Quando a resposta estiver distribuída em vários trechos do contexto, combine essas informações de forma coerente.
- Nunca invente nomes, datas, valores, cargos, processos ou documentos.
- Sempre cite a(s) fonte(s) utilizadas ao final da resposta, utilizando o formato informado no contexto (por exemplo: [1] Portaria.pdf, pág. 3).
- Responda em português, de forma objetiva e organizada.
- Se existirem divergências entre documentos, informe todas elas e indique as respectivas fontes.

========================
CONTEXTO RECUPERADO
========================

{context}

========================
HISTÓRICO DA CONVERSA
========================

{history}

========================
PERGUNTA
========================

{question}

========================
RESPOSTA
========================
"""
)

# ── Tipos de dados ────────────────────────────────────────────────────────────

@dataclass
class Message:
    role:    str   # "user" | "assistant"
    content: str


@dataclass
class RAGResponse:
    answer:  str
    sources: list[Chunk] = field(default_factory=list)
    error:   str | None = None


# ── Montagem do prompt (compartilhada entre answer() e answer_stream()) ───────

def _select_within_budget(chunks: list[Chunk], max_chars: int) -> list[Chunk]:
    """
    Seleciona, na ordem de relevância em que chegaram (mais similar primeiro),
    quantos chunks INTEIROS cabem no orçamento de caracteres.

    Corta pela lista de chunks, nunca no meio do texto de um chunk — um corte
    cego de string podia partir um trecho no meio da frase (ou da própria
    citação da fonte), entregando contexto malformado ao LLM. Sempre inclui
    ao menos 1 chunk, mesmo que ele sozinho exceda o orçamento.
    """
    selected: list[Chunk] = []
    total = 0
    for c in chunks:
        block_len = len(c.text) + 80  # aproxima o overhead do rótulo/fonte formatado
        if selected and total + block_len > max_chars:
            break
        selected.append(c)
        total += block_len
    return selected


def _build_prompt(
    query:      str,
    collection: Any,
    n_results:  int,
    history:    list[Message] | None,
) -> tuple[list[Chunk], str]:
    """Recupera os chunks relevantes e monta o prompt final. Retorna (chunks usados, prompt)."""
    all_chunks = retrieve(query, collection, n_results=n_results)
    chunks     = _select_within_budget(all_chunks, MAX_CONTEXT_CHARS)
    context    = format_context(chunks)
    if len(chunks) < len(all_chunks):
        omitted  = len(all_chunks) - len(chunks)
        context += f"\n\n[...{omitted} trecho(s) adicional(is) omitido(s) por limite de tamanho de contexto...]"

    history_block = ""
    if history:
        # Mantém só as últimas MAX_HISTORY_TURNS mensagens — sem isso o histórico
        # cresce a cada turno e eventualmente estoura a janela de contexto sozinho.
        recent = history[-MAX_HISTORY_TURNS:]
        turns = [
            f"{'Usuário' if m.role == 'user' else 'Assistente'}: {m.content}"
            for m in recent
        ]
        history_block = "Conversa anterior:\n" + "\n".join(turns) + "\n\n"

    prompt_str = _PROMPT_TEMPLATE.format(
        context=context,
        history=history_block,
        question=query,
    )
    return chunks, prompt_str


# ── API pública ───────────────────────────────────────────────────────────────

def answer(
    query:       str,
    collection:  Any = None,   # mantido para compatibilidade (ignorado)
    model:       str = DEFAULT_MODEL,
    n_results:   int = DEFAULT_N_RESULTS,
    history:     list[Message] | None = None,
    temperature: float = 0.3,
    max_tokens:  int   = 2048,
) -> RAGResponse:
    """
    Pipeline RAG completo (sem streaming):
      1. _build_prompt()     — busca semântica + montagem do prompt
      2. llm_client.invoke() — chama o Ollama local e espera a resposta completa
    """
    try:
        chunks, prompt_str = _build_prompt(query, collection, n_results, history)

        llm_answer = invoke(
            prompt=prompt_str,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not llm_answer:
            return RAGResponse(
                answer="Não foi possível obter resposta do LLM. Verifique se o Ollama está rodando.",
                sources=chunks,
                error="O LLM retornou vazio.",
            )

        return RAGResponse(answer=llm_answer.strip(), sources=chunks)

    except Exception as exc:
        return RAGResponse(
            answer="Ocorreu um erro ao gerar a resposta.",
            sources=[],
            error=str(exc),
        )


def answer_stream(
    query:       str,
    collection:  Any = None,   # mantido para compatibilidade (ignorado)
    model:       str = DEFAULT_MODEL,
    n_results:   int = DEFAULT_N_RESULTS,
    history:     list[Message] | None = None,
    temperature: float = 0.3,
    max_tokens:  int   = 2048,
) -> tuple[list[Chunk], Iterator[str], str | None]:
    """
    Versão em streaming de answer() — pensada para st.write_stream() no Streamlit,
    já que a espera pela resposta completa do Ollama (stream=False) faz a UI
    parecer travada em modelos locais mais lentos.

    Retorna (sources, token_iterator, error). Se `error` não for None, a busca
    de contexto falhou e `token_iterator` estará vazio. Erros levantados durante
    a geração (ex.: Ollama caiu no meio do streaming) surgem ao iterar
    `token_iterator` — quem consome deve envolver a iteração em try/except.
    """
    try:
        chunks, prompt_str = _build_prompt(query, collection, n_results, history)
    except Exception as exc:
        return [], iter(()), str(exc)

    token_iter = invoke_stream(
        prompt=prompt_str,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return chunks, token_iter, None

