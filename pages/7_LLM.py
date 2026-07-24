import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from llm_client import CHAT_MODELS, invoke_stream
from rag.chain import DEFAULT_MODEL, Message
from ui_common import render_ollama_status

# ── Configuração da página ────────────────────────────────────────────────────────────────────
st.title("💬 Chat LLM")
st.caption("Converse diretamente com o modelo — sem busca em base documental (sem RAG).")


# ── Configurações do sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ LLM Settings")
    model = st.selectbox(
        "Modelo LLM",
        options=CHAT_MODELS,
        index=CHAT_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in CHAT_MODELS else 0,
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    max_tokens = st.slider(
        "Max tokens da resposta", min_value=128, max_value=4096, value=1024, step=128,
        help="Valores menores geram respostas mais rápidas em modelos locais.",
    )
    if st.button("🗑️ Clear conversation"):
        st.session_state.llm_messages = []
        st.rerun()

    st.divider()
    render_ollama_status()


# ── Estado da sessão ─────────────────────────────────────────────────────────────
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages: list[dict] = []   # {"role": str, "content": str}


MAX_HISTORY_TURNS = 6  # limita o histórico injetado no prompt para não estourar num_ctx


def _build_prompt(query: str, history: list[Message]) -> str:
    """Monta o prompt com o histórico da conversa — sem instruções de RAG/contexto."""
    if not history:
        return query

    recent = history[-MAX_HISTORY_TURNS:]
    turns = [
        f"{'Usuário' if m.role == 'user' else 'Assistente'}: {m.content}"
        for m in recent
    ]
    return "Conversa anterior:\n" + "\n".join(turns) + f"\n\nUsuário: {query}\nAssistente:"


# ── Exibir histórico da conversa ──────────────────────────────────────────────────
for msg in st.session_state.llm_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Entrada do chat ──────────────────────────────────────────────────────────────
user_query = st.chat_input("Converse com o modelo...")

if user_query:
    st.session_state.llm_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    history = [
        Message(role=m["role"], content=m["content"])
        for m in st.session_state.llm_messages[:-1]
    ]
    prompt = _build_prompt(user_query, history)

    # Streaming direto via invoke_stream — sem retrieval, sem ChromaDB.
    with st.chat_message("assistant"):
        full_answer = ""
        error = None
        try:
            token_iter = invoke_stream(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            full_answer = st.write_stream(token_iter)
        except Exception as e:
            error = str(e)
            st.error(f"Erro: {error}")

        if not full_answer and not error:
            full_answer = "Não foi possível obter resposta do LLM. Verifique se o Ollama está rodando."
            st.markdown(full_answer)

    st.session_state.llm_messages.append({"role": "assistant", "content": full_answer})
