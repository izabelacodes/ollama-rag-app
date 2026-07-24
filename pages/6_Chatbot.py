import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from client import get_client, get_or_create_collection
from llm_client import CHAT_MODELS
from rag.chain import DEFAULT_MODEL, DEFAULT_N_RESULTS, Message, answer_stream
from rag.retriever import get_embeddings, get_vectorstore
from ui_common import render_ollama_status

# ── Configuração da página ────────────────────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.caption("Olá, como posso te ajudar hoje?")


# ── Conexão com o ChromaDB ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def _load_rag():
    """Carrega embeddings + vectorstore uma única vez (cacheado pelo Streamlit)."""
    get_embeddings()   # aquece o singleton
    get_vectorstore()  # conecta ao ChromaDB
    client     = get_client()
    collection = get_or_create_collection(client)
    return collection


try:
    collection = _load_rag()
except Exception as e:
    st.error(f"Não foi possível inicializar o RAG: {e}")
    st.stop()

if collection.count() == 0:
    st.warning(
        "The collection is empty. Ingest documents first via the **Ingest PDFs** "
        "or **Add Documents** pages."
    )


# ── Configurações do sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ RAG Settings")
    model = st.selectbox(
        "Modelo LLM",
        options=CHAT_MODELS,
        index=CHAT_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in CHAT_MODELS else 0,
    )
    n_results = st.slider(
        "Context chunks (k)", min_value=1, max_value=30, value=DEFAULT_N_RESULTS
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    max_tokens = st.slider(
        "Max tokens da resposta", min_value=128, max_value=4096, value=1024, step=128,
        help="Valores menores geram respostas mais rápidas em modelos locais.",
    )
    show_sources = st.toggle("Show source citations", value=True)
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Collection: **{collection.name}** ({collection.count()} docs)")

    render_ollama_status()


# ── Estado da sessão ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []   # {"role": str, "content": str, "sources": list}


# ── Helpers de renderização ───────────────────────────────────────────────────────
def _render_sources(chunks: list) -> None:
    """Renderiza a lista de fontes recuperadas em um expander."""
    with st.expander("📚 Fontes"):
        for chunk in chunks:
            label = f"`{chunk.id}`"
            if chunk.source:
                label += f" — **{chunk.source}**"
                if chunk.page is not None:
                    label += f", pág. {chunk.page}"
            if chunk.section:
                label += f" · *{chunk.section}*"
            label += f" (score {chunk.score:.3f})"
            st.markdown(label)
            st.caption(chunk.text[:300] + ("…" if len(chunk.text) > 300 else ""))


def _render_embedding_mismatch_error() -> None:
    st.error(
        "⚠️ **Embeddings incompatíveis** — os documentos na coleção foram indexados "
        "com uma função de embedding diferente da atual.\n\n"
        "**Solução:** Vá em **Delete Document** → apague toda a coleção e "
        "depois re-ingira os PDFs em **Ingerir PDFs**."
    )


# ── Exibir histórico da conversa ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])


# ── Entrada do chat ──────────────────────────────────────────────────────────────
user_query = st.chat_input("Realize perguntas baseada em sua base de conhecimento....")

if user_query:
    # Adiciona e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Constrói o histórico para a cadeia (exclui o turno atual)
    history = [
        Message(role=m["role"], content=m["content"])
        for m in st.session_state.messages[:-1]
    ]

    # Chama a cadeia RAG em modo streaming — a resposta aparece token a token
    # em vez de deixar a UI parada esperando o Ollama terminar tudo de uma vez.
    with st.chat_message("assistant"):
        sources, token_iter, error = answer_stream(
            query=user_query,
            collection=collection,
            model=model,
            n_results=n_results,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        full_answer = ""
        if error:
            if "EMBEDDING_MISMATCH" in error:
                _render_embedding_mismatch_error()
            else:
                st.error(f"Erro: {error}")
        else:
            try:
                full_answer = st.write_stream(token_iter)
            except Exception as e:
                error = str(e)
                if "EMBEDDING_MISMATCH" in error:
                    _render_embedding_mismatch_error()
                else:
                    st.error(f"Erro: {error}")

        if not full_answer and not error:
            full_answer = "Não foi possível obter resposta do LLM. Verifique se o Ollama está rodando."
            st.markdown(full_answer)

        if show_sources and sources:
            _render_sources(sources)

    # Persiste a mensagem do assistente
    st.session_state.messages.append(
        {
            "role":    "assistant",
            "content": full_answer,
            "sources": sources,
        }
    )
