import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client import get_client, get_or_create_collection, list_documents, delete_document
from ui_common import render_ollama_status

render_ollama_status()

st.title("🗑️ Remover Documento")
st.caption("Remova um documento da coleção pelo seu ID, ou apague todos os documentos de uma vez.")

@st.cache_resource(show_spinner=False)
def _client():
    return get_client()

try:
    client = _client()
    collection = get_or_create_collection(client)
except Exception as e:
    st.error(f"Não foi possível conectar ao ChromaDB: {e}")
    st.stop()

if collection.count() == 0:
    st.info("A coleção está vazia — nada para remover.")
    st.stop()

# ── Selecionar entre os IDs existentes ───────────────────────────────────────────
data         = list_documents(collection)
existing_ids = data["ids"]

tab_single, tab_all = st.tabs(["🗑️ Remover documento", "💣 Apagar toda a coleção"])

# ────────────────────────────────────────────────────────────────
# Aba 1 — Remover documento individual
# ────────────────────────────────────────────────────────────────
with tab_single:
    st.subheader("Selecionar Documento")
    selected_id = st.selectbox("ID do documento", options=existing_ids)

    if selected_id:
        idx  = existing_ids.index(selected_id)
        doc  = data["documents"][idx] if data.get("documents") else "(sem conteúdo)"
        meta = data["metadatas"][idx] if data.get("metadatas") else {}

        with st.container(border=True):
            st.markdown(f"**ID:** `{selected_id}`")
            st.write(doc)
            if meta:
                with st.expander("Metadados"):
                    st.json(meta)

        st.divider()
        st.subheader("Remover manualmente por ID")
        manual_id = st.text_input("Ou digite um ID diretamente", placeholder="id-do-documento-aqui")

        target_id = manual_id.strip() if manual_id.strip() else selected_id

        confirm = st.checkbox(f"Confirmo que desejo remover `{target_id}`")

        if st.button("Remover Documento", type="primary", disabled=not confirm):
            try:
                delete_document(collection, target_id)
                st.success(f"Documento `{target_id}` removido com sucesso.")
                st.rerun()
            except Exception as e:
                st.error(f"Falha ao remover: {e}")

# ────────────────────────────────────────────────────────────────
# Aba 2 — Apagar toda a coleção
# ────────────────────────────────────────────────────────────────
with tab_all:
    total = collection.count()
    st.warning(
        f"Esta ação irá remover **todos os {total} documento(s)** "
        f"da coleção **`{collection.name}`**. Esta operação é irreversível."
    )

    st.divider()

    # Confirmação dupla
    st.subheader("Confirmação dupla obrigatória")

    confirm1 = st.checkbox(
        f"Entendo que todos os {total} documento(s) serão permanentemente removidos."
    )
    confirm2 = st.checkbox(
        "Confirmo que desejo apagar toda a coleção e não há cópias de segurança necessárias.",
        disabled=not confirm1,
    )

    if st.button(
        "💣 Apagar todos os documentos",
        type="primary",
        disabled=not (confirm1 and confirm2),
    ):
        try:
            all_ids = data["ids"]
            collection.delete(ids=all_ids)
            st.success(f"{len(all_ids)} documento(s) removido(s) com sucesso da coleção `{collection.name}`.")
            st.rerun()
        except Exception as e:
            st.error(f"Falha ao apagar a coleção: {e}")

