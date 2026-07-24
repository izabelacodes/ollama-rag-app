import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client import get_client, get_or_create_collection, list_documents
from ui_common import render_ollama_status

render_ollama_status()

st.title("📊 Collection Info")
st.caption("Overview of the active ChromaDB collection and its stored documents.")

@st.cache_resource(show_spinner=False)
def _client():
    return get_client()

try:
    client = _client()
    collection = get_or_create_collection(client)
except Exception as e:
    st.error(f"Could not connect to ChromaDB: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Collection", collection.name)
col2.metric("Total Documents", collection.count())
col3.metric("Host", f"{os.getenv('CHROMA_HOST', 'localhost')}:{os.getenv('CHROMA_PORT', '8000')}")

st.divider()
st.subheader("Stored Documents")

if collection.count() == 0:
    st.info("The collection is empty. Add documents from the **Add Documents** page.")
else:
    data = list_documents(collection)
    rows = []
    for i, doc_id in enumerate(data["ids"]):
        rows.append({
            "ID": doc_id,
            "Document": data["documents"][i] if data.get("documents") else "",
            "Metadata": data["metadatas"][i] if data.get("metadatas") else {},
        })
    st.dataframe(rows, use_container_width=True)
