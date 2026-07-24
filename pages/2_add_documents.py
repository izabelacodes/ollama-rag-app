import streamlit as st
import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client import get_client, get_or_create_collection, add_documents
from ui_common import render_ollama_status

render_ollama_status()

st.title("➕ Add Documents")
st.caption("Insert one or more documents into the collection. Leave the ID blank to auto-generate one.")

@st.cache_resource(show_spinner=False)
def _client():
    return get_client()

try:
    client = _client()
    collection = get_or_create_collection(client)
except Exception as e:
    st.error(f"Could not connect to ChromaDB: {e}")
    st.stop()

# ── Formulário de documento único ────────────────────────────────────────────────
st.subheader("Single Document")

with st.form("single_doc_form"):
    doc_text = st.text_area("Document text", height=120, placeholder="Enter your document content here…")
    doc_id   = st.text_input("Document ID (optional)", placeholder="Leave blank to auto-generate")
    metadata_raw = st.text_input("Metadata (optional, JSON)", placeholder='{"source": "web", "author": "alice"}')
    submitted = st.form_submit_button("Add Document", type="primary")

if submitted:
    if not doc_text.strip():
        st.warning("Document text cannot be empty.")
    else:
        final_id = doc_id.strip() or str(uuid.uuid4())
        try:
            meta = {}
            if metadata_raw.strip():
                import json
                meta = json.loads(metadata_raw)
            collection.add(documents=[doc_text], ids=[final_id], metadatas=[meta] if meta else None)
            st.success(f"Document added with ID `{final_id}`.")
        except Exception as e:
            st.error(f"Failed to add document: {e}")

st.divider()

# ── Upload em lote via CSV ───────────────────────────────────────────────────────
st.subheader("Bulk Upload (CSV)")
st.caption("Upload a CSV file with columns: `id` (optional) and `document`.")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded:
    import pandas as pd

    df = pd.read_csv(uploaded)

    if "document" not in df.columns:
        st.error("The CSV must have a `document` column.")
    else:
        if "id" not in df.columns:
            df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        st.dataframe(df, use_container_width=True)

        if st.button("Upload to Collection", type="primary"):
            try:
                add_documents(collection, df["document"].tolist(), df["id"].tolist())
                st.success(f"Uploaded {len(df)} document(s) successfully.")
            except Exception as e:
                st.error(f"Upload failed: {e}")
