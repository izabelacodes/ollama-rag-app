import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client import get_client, get_or_create_collection, query_collection
from ui_common import render_ollama_status

render_ollama_status()

st.title("🔍 Query Collection")
st.caption("Search for semantically similar documents using natural language.")

@st.cache_resource(show_spinner=False)
def _client():
    return get_client()

try:
    client = _client()
    collection = get_or_create_collection(client)
except Exception as e:
    st.error(f"Could not connect to ChromaDB: {e}")
    st.stop()

if collection.count() == 0:
    st.info("The collection is empty. Add documents from the **Add Documents** page first.")
    st.stop()

with st.form("query_form"):
    query_text = st.text_area("Query text", height=100, placeholder="Type your search query…")
    n_results  = st.slider("Number of results", min_value=1, max_value=min(20, collection.count()), value=3)
    run_query  = st.form_submit_button("Search", type="primary")

if run_query:
    if not query_text.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching…"):
            results = query_collection(collection, [query_text], n_results=n_results)

        docs      = results["documents"][0]
        distances = results["distances"][0]
        ids       = results["ids"][0]
        metadatas = results.get("metadatas", [[]])[0]

        st.subheader(f"{len(docs)} result(s)")

        for rank, (doc_id, doc, dist, meta) in enumerate(zip(ids, docs, distances, metadatas), start=1):
            similarity = 1 - dist  # distância cosseno → similaridade
            with st.container(border=True):
                header_col, score_col = st.columns([4, 1])
                header_col.markdown(f"**#{rank} — `{doc_id}`**")
                score_col.metric("Similarity", f"{similarity:.4f}")
                st.write(doc)
                if meta:
                    with st.expander("Metadata"):
                        st.json(meta)
