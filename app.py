import streamlit as st

st.set_page_config(
    page_title="ChromaDB Manager",
    page_icon="🔵",
    layout="wide",
)

pages = {
    "ChromaDB": [
        st.Page("pages/1_collection_info.py", title="Collection Info", icon="📊"),
        st.Page("pages/2_add_documents.py",   title="Add Documents",   icon="➕"),
        st.Page("pages/3_query.py",            title="Query",           icon="🔍"),
        st.Page("pages/4_delete_document.py",  title="Delete Document", icon="🗑️"),
        st.Page("pages/5_ingest_pdf.py",       title="Ingest PDFs",     icon="📄"),
        st.Page("pages/6_Chatbot.py",          title="RAG Chatbot",     icon="🤖"),
        st.Page("pages/7_LLM.py",              title="Chat LLM",        icon="💬"),
    ]
}

pg = st.navigation(pages)
pg.run()
