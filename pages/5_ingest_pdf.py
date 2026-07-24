import streamlit as st
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.pdf_ingestor import ingest_pdf, CHUNK_SIZE, CHUNK_OVERLAP
from ui_common import render_ollama_status

render_ollama_status()

st.title("📄 Ingerir PDFs")
st.caption("Faça upload de PDFs e ingira o conteúdo na coleção ChromaDB via LangChain.")

with st.expander("ℹ️ Como funciona a ingestão", expanded=False):
    st.markdown("""
    O pipeline processa cada PDF em 6 etapas:

    ```
    PDF
     └─ PyPDFLoader (LangChain)
         └─ Limpeza de texto  (remove números de pág., cabeçalhos, rodapés)
             └─ Detecção de seções/artigos
             │   Art., Capítulo, Seção, Título, §, MAIOÚsCULAS, 1.1., ...
             └─ Sub-chunking  (RecursiveCharacterTextSplitter, só se > chunk_size)
                 └─ OllamaEmbeddings  (bge-m3, local)
                     └─ ChromaDB  (distância cosine, upsert idem potente)
    ```

    Cada chunk é armazenado com metadados de **página** e **seção**.
    Re-ingerir o mesmo arquivo não cria duplicatas.
    """)

@st.cache_resource(show_spinner=False)
def _get_collection_info():
    """Retorna contagem e nome da coleção para exibição."""
    from client import get_client, get_or_create_collection
    client     = get_client()
    collection = get_or_create_collection(client)
    return collection.name, collection.count()

try:
    col_name, col_count = _get_collection_info()
    st.caption(f"Coleção: **{col_name}** — {col_count} documento(s)")
except Exception as e:
    st.warning(f"Não foi possível conectar ao ChromaDB: {e}")

# ── Configurações ─────────────────────────────────────────────────────────────
with st.expander("⚙️ Configurações de chunking", expanded=False):
    chunk_size = st.number_input(
        "Tamanho máximo do sub-chunk (caracteres)",
        min_value=200, max_value=8000,
        value=CHUNK_SIZE, step=100,
        help="Limite de caracteres por chunk. Seções menores que esse valor ficam intactas.",
    )
    chunk_overlap = st.number_input(
        "Sobreposição entre sub-chunks (caracteres)",
        min_value=0, max_value=800,
        value=CHUNK_OVERLAP, step=50,
        help="Sobreposição aplicada apenas quando uma seção é dividida.",
    )

st.divider()

# ── Upload de arquivos ────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Faça upload de um ou mais arquivos PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} arquivo(s) selecionado(s):**")
    for f in uploaded_files:
        st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

    if st.button("Ingerir no ChromaDB", type="primary"):
        progress    = st.progress(0, text="Iniciando ingestão…")
        results_log = []

        for idx, uploaded in enumerate(uploaded_files):
            progress.progress(
                idx / len(uploaded_files),
                text=f"Processando {uploaded.name}…",
            )

            # Escreve em arquivo temp para que o PyPDFLoader consiga ler
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                result = ingest_pdf(
                    tmp_path,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    original_filename=uploaded.name,  # usa o nome real em vez de tmpXXX
                )
                results_log.append(result)
            finally:
                os.unlink(tmp_path)

        progress.progress(1.0, text="Concluído!")

        # ── Resumo dos resultados ─────────────────────────────────────────────
        st.subheader("Resultado da Ingestão")
        for res in results_log:
            if res.errors:
                st.error(f"**{res.filename}** — falhou: {'; '.join(res.errors)}")
            else:
                st.success(
                    f"**{res.filename}** — {res.pages} pág., "
                    f"**{res.sections_found} seções** detectadas, "
                    f"{res.chunks_added} chunk(s) armazenado(s)."
                )

        # Recarrega a contagem
        st.cache_resource.clear()
        try:
            col_name, col_count = _get_collection_info()
            st.metric("Total de documentos na coleção", col_count)
        except Exception:
            pass
