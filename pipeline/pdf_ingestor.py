"""
Pipeline de Ingestão de PDFs — LangChain + Chunking por Seções/Artigos
------------------------------------------------------------------------
PDFs
  -> PyPDFLoader          (LangChain Community)
  -> Limpeza de texto     (remove pág./cabeçalhos/rodapés, só nas bordas de cada página)
  -> Detecção de seções   (Art., Capítulo, Seção, §, MAIÚSCULAS, numeração… com merge de fragmentos)
  -> Sub-chunking         (RecursiveCharacterTextSplitter só quando > chunk_size)
  -> OllamaEmbeddings     (bge-m3, Ollama local)
  -> ChromaDB             (LangChain Chroma via HTTP, distância cosine)
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.embeddings import get_embeddings

load_dotenv()

# ── Configurações ─────────────────────────────────────────────────────────────
CHUNK_SIZE        = int(os.getenv("INGEST_CHUNK_SIZE",       "1500"))
CHUNK_OVERLAP     = int(os.getenv("INGEST_CHUNK_OVERLAP",    "200"))
MIN_SECTION_CHARS = int(os.getenv("INGEST_MIN_SECTION_CHARS", "80"))
CHROMA_HOST       = os.getenv("CHROMA_HOST",     "localhost")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME   = os.getenv("CHROMA_COLLECTION_NAME", "my_collection")

MAX_CHUNK_CHARS = CHUNK_SIZE  # compatibilidade


@dataclass
class IngestResult:
    filename:       str
    pages:          int
    chunks_added:   int
    sections_found: int = 0
    skipped_ids:    list[str] = field(default_factory=list)
    errors:         list[str] = field(default_factory=list)


# ── Detecção de seções/artigos ────────────────────────────────────────────────

_SECTION_RE = re.compile(
    r'^(?:'
    # Artigos e parágrafos legais
    r'Art\.?\s*\d+[°ºo]?\.?'
    r'|ARTIGO\s+\d+'
    r'|Artigo\s+\d+'
    r'|§\s*\d+[°ºo]?'
    r'|Parágrafo\s+(?:[Úú]nico|\d+)'
    # Estrutura de documentos
    r'|CAP[ÍI]TULO\s+[IVXLCDM\d]+'
    r'|Cap[ií]tulo\s+[IVXLCDM\d]+'
    r'|T[ÍI]TULO\s+[IVXLCDM\d]+'
    r'|T[íi]tulo\s+[IVXLCDM\d]+'
    r'|SE[ÇC][ÃA]O\s+[IVXLCDM\d]+'
    r'|Se[çc][ãa]o\s+[IVXLCDM\d]+'
    r'|SUBSE[ÇC][ÃA]O\s+[IVXLCDM\d]+'
    r'|Subse[çc][ãa]o\s+[IVXLCDM\d]+'
    r'|PARTE\s+[IVXLCDM\d]+'
    r'|Parte\s+[IVXLCDM\d]+'
    r'|ANEXO\s*[IVXLCDM\d]*'
    r'|Anexo\s+[IVXLCDM\d]+'
    r'|AP[ÊE]NDICE\s+[IVXLCDM\d]+'
    r'|Ap[êe]ndice\s+[IVXLCDM\d]+'
    # Numeração hierárquica: 1.  /  1.1.  /  1.1.1.
    r'|\d+\.\d+(?:\.\d+)*\.?\s+\S'
    r'|\d+\.\s+[A-ZÁÀÃÂÉÊÍÓÔÕÚÇ]'
    # Linha inteira em MAIÚSCULAS (título de seção)
    r'|[A-ZÁÀÃÂÉÊÍÓÔÕÚÇ][A-ZÁÀÃÂÉÊÍÓÔÕÚÇ\s\-\/]{3,}$'
    r')',
    re.MULTILINE,
)


def _is_section_header(line: str) -> bool:
    """Retorna True se a linha é um cabeçalho de seção."""
    s = line.strip()
    return bool(_SECTION_RE.match(s)) and 2 <= len(s) <= 150


# ── Limpeza de texto ──────────────────────────────────────────────────────────

_PAGE_EDGE_LINES = 2  # nº de linhas (não-vazias) checadas no topo/rodapé de cada página


def _clean_pdf_pages(pages: list[str]) -> list[str]:
    """
    Remove artefatos comuns de PDFs — números de página isolados e
    cabeçalhos/rodapés curtos em maiúsculas — mas SOMENTE nas bordas
    (topo/rodapé) de cada página, onde esses artefatos realmente aparecem.

    Antes a limpeza rodava sobre o texto inteiro, o que apagava qualquer
    linha curta que fosse só um número ou só maiúsculas em qualquer lugar
    do documento — inclusive valores legítimos no meio de tabelas (ex.:
    o pypdf costuma extrair cada célula de uma tabela em sua própria
    linha, então um total como "220" numa "Quadro de Vagas" era
    indistinguível de um número de página e acabava sendo descartado
    antes mesmo de virar embedding).
    """
    cleaned_pages: list[str] = []

    for text in pages:
        lines = text.splitlines()
        non_blank_idxs = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idxs = set(non_blank_idxs[:_PAGE_EDGE_LINES]) | set(non_blank_idxs[-_PAGE_EDGE_LINES:])

        cleaned = []
        for i, line in enumerate(lines):
            s = line.strip()
            if i in edge_idxs:
                # Número de página isolado (1–4 dígitos)
                if re.match(r'^\d{1,4}$', s):
                    continue
                # Cabeçalho/rodapé: ≤ 2 palavras, todo maiúsculo, < 30 chars
                if s.isupper() and len(s.split()) <= 2 and len(s) < 30:
                    continue
            cleaned.append(line)

        cleaned_pages.append("\n".join(cleaned))

    return cleaned_pages


def _collapse_blank_lines(text: str) -> str:
    """Colapsa 3+ linhas em branco seguidas para 2."""
    return re.sub(r'\n{3,}', '\n\n', text).strip()


# ── Divisão por seções ────────────────────────────────────────────────────────

def _get_page_at_offset(offset: int, page_offsets: list[tuple[int, int]]) -> int:
    """Retorna o número de página para um dado offset de caractere."""
    for i in range(len(page_offsets) - 1, -1, -1):
        if offset >= page_offsets[i][0]:
            return page_offsets[i][1]
    return 1


def _merge_fragmented_sections(sections: list[dict]) -> list[dict]:
    """
    Funde seções muito curtas (< MIN_SECTION_CHARS) à seção seguinte.

    O detector de seções trata qualquer linha isolada em MAIÚSCULAS como um
    novo cabeçalho — o que é ótimo para "CAPÍTULO II", mas em tabelas (ex.:
    "CARGO VAGAS LOCALIZAÇÃO") cada linha maiúscula vira um "cabeçalho" novo,
    fatiando a tabela em vários fragmentos quase vazios e desconexos. Fundir
    esses fragmentos minúsculos preserva a coesão dos dados tabulares.
    """
    if len(sections) <= 1:
        return sections

    merged: list[dict] = [sections[0]]
    for sec in sections[1:]:
        prev = merged[-1]
        if len(prev["text"]) < MIN_SECTION_CHARS:
            merged[-1] = {
                **sec,
                "text":    prev["text"] + "\n" + sec["text"],
                "section": prev["section"] or sec["section"],
                "page":    prev["page"],
            }
        else:
            merged.append(sec)

    return merged


def _split_into_sections(
    text: str,
    page_offsets: list[tuple[int, int]],
) -> list[dict]:
    """
    Divide o texto do documento por cabeçalhos de seção detectados.

    Retorna lista de dicts: {'text', 'section', 'page'}.
    Se nenhuma seção for detectada, retorna o documento inteiro como única seção.
    """
    # Coletar cabeçalhos de seção com seus offsets
    header_matches: list[tuple[int, str]] = []
    for m in re.finditer(r'^.+$', text, re.MULTILINE):
        if _is_section_header(m.group(0)):
            header_matches.append((m.start(), m.group(0).strip()))

    if not header_matches:
        return [{"text": text.strip(), "section": "", "page": _get_page_at_offset(0, page_offsets)}]

    sections: list[dict] = []

    # Conteúdo anterior à primeira seção
    pre = text[: header_matches[0][0]].strip()
    if pre:
        sections.append({"text": pre, "section": "", "page": _get_page_at_offset(0, page_offsets)})

    # Seções detectadas
    for i, (start, header) in enumerate(header_matches):
        end     = header_matches[i + 1][0] if i + 1 < len(header_matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append({
                "text":    content,
                "section": header,
                "page":    _get_page_at_offset(start, page_offsets),
            })

    return _merge_fragmented_sections(sections)


# ── Vectorstore LangChain ─────────────────────────────────────────────────────

_vectorstore: Chroma | None = None


def get_vectorstore(embeddings=None) -> Chroma:
    """Retorna o vectorstore LangChain (singleton — criado apenas uma vez)."""
    global _vectorstore
    if _vectorstore is None:
        client       = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings or get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vectorstore


def _chunk_id(filename: str, index: int) -> str:
    """ID determinístico e resistente a colisões para um chunk."""
    raw = f"{filename}::i{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ── Pipeline principal ────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path:      str | Path,
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size:    int = 100,
    collection=None,
    max_chunk_chars: int | None = None,
    overlap: int | None = None,
    original_filename: str | None = None,
) -> IngestResult:
    """
    Pipeline de ingestão com chunking hierárquico:

      1. PyPDFLoader          — carrega o PDF página a página
      2. Limpeza de texto     — remove artefatos de PDF (restrita às bordas da página)
      3. Detecção de seções   — divide por Art., Capítulo, §, MAIÚSCULAS…
                                 (fragmentos minúsculos são fundidos de volta)
      4. Sub-chunking         — RecursiveCharacterTextSplitter se seção > chunk_size
      5. OllamaEmbeddings     — gera vetores de embedding (bge-m3)
      6. ChromaDB (upsert)    — armazena com metadados de seção e página (idempotente)
    """
    if max_chunk_chars is not None:
        chunk_size = max_chunk_chars

    pdf_path = Path(pdf_path)
    display_name = original_filename or pdf_path.name
    result   = IngestResult(filename=display_name, pages=0, chunks_added=0)

    try:
        # 1. Carregar PDF
        loader   = PyPDFLoader(str(pdf_path))
        raw_docs = loader.load()
        result.pages = len(raw_docs)

        # 2. Limpar cada página individualmente (limpeza ciente de borda de página)
        cleaned_pages = _clean_pdf_pages([doc.page_content for doc in raw_docs])

        # 3. Construir texto completo com mapa de offsets de página
        page_offsets: list[tuple[int, int]] = []
        full_text = ""
        for doc, cleaned in zip(raw_docs, cleaned_pages):
            page_offsets.append((len(full_text), doc.metadata.get("page", 0) + 1))
            full_text += cleaned + "\n\n"
        full_text = _collapse_blank_lines(full_text)

        # 4. Dividir por seções
        sections = _split_into_sections(full_text, page_offsets)
        result.sections_found = len(sections)

        # 5. Sub-chunking das seções maiores que chunk_size
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            length_function=len,
        )

        final_chunks: list[dict] = []
        for section in sections:
            if len(section["text"]) <= chunk_size:
                final_chunks.append(section)
            else:
                for sub in splitter.split_text(section["text"]):
                    final_chunks.append({**section, "text": sub})

        # 6. Preparar textos, IDs e metadados
        texts     = [c["text"] for c in final_chunks]
        ids       = [_chunk_id(display_name, i) for i in range(len(texts))]
        metadatas = [
            {
                "source":       display_name,
                "page":         c["page"],
                "total_pages":  result.pages,
                "section":      c["section"],
                "chunk_index":  i,
                "doc_id":       ids[i],
            }
            for i, c in enumerate(final_chunks)
        ]

        # 7. Gerar embeddings e fazer upsert no ChromaDB
        emb         = get_embeddings()
        vectorstore = get_vectorstore(emb)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids   = ids[i : i + batch_size]
            batch_meta  = metadatas[i : i + batch_size]
            batch_vecs  = emb.embed_documents(batch_texts)

            vectorstore._collection.upsert(
                documents=batch_texts,
                embeddings=batch_vecs,
                ids=batch_ids,
                metadatas=batch_meta,
            )
            result.chunks_added += len(batch_texts)

    except Exception as exc:
        result.errors.append(str(exc))

    return result


def ingest_multiple_pdfs(
    pdf_paths:     list[str | Path],
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[IngestResult]:
    """Ingere uma lista de PDFs e retorna um IngestResult por arquivo."""
    return [ingest_pdf(p, chunk_size=chunk_size, chunk_overlap=chunk_overlap) for p in pdf_paths]


# ── Ponto de entrada CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    if len(sys.argv) < 2:
        print("Uso: python -m pipeline.pdf_ingestor <arquivo1.pdf> [arquivo2.pdf ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        res = ingest_pdf(path)
        if res.errors:
            print(f"[ERRO] {res.filename}: {res.errors}")
        else:
            print(
                f"[OK] {res.filename} — {res.pages} pág., "
                f"{res.sections_found} seções, {res.chunks_added} chunks."
            )
