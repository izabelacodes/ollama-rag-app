import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my_collection")


def get_client() -> chromadb.HttpClient:
    """Retorna um cliente HTTP do ChromaDB conectado ao servidor local."""
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def get_or_create_collection(client: chromadb.HttpClient) -> chromadb.Collection:
    """Obtém uma coleção existente ou cria uma nova se não existir."""
    return client.get_or_create_collection(name=COLLECTION_NAME)


def add_documents(collection: chromadb.Collection, documents: list[str], ids: list[str]) -> None:
    """Adiciona documentos à coleção usando embeddings do backend configurado."""
    from rag.embeddings import get_embeddings
    emb        = get_embeddings()
    embeddings = emb.embed_documents(documents)
    collection.add(documents=documents, ids=ids, embeddings=embeddings)
    print(f"Added {len(documents)} document(s) to '{COLLECTION_NAME}'.")


def query_collection(collection: chromadb.Collection, query_texts: list[str], n_results: int = 3) -> dict:
    """Consulta a coleção usando embeddings gerados pelo backend configurado (Ollama local)."""
    from rag.embeddings import get_embeddings
    emb        = get_embeddings()
    query_vecs = emb.embed_documents(query_texts)
    results    = collection.query(query_embeddings=query_vecs, n_results=n_results)
    return results


def list_documents(collection: chromadb.Collection) -> dict:
    """Retorna todos os documentos armazenados na coleção."""
    return collection.get()


def delete_document(collection: chromadb.Collection, doc_id: str) -> None:
    """Remove um único documento pelo seu ID."""
    collection.delete(ids=[doc_id])


def main() -> None:
    client = get_client()
    print(f"Connected to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")

    collection = get_or_create_collection(client)
    print(f"Using collection: '{COLLECTION_NAME}' (total docs: {collection.count()})")

    # --- Exemplo: adicionar documentos ---
    sample_docs = [
        "ChromaDB is an open-source embedding database.",
        "It lets you store and query vector embeddings easily.",
        "You can run ChromaDB locally or in the cloud.",
    ]
    sample_ids = ["doc1", "doc2", "doc3"]
    add_documents(collection, sample_docs, sample_ids)

    # --- Exemplo: consulta ---
    query = ["What is ChromaDB?"]
    results = query_collection(collection, query, n_results=2)

    print("\nQuery results:")
    for doc, distance in zip(results["documents"][0], results["distances"][0]):
        print(f"  [{distance:.4f}] {doc}")


if __name__ == "__main__":
    main()
