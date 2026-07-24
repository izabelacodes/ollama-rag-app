"""Cliente LLM — chama o servidor Ollama local para completar conversas e gerar embeddings."""

import json
import os
from collections.abc import Iterator

import requests
from dotenv import load_dotenv

load_dotenv()

# URL lida do .env — com fallback para o servidor local padrão do Ollama
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Janela de contexto (tokens de entrada + saída). O Ollama usa 2048 por padrão
# quando o cliente não especifica nada, o que é pouco para prompts de RAG
# (várias passagens de contexto + histórico) — sem espaço sobrando, o modelo
# termina com done_reason="length" sem gerar nenhum token de resposta.
_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))


def _wrap_text(text: str, max_length: int = 1000) -> str:
    lines = text.split()
    wrapped, current = [], ""
    for word in lines:
        if len(current) + len(word) + 1 <= max_length:
            current = f"{current} {word}" if current else word
        else:
            wrapped.append(current)
            current = word
    if current:
        wrapped.append(current)
    return "\n".join(wrapped)


def invoke(
    prompt: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 10000,
    stream: bool = False,
) -> str | None:
    """Envia uma mensagem ao modelo LLM local (Ollama) e retorna a resposta como string."""
    url = f"{_OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": _NUM_CTX,
        },
    }

    try:
        result = requests.post(url, json=payload, timeout=120)  # 2 min para completar a resposta
    except requests.exceptions.Timeout:
        raise Exception("Timeout na chamada ao LLM (>120s). Verifique se o Ollama está respondendo.")
    except requests.exceptions.ConnectionError:
        raise Exception(
            f"Não foi possível conectar ao Ollama em {_OLLAMA_BASE_URL}. "
            "Verifique se o serviço está rodando (`ollama serve`)."
        )

    if result.ok:
        content = result.json()["message"]["content"]
        return _wrap_text(content)

    print(result.text)
    raise Exception("Ocorreu um erro na chamada ao LLM")


def invoke_stream(
    prompt: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 10000,
) -> Iterator[str]:
    """Envia uma mensagem ao Ollama e produz pedaços da resposta conforme chegam (streaming)."""
    url = f"{_OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": _NUM_CTX,
        },
    }

    got_content = False
    last_chunk: dict = {}

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as result:
            if not result.ok:
                raise Exception(f"Ocorreu um erro na chamada ao LLM: {result.status_code} {result.text[:200]}")

            for line in result.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                last_chunk = chunk

                if chunk.get("error"):
                    raise Exception(f"Ollama retornou um erro: {chunk['error']}")

                content = chunk.get("message", {}).get("content", "")
                if content:
                    got_content = True
                    yield content
                if chunk.get("done"):
                    break
    except requests.exceptions.Timeout:
        raise Exception("Timeout na chamada ao LLM (>120s). Verifique se o Ollama está respondendo.")
    except requests.exceptions.ConnectionError:
        raise Exception(
            f"Não foi possível conectar ao Ollama em {_OLLAMA_BASE_URL}. "
            "Verifique se o serviço está rodando (`ollama serve`)."
        )

    if not got_content:
        reason = last_chunk.get("done_reason")
        detail = f" (motivo: {reason})" if reason else ""
        raise Exception(
            f"O modelo '{model}' não retornou nenhum conteúdo{detail}. "
            "Verifique se o modelo está instalado (`ollama list`) e os logs do `ollama serve`."
        )


def ping(timeout: float = 3.0) -> tuple[bool, list[str]]:
    """
    Verifica se o servidor Ollama está acessível.

    Retorna (online, modelos_instalados). Usado para o indicador de status na UI.
    """
    try:
        result = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=timeout)
    except requests.exceptions.RequestException:
        return False, []

    if not result.ok:
        return False, []

    models = [m["name"].split(":")[0] for m in result.json().get("models", [])]
    return True, models


# ── Modelos disponíveis localmente via Ollama ─────────────────────────────────

CHAT_MODELS = [
    "gemma4",
]

EMBEDDING_MODELS = [
    "bge-m3",
]


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed(
    texts: list[str],
    model: str = "bge-m3",
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos via Ollama local.

    Endpoint: POST {OLLAMA_BASE_URL}/api/embed

    Parâmetros:
        texts      — lista de strings a serem embeddadas
        model      — modelo de embedding (padrão: bge-m3, 1024 dims)
        batch_size — textos por requisição (evita payload muito grande)

    Retorna lista de vetores float (um por texto).
    """
    url = f"{_OLLAMA_BASE_URL}/api/embed"

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            result = requests.post(
                url,
                json={"model": model, "input": batch},
                timeout=60,  # 60s por batch
            )
        except requests.exceptions.Timeout:
            raise Exception(f"Timeout ao gerar embeddings (batch {i // batch_size + 1}).")
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Não foi possível conectar ao Ollama em {_OLLAMA_BASE_URL}. "
                "Verifique se o serviço está rodando (`ollama serve`)."
            )

        if not result.ok:
            raise Exception(f"Erro ao gerar embeddings: {result.status_code} {result.text[:200]}")

        all_embeddings.extend(result.json()["embeddings"])

    return all_embeddings


def embed_query(text: str, model: str = "bge-m3") -> list[float]:
    """
    Gera embedding para um único texto (ex.: query do usuário).

    Atalho para `embed([text], model)[0]`.
    """
    return embed([text], model=model)[0]
