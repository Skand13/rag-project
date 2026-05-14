"""Module de génération d'embeddings — API OpenRouter ou local."""

import os
import time
import httpx
from loguru import logger
from dotenv import load_dotenv
from src.chunking.chunker import Chunk

load_dotenv()

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openrouter")  # "openrouter" ou "local"


class LocalEmbedder:
    """Embeddings locaux via sentence-transformers (gratuit, sans API)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.dim = 384
        logger.info(f"Modèle local chargé : {model} (dim={self.dim})")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        texts = [chunk.text for chunk in chunks]
        logger.info(f"Embedding local de {len(texts)} textes...")
        embeddings = self.embed_batch(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata["embedding"] = embedding
        logger.info(f"Total embeddings générés : {len(embeddings)}")
        return chunks


class OpenRouterEmbedder:
    """Embeddings via API OpenRouter."""

    def __init__(self, model: str = None, batch_size: int = 100):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = model or os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self.batch_size = batch_size
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY non définie dans .env")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/logistore-rag-time",
        }
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        texts = [chunk.text for chunk in chunks]
        total_batches = (len(texts) - 1) // self.batch_size + 1
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            logger.info(f"Embedding batch {i // self.batch_size + 1}/{total_batches}...")
            try:
                all_embeddings.extend(self.embed_batch(batch))
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Erreur batch {i} : {e}")
                raise
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.metadata["embedding"] = embedding
        logger.info(f"Total embeddings générés : {len(all_embeddings)}")
        return chunks


def get_embedder():
    """Factory : retourne le bon embedder selon EMBEDDING_BACKEND."""
    if EMBEDDING_BACKEND == "local":
        return LocalEmbedder(model=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    return OpenRouterEmbedder()