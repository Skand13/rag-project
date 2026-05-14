"""Client OpenSearch : connexion et indexation des chunks."""

import os
from loguru import logger
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

from src.chunking.chunker import Chunk

load_dotenv()


def get_client() -> OpenSearch:
    """Crée et retourne un client OpenSearch authentifié."""
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", 9200))
    user = os.getenv("OPENSEARCH_USER", "admin")
    password = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD")

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, password),
        use_ssl=True,
        verify_certs=False,           # Certificat auto-signé en dev
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
    )
    logger.info(f"Connecté à OpenSearch sur {host}:{port}")
    return client


class OpenSearchIndexer:
    """Gère la création de l'index et l'indexation des chunks."""

    INDEX_NAME = "logistore_tickets"
    EMBEDDING_DIM =  int(os.getenv("EMBEDDING_DIM", 384)) # text-embedding-3-small

    INDEX_MAPPING = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id":  {"type": "keyword"},
                "text":      {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
                # Metadata filtrables
                "type":     {"type": "keyword"},
                "queue":    {"type": "keyword"},
                "priority": {"type": "keyword"},
                "language": {"type": "keyword"},
                "version":  {"type": "integer"},
                "tags":     {"type": "keyword"},
            }
        },
    }

    def __init__(self):
        self.client = get_client()

    def create_index(self, recreate: bool = False) -> None:
        """Crée l'index hybride BM25 + k-NN. Supprime si recreate=True."""
        exists = self.client.indices.exists(index=self.INDEX_NAME)
        if exists:
            if recreate:
                self.client.indices.delete(index=self.INDEX_NAME)
                logger.warning(f"Index '{self.INDEX_NAME}' supprimé.")
            else:
                logger.info(f"Index '{self.INDEX_NAME}' déjà existant — skip.")
                return

        self.client.indices.create(
            index=self.INDEX_NAME,
            body=self.INDEX_MAPPING,
        )
        logger.info(f"Index '{self.INDEX_NAME}' créé avec succès.")

    def _build_actions(self, chunks: list[Chunk]) -> list[dict]:
        """Prépare les actions bulk pour OpenSearch."""
        actions = []
        for chunk in chunks:
            embedding = chunk.metadata.get("embedding")
            if embedding is None:
                logger.warning(f"Chunk {chunk.chunk_id} sans embedding — ignoré.")
                continue

            doc = {
                "_index": self.INDEX_NAME,
                "_id":    chunk.chunk_id,
                "_source": {
                    "chunk_id":  chunk.chunk_id,
                    "text":      chunk.text,
                    "embedding": embedding,
                    "type":      chunk.metadata.get("type", ""),
                    "queue":     chunk.metadata.get("queue", ""),
                    "priority":  chunk.metadata.get("priority", ""),
                    "language":  chunk.metadata.get("language", ""),
                    "version":   chunk.metadata.get("version"),
                    "tags":      chunk.metadata.get("tags", []),
                },
            }
            actions.append(doc)
        return actions

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 500) -> None:
        """Indexe les chunks par batch via l'API bulk."""
        actions = self._build_actions(chunks)
        total = len(actions)
        logger.info(f"Indexation de {total} documents...")

        success, failed = bulk(
            self.client,
            actions,
            chunk_size=batch_size,
            raise_on_error=False,
            
        )
        logger.info(f"Indexés : {success} | Échecs : {len(failed)}")
        if failed:
            logger.warning(f"Premiers échecs : {failed[:3]}")

        # Forcer le refresh pour que le count soit immédiatement à jour
        self.client.indices.refresh(index=self.INDEX_NAME)

    def count(self) -> int:
        """Retourne le nombre de documents dans l'index."""
        result = self.client.count(index=self.INDEX_NAME)
        return result["count"]