"""Utilitaires de gestion du cycle de vie de l'index OpenSearch."""

from loguru import logger
from src.indexing.opensearch_client import OpenSearchIndexer


def recreate_index() -> None:
    """Supprime et recrée l'index depuis zéro."""
    indexer = OpenSearchIndexer()
    indexer.create_index(recreate=True)
    logger.info("Index recréé proprement.")


def check_index_health() -> dict:
    """Retourne l'état de l'index (nombre de docs, statut)."""
    indexer = OpenSearchIndexer()
    client = indexer.client

    count = indexer.count()
    health = client.cluster.health(index=indexer.INDEX_NAME)

    report = {
        "index":   indexer.INDEX_NAME,
        "docs":    count,
        "status":  health.get("status"),
        "shards":  health.get("active_shards"),
    }
    logger.info(f"État index : {report}")
    return report