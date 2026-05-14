"""Script d'ingestion complet : de CSV à index OpenSearch."""

import argparse
from loguru import logger

from src.ingestion.loader import load_tickets
from src.ingestion.preprocessor import preprocess_tickets
from src.chunking.chunker import chunk_dataframe
from src.embeddings.embedder import get_embedder
from src.indexing.opensearch_client import OpenSearchIndexer


def run_ingestion(csv_path: str, limit: int = None, recreate: bool = False):
    logger.info("=== Démarrage du pipeline d'ingestion ===")

    # 1. Chargement
    df = load_tickets(csv_path)
    if limit:
        df = df.head(limit)
        logger.info(f"Mode limité : {limit} tickets")

    # 2. Prétraitement
    df = preprocess_tickets(df)

    # 3. Chunking
    chunks = chunk_dataframe(df)

    # 4. Embeddings — les vecteurs sont attachés dans chunk.metadata["embedding"]
    embedder = get_embedder()
    chunks = embedder.embed_chunks(chunks)

    # 5. Indexation
    indexer = OpenSearchIndexer()
    indexer.create_index(recreate=recreate)
    indexer.index_chunks(chunks)

    # 6. Vérification
    total = indexer.count()
    logger.info(f"=== Ingestion terminée — {total} documents dans l'index ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline d'ingestion RAG-time")
    parser.add_argument(
        "--csv",
        default="data/raw/dataset-tickets-multilingual.csv",
        help="Chemin vers le fichier CSV source",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter à N tickets (test)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Supprimer et recréer l'index avant ingestion",
    )
    args = parser.parse_args()
    run_ingestion(args.csv, args.limit, args.recreate)