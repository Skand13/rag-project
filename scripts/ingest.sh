#!/bin/bash
set -e
source .env

echo "=== Pipeline d'ingestion RAG-time ==="
echo "1. Téléchargement des données..."
python -m src.ingestion.loader

echo "2. Prétraitement et nettoyage..."
python -m src.ingestion.preprocessor

echo "3. Chunking..."
python -m src.chunking.chunker

echo "4. Génération des embeddings..."
python -m src.embeddings.embedder

echo "5. Indexation..."
python -m src.indexing.index_documents

echo "=== Ingestion terminée ==="