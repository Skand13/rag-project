import uuid
import time
import json
import pandas as pd
import ollama

from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


CSV_FILE = r"C:\Users\sofiahp\Desktop\Projet-rag\archiveclean_tickets.csv"
QDRANT_PATH = r"C:\Users\sofiahp\Desktop\Projet-rag\qdrant_data_test_100"
METRICS_FILE = r"C:\Users\sofiahp\Desktop\Projet-rag\indexation_metrics_100.csv"

COLLECTION_NAME = "tickets_rag_test_100"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
VECTOR_SIZE = 1024
BATCH_SIZE = 10
LIMIT_TICKETS = 100


qdrant_client = QdrantClient(path=QDRANT_PATH)


def create_collection():
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )


def get_embedding(text):
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )

    return response["embedding"]


def prepare_payload(row):
    return {
        "ticket_id": int(row["ticket_id"]),
        "raw_text": str(row["raw_text"]),
        "clean_text": str(row["clean_text"]),
        "language": str(row["language"]),
        "intent": str(row["intent"]),
        "priority": str(row["priority"]),
        "text_length": int(row["text_length"]),
        "source_file": str(row["source_file"])
    }


def save_metrics(metrics):
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_FILE, index=False, encoding="utf-8")


def vectorize_csv():
    if not Path(CSV_FILE).exists():
        raise FileNotFoundError(f"CSV introuvable : {CSV_FILE}")

    start_time = time.time()

    print("Chargement CSV...")
    df = pd.read_csv(CSV_FILE)

    total_before_cleaning = len(df)

    df = df.dropna(subset=["clean_text"])
    df = df[df["clean_text"].astype(str).str.len() > 10]
    df = df.drop_duplicates(subset=["clean_text"])
    df = df.reset_index(drop=True)

    total_after_cleaning = len(df)

    df = df.head(LIMIT_TICKETS).reset_index(drop=True)

    print(f"{total_after_cleaning} tickets propres détectés")
    print(f"Test limité à {len(df)} tickets")

    create_collection()

    indexed_count = 0
    failed_count = 0
    embedding_times = []

    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[start:start + BATCH_SIZE]
        points = []

        for _, row in batch_df.iterrows():
            try:
                text = str(row["clean_text"])

                embedding_start = time.time()
                embedding = get_embedding(text)
                embedding_end = time.time()

                embedding_times.append(embedding_end - embedding_start)

                if len(embedding) != VECTOR_SIZE:
                    raise ValueError(
                        f"Taille vecteur incorrecte : {len(embedding)} au lieu de {VECTOR_SIZE}"
                    )

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=prepare_payload(row)
                )

                points.append(point)
                indexed_count += 1

            except Exception as e:
                failed_count += 1
                print(f"Erreur ticket : {e}")

        if points:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

    total_time = time.time() - start_time
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)

    metrics = {
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "vector_size": VECTOR_SIZE,
        "limit_tickets": LIMIT_TICKETS,
        "csv_file": CSV_FILE,
        "qdrant_path": QDRANT_PATH,
        "total_rows_before_cleaning": total_before_cleaning,
        "total_rows_after_cleaning": total_after_cleaning,
        "indexed_tickets": indexed_count,
        "failed_tickets": failed_count,
        "success_rate": round(indexed_count / len(df), 4),
        "failure_rate": round(failed_count / len(df), 4),
        "total_indexation_time_seconds": round(total_time, 2),
        "average_embedding_time_seconds": round(
            sum(embedding_times) / len(embedding_times), 4
        ) if embedding_times else 0,
        "tickets_per_second": round(indexed_count / total_time, 2),
        "qdrant_points_count": collection_info.points_count,
        "language_distribution": json.dumps(
            df["language"].value_counts().to_dict(),
            ensure_ascii=False
        ),
        "intent_distribution": json.dumps(
            df["intent"].value_counts().to_dict(),
            ensure_ascii=False
        ),
        "priority_distribution": json.dumps(
            df["priority"].value_counts().to_dict(),
            ensure_ascii=False
        )
    }

    save_metrics(metrics)

    print("\nVectorisation test terminée")
    print(f"Tickets indexés : {indexed_count}")
    print(f"Tickets échoués : {failed_count}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Base locale : {QDRANT_PATH}")
    print(f"Métriques : {METRICS_FILE}")


if __name__ == "__main__":
    vectorize_csv()