"""Moteur de recherche hybride BM25 + vectoriel avec fusion RRF."""

import os
from loguru import logger
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection

load_dotenv()

INDEX_NAME = "logistore_tickets"
RRF_K = 60  # Constante de lissage RRF (valeur empirique recommandée)
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Client OpenSearch
# ---------------------------------------------------------------------------

def get_client() -> OpenSearch:
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", 9200))
    user = os.getenv("OPENSEARCH_USER", "admin")
    password = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD")

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, password),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
    )


# ---------------------------------------------------------------------------
# Embedder (réutilise le backend configuré dans .env)
# ---------------------------------------------------------------------------

def get_query_embedding(query: str) -> list[float]:
    """Encode la requête avec le même backend que l'ingestion."""
    backend = os.getenv("EMBEDDING_BACKEND", "local")

    if backend == "local":
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        return model.encode([query])[0].tolist()

    # OpenRouter
    import httpx
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    response = httpx.post(
        f"{os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')}/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model_name, "input": [query]},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Recherche BM25
# ---------------------------------------------------------------------------

def bm25_search(
    client: OpenSearch,
    query: str,
    top_k: int = 20,
    filters: dict = None,
) -> list[dict]:
    """
    Recherche full-text BM25 sur le champ 'text'.
    filters : dict optionnel, ex. {"type": "incident", "priority": "high"}
    """
    must_clauses = [{"match": {"text": {"query": query, "operator": "or"}}}]
    filter_clauses = []

    if filters:
        for field, value in filters.items():
            if value:
                filter_clauses.append({"term": {field: value.lower()}})

    body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses,
            }
        },
        "_source": {"excludes": ["embedding"]},
    }

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]
    logger.debug(f"BM25 → {len(hits)} résultats")
    return hits


# ---------------------------------------------------------------------------
# Recherche vectorielle k-NN
# ---------------------------------------------------------------------------

def vector_search(
    client: OpenSearch,
    query_vector: list[float],
    top_k: int = 20,
    filters: dict = None,
) -> list[dict]:
    """
    Recherche par similarité cosinus sur le champ 'embedding'.
    filters : dict optionnel, ex. {"language": "en"}
    """
    filter_clauses = []
    if filters:
        for field, value in filters.items():
            if value:
                filter_clauses.append({"term": {field: value.lower()}})

    knn_query = {
        "vector": query_vector,
        "k": top_k,
    }
    if filter_clauses:
        knn_query["filter"] = {"bool": {"filter": filter_clauses}}

    body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": knn_query
            }
        },
        "_source": {"excludes": ["embedding"]},
    }

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]
    logger.debug(f"Vector → {len(hits)} résultats")
    return hits


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    bm25_hits: list[dict],
    vector_hits: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Fusionne deux listes de résultats OpenSearch via RRF.
    score_rrf(d) = Σ 1 / (k + rank(d))
    Retourne une liste triée par score RRF décroissant.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(bm25_hits):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        docs[doc_id] = hit
        # Conserver le score BM25 brut dans la source
        docs[doc_id]["_bm25_score"] = hit["_score"]
        docs[doc_id]["_bm25_rank"] = rank + 1

    for rank, hit in enumerate(vector_hits):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        if doc_id not in docs:
            docs[doc_id] = hit
        docs[doc_id]["_vector_score"] = hit["_score"]
        docs[doc_id]["_vector_rank"] = rank + 1

    # Tri par score RRF décroissant
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, rrf_score in ranked:
        doc = docs[doc_id]
        doc["_rrf_score"] = round(rrf_score, 6)
        results.append(doc)

    return results


# ---------------------------------------------------------------------------
# Interface principale
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict = None,
    bm25_candidates: int = 20,
    vector_candidates: int = 20,
) -> list[dict]:
    """
    Recherche hybride complète : BM25 + vectoriel + RRF.

    Paramètres
    ----------
    query           : texte de la requête utilisateur
    top_k           : nombre de résultats finaux à retourner
    filters         : dict de filtres metadata (type, queue, priority, language)
    bm25_candidates : nombre de candidats BM25 avant fusion
    vector_candidates : nombre de candidats vectoriels avant fusion

    Retourne
    --------
    Liste de dicts avec les champs :
      _id, _rrf_score, _bm25_score, _bm25_rank,
      _vector_score, _vector_rank, _source
    """
    logger.info(f"Recherche hybride : '{query}' | filters={filters} | top_k={top_k}")

    client = get_client()

    # 1. Embedding de la requête
    query_vector = get_query_embedding(query)

    # 2. BM25
    bm25_hits = bm25_search(client, query, top_k=bm25_candidates, filters=filters)

    # 3. Vectoriel
    vector_hits = vector_search(client, query_vector, top_k=vector_candidates, filters=filters)

    # 4. Fusion RRF
    fused = reciprocal_rank_fusion(bm25_hits, vector_hits)

    # 5. Retourner les top_k
    final = fused[:top_k]

    logger.info(f"Résultats retournés : {len(final)}")
    for i, hit in enumerate(final):
        logger.debug(
            f"  [{i+1}] id={hit['_id']} | rrf={hit['_rrf_score']} "
            f"| bm25_rank={hit.get('_bm25_rank', '-')} "
            f"| vec_rank={hit.get('_vector_rank', '-')}"
        )

    return final


def format_results(hits: list[dict]) -> list[dict]:
    """
    Formate les résultats pour l'affichage (Streamlit ou CLI).
    Retourne une liste de dicts propres sans les champs internes OpenSearch.
    """
    formatted = []
    for hit in hits:
        source = hit.get("_source", {})
        formatted.append({
            "id":           hit["_id"],
            "rrf_score":    hit.get("_rrf_score", 0),
            "bm25_rank":    hit.get("_bm25_rank", "-"),
            "vector_rank":  hit.get("_vector_rank", "-"),
            "subject":      source.get("subject", ""),
            "text":         source.get("text", ""),
            "type":         source.get("type", ""),
            "queue":        source.get("queue", ""),
            "priority":     source.get("priority", ""),
            "language":     source.get("language", ""),
            "tags":         source.get("tags", []),
        })
    return formatted


# ---------------------------------------------------------------------------
# Test CLI rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "product defective not working"
    results = hybrid_search(query, top_k=5)
    formatted = format_results(results)

    print(f"\n=== Résultats pour : '{query}' ===\n")
    for i, r in enumerate(formatted, 1):
        print(f"[{i}] {r['subject']}")
        print(f"     Score RRF : {r['rrf_score']} | BM25 rank : {r['bm25_rank']} | Vec rank : {r['vector_rank']}")
        print(f"     Type : {r['type']} | Queue : {r['queue']} | Priority : {r['priority']}")
        print(f"     Tags : {', '.join(r['tags']) if r['tags'] else '-'}")
        print()