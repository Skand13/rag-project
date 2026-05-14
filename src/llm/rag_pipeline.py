"""
Pipeline RAG bout-en-bout pour LogiStore.
Query → Retrieval hybride → Normalisation → Génération LLM
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from src.retrieval.hybrid_search import hybrid_search
from src.llm.llm_client import rag_answer

DEFAULT_MODEL     = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
TOP_K_RETRIEVAL   = 20
TOP_K_CONTEXT     = 5


def to_rag_tickets(hybrid_results: list[dict]) -> list[dict]:
    """Normalise les résultats hybrid_search au format attendu par rag_answer.

    Le champ 'text' contient 'subject | body | answer' concaténés à l'ingestion.
    """
    tickets = []
    for r in hybrid_results:
        src = r.get("_source", {})
        text = src.get("text", "")
        parts = text.split(" | ", 2)
        tickets.append({
            "subject":      parts[0].strip() if len(parts) > 0 else "",
            "body_preview": parts[1].strip()[:400] if len(parts) > 1 else text[:400],
            "answer":       parts[2].strip()[:500] if len(parts) > 2 else "",
            "language":     src.get("language", ""),
            "type":         src.get("type", ""),
            "priority":     src.get("priority", ""),
            "queue":        src.get("queue", ""),
        })
    return tickets


def run_rag(
    query: str,
    lang: str,
    model: str = DEFAULT_MODEL,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_context: int = TOP_K_CONTEXT,
    filters: dict = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    verbose: bool = False,
) -> dict:
    """Pipeline RAG complet.

    Args:
        query          : Question de l'utilisateur.
        lang           : Langue ('fr', 'en', 'de', 'es', 'pt').
        model          : Modèle LLM OpenRouter.
        top_k_retrieval: Candidats récupérés par le retrieval.
        top_k_context  : Tickets passés au LLM.
        filters        : Filtres additionnels (type, priority, queue).
        temperature    : Température de génération.
        max_tokens     : Longueur max de la réponse.
        verbose        : Affiche les étapes intermédiaires.

    Returns:
        Dict avec answer, model, lang, n_tickets, tickets, retrieval.
    """
    lang = lang.lower()

    # Merge langue + filtres additionnels
    all_filters = {"language": lang}
    if filters:
        all_filters.update(filters)

    # --- Étape 1 : Retrieval hybride ---
    if verbose:
        print(f"[1/3] Retrieval — query='{query}' | filters={all_filters}")

    retrieval_results = hybrid_search(
        query=query,
        top_k=top_k_retrieval,
        filters=all_filters,
    )

    if not retrieval_results:
        return {
            "answer":    "Aucun ticket pertinent trouvé pour cette requête.",
            "model":     model,
            "lang":      lang,
            "n_tickets": 0,
            "tickets":   [],
            "retrieval": [],
        }

    # --- Étape 2 : Normalisation ---
    top_results  = retrieval_results[:top_k_context]
    tickets_full = to_rag_tickets(top_results)

    if verbose:
        print(f"[2/3] Contexte — {len(tickets_full)} tickets")
        for t in tickets_full:
            print(f"  • [{t['language']}] {t['subject'][:60]}")

    # --- Étape 3 : Génération LLM ---
    if verbose:
        print(f"[3/3] Génération — modèle={model}")

    result = rag_answer(
        query=query,
        tickets=tickets_full,
        lang=lang,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return {
        "answer":    result["answer"],
        "model":     model,
        "lang":      lang,
        "n_tickets": len(tickets_full),
        "tickets":   tickets_full,
        "retrieval": top_results,
    }