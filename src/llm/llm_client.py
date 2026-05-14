"""
Client LLM via OpenRouter API.
Supporte la génération de réponses RAG multilingues.
"""

import os
import httpx
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

# Prompts système par langue
SYSTEM_PROMPTS = {
    "en": "You are a helpful customer support assistant for LogiStore. Answer the user's question clearly and concisely based on the provided context tickets. If the context is insufficient, say so honestly.",
    "fr": "Tu es un assistant support client pour LogiStore. Réponds à la question de l'utilisateur de manière claire et concise en te basant sur les tickets de contexte fournis. Si le contexte est insuffisant, dis-le honnêtement.",
    "de": "Du bist ein hilfreicher Kundensupport-Assistent für LogiStore. Beantworte die Frage des Nutzers klar und präzise basierend auf den bereitgestellten Kontext-Tickets. Wenn der Kontext unzureichend ist, sage das ehrlich.",
    "es": "Eres un asistente de soporte al cliente de LogiStore. Responde la pregunta del usuario de forma clara y concisa basándote en los tickets de contexto proporcionados. Si el contexto es insuficiente, dilo honestamente.",
    "pt": "És um assistente de suporte ao cliente da LogiStore. Responde à pergunta do utilizador de forma clara e concisa com base nos tickets de contexto fornecidos. Se o contexto for insuficiente, diz-o honestamente.",
}

RAG_PROMPT_TEMPLATE = """Voici {n} tickets similaires extraits de notre base de données :

{context}

---
Question : {query}

Réponds dans la même langue que la question."""


def build_context(tickets: list[dict], max_chars: int = 3000) -> str:
    """Formate les tickets récupérés en bloc de contexte pour le prompt.

    Args:
        tickets: Liste de dicts avec au moins 'subject', 'body_preview', 'answer'.
        max_chars: Limite de caractères totaux pour éviter de dépasser la fenêtre.

    Returns:
        Bloc texte formaté.
    """
    parts = []
    total = 0
    for i, t in enumerate(tickets, 1):
        subject = t.get("subject", "").strip()
        body = t.get("body_preview", t.get("body", "")).strip()[:300]
        answer = t.get("answer", "").strip()[:400]
        block = f"[Ticket {i}]\nSujet : {subject}\nProblème : {body}\nRésolution : {answer}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_prompt(query: str, tickets: list[dict], lang: str = "fr") -> list[dict]:
    """Construit les messages (system + user) pour l'appel LLM.

    Args:
        query: Question de l'utilisateur.
        tickets: Tickets récupérés par le pipeline retrieval.
        lang: Langue détectée de la requête.

    Returns:
        Liste de messages au format OpenAI chat.
    """
    lang = lang.lower()
    system_msg = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    context = build_context(tickets)
    user_msg = RAG_PROMPT_TEMPLATE.format(
        n=len(tickets),
        context=context,
        query=query,
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def call_llm(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """Appelle l'API OpenRouter et retourne la réponse générée.

    Args:
        messages: Liste de messages chat (system + user).
        model: Identifiant du modèle OpenRouter.
        temperature: Créativité (0 = déterministe, 1 = créatif).
        max_tokens: Longueur max de la réponse.

    Returns:
        Texte de la réponse générée.

    Raises:
        ValueError: Si la clé API est absente.
        httpx.HTTPStatusError: Si l'API retourne une erreur HTTP.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY manquante dans le .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://logistore-rag.local",  # requis par OpenRouter
        "X-Title": "LogiStore RAG",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def rag_answer(
    query: str,
    tickets: list[dict],
    lang: str = "fr",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> dict:
    """Pipeline RAG complet : contexte → prompt → LLM → réponse.

    Args:
        query: Question de l'utilisateur.
        tickets: Top-K tickets récupérés par le retrieval.
        lang: Langue de la requête.
        model: Modèle LLM à utiliser.
        temperature: Température de génération.
        max_tokens: Longueur max de la réponse.

    Returns:
        Dict avec 'answer', 'model', 'n_tickets_used', 'lang'.
    """
    messages = build_prompt(query, tickets, lang=lang)
    answer = call_llm(messages, model=model, temperature=temperature, max_tokens=max_tokens)

    return {
        "answer": answer,
        "model": model,
        "n_tickets_used": len(tickets),
        "lang": lang,
    }