"""
Frontend Streamlit — RAG-time LogiStore
Interface de recherche et génération de réponses RAG multilingue.
"""

import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# --- Path setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
load_dotenv(find_dotenv(), override=True)

from src.retrieval.hybrid_search import hybrid_search
from src.llm.rag_pipeline import run_rag, to_rag_tickets

# --- Config page ---
st.set_page_config(
    page_title="RAG-time LogiStore",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS custom ---
st.markdown("""
<style>
    .answer-box {
        background: #E8F4F5;
        border-left: 4px solid #01696F;
        border-radius: 0 8px 8px 0;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.97rem;
        line-height: 1.7;
        color: #28251D !important;  /* ← force couleur texte foncée */
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")

    lang = st.selectbox(
        "Langue",
        options=["fr", "en", "de", "es", "pt"],
        format_func=lambda x: {
            "fr": "🇫🇷 Français",
            "en": "🇬🇧 English",
            "de": "🇩🇪 Deutsch",
            "es": "🇪🇸 Español",
            "pt": "🇵🇹 Português",
        }[x],
    )

    st.markdown("#### Filtres optionnels")
    ticket_type = st.selectbox(
        "Type de ticket",
        options=["(tous)", "incident", "request", "change", "problem"],
    )
    priority = st.selectbox(
        "Priorité",
        options=["(tous)", "low", "medium", "high", "critical"],
    )

    st.divider()
    st.markdown("#### Paramètres RAG")
    top_k_retrieval = st.slider("Candidats retrieval", 5, 50, 20)
    top_k_context   = st.slider("Tickets en contexte", 1, 10, 5)
    max_tokens      = st.slider("Longueur réponse (tokens)", 256, 1024, 512, step=128)
    
    AVAILABLE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "minimax/minimax-m2.5:free",
]

model = st.selectbox(
    "Modèle LLM",
    options=AVAILABLE_MODELS,
    index=0,
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 RAG-time LogiStore</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Recherche intelligente de tickets SAV multilingue · '
    'OpenSearch 2.13 + Hybrid RRF + LLM</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# Barre de recherche
# ─────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="Requête",
        placeholder="Ex: Mon colis est perdu, que faire ?",
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("Rechercher", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# Recherche & Affichage
# ─────────────────────────────────────────────
if search_btn and query.strip():

    # Construction des filtres
    filters = {}
    if ticket_type != "(tous)":
        filters["type"] = ticket_type
    if priority != "(tous)":
        filters["priority"] = priority

    # Tabs : Réponse RAG | Tickets sources
    tab_rag, tab_tickets, tab_debug = st.tabs([
        "💬 Réponse RAG",
        "📄 Tickets sources",
        "🛠 Debug"
    ])

    with st.spinner("Recherche en cours..."):
        t0 = time.time()

        # Retrieval hybride
        retrieval_results = hybrid_search(
            query=query,
            top_k=top_k_retrieval,
            filters={"language": lang, **filters},
        )
        t_retrieval = time.time() - t0

        # Tickets complets pour le LLM
        tickets_full = to_rag_tickets(retrieval_results[:top_k_context])

        # Génération RAG
        t1 = time.time()
        try:
            rag_result = run_rag(
                query=query,
                lang=lang,
                model=model,
                top_k_retrieval=top_k_retrieval,
                top_k_context=top_k_context,
                filters=filters,
                max_tokens=max_tokens,
            )
            answer = rag_result["answer"]
            rag_error = None
        except Exception as e:
            answer = None
            rag_error = str(e)

        t_rag = time.time() - t1
        t_total = time.time() - t0

    # ── Tab 1 : Réponse RAG ──
    with tab_rag:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Temps total", f"{t_total:.2f}s")
        col_m2.metric("Retrieval", f"{t_retrieval:.2f}s")
        col_m3.metric("Tickets trouvés", len(retrieval_results))

        if rag_error:
            st.error(f"Erreur LLM : {rag_error}")
            st.info("Les tickets sources sont disponibles dans l'onglet 'Tickets sources'.")
        else:
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Modèle : `{model}` · Basé sur {top_k_context} tickets")

    # ── Tab 2 : Tickets sources ──
    with tab_tickets:
        if not retrieval_results:
            st.warning("Aucun ticket trouvé pour cette requête.")
        else:
            for i, r in enumerate(retrieval_results[:top_k_context], 1):
                src = r.get("_source", {})
                text = src.get("text", "")
                parts = text.split(" | ", 2)
                subject     = parts[0][:80] if len(parts) > 0 else "—"
                body        = parts[1][:200] if len(parts) > 1 else text[:200]
                answer_text = parts[2][:300] if len(parts) > 2 else ""

                rrf_score = r.get("_rrf_score", 0)
                t_lang    = src.get("language", lang)
                t_type    = src.get("type", "—")
                t_prio    = src.get("priority", "—")

                with st.expander(f"#{i} · {subject} — RRF: {rrf_score:.4f}"):
                    st.markdown(
                        f'<span class="ticket-badge badge-lang">{t_lang.upper()}</span>'
                        f'<span class="ticket-badge badge-type">{t_type}</span>'
                        f'<span class="ticket-badge badge-prio">{t_prio}</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Problème :** {body}")
                    if answer_text:
                        st.markdown(f"**Résolution :** {answer_text}")
                    st.caption(
                        f"BM25 rank: {r.get('_bm25_rank', 'N/A')} · "
                        f"Vector rank: {r.get('_vector_rank', 'N/A')} · "
                        f"ID: {r.get('_id', '—')}"
                    )

    # ── Tab 3 : Debug ──
    with tab_debug:
        st.markdown("**Paramètres de la requête**")
        st.json({
            "query": query,
            "lang": lang,
            "filters": filters,
            "top_k_retrieval": top_k_retrieval,
            "top_k_context": top_k_context,
            "model": model,
            "max_tokens": max_tokens,
        })
        st.markdown("**Scores retrieval (top-10)**")
        import pandas as pd
        debug_rows = []
        for r in retrieval_results[:10]:
            debug_rows.append({
                "_id":          r.get("_id"),
                "rrf_score":    round(r.get("_rrf_score", 0), 5),
                "bm25_rank":    r.get("_bm25_rank", "—"),
                "vector_rank":  r.get("_vector_rank", "—"),
                "type":         r.get("_source", {}).get("type", "—"),
                "priority":     r.get("_source", {}).get("priority", "—"),
            })
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)

elif search_btn and not query.strip():
    st.warning("Saisis une requête avant de rechercher.")

else:
    # État initial
    st.markdown("---")
    st.markdown("#### Comment utiliser")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**1. Choisis ta langue**\nSélectionne la langue dans la barre latérale")
    with cols[1]:
        st.markdown("**2. Pose ta question**\nSaisis le problème du client dans le champ de recherche")
    with cols[2]:
        st.markdown("**3. Obtiens une réponse**\nLe système retrouve les tickets similaires et génère une réponse")