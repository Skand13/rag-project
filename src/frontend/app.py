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
        color: #28251D !important;
    }

    /* Bloc métriques principal */
    .metrics-bar {
        background: #F0FAFA;
        border: 1px solid #B2D8D8;
        border-radius: 10px;
        padding: 1rem 1.5rem 0.75rem 1.5rem;
        margin: 1rem 0 1.5rem 0;
    }
    .metrics-bar-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #01696F;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.6rem;
    }

    /* Indicateur qualité RRF */
    .quality-good  { color: #437A22; font-weight: 700; }
    .quality-mid   { color: #964219; font-weight: 700; }
    .quality-low   { color: #A12C7B; font-weight: 700; }

    /* Badge latence */
    .latency-ok   { color: #437A22; font-weight: 600; }
    .latency-slow { color: #964219; font-weight: 600; }
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

    st.divider()
    st.markdown("#### À propos")
    st.caption(
        "OpenSearch 2.13 · BM25 + kNN HNSW · RRF k=60\n"
        "Embedding : MiniLM-L12-v2 (dim=384, local)"
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("## 🔍 RAG-time LogiStore")
st.caption(
    "Recherche intelligente de tickets SAV multilingue · "
    "OpenSearch 2.13 + Hybrid RRF + LLM"
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
# Helper : qualité du score RRF
# ─────────────────────────────────────────────
def rrf_quality_label(score: float) -> str:
    """
    RRF k=60 : score max théorique ≈ 1/61 ≈ 0.0164 (rang 1 BM25 + rang 1 kNN).
    On ramène ça sur une échelle lisible.
    """
    if score >= 0.013:
        return "🟢 Haute"
    elif score >= 0.008:
        return "🟡 Moyenne"
    else:
        return "🔴 Faible"

def rrf_quality_css(score: float) -> str:
    if score >= 0.013:
        return "quality-good"
    elif score >= 0.008:
        return "quality-mid"
    else:
        return "quality-low"

def latency_css(seconds: float, threshold: float) -> str:
    return "latency-ok" if seconds < threshold else "latency-slow"


# ─────────────────────────────────────────────
# Recherche & Affichage
# ─────────────────────────────────────────────
if search_btn and query.strip():

    filters = {}
    if ticket_type != "(tous)":
        filters["type"] = ticket_type
    if priority != "(tous)":
        filters["priority"] = priority

    with st.spinner("Recherche en cours…"):
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

        t_llm   = time.time() - t1
        t_total = time.time() - t0

    # ── Calcul métriques ──────────────────────────────────────────────
    top_rrf     = retrieval_results[0].get("_rrf_score", 0) if retrieval_results else 0
    avg_rrf     = (
        sum(r.get("_rrf_score", 0) for r in retrieval_results[:top_k_context])
        / top_k_context
        if retrieval_results else 0
    )
    n_found     = len(retrieval_results)
    # Proportion des top-k_context tickets ayant un rank BM25 ET vector (vrais hybrides)
    n_hybrid    = sum(
        1 for r in retrieval_results[:top_k_context]
        if r.get("_bm25_rank") not in (None, "—") and r.get("_vector_rank") not in (None, "—")
    )

    # ── Bloc métriques global (affiché au-dessus des onglets) ─────────
    st.markdown("---")
    st.markdown(
        '<div class="metrics-bar-title">📊 Métriques de la requête</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        label="⏱ Temps total",
        value=f"{t_total:.2f}s",
        delta=None,
        help="Retrieval + génération LLM",
    )
    c2.metric(
        label="🔎 Retrieval",
        value=f"{t_retrieval:.2f}s",
        help="Temps de la requête hybride OpenSearch (BM25 + kNN + RRF)",
    )
    c3.metric(
        label="🤖 Génération LLM",
        value=f"{t_llm:.2f}s" if not rag_error else "Erreur",
        help=f"Modèle : {model}",
    )
    c4.metric(
        label="📄 Tickets trouvés",
        value=n_found,
        help=f"{top_k_context} utilisés en contexte LLM sur {n_found} candidats",
    )
    c5.metric(
        label="🏆 Score RRF top-1",
        value=f"{top_rrf:.5f}",
        delta=f"moy top-{top_k_context}: {avg_rrf:.5f}",
        help="RRF k=60 · max théorique ≈ 0.0164 (rang 1 BM25 + rang 1 kNN)",
    )
    c6.metric(
        label="⚡ Qualité retrieval",
        value=rrf_quality_label(top_rrf),
        help=(
            "🟢 Haute  : score RRF ≥ 0.013\n"
            "🟡 Moyenne: score RRF ≥ 0.008\n"
            "🔴 Faible : score RRF < 0.008"
        ),
    )

    # Ligne secondaire — détails techniques
    with st.expander("Détails techniques des métriques", expanded=False):
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric(
            "Tickets hybrides (BM25 ∩ kNN)",
            f"{n_hybrid} / {top_k_context}",
            help="Tickets présents à la fois dans les résultats BM25 et kNN avant fusion RRF",
        )
        dc2.metric(
            "Score RRF moyen (contexte)",
            f"{avg_rrf:.5f}",
        )
        dc3.metric(
            "Candidats retrieval",
            top_k_retrieval,
            help="Nombre de tickets candidats demandés à OpenSearch avant reranking RRF",
        )
        dc4.metric(
            "Tokens max réponse",
            max_tokens,
        )

        # Interprétation automatique
        st.markdown("**Interprétation automatique**")
        interpretations = []

        if top_rrf >= 0.013:
            interpretations.append(
                "✅ **Retrieval de haute qualité** — le ticket le plus pertinent est "
                "bien classé à la fois en BM25 et en kNN."
            )
        elif top_rrf >= 0.008:
            interpretations.append(
                "⚠️ **Retrieval moyen** — les résultats sont pertinents mais la requête "
                "pourrait bénéficier d'une reformulation ou d'un re-ranker cross-encoder."
            )
        else:
            interpretations.append(
                "🔴 **Retrieval faible** — peu de correspondance lexicale ET sémantique. "
                "Vérifier la langue sélectionnée ou reformuler la requête."
            )

        if n_hybrid < top_k_context // 2:
            interpretations.append(
                f"⚠️ Seulement **{n_hybrid}/{top_k_context} tickets hybrides** — "
                "la majorité des tickets remontés ne sont présents que dans un seul index "
                "(BM25 ou kNN), pas les deux. La fusion RRF est moins efficace."
            )

        if t_retrieval > 2.0:
            interpretations.append(
                f"🐢 **Retrieval lent** ({t_retrieval:.2f}s) — OpenSearch pourrait être "
                "sous charge ou le cluster n'est pas optimisé (1 nœud Docker)."
            )

        if t_llm > 15.0:
            interpretations.append(
                f"🐢 **LLM lent** ({t_llm:.2f}s) — latence OpenRouter variable "
                "(modèle gratuit sans SLA)."
            )

        if rag_error:
            interpretations.append(
                f"❌ **Erreur LLM** — la génération a échoué : `{rag_error}`"
            )

        for msg in interpretations:
            st.markdown(msg)

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_rag, tab_tickets, tab_debug = st.tabs([
        "💬 Réponse RAG",
        "📄 Tickets sources",
        "🛠 Debug",
    ])

    # ── Tab 1 : Réponse RAG ──────────────────────────────────────────
    with tab_rag:
        if rag_error:
            st.error(f"Erreur LLM : {rag_error}")
            st.info("Les tickets sources sont disponibles dans l'onglet 'Tickets sources'.")
        else:
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Modèle : `{model}` · "
                f"Basé sur {top_k_context} tickets · "
                f"Généré en {t_llm:.1f}s"
            )

    # ── Tab 2 : Tickets sources ──────────────────────────────────────
    with tab_tickets:
        if not retrieval_results:
            st.warning("Aucun ticket trouvé pour cette requête.")
        else:
            for i, r in enumerate(retrieval_results[:top_k_context], 1):
                src   = r.get("_source", {})
                text  = src.get("text", "")
                parts = text.split(" | ", 2)
                subject     = parts[0][:80]  if len(parts) > 0 else "—"
                body        = parts[1][:200] if len(parts) > 1 else text[:200]
                answer_text = parts[2][:300] if len(parts) > 2 else ""

                rrf_score   = r.get("_rrf_score", 0)
                t_lang      = src.get("language", lang)
                t_type      = src.get("type", "—")
                t_prio      = src.get("priority", "—")

                # Couleur indicateur qualité par ticket
                quality     = rrf_quality_label(rrf_score)

                with st.expander(
                    f"#{i} · {subject} · RRF: {rrf_score:.5f} {quality}"
                ):
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("Langue", t_lang.upper())
                    tc2.metric("Type",   t_type)
                    tc3.metric("Priorité", t_prio)
                    tc4.metric("Score RRF", f"{rrf_score:.5f}")

                    st.markdown(f"**Problème :** {body}")
                    if answer_text:
                        st.markdown(f"**Résolution :** {answer_text}")

                    st.caption(
                        f"BM25 rank: {r.get('_bm25_rank', 'N/A')} · "
                        f"Vector rank: {r.get('_vector_rank', 'N/A')} · "
                        f"ID: {r.get('_id', '—')}"
                    )

    # ── Tab 3 : Debug ────────────────────────────────────────────────
    with tab_debug:
        st.markdown("**Paramètres de la requête**")
        st.json({
            "query":           query,
            "lang":            lang,
            "filters":         filters,
            "top_k_retrieval": top_k_retrieval,
            "top_k_context":   top_k_context,
            "model":           model,
            "max_tokens":      max_tokens,
        })

        st.markdown("**Scores retrieval (top-10)**")
        import pandas as pd
        debug_rows = []
        for r in retrieval_results[:10]:
            rrf = r.get("_rrf_score", 0)
            debug_rows.append({
                "_id":         r.get("_id"),
                "rrf_score":   round(rrf, 6),
                "bm25_rank":   r.get("_bm25_rank", "—"),
                "vector_rank": r.get("_vector_rank", "—"),
                "qualite":     rrf_quality_label(rrf),
                "type":        r.get("_source", {}).get("type", "—"),
                "priority":    r.get("_source", {}).get("priority", "—"),
                "langue":      r.get("_source", {}).get("language", "—"),
            })
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)

        st.markdown("**Résumé des timings**")
        st.json({
            "t_retrieval_s": round(t_retrieval, 3),
            "t_llm_s":       round(t_llm, 3),
            "t_total_s":     round(t_total, 3),
        })

elif search_btn and not query.strip():
    st.warning("Saisis une requête avant de rechercher.")

else:
    st.markdown("---")
    st.markdown("#### Comment utiliser")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**1. Choisis ta langue**\nSélectionne la langue dans la barre latérale")
    with cols[1]:
        st.markdown("**2. Pose ta question**\nSaisis le problème du client dans le champ de recherche")
    with cols[2]:
        st.markdown("**3. Obtiens une réponse**\nLe système retrouve les tickets similaires et génère une réponse")
