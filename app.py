import time
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ollama

from pathlib import Path
from qdrant_client import QdrantClient


QDRANT_PATH = r"C:\Users\sofiahp\Desktop\Projet-rag\qdrant_data_test_100"
METRICS_FILE = r"C:\Users\sofiahp\Desktop\Projet-rag\indexation_metrics_100.csv"

COLLECTION_NAME = "tickets_rag_test_100"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
GENERATION_MODEL = "mistral:7b"

DEFAULT_TOP_K = 5


st.set_page_config(
    page_title="RAG Ticket Center",
    page_icon="🤖",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #ECFEFF 45%, #EFF6FF 100%);
    color: #0F172A;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F1F5F9 100%);
    border-right: 1px solid #E2E8F0;
}

.main-header {
    background: linear-gradient(135deg, #1ABC9C, #2ECC71, #3498DB);
    padding: 32px;
    border-radius: 28px;
    color: white;
    box-shadow: 0 18px 45px rgba(52, 152, 219, 0.25);
    margin-bottom: 28px;
}

.main-title {
    font-size: 42px;
    font-weight: 900;
    margin: 0;
}

.main-subtitle {
    font-size: 17px;
    opacity: 0.95;
    margin-top: 8px;
}

.card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #E2E8F0;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    border-radius: 22px;
    padding: 22px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
}

.metric-title {
    font-size: 14px;
    color: #64748B;
    font-weight: 700;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 34px;
    color: #0F172A;
    font-weight: 900;
}

.metric-help {
    font-size: 12px;
    color: #94A3B8;
}

.chat-answer {
    background: linear-gradient(135deg, #FFFFFF, #F0FDFA);
    border-left: 6px solid #1ABC9C;
    border-radius: 24px;
    padding: 26px;
    box-shadow: 0 14px 35px rgba(26, 188, 156, 0.12);
}

.ticket-card {
    background: white;
    border-radius: 22px;
    padding: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    margin-bottom: 14px;
}

.ticket-title {
    font-size: 18px;
    font-weight: 800;
    color: #0F172A;
}

.badge {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    margin-right: 6px;
}

.badge-turquoise {
    background: #CCFBF1;
    color: #0F766E;
}

.badge-green {
    background: #DCFCE7;
    color: #166534;
}

.badge-blue {
    background: #DBEAFE;
    color: #1D4ED8;
}

.badge-red {
    background: #FEE2E2;
    color: #B91C1C;
}

.badge-orange {
    background: #FFEDD5;
    color: #C2410C;
}

.section-title {
    font-size: 24px;
    font-weight: 900;
    color: #0F172A;
    margin-bottom: 12px;
}

.small-muted {
    color: #64748B;
    font-size: 14px;
}

div.stButton > button {
    background: linear-gradient(135deg, #1ABC9C, #3498DB);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 12px 24px;
    font-weight: 800;
    box-shadow: 0 10px 22px rgba(52, 152, 219, 0.25);
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #16A085, #2980B9);
    color: white;
}

textarea {
    border-radius: 18px !important;
}

[data-testid="stMetricValue"] {
    color: #0F172A;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_qdrant():
    return QdrantClient(path=QDRANT_PATH)


@st.cache_data
def load_metrics_file():
    if Path(METRICS_FILE).exists():
        return pd.read_csv(METRICS_FILE)
    return pd.DataFrame()


def get_embedding(text):
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )
    return response["embedding"]


def search_qdrant(query, top_k):
    query_vector = get_embedding(query)

    return qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )


def build_context(results):
    blocks = []

    for i, result in enumerate(results, start=1):
        p = result.payload

        blocks.append(f"""
Document {i}
Score similarité : {round(result.score, 4)}
Ticket ID : {p.get("ticket_id")}
Langue : {p.get("language")}
Intention : {p.get("intent")}
Priorité : {p.get("priority")}
Texte : {p.get("clean_text")}
""")

    return "\n".join(blocks)


def generate_answer(question, context):
    prompt = f"""
Tu es un assistant support IT basé sur un système RAG.

Règles :
- Réponds uniquement avec le contexte fourni.
- Ne mens jamais.
- Si le contexte est insuffisant, dis que tu ne peux pas confirmer.
- Justifie rapidement avec les tickets retrouvés.
- Réponse claire, structurée et utile.

Contexte :
{context}

Question :
{question}

Réponse :
"""

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Tu es un assistant RAG fiable spécialisé dans l’analyse de tickets support."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def compute_proxy_metrics(results, latency):
    if not results:
        return {
            "trust_score": 0,
            "context_relevance": 0,
            "context_recall_proxy": 0,
            "precision_proxy": 0,
            "map_proxy": 0,
            "accuracy_proxy": 0,
            "reliability": 0,
            "max_similarity": 0,
            "avg_similarity": 0,
            "latency": latency
        }

    scores = [float(r.score) for r in results]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)

    relevant_docs = [s for s in scores if s >= 0.45]
    very_relevant_docs = [s for s in scores if s >= 0.60]

    precision_proxy = len(relevant_docs) / len(scores)
    context_recall_proxy = min(len(relevant_docs) / 3, 1)
    context_relevance = avg_score
    accuracy_proxy = (0.55 * max_score) + (0.45 * avg_score)

    average_precision_sum = 0
    relevant_seen = 0

    for rank, score in enumerate(scores, start=1):
        if score >= 0.45:
            relevant_seen += 1
            average_precision_sum += relevant_seen / rank

    map_proxy = average_precision_sum / max(len(relevant_docs), 1)

    reliability = (
        0.35 * context_relevance +
        0.25 * precision_proxy +
        0.20 * context_recall_proxy +
        0.20 * accuracy_proxy
    )

    trust_score = (
        0.40 * reliability +
        0.30 * max_score +
        0.20 * map_proxy +
        0.10 * min(len(very_relevant_docs) / 2, 1)
    )

    return {
        "trust_score": round(trust_score, 3),
        "context_relevance": round(context_relevance, 3),
        "context_recall_proxy": round(context_recall_proxy, 3),
        "precision_proxy": round(precision_proxy, 3),
        "map_proxy": round(map_proxy, 3),
        "accuracy_proxy": round(accuracy_proxy, 3),
        "reliability": round(reliability, 3),
        "max_similarity": round(max_score, 3),
        "avg_similarity": round(avg_score, 3),
        "latency": round(latency, 2)
    }


def metric_card(title, value, help_text, color="#1ABC9C"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-help">{help_text}</div>
    </div>
    """, unsafe_allow_html=True)


qdrant_client = load_qdrant()
index_metrics_df = load_metrics_file()


if "results" not in st.session_state:
    st.session_state.results = []

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "runtime_metrics" not in st.session_state:
    st.session_state.runtime_metrics = {}

if "question" not in st.session_state:
    st.session_state.question = ""


st.sidebar.markdown("## 🤖 RAG Center")
st.sidebar.markdown("Analyse intelligente des tickets support")

top_k = st.sidebar.slider(
    "Tickets récupérés",
    min_value=1,
    max_value=10,
    value=DEFAULT_TOP_K
)

page = st.sidebar.radio(
    "Navigation",
    [
        "💬 Chat RAG",
        "📋 Listing tickets",
        "📊 Monitoring",
        "🧪 Métriques indexation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Modèles")
st.sidebar.markdown(f"**Embedding**  \n`{EMBEDDING_MODEL}`")
st.sidebar.markdown(f"**Génération**  \n`{GENERATION_MODEL}`")
st.sidebar.markdown(f"**Qdrant**  \n`{COLLECTION_NAME}`")


st.markdown("""
<div class="main-header">
    <div class="main-title">🤖 RAG Ticket Intelligence</div>
    <div class="main-subtitle">
        Qdrant local • Qwen Embeddings • Mistral 7B • Monitoring qualité • Analyse contexte
    </div>
</div>
""", unsafe_allow_html=True)


if page == "💬 Chat RAG":
    left, right = st.columns([1.25, 0.75])

    with left:
        st.markdown('<div class="section-title">💬 Assistant RAG</div>', unsafe_allow_html=True)

        question = st.text_area(
            "Question",
            placeholder="Exemple : quels tickets parlent d’un problème de connexion, VPN ou mot de passe ?",
            height=140
        )

        if st.button("🚀 Lancer l’analyse"):
            if not question.strip():
                st.warning("Écris une question avant de lancer.")
            else:
                st.session_state.question = question
                start_time = time.time()

                with st.spinner("🔎 Recherche sémantique dans Qdrant..."):
                    results = search_qdrant(question, top_k)

                context = build_context(results)

                with st.spinner("🧠 Génération de la réponse avec Mistral..."):
                    answer = generate_answer(question, context)

                latency = time.time() - start_time
                runtime_metrics = compute_proxy_metrics(results, latency)

                st.session_state.results = results
                st.session_state.answer = answer
                st.session_state.runtime_metrics = runtime_metrics

        if st.session_state.answer:
            st.markdown(f"""
            <div class="chat-answer">
                <h2>🧠 Réponse Mistral</h2>
                <p>{st.session_state.answer}</p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">⚡ Score de confiance</div>', unsafe_allow_html=True)

        if st.session_state.runtime_metrics:
            m = st.session_state.runtime_metrics

            metric_card(
                "Score de confiance",
                m["trust_score"],
                "Score global basé sur similarité, contexte et ranking",
                "#1ABC9C"
            )

            metric_card(
                "Fiabilité",
                m["reliability"],
                "Estimation de stabilité du contexte récupéré",
                "#2ECC71"
            )

            metric_card(
                "Pertinence contexte",
                m["context_relevance"],
                "Moyenne des scores de similarité Qdrant",
                "#3498DB"
            )

            metric_card(
                "Latence",
                f'{m["latency"]}s',
                "Temps total retrieval + génération",
                "#F97316"
            )
        else:
            st.info("Lance une question pour afficher les scores.")


elif page == "📋 Listing tickets":
    st.markdown('<div class="section-title">📋 Tickets retrouvés</div>', unsafe_allow_html=True)

    if not st.session_state.results:
        st.info("Aucun ticket affiché. Lance d’abord une question dans Chat RAG.")
    else:
        for result in st.session_state.results:
            p = result.payload

            priority = str(p.get("priority", "unknown")).lower()
            priority_class = "badge-green"

            if priority == "high":
                priority_class = "badge-red"
            elif priority == "medium":
                priority_class = "badge-orange"

            st.markdown(f"""
            <div class="ticket-card">
                <div class="ticket-title">🎫 Ticket #{p.get("ticket_id")}</div>
                <br>
                <span class="badge badge-turquoise">Score {round(result.score, 4)}</span>
                <span class="badge badge-blue">{p.get("language")}</span>
                <span class="badge badge-green">{p.get("intent")}</span>
                <span class="badge {priority_class}">{p.get("priority")}</span>
                <p style="margin-top:16px; color:#334155;">{p.get("clean_text")}</p>
                <p class="small-muted">
                    Source : {p.get("source_file")} • Longueur : {p.get("text_length")}
                </p>
            </div>
            """, unsafe_allow_html=True)


elif page == "📊 Monitoring":
    st.markdown('<div class="section-title">📊 Monitoring qualité RAG</div>', unsafe_allow_html=True)

    if not st.session_state.runtime_metrics:
        st.info("Lance une question dans Chat RAG pour générer les métriques.")
    else:
        m = st.session_state.runtime_metrics

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("MAP Proxy", m["map_proxy"], "Qualité du classement des tickets", "#1ABC9C")

        with c2:
            metric_card("Recall Proxy", m["context_recall_proxy"], "Couverture estimée du contexte utile", "#2ECC71")

        with c3:
            metric_card("Accuracy Proxy", m["accuracy_proxy"], "Exactitude estimée via similarité", "#3498DB")

        with c4:
            metric_card("Confiance", m["trust_score"], "Score global de confiance", "#F97316")

        st.markdown("### 📈 Radar qualité")

        radar_metrics = [
            "trust_score",
            "reliability",
            "context_relevance",
            "context_recall_proxy",
            "precision_proxy",
            "map_proxy",
            "accuracy_proxy"
        ]

        radar_values = [m[x] for x in radar_metrics]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=[
                "Confiance",
                "Fiabilité",
                "Pertinence",
                "Context Recall",
                "Precision",
                "MAP",
                "Accuracy"
            ],
            fill="toself",
            name="RAG Quality",
            line_color="#1ABC9C"
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Scores Qdrant des tickets récupérés")

        scores_df = pd.DataFrame([
            {
                "ticket_id": r.payload.get("ticket_id"),
                "score": r.score,
                "intent": r.payload.get("intent"),
                "priority": r.payload.get("priority")
            }
            for r in st.session_state.results
        ])

        fig_scores = px.bar(
            scores_df,
            x="ticket_id",
            y="score",
            color="intent",
            title="Similarité par ticket récupéré",
            color_discrete_sequence=["#1ABC9C", "#2ECC71", "#3498DB", "#F97316"]
        )

        st.plotly_chart(fig_scores, use_container_width=True)

        st.markdown("""
        <div class="card">
            <h3>🧾 Justification des métriques</h3>
            <p>
                Les métriques affichées ici sont des <b>proxies</b> car il n’y a pas encore de vérité terrain labellisée.
                Pour obtenir un vrai MAP, Recall ou Accuracy académique, il faudra créer un fichier d’évaluation avec :
                question, tickets attendus, réponse attendue.
            </p>
            <p>
                Actuellement, le score est calculé à partir des scores de similarité Qdrant, du classement des tickets,
                de la couverture du contexte et de la stabilité des résultats.
            </p>
        </div>
        """, unsafe_allow_html=True)


elif page == "🧪 Métriques indexation":
    st.markdown('<div class="section-title">🧪 Métriques d’indexation Qdrant</div>', unsafe_allow_html=True)

    if index_metrics_df.empty:
        st.warning("Fichier de métriques introuvable.")
    else:
        row = index_metrics_df.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Tickets indexés", int(row["indexed_tickets"]), "Nombre de vecteurs stockés", "#1ABC9C")

        with c2:
            metric_card("Échecs", int(row["failed_tickets"]), "Tickets non vectorisés", "#EF4444")

        with c3:
            metric_card("Taux succès", row["success_rate"], "Indexation réussie", "#2ECC71")

        with c4:
            metric_card("Temps total", f'{row["total_indexation_time_seconds"]}s', "Durée indexation", "#3498DB")

        st.markdown("### 🌍 Distribution des langues")

        try:
            lang_data = json.loads(row["language_distribution"])
            lang_df = pd.DataFrame(list(lang_data.items()), columns=["Langue", "Nombre"])

            fig_lang = px.pie(
                lang_df,
                names="Langue",
                values="Nombre",
                hole=0.45,
                color_discrete_sequence=["#1ABC9C", "#2ECC71", "#3498DB", "#F97316"]
            )

            st.plotly_chart(fig_lang, use_container_width=True)
        except Exception:
            st.write(row.get("language_distribution"))

        st.markdown("### 🧩 Distribution des intentions")

        try:
            intent_data = json.loads(row["intent_distribution"])
            intent_df = pd.DataFrame(list(intent_data.items()), columns=["Intent", "Nombre"])

            fig_intent = px.bar(
                intent_df,
                x="Intent",
                y="Nombre",
                color="Intent",
                color_discrete_sequence=["#1ABC9C", "#2ECC71", "#3498DB", "#F97316"]
            )

            st.plotly_chart(fig_intent, use_container_width=True)
        except Exception:
            st.write(row.get("intent_distribution"))