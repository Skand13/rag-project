# Architecture Système — RAG-time LogiStore

**Projet** : RAG-time — LogiStore
**Version** : 1.0
**Date** : Mai 2026

---

## 1. Vue d'ensemble du système

```mermaid
flowchart TD

    %% =====================
    %% UTILISATEURS
    %% =====================
    subgraph Users["Utilisateurs"]
        U1["👤 Agent SAV\nStreamlit UI"]
        U2["🔧 Admin technique\nCLI / scripts"]
        U3["📊 Développeur\nJupyter notebooks"]
    end

    %% =====================
    %% FRONTEND
    %% =====================
    subgraph Frontend["Frontend — Streamlit :8501"]
        F1["Onglet RAG\nRéponse LLM + sources"]
        F2["Onglet Tickets\nTop-5 + scores RRF"]
        F3["Onglet Debug\nScores BM25 / kNN / RRF"]
    end

    %% =====================
    %% BACKEND PYTHON
    %% =====================
    subgraph Backend["Backend Python (src/)"]

        subgraph Ingestion["ingestion/"]
            B1["loader.py"]
            B2["preprocessor.py"]
            B3["chunker.py"]
        end

        subgraph Embeddings["embeddings/"]
            B4["embedder.py\nMiniLM-L12-v2\ndim=384 · local"]
        end

        subgraph Indexing["indexing/"]
            B5["opensearch_client.py"]
            B6["index_manager.py"]
        end

        subgraph Retrieval["retrieval/"]
            B7["hybrid_search.py\nBM25 + kNN + RRF k=60"]
        end

        subgraph LLMLayer["llm/"]
            B8["llm_client.py\ncall_llm() · build_prompt()"]
            B9["rag_pipeline.py\nrun_rag() · fetch_full_tickets()"]
        end

    end

    %% =====================
    %% INFRASTRUCTURE LOCALE
    %% =====================
    subgraph Infra["Infrastructure Docker (WSL2)"]
        OS[("OpenSearch 2.13\n:9200\nBM25 + kNN HNSW\n3 999 tickets indexés")]
    end

    %% =====================
    %% APIS EXTERNES
    %% =====================
    subgraph External["APIs externes (OpenRouter)"]
        EX1["nvidia/nemotron-3-super-120b-a12b:free\nGénération RAG"]
        EX2["openai/gpt-oss-20b:free\nLLM-as-Judge"]
    end

    %% =====================
    %% DONNÉES
    %% =====================
    subgraph Data["Données"]
        D1["📄 CSV Kaggle\n4K tickets · 5 langues\n(data/raw/)"]
        D2["📊 Eval results\ndata/eval/\nCSV + radar PNG"]
        D3["🔒 .env\nCredentials\n(non commité)"]
    end

    %% =====================
    %% CI/CD & OUTILLAGE
    %% =====================
    subgraph CICD["CI/CD — GitHub Actions"]
        G1["lint.yml\nruff check"]
        G2["tests.yml\npytest"]
        G3["nbstripout\nprotection notebooks"]
    end

    %% =====================
    %% NOTEBOOKS
    %% =====================
    subgraph Notebooks["Notebooks (notebooks/)"]
        N1["eda_tickets_v2.ipynb"]
        N2["preprocessing_search_tests.ipynb"]
        N3["evaluation.ipynb\n✅ MRR=0.391\nFaithfulness=0.57/5"]
    end

    %% =====================
    %% FLUX UTILISATEURS
    %% =====================
    U1 -->|"requête texte + langue"| F1
    U2 -->|"python -m scripts.pipeline_ingestion"| B1
    U3 --> Notebooks

    %% =====================
    %% FLUX FRONTEND → BACKEND
    %% =====================
    F1 --> B7
    F1 --> B9
    F2 --> B7
    F3 --> B7

    %% =====================
    %% FLUX INGESTION
    %% =====================
    D1 --> B1 --> B2 --> B3 --> B4
    B4 --> B5
    B3 --> B5
    B5 --> B6
    B6 -->|"index BM25 + kNN"| OS

    %% =====================
    %% FLUX QUERY
    %% =====================
    B7 -->|"BM25 + kNN + filtre langue"| OS
    OS -->|"Top-K résultats"| B7
    B7 -->|"Top-5 fusionnés RRF"| B9
    B9 --> B8
    B8 -->|"HTTPS · Bearer token"| EX1
    EX1 -->|"réponse générée"| B8
    B8 --> F1

    %% =====================
    %% FLUX ÉVALUATION
    %% =====================
    N3 --> B7
    N3 --> B8
    B8 -->|"judge"| EX2
    EX2 --> N3
    N3 --> D2

    %% =====================
    %% CONFIG & SÉCURITÉ
    %% =====================
    D3 -->|"dotenv"| B8
    D3 -->|"dotenv"| B5

    %% =====================
    %% CI/CD
    %% =====================
    CICD -->|"push / PR"| Backend
    CICD -->|"push / PR"| Notebooks
```

---

## 2. Flux réseau

┌─────────────────────────────────────────────────────────────────────┐
│ Machine locale Windows (WSL2) │
│ │
│ ┌──────────────┐ HTTP :8501 ┌─────────────────────────────┐ │
│ │ Browser │ ◄──────────────► │ Streamlit app.py │ │
│ │ Agent SAV │ │ src/frontend/ │ │
│ └──────────────┘ └──────────────┬──────────────┘ │
│ │ │
│ Python src/ │
│ retrieval/ · llm/ │
│ │ │
│ ┌────────────────────┼──────────────┐ │
│ │ │ │ │
│ HTTPS :9200│ HTTPS │OpenRouter │ │
│ ▼ ▼ │ │
│ ┌──────────────┐ ┌──────────────────────┐ │ │
│ │ OpenSearch │ │ api.openrouter.ai │ │ │
│ │ 2.13 Docker │ │ Nemotron 120B │ │ │
│ │ (WSL2) │ │ GPT-OSS-20B │ │ │
│ └──────────────┘ └──────────────────────┘ │ │
│ │ │
└───────────────────────────────────────────────────────────────────┘ │
│
GitHub Actions (lint + tests) ◄──────────────────────────────┘
github.com/Skand13/rag-project


---

## 3. Composants — Fiche technique

| Composant | Technologie | Port | Rôle | Statut |
|---|---|---|---|---|
| **Frontend** | Streamlit | 8501 | Interface agent SAV (3 onglets) | ✅ |
| **Retrieval** | hybrid_search.py | — | BM25 + kNN + RRF k=60 | ✅ |
| **LLM client** | llm_client.py + OpenRouter | 443 (HTTPS) | Génération RAG multilingue | ✅ |
| **RAG pipeline** | rag_pipeline.py | — | Orchestration retrieval → LLM | ✅ |
| **Ingestion** | loader + preprocessor + chunker | — | CSV → chunks + metadata | ✅ |
| **Embedder** | MiniLM-L12-v2 (local) | — | Vecteurs dim=384 | ✅ |
| **Index** | OpenSearch 2.13 (Docker) | 9200 | BM25 + kNN HNSW, 3 999 docs | ✅ |
| **LLM génération** | Nemotron 120B (OpenRouter) | 443 | Réponses RAG multilingues | ✅ |
| **LLM évaluation** | GPT-OSS-20B (OpenRouter) | 443 | LLM-as-Judge JSON strict | ✅ |
| **CI lint** | GitHub Actions + ruff | — | Vérification style à chaque push | ✅ |
| **CI tests** | GitHub Actions + pytest | — | Tests unitaires à chaque push | ✅ |
| **Notebooks** | Jupyter :8889 | 8889 | EDA, preprocessing, évaluation | ✅ |

---

## 4. Sécurité et credentials
┌─────────────────────────────────────────────────────────┐
│ Gestion des secrets │
│ │
│ .env (non commité, dans .gitignore) │
│ ├── OPENROUTER_API_KEY=... │
│ ├── OPENSEARCH_PASSWORD=R@gTime2026#Store │
│ └── EMBEDDING_DIM=384 │
│ │
│ .env.example (commité — template sans valeurs) │
│ │
│ nbstripout (hook git pre-commit) │
│ └── Supprime les outputs de notebooks avant commit │
│ → empêche la fuite de clés API dans les outputs │
└─────────────────────────────────────────────────────────┘


**Incident résolu** : clé OpenRouter committée dans un notebook → réécriture historique git (`git filter-repo`), nouvelle clé générée, nbstripout installé.

---

## 5. Intégration SI — Trajectoire

```mermaid
flowchart LR

    subgraph MVP["✅ MVP (réalisé)"]
        M1["CSV Kaggle\n4K tickets · 5 langues"]
        M2["OpenSearch local\nDocker WSL2"]
        M3["OpenRouter API\nNemotron · GPT-OSS"]
        M4["Streamlit\nInterface agent"]
    end

    subgraph Phase2["Phase 2 — Données réelles"]
        P2A["API Zendesk\nTickets en temps réel"]
        P2B["GED SharePoint\nProcédures SAV"]
        P2C["Pipeline incrémental\nMise à jour index"]
    end

    subgraph Phase3["Phase 3 — Contexte enrichi"]
        P3A["CRM Salesforce\nContexte client"]
        P3B["PIM Akeneo\nFiches produits"]
        P3C["Dashboard analytique\nClustering tickets"]
    end

    subgraph Phase4["Phase 4 — Assistant"]
        P4A["Chatbot multi-tours\nMémoire de session"]
        P4B["Plugin Zendesk natif\nIntégration directe"]
        P4C["Alertes proactives\nDétection pics tickets"]
    end

    subgraph Phase5["Phase 5 — Production"]
        P5A["Cloud privé\nAzure / AWS"]
        P5B["SSO / RBAC\nActive Directory"]
        P5C["Monitoring\nPrometheus · Grafana\nEvidently AI"]
    end

    MVP --> Phase2 --> Phase3 --> Phase4 --> Phase5
```

---

## 6. Qualité et tests

| Couche | Outil | Couverture | Statut |
|---|---|---|---|
| Style / lint | ruff (GitHub Actions) | Tout le code src/ | ✅ Au vert |
| Tests unitaires | pytest | test_pipeline.py | ✅ Au vert |
| Notebooks | nbstripout | Tous les .ipynb | ✅ Actif |
| Évaluation retrieval | evaluation.ipynb | 50 requêtes · 5 langues | ✅ MRR=0.391 |
| Évaluation RAG | LLM-as-Judge (GPT-OSS-20B) | 50 requêtes · 5 langues | ✅ Context Precision=3.06/5 |