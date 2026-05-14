# RAG-time LogiStore

Système RAG (Retrieval-Augmented Generation) pour la recherche intelligente de tickets SAV multilingues.

## Stack technique

| Composant | Technologie |
|---|---|
| Moteur de recherche | OpenSearch 2.13 (BM25 + k-NN HNSW) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (dim=384) |
| Fusion hybride | RRF k=60 (Reciprocal Rank Fusion) |
| LLM | OpenRouter API (nvidia/nemotron-3-super-120b-a12b) |
| Frontend | Streamlit |
| Langues | EN, FR, DE, ES, PT |

## Installation

### Prérequis
- Python 3.10+
- Docker Desktop + WSL2
- Compte OpenRouter (clé API)

### Setup

```bash
# 1. Clone le dépôt
git clone https://github.com/Skand13/rag-project.git
cd rag-project

# 2. Crée le venv
python -m venv .venv
.venv\Scripts\Activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Installe les dépendances
pip install -r requirements.txt

# 4. Configure les variables d'environnement
cp .env.example .env
# Édite .env avec tes valeurs
```

### Démarrer OpenSearch

```bash
docker compose -f docker/docker-compose.yml up -d
```

### Indexer les données

```bash
python -m scripts.pipeline_ingestion --recreate
```

### Lancer le frontend

```bash
streamlit run src/frontend/app.py
```

## Structure du projet

rag-project/
├── docker/ # Docker Compose (OpenSearch)
├── src/
│ ├── ingestion/ # Chargement et preprocessing des données
│ ├── chunking/ # Stratégie de chunking
│ ├── embeddings/ # Modèles d'embedding (local + OpenRouter)
│ ├── indexing/ # Indexation OpenSearch
│ ├── retrieval/ # Recherche hybride BM25 + vectoriel + RRF
│ ├── llm/ # Client LLM + pipeline RAG
│ └── frontend/ # Interface Streamlit
├── scripts/ # Scripts d'ingestion
├── notebooks/ # EDA, preprocessing, évaluation
├── docs/ # Documentation (cadrage, veille)
├── data/
│ ├── raw/ # Dataset brut (non commité)
│ └── processed/ # Dataset prétraité (non commité)
├── tests/ # Tests unitaires
├── .env.example # Template de configuration
└── requirements.txt


## Dataset

[Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) — 4000 tickets, 5 langues (EN/FR/DE/ES/PT), 17 colonnes.

## CI/CD

- **Lint** : ruff (vérifié à chaque push)
- **Tests** : pytest (vérifié à chaque push)

## Contribution

1. Crée une branche depuis `main`
2. Ouvre une Pull Request vers `main`
3. Les checks CI doivent passer avant merge