# RAG-time LogiStore

[![CI Lint](https://github.com/Skand13/rag-project/actions/workflows/lint.yml/badge.svg?branch=feature/logistore-rag)](https://github.com/Skand13/rag-project/actions/workflows/lint.yml)
[![CI Tests](https://github.com/Skand13/rag-project/actions/workflows/tests.yml/badge.svg?branch=feature/logistore-rag)](https://github.com/Skand13/rag-project/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/Skand13/rag-project/blob/main/LICENSE)
[![OpenSearch 2.13](https://img.shields.io/badge/OpenSearch-2.13-005EB8?logo=opensearch)](https://opensearch.org/)

Système RAG (Retrieval-Augmented Generation) **multilingue** pour la recherche intelligente de tickets SAV.
Conçu pour indexer et interroger **plus de 200 000 tickets** en 5 langues (EN, FR, DE, ES, PT).

---

## Table des matières

1. [Architecture](#architecture)
2. [Stack technique](#stack-technique)
3. [Métriques d'évaluation](#métriques-dévaluation)
4. [Installation](#installation)
5. [Structure du projet](#structure-du-projet)
6. [Décisions techniques](#décisions-techniques)
7. [Axes d'amélioration](#axes-damélioration)
8. [CI/CD](#cicd)
9. [Dataset](#dataset)
10. [Contribution](#contribution)

---

## Architecture

Pipeline RAG complet — de la requête utilisateur à la réponse générée :

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE RAG-TIME LOGISTORE                  │
└─────────────────────────────────────────────────────────────────────┘

  Utilisateur
      │
      ▼
 ┌──────────┐
 │  Query   │  (texte libre, multilingue : FR / EN / DE / ES / PT)
 └──────────┘
      │
      ▼
 ┌────────────────────────────┐
 │  Embedding (local)         │  paraphrase-multilingual-MiniLM-L12-v2
 │  dim = 384                 │  (modèle tournant entièrement en local)
 └────────────────────────────┘
      │
      ├──────────────────────────────────────┐
      │  Vecteur dense                       │  Texte brut
      ▼                                      ▼
 ┌─────────────────────┐        ┌────────────────────────┐
 │  kNN HNSW           │        │  BM25                  │
 │  (vectoriel)        │        │  (lexical)             │
 │  OpenSearch 2.13    │        │  OpenSearch 2.13       │
 └─────────────────────┘        └────────────────────────┘
      │  Top-K résultats               │  Top-K résultats
      └──────────────┬─────────────────┘
                     ▼
            ┌────────────────┐
            │  RRF  k=60     │  Reciprocal Rank Fusion
            │  (fusion)      │  (Cormack 2009)
            └────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  Top-N chunks  │  contexte injecté dans le prompt
            └────────────────┘
                     │
                     ▼
 ┌──────────────────────────────────────────┐
 │  LLM (nvidia/nemotron-3-super-120b-a12b) │  via OpenRouter API
 │  Génération de la réponse finale         │
 └──────────────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │    Réponse     │  affichée dans l'interface Streamlit
            └────────────────┘
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Moteur de recherche | OpenSearch 2.13 (BM25 + k-NN HNSW) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (dim=384) |
| Fusion hybride | RRF k=60 (Reciprocal Rank Fusion) |
| LLM génération | `nvidia/nemotron-3-super-120b-a12b` (via OpenRouter) |
| LLM évaluation | `openai/gpt-oss-20b` (via OpenRouter) |
| Frontend | Streamlit |
| Langues supportées | EN, FR, DE, ES, PT |
| CI/CD | GitHub Actions (ruff lint + pytest) |
| Orchestration | Docker Compose + WSL2 |

---

## Métriques d'évaluation

### Retrieval — BM25 + kNN, RRF k=60

| Langue | MRR | nDCG@5 | Recall@5 |
|---|---|---|---|
| **Global** | **0.391** | **0.423** | **0.520** |
| FR | 0.600 | 0.626 | 0.700 |
| EN | 0.453 | 0.515 | 0.700 |
| DE | 0.350 | 0.363 | 0.400 |
| ES | 0.275 | 0.306 | 0.400 |
| PT | 0.275 | 0.306 | 0.400 |

### RAG — Nemotron 120B (génération) · GPT-OSS-20B (juge) · 50 requêtes

| Métrique | Score /5 |
|---|---|
| Faithfulness | 0.57 |
| Answer Relevancy | 2.37 |
| Context Precision | 3.06 |

> **Lecture** : le retrieval est très bon en FR/EN (Recall@5 = 0.70). La Faithfulness faible (0.57/5) indique que le LLM génère parfois des informations hors contexte — voir [Axes d'amélioration](#axes-damélioration).

---

## Installation

### Prérequis

- Python 3.10+
- Docker Desktop (avec backend WSL2 recommandé sous Windows)
- Compte [OpenRouter](https://openrouter.ai/) — clé API gratuite

### Setup

```bash
# 1. Cloner le dépôt
git clone https://github.com/Skand13/rag-project.git
cd rag-project

# 2. Créer et activer l'environnement virtuel
python -m venv .venv
# Windows
.venv\Scripts\Activate
# Linux / macOS
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos valeurs (voir section Variables d'environnement ci-dessous)
```

### Variables d'environnement (`.env.example`)

```dotenv
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=

OPENROUTER_API_KEY=
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM=384
```

### Démarrer OpenSearch

```bash
docker compose -f docker/docker-compose.yml up -d
```

### Indexer les données

```bash
python -m scripts.pipeline_ingestion --recreate
```

> L'option `--recreate` supprime et recrée l'index OpenSearch. Omettre pour un ajout incrémental.

### Lancer le frontend

```bash
streamlit run src/frontend/app.py
```

L'interface est accessible sur `http://localhost:8501`.

---

## Structure du projet

```
rag-project/
├── .github/
│   └── workflows/
│       ├── lint.yml              # ruff check (lint automatique)
│       └── tests.yml             # pytest (tests automatiques)
├── docker/
│   └── docker-compose.yml        # OpenSearch + OpenSearch Dashboards
├── src/
│   ├── ingestion/                # Chargement et preprocessing des tickets
│   ├── chunking/                 # Stratégie de chunking (1 ticket = 1 chunk)
│   ├── embeddings/               # Modèle d'embedding local (MiniLM)
│   ├── indexing/                 # Indexation OpenSearch (BM25 + kNN)
│   ├── retrieval/                # Recherche hybride BM25 + vectoriel + RRF
│   ├── llm/                      # Client LLM + pipeline RAG complet
│   └── frontend/                 # Interface Streamlit
├── scripts/
│   └── pipeline_ingestion.py     # Script d'ingestion des données
├── notebooks/
│   ├── eda_tickets_v2.ipynb      # Analyse exploratoire des données
│   ├── preprocessing_search_tests.ipynb
│   └── evaluation.ipynb          # Évaluation retrieval + RAG
├── docs/
│   ├── cadrage.md                # Cadrage du projet
│   └── veille_technologique.md   # Veille sur les composants RAG
├── data/
│   ├── raw/                      # Dataset brut (non commité — .gitignore)
│   └── eval/                     # Résultats d'évaluation
├── tests/
│   └── test_pipeline.py          # Tests unitaires du pipeline
├── .env.example                  # Template de configuration
├── requirements.txt
└── README.md
```

---

## Décisions techniques

| Décision | Choix retenu | Justification |
|---|---|---|
| Moteur de recherche | OpenSearch 2.13 | BM25 natif + vectoriel dans un seul service, licence Apache 2.0 |
| Modèle embedding | `paraphrase-multilingual-MiniLM-L12-v2` | 50+ langues, dim=384, entièrement local et gratuit |
| Stratégie de chunking | 1 ticket = 1 chunk | Unités sémantiques naturelles, médiane < 512 tokens |
| Fusion des scores | RRF k=60 | Cormack 2009 — pas de calibration de score nécessaire |
| LLM RAG | `nvidia/nemotron-3-super-120b-a12b:free` | Disponible gratuitement sur OpenRouter, haute capacité de raisonnement |
| LLM évaluation | `openai/gpt-oss-20b:free` | Suit les instructions JSON (Nemotron est un modèle "thinking", inadapté au parsing structuré) |
| Filtre kNN | Native kNN filter | Le `bool/filter` échoue pour FR/DE dans OpenSearch 2.13 HNSW — le filtre natif est la solution recommandée |

---

## Axes d'amélioration

1. **Faithfulness faible (0.57/5)** — Ajouter une instruction système explicite : _"Réponds UNIQUEMENT à partir du contexte fourni, sans inférence externe."_
2. **PT/ES sous-performants (MRR ≈ 0.275)** — Envisager un re-ranker multilingue de type cross-encoder (ex. `cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuné multilingue).
3. **DE retrieval moyen (MRR = 0.350)** — Augmenter le poids BM25 dans la fusion RRF pour les langues morphologiquement riches (allemand, langues agglutinantes).
4. **Query expansion** — Améliorer la Context Precision via expansion de requête (HyDE, pseudo-relevance feedback) afin de mieux capturer les variations lexicales.

---

## CI/CD

Les workflows GitHub Actions sont déclenchés à chaque `push` et `pull_request` sur toutes les branches.

| Workflow | Fichier | Outil | Description |
|---|---|---|---|
| Lint | `.github/workflows/lint.yml` | [ruff](https://github.com/astral-sh/ruff) | Vérification du style et des erreurs statiques Python |
| Tests | `.github/workflows/tests.yml` | [pytest](https://pytest.org/) | Exécution de la suite de tests unitaires |

Les deux checks **doivent passer** avant tout merge dans `main`.

---

## Dataset

[Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) — 4 000 tickets, 5 langues (EN / FR / DE / ES / PT), 17 colonnes.

Le fichier brut est placé dans `data/raw/` (exclu du dépôt via `.gitignore`).

---

## Contribution

```
main
 └── feature/<nom-feature>   ← créer votre branche ici
          │
          └── Pull Request → main
                   │
                   └── CI lint ✔ + CI tests ✔ → merge autorisé
```

1. Créer une branche depuis `main` : `git checkout -b feature/<nom>` ou `fix/<nom>`
2. Commiter vos modifications avec des messages clairs
3. Ouvrir une **Pull Request** vers `main`
4. S'assurer que les deux checks CI (lint + tests) passent
5. Demander une review avant de merger

---
