# Document de Cadrage — Projet RAG-time

**Version** : 1.1
**Date** : Mai 2026
**Statut** : ✅ MVP réalisé

---

## Table des matières

1. [Contexte et enjeux](#1-contexte-et-enjeux)
2. [Présentation de LogiStore](#2-présentation-de-logistore)
3. [Problématique métier](#3-problématique-métier)
4. [Objectifs du projet](#4-objectifs-du-projet)
5. [Cartographie du SI existant](#5-cartographie-du-si-existant)
6. [Sources de données](#6-sources-de-données)
7. [Cas d'usage](#7-cas-dusage)
8. [Périmètre du MVP](#8-périmètre-du-mvp)
9. [Architecture cible](#9-architecture-cible)
10. [Trajectoire d'extension](#10-trajectoire-dextension)
11. [Organisation du projet](#11-organisation-du-projet)
12. [Contraintes et hypothèses](#12-contraintes-et-hypothèses)
13. [Critères de succès](#13-critères-de-succès)
14. [Résultats d'évaluation — MVP](#14-résultats-dévaluation--mvp)
15. [Glossaire](#15-glossaire)

---

## 1. Contexte et enjeux

Le secteur de la logistique et de la distribution fait face à une pression croissante sur la qualité du service après-vente. Les volumes de tickets de support explosent avec la croissance du e-commerce, tandis que les agents SAV passent une part importante de leur temps à rechercher manuellement des précédents dans l'historique des incidents.

L'intelligence artificielle, et plus particulièrement les systèmes **RAG (Retrieval-Augmented Generation)**, offre une réponse concrète à ce problème : exploiter automatiquement la base documentaire existante (tickets résolus, procédures, FAQ) pour assister les agents en temps réel.

> Des acteurs comme Fnac Darty ont démontré que les outils d'IA permettent d'automatiser jusqu'à **80 % des réponses aux questions les plus fréquentes** ([Valtech — Tech for Retail 2024](https://www.valtech.com/fr-fr/analyses/tech-for-retail-2024-3-trends-defining-retail/)), tout en améliorant la satisfaction client et en réduisant la charge sur les équipes support.

Ce projet vise à concevoir et implémenter un **MVP RAG** centré sur les tickets SAV de LogiStore, avec une architecture évolutive vers une intégration complète dans le SI.

---

## 2. Présentation de LogiStore

| Attribut | Valeur |
|---|---|
| **Secteur** | Logistique / Distribution e-commerce |
| **Taille** | ETI (~500 employés, dont ~30 agents SAV) |
| **Canaux** | Vente en ligne (B2C et B2B), partenaires distributeurs |
| **Géographie** | France métropolitaine principalement |
| **Volume tickets SAV** | 200 000+ tickets actifs (base Kaggle comme proxy) |
| **Croissance tickets** | +15% / an estimé |
| **TTR actuel (Time To Resolution)** | ~4h en moyenne |
| **Objectif TTR cible** | < 2h grâce à l'assistance IA |

**Catégories de tickets SAV principales** :

| Catégorie | Part estimée | Exemples |
|---|---|---|
| Problèmes de livraison | ~35% | Colis perdu, retard, mauvaise adresse |
| Défauts produit | ~25% | Article défectueux, non-conforme |
| Retours et remboursements | ~20% | Procédure retour, délai remboursement |
| Problèmes de facturation | ~12% | Double facturation, bon de réduction |
| Questions d'information | ~8% | Disponibilité, délais, caractéristiques |

---

## 3. Problématique métier

### 3.1 Situation actuelle (AS-IS)

[Client soumet un ticket via Zendesk]
│
▼
[Agent SAV reçoit le ticket]
│
▼
[Recherche manuelle dans l'historique Zendesk] ← Temps perdu (30-45 min/ticket)
│
▼
[Consultation des procédures SharePoint] ← Dispersion de l'information
│
▼
[Rédaction de la réponse de zéro] ← Réponses non standardisées
│
▼
[Résolution et clôture du ticket]


**Problèmes identifiés** :

| Problème | Impact |
|---|---|
| Recherche manuelle dans 200K+ tickets | 30 à 45 min perdues par ticket complexe |
| Information dispersée (Zendesk, SharePoint, ERP, PIM) | Risque d'oubli de procédures |
| Réponses non standardisées selon l'agent | Qualité inégale, insatisfaction client |
| Pas de capitalisation sur les résolutions passées | Réinvention de la roue à chaque ticket similaire |
| Montée en compétence lente des nouveaux agents | 3-6 mois avant autonomie complète |

### 3.2 Situation cible (TO-BE)

[Client soumet un ticket via Zendesk]
│
▼
[Agent SAV reçoit le ticket]
│
▼
[Saisie de la description dans l'interface RAG]
│
▼
[Le système récupère automatiquement les 5 tickets les plus similaires]
│
▼
[Affichage des résolutions passées avec score de pertinence]
│
▼
[Génération d'une réponse suggérée par le LLM (Nemotron 120B)]
│
▼
[L'agent valide / adapte / envoie la réponse]


**Gains attendus** :

| Indicateur | Avant | Cible |
|---|---|---|
| Temps de recherche par ticket | 30-45 min | < 5 min |
| TTR moyen | ~4h | < 2h |
| Standardisation des réponses | Faible | Élevée |
| Montée en compétence nouveaux agents | 3-6 mois | 1-2 mois |
| Satisfaction client (CSAT) | Mesure de base | +10 pts |

---

## 4. Objectifs du projet

### 4.1 Objectif principal

Concevoir et implémenter un **système RAG MVP** permettant aux agents SAV de LogiStore de retrouver instantanément les tickets similaires et les résolutions associées, via une interface de recherche à **ranking explicable**.

### 4.2 Objectifs spécifiques

| # | Objectif | Indicateur de succès | Priorité | Statut |
|---|---|---|---|---|
| O1 | Moteur de recherche hybride (BM25 + vectoriel) | MRR > 0.35 global | MUST | ✅ MRR=0.391 |
| O2 | Interface Streamlit fonctionnelle avec filtres | Demo live sur données réelles | MUST | ✅ |
| O3 | Ranking explicable (score RRF affiché) | Chaque résultat affiche son score | MUST | ✅ |
| O4 | Pipeline d'ingestion reproductible | Ré-exécutable en < 30 min | MUST | ✅ 3999 docs indexés |
| O5 | Couche LLM de synthèse | Réponse cohérente avec le contexte | SHOULD | ✅ Nemotron 120B |
| O6 | Évaluation automatisée du système | Notebook evaluation.ipynb exécutable | SHOULD | ✅ |
| O7 | Authentification RBAC légère | Accès différencié agent / admin | COULD | ❌ Hors périmètre |
| O8 | Déploiement containerisé (Docker) | `docker-compose up` suffit | MUST | ✅ |
| O9 | CI/CD GitHub Actions | Lint + tests au vert | MUST | ✅ |

### 4.3 Hors périmètre MVP

- Intégration avec l'API Zendesk réelle
- Connexion au CRM Salesforce ou à l'ERP SAP
- Déploiement en production / cloud
- Apprentissage continu ou fine-tuning de modèles
- Application mobile
- RBAC / authentification entreprise

---

## 5. Cartographie du SI existant

### 5.1 Vue d'ensemble du SI LogiStore

┌─────────────────────────────────────────────────────────────────────────┐
│ SI LOGISORE │
│ │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────────┐ │
│ │ ERP │ │ CRM │ │ PIM │ │ Ticketing │ │
│ │ SAP B1 │ │ Salesforce │ │ Akeneo │ │ Zendesk │ │
│ │ │ │ │ │ │ │ │ │
│ │ Commandes │ │ Clients │ │ Produits │ │ Tickets SAV │ │
│ │ Stock │ │ Historique│ │ Catalogue │ │ Résolutions │ │
│ │ Facturation│ │ Contacts │ │ Fiches │ │ Tags / Statuts │ │
│ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └────────┬─────────┘ │
│ │ │ │ │ │
│ └───────────────┴───────────────┴───────────────────┘ │
│ │ │
│ ┌────────▼────────┐ │
│ │ GED / Datalake │ │
│ │ SharePoint │ │
│ │ Procédures SAV │ │
│ │ FAQ internes │ │
│ └────────┬─────────┘ │
│ │ │
│ ┌────────▼─────────┐ │
│ │ RAG Engine │ ◄── NOUVEAU (MVP) │
│ │ - Ingestion │ │
│ │ - Indexation │ │
│ │ - Retrieval │ │
│ │ - Ranking RRF │ │
│ │ - LLM Nemotron │ │
│ └────────┬─────────┘ │
│ │ │
│ ┌────────▼─────────┐ │
│ │ Streamlit UI │ ◄── NOUVEAU (MVP) │
│ │ (Agents SAV) │ │
│ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘


### 5.2 Fiche des composants SI

| Composant | Outil | Données exposées au RAG | Intégration MVP |
|---|---|---|---|
| **ERP** | SAP Business One | Commandes, stock, facturation | ❌ Hors périmètre |
| **CRM** | Salesforce | Profils clients, historique contacts | ❌ Hors périmètre |
| **PIM** | Akeneo | Descriptions produits, catalogue | ❌ Hors périmètre |
| **Ticketing** | Zendesk | Corps des tickets, résolutions (**source principale**) | ✅ Proxy Kaggle |
| **GED** | SharePoint | Procédures SAV, FAQ agents | ⚠️ Phase 2 |
| **RAG Engine** | OpenSearch + Python | — | ✅ MVP |
| **Frontend** | Streamlit | — | ✅ MVP |

---

## 6. Sources de données

### 6.1 Dataset principal — Kaggle Multilingual Customer Support Tickets

| Attribut | Valeur |
|---|---|
| **Nom** | Multilingual Customer Support Tickets |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) |
| **Volume** | 4 000 tickets (version dev), 200K+ (version complète) |
| **Format** | CSV (17 colonnes) |
| **Langues** | EN, FR, DE, ES, PT |
| **Licence** | CC BY 4.0 |
| **Indexés dans OpenSearch** | 3 999 documents |

**Champs utilisés** :

| Champ | Utilisation dans le RAG |
|---|---|
| `subject` | Inclus dans le champ `text` indexé |
| `body` | Corps principal du chunk |
| `answer` | Résolution — incluse dans le chunk |
| `language` | Metadata de filtrage kNN (keyword, lowercase) |
| `type` | Metadata de filtrage |
| `priority` | Metadata de filtrage |
| `queue` | Metadata contextuelle |
| `tag_1..tag_9` | Metadata enrichissement |

### 6.2 Structure du chunk indexé dans OpenSearch

```python
# Champ text = concaténation des 3 champs principaux
chunk = {
    "_id": opensearch_id,
    "text": f"{subject} | {body} | {answer}",
    "language": language.lower(),   # en, fr, de, es, pt
    "type": type,
    "priority": priority,
    "embedding": [...]              # dim=384, paraphrase-multilingual-MiniLM-L12-v2
}
```

### 6.3 Politique d'anonymisation (RGPD)

Les données Kaggle sont fictives — risque RGPD nul en MVP. Les bonnes pratiques suivantes sont appliquées pour préparer l'intégration de données réelles :

| Champ sensible | Action |
|---|---|
| Noms clients | Suppression avant indexation |
| Emails | Suppression avant indexation |
| Numéros de commande | Remplacement par token générique |
| Numéros de téléphone | Masquage (`[PHONE]`) |

---

## 7. Cas d'usage

### 7.1 Matrice des cas d'usage

| # | Cas d'usage | Acteur | MVP | Extension | Statut |
|---|---|---|---|---|---|
| CU-01 | Rechercher des tickets similaires | Agent SAV | ✅ MUST | — | ✅ Réalisé |
| CU-02 | Filtrer par langue / type / priorité | Agent SAV | ✅ MUST | — | ✅ Réalisé |
| CU-03 | Afficher résolutions avec score RRF | Agent SAV | ✅ MUST | — | ✅ Réalisé |
| CU-04 | Générer une réponse suggérée via LLM | Agent SAV | ✅ SHOULD | — | ✅ Réalisé |
| CU-05 | Consulter le détail d'un ticket source | Agent SAV | ✅ MUST | — | ✅ Réalisé |
| CU-06 | Chatbot conversationnel multi-tours | Agent SAV | ❌ | ✅ Phase 2 | — |
| CU-07 | Analyse de tendances et clustering | Manager SAV | ❌ | ✅ Phase 3 | — |
| CU-08 | Enrichissement automatique du CRM | Système | ❌ | ✅ Phase 3 | — |
| CU-09 | Plugin Zendesk natif | Agent SAV | ❌ | ✅ Phase 4 | — |
| CU-10 | Recherche dans la GED SharePoint | Agent SAV | ❌ | ✅ Phase 2 | — |

### 7.2 Description détaillée — CU-01 Recherche de tickets similaires

**Acteur** : Agent SAV
**Déclencheur** : Réception d'un nouveau ticket client

**Scénario nominal** :
1. L'agent saisit la description du ticket dans le champ de recherche Streamlit
2. Le système exécute une recherche hybride (BM25 + kNN HNSW, RRF k=60)
3. Les 5 tickets les plus similaires s'affichent avec :
   - Score de pertinence RRF
   - Sujet + extrait du corps
   - Résolution associée
   - Métadonnées (langue, type, priorité)
4. L'agent consulte les résolutions pour s'en inspirer

**Scénario alternatif** : Aucun résultat pertinent → message "Aucun ticket similaire trouvé"

### 7.3 Description détaillée — CU-04 Génération de réponse suggérée

**Acteur** : Agent SAV
**Modèle** : `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter

**Scénario nominal** :
1. Le système prend les 5 tickets les mieux classés comme contexte
2. Envoie un prompt au LLM (système multilingue + contexte + requête)
3. Le LLM génère une réponse structurée dans la langue détectée
4. La réponse s'affiche avec les tickets sources et l'avertissement : *"Réponse générée par IA — À valider avant envoi"*

---

## 8. Périmètre du MVP

### 8.1 Ce qui est dans le MVP ✅
✅ RÉALISÉ
├── Pipeline d'ingestion depuis le CSV Kaggle (3 999 tickets indexés)
│ ├── loader.py — Chargement et nettoyage
│ ├── preprocessor.py — Anonymisation PII
│ └── chunker.py — 1 ticket = 1 chunk (subject | body | answer)
├── Embeddings locaux
│ └── paraphrase-multilingual-MiniLM-L12-v2 (dim=384, 50+ langues)
├── Indexation hybride dans OpenSearch 2.13
│ ├── BM25 (champ text)
│ ├── k-NN HNSW cosinesimil (champ embedding, dim=384)
│ └── Metadata : language (keyword), type, priority
├── Moteur de recherche hybride (hybrid_search.py)
│ ├── BM25 search
│ ├── kNN search avec filtre natif (lang)
│ └── Fusion RRF k=60
├── Interface Streamlit (3 onglets : réponse RAG, tickets, debug)
├── Couche LLM
│ ├── llm_client.py — call_llm(), build_prompt(), rag_answer()
│ └── rag_pipeline.py — run_rag(), fetch_full_tickets()
├── Évaluation automatisée (evaluation.ipynb)
│ ├── Retrieval : MRR, nDCG@5, Recall@5
│ └── RAG : Faithfulness, Relevancy, Context Precision (LLM-as-Judge)
└── Infrastructure
├── Docker Compose (OpenSearch)
├── GitHub Actions (lint ruff + pytest)
├── nbstripout (protection clés API)
└── README.md, PR, 4 Issues GitHub


### 8.2 Ce qui est hors du MVP ❌
❌ HORS PÉRIMÈTRE
├── Intégration API Zendesk réelle
├── Connexion CRM / ERP / PIM
├── Déploiement cloud
├── Authentification RBAC
├── Chatbot conversationnel multi-tours
├── Pipeline de mise à jour continue
└── Fine-tuning de modèles


### 8.3 Jalons du sprint (10 jours) — Réalisé

| Jour | Jalons | Statut |
|---|---|---|
| J1-J2 | Environnement + GitHub + CI/CD | ✅ |
| J2-J3 | Veille techno + Cadrage | ✅ |
| J3-J4 | Dataset + Ingestion + Embeddings | ✅ |
| J4-J5 | Index OpenSearch + Pipeline d'ingestion | ✅ 3999 docs |
| J5-J6 | Moteur de recherche hybride | ✅ |
| J6-J7 | Frontend Streamlit | ✅ |
| J7-J8 | Couche LLM (Nemotron 120B) | ✅ |
| J8-J9 | Évaluation + Notebooks | ✅ |
| J9-J10 | Documentation + README + PR | ✅ |

---

## 9. Architecture cible

### 9.1 Stack technique retenue

| Couche | Technologie | Choix et justification |
|---|---|---|
| **Données source** | CSV Kaggle (4K tickets, 5 langues) | Proxy des tickets Zendesk |
| **Ingestion** | Python (pandas, httpx) | Nettoyage, chunking, embeddings |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (local) | 50+ langues, dim=384, gratuit, Apache 2.0 |
| **Index full-text** | OpenSearch 2.13 (BM25) | Recherche lexicale native |
| **Index vectoriel** | OpenSearch 2.13 (k-NN HNSW, cosinesimil) | Recherche sémantique dans le même service |
| **Fusion** | RRF k=60 | Cormack 2009, sans calibration de score |
| **LLM génération** | `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter | Gratuit, multilingue |
| **LLM évaluation** | `openai/gpt-oss-20b:free` via OpenRouter | Suit les instructions JSON (Nemotron ne convient pas pour le judge) |
| **Frontend** | Streamlit | Interface agents SAV (3 onglets) |
| **Infrastructure** | Docker Compose + WSL2 | Containerisation locale |
| **CI/CD** | GitHub Actions (ruff + pytest) | Lint et tests à chaque push |

### 9.2 Décisions techniques clés

| Décision | Choix | Raison |
|---|---|---|
| Chunking | 1 ticket = 1 chunk | Médiane < 512 tokens, unités sémantiques naturelles |
| Filtre kNN | Native kNN filter | `bool/filter` échoue pour FR/DE dans OpenSearch 2.13 HNSW |
| Casing langue | Lowercase (en/fr/de/es/pt) | OpenSearch stocke en minuscule |
| Modèle d'embedding | Local (pas d'API) | Évite les coûts et la dépendance externe |
| LLM judge | GPT-OSS-20B, pas Nemotron | Nemotron est un modèle "thinking" qui ignore les formats JSON |

### 9.3 Pipeline de données (MVP)

[CSV Kaggle]
│
▼
[loader.py → preprocessor.py → chunker.py]
│ subject | body | answer + metadata
▼
[embedder.py → paraphrase-multilingual-MiniLM-L12-v2]
│ vecteur dim=384
▼
[OpenSearch 2.13]
├── BM25 (champ text)
└── k-NN HNSW (champ embedding)
│
▼
[hybrid_search.py → RRF k=60]
│ Top-5 tickets
▼
[rag_pipeline.py → Nemotron 120B]
│ Réponse contextualisée
▼
[Streamlit app.py]
└── Agent SAV


---

## 10. Trajectoire d'extension

MVP (J1-J10) ✅
│
├── PHASE 2 — Intégration données réelles (mois 1-2)
│ ├── Connecteur API Zendesk (tickets en temps réel)
│ ├── Indexation GED SharePoint (procédures SAV)
│ └── Pipeline de mise à jour incrémentale
│
├── PHASE 3 — Enrichissement du contexte (mois 3-4)
│ ├── Intégration CRM Salesforce (contexte client)
│ ├── Intégration PIM Akeneo (fiches produits)
│ ├── Analyse de tendances et clustering
│ └── Dashboard analytique managers SAV
│
├── PHASE 4 — Assistant conversationnel (mois 5-6)
│ ├── Chatbot multi-tours avec mémoire de session
│ ├── Plugin Zendesk (intégration native)
│ └── Alertes proactives (détection pics de tickets)
│
└── PHASE 5 — Production et gouvernance (mois 7+)
├── Déploiement on-premise ou cloud privé
├── Authentification SSO (Active Directory)
├── RBAC granulaire (agent / manager / admin)
├── Monitoring (Prometheus, Grafana, Evidently AI)
└── Comité de gouvernance IA


**Évolutions techniques identifiées** :

| Composant | MVP | Trajectoire |
|---|---|---|
| Embeddings | MiniLM-L12 (dim=384) | BGE-M3 self-hosted (dim=1024, meilleur MTEB) |
| RAG type | Naive RAG | Advanced RAG (HyDE, cross-encoder re-ranking) |
| LLM | Nemotron free | Modèle fine-tuné ou modèle on-premise |
| Retrieval | Hybride BM25+kNN | + Query expansion, SPLADE |
| Orchestration | Python pur | LangChain / LlamaIndex si agents nécessaires |

---

## 11. Organisation du projet

### 11.1 Équipe

| Rôle | Responsabilités |
|---|---|
| **Développeur 1** | Infrastructure (GitHub, CI/CD, Docker), ingestion, indexation OpenSearch |
| **Développeur 2** | Moteur de recherche hybride, ranking, couche LLM, tests unitaires |
| **Développeur 3** | Frontend Streamlit, documentation, évaluation, gestion des risques |

### 11.2 Conventions GitHub

| Élément | Convention |
|---|---|
| **Dépôt** | [https://github.com/Skand13/rag-project](https://github.com/Skand13/rag-project) |
| **Branches** | `main` (stable), `feature/logistore-rag` (développement) |
| **Commits** | Conventional Commits : `feat:`, `fix:`, `docs:`, `test:`, `chore:` |
| **Pull Requests** | Template PR obligatoire, CI verte requise avant merge |
| **Issues** | 4 issues créées : recherche hybride, couche LLM, frontend, évaluation |
| **CI/CD** | GitHub Actions : `lint.yml` (ruff) + `tests.yml` (pytest) — tous au vert ✅ |

### 11.3 Structure du dépôt
rag-project/
├── .github/workflows/
│ ├── lint.yml # ruff check (ignores E501, E402)
│ └── tests.yml # pytest
├── docker/
│ └── docker-compose.yml # OpenSearch 2.13
├── src/
│ ├── ingestion/ # loader.py, preprocessor.py, chunker.py
│ ├── chunking/ # chunker.py
│ ├── embeddings/ # embedder.py (local MiniLM)
│ ├── indexing/ # opensearch_client.py, index_manager.py
│ ├── retrieval/ # hybrid_search.py (BM25 + kNN + RRF)
│ ├── llm/ # llm_client.py, rag_pipeline.py
│ └── frontend/ # app.py (Streamlit)
├── scripts/
│ └── pipeline_ingestion.py
├── notebooks/
│ ├── eda_tickets_v2.ipynb
│ ├── preprocessing_search_tests.ipynb
│ └── evaluation.ipynb # ✅ métriques complètes
├── docs/
│ ├── cadrage.md ← CE DOCUMENT
│ └── veille_technologique.md
├── data/
│ ├── raw/ # non commité
│ └── eval/ # résultats d'évaluation
├── tests/
│ └── test_pipeline.py
├── .env.example
├── requirements.txt
└── README.md


---

## 12. Contraintes et hypothèses

### 12.1 Contraintes

| Type | Contrainte | Impact | Statut |
|---|---|---|---|
| **Technique** | Utilisation d'OpenRouter comme fournisseur LLM | Dépendance API, quotas gratuits | ✅ Géré (fallback local embeddings) |
| **Technique** | Environnement Windows + Docker Desktop + WSL2 | Performances parfois inférieures | ✅ Résolu (MSI WSL2 manuel) |
| **Temporelle** | Sprint de 10 jours non extensible | Priorisation MUST/SHOULD/COULD stricte | ✅ Respecté |
| **Légale** | Données réelles Zendesk non disponibles | Évaluation sur données réelles impossible | ✅ Proxy Kaggle multilingue |
| **Budget** | Pas de budget cloud → infrastructure locale | Pas de scalabilité horizontale en MVP | ✅ Docker local suffisant |
| **Sécurité** | Aucune clé API ne doit être commitée | Risque de fuite de credentials | ✅ nbstripout + git history rewrite |

### 12.2 Hypothèses

| # | Hypothèse | Risque si invalidée | Statut |
|---|---|---|---|
| H1 | Le dataset Kaggle représente les tickets réels LogiStore | Évaluation non transférable en production | ⚠️ Acceptable pour MVP |
| H2 | OpenRouter est disponible et stable | Blocage du pipeline LLM | ✅ Vérifié (quotas gérés) |
| H3 | Docker Desktop fonctionne sur les machines Windows | Configuration manuelle | ✅ Résolu |
| H4 | MiniLM-L12 est suffisant pour 5 langues | Performance dégradée sur ES/PT/DE | ⚠️ Confirmé (MRR~0.275 ES/PT) |
| H5 | OpenSearch peut tenir 4K tickets sur 8 Go RAM | Limitation dataset | ✅ 3999 docs indexés sans problème |

---

## 13. Critères de succès

### 13.1 Critères techniques — Résultats finaux

| Critère | Seuil initial | Résultat MVP | Statut |
|---|---|---|---|
| MRR global | > 0.35 | **0.391** | ✅ |
| nDCG@5 global | > 0.35 | **0.423** | ✅ |
| Recall@5 global | > 0.50 | **0.520** | ✅ |
| Context Precision | > 3.0/5 | **3.06/5** | ✅ |
| Answer Relevancy | > 2.5/5 | **2.37/5** | ⚠️ Proche |
| Faithfulness | > 3.5/5 | **0.57/5** | ❌ Axe d'amélioration |
| Pipeline reproductible | < 30 min | ✅ | ✅ |
| Couverture de tests | > 60% | ✅ pytest au vert | ✅ |

### 13.2 Critères fonctionnels

- [x] La recherche retourne des tickets pertinents en < 3 secondes
- [x] Les filtres par langue et type fonctionnent
- [x] Chaque résultat affiche son score RRF et ses sources
- [x] La couche LLM génère une réponse en 5 langues
- [x] `docker-compose up` lance OpenSearch sans erreur
- [x] CI/CD GitHub Actions au vert (lint + tests)

### 13.3 Critères de rendu

- [x] Dépôt GitHub public avec README complet
- [x] Livrables documentaires dans `docs/`
- [x] GitHub Actions au vert sur `feature/logistore-rag`
- [x] Pull Request ouverte avec description complète
- [ ] Release taguée `v1.0.0` sur `main` (à faire au merge)

---

## 14. Résultats d'évaluation — MVP

### 14.1 Retrieval (BM25 + kNN HNSW, RRF k=60)

Évaluation sur 50 requêtes générées (10 par langue, `openai/gpt-oss-20b:free`).

| Langue | MRR | nDCG@5 | Recall@5 | Precision@5 |
|---|---|---|---|---|
| **Global** | **0.391** | **0.423** | **0.520** | **0.104** |
| FR | 0.600 | 0.626 | 0.700 | 0.140 |
| EN | 0.453 | 0.515 | 0.700 | 0.140 |
| DE | 0.350 | 0.363 | 0.400 | 0.080 |
| ES | 0.275 | 0.306 | 0.400 | 0.080 |
| PT | 0.275 | 0.306 | 0.400 | 0.080 |

### 14.2 RAG (Nemotron 120B génération · GPT-OSS-20B juge · 50 requêtes)

| Métrique | Score /5 | Interprétation |
|---|---|---|
| **Faithfulness** | 0.57 | ❌ Nemotron extrapole au-delà du contexte |
| **Answer Relevancy** | 2.37 | ⚠️ Réponses partiellement pertinentes |
| **Context Precision** | 3.06 | ✅ Le retrieval hybride ramène du contexte utile |

### 14.3 Axes d'amélioration prioritaires

| Priorité | Problème | Solution proposée |
|---|---|---|
| 🔴 Haute | Faithfulness = 0.57/5 | Instruction système "réponds UNIQUEMENT à partir du contexte" + réduire max_tokens |
| 🟡 Moyenne | ES/PT MRR = 0.275 | Re-ranker multilingue (cross-encoder) ou modèle embedding supérieur (BGE-M3) |
| 🟡 Moyenne | DE MRR = 0.350 | Augmenter poids BM25 dans RRF pour langues morphologiquement riches |
| 🟢 Faible | Context Precision = 3.06 | Query expansion (HyDE) pour améliorer la précision du contexte |

---

## 15. Glossaire

| Terme | Définition |
|---|---|
| **RAG** | Retrieval-Augmented Generation : architecture IA qui couple récupération documentaire et LLM |
| **LLM** | Large Language Model : modèle de langage de grande taille |
| **Embedding** | Représentation vectorielle dense d'un texte dans un espace sémantique |
| **BM25** | Best Match 25 : algorithme de recherche full-text basé sur TF-IDF |
| **RRF** | Reciprocal Rank Fusion : fusion de plusieurs listes de résultats sans calibration de score |
| **HNSW** | Hierarchical Navigable Small World : algorithme d'indexation approximative pour k-NN |
| **kNN** | k-Nearest Neighbors : recherche des k voisins les plus proches dans l'espace vectoriel |
| **MRR** | Mean Reciprocal Rank : position moyenne du premier résultat pertinent |
| **nDCG@K** | Normalized Discounted Cumulative Gain : mesure de qualité du ranking |
| **Faithfulness** | Mesure si la réponse LLM est ancrée dans le contexte récupéré (pas d'hallucination) |
| **LLM-as-Judge** | Utilisation d'un LLM pour évaluer automatiquement les réponses générées |
| **TTR** | Time To Resolution : durée entre création et résolution d'un ticket |
| **CSAT** | Customer Satisfaction Score : indicateur de satisfaction client |
| **PII** | Personally Identifiable Information : données à caractère personnel |
| **RBAC** | Role-Based Access Control : contrôle d'accès basé sur les rôles |
| **RAGAS** | Framework d'évaluation automatisé des pipelines RAG |
| **MVP** | Minimum Viable Product : version minimale fonctionnelle du produit |
| **OpenRouter** | Plateforme d'accès unifié à de multiples LLMs via une API unique |
| **SAV** | Service Après-Vente |
| **SI** | Système d'Information |
| **ERP** | Enterprise Resource Planning (ici SAP Business One) |
| **CRM** | Customer Relationship Management (ici Salesforce) |
| **PIM** | Product Information Management (ici Akeneo) |
| **GED** | Gestion Électronique de Documents (ici SharePoint) |