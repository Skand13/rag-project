# Veille Technologique — Systèmes RAG (Retrieval-Augmented Generation)


**Stack** : OpenSearch 2.13 · paraphrase-multilingual-MiniLM-L12-v2 · Nemotron 120B · Streamlit

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Architectures RAG](#2-architectures-rag)
   - 2.1 [RAG Naïf](#21-rag-naïf-naive-rag)
   - 2.2 [RAG Avancé](#22-rag-avancé-advanced-rag)
   - 2.3 [RAG Modulaire](#23-rag-modulaire-modular-rag)
   - 2.4 [GraphRAG](#24-graphrag)
   - 2.5 [Comparatif des architectures](#25-comparatif-des-architectures)
3. [Stratégies de Chunking](#3-stratégies-de-chunking)
4. [Modèles d'Embeddings](#4-modèles-dembeddings)
5. [Recherche Hybride](#5-recherche-hybride)
6. [Moteurs d'Indexation](#6-moteurs-dindexation)
7. [Évaluation des Systèmes RAG](#7-évaluation-des-systèmes-rag)
8. [Frameworks d'Orchestration](#8-frameworks-dorchestration)
9. [Recommandations pour le MVP LogiStore](#9-recommandations-pour-le-mvp-logistore)
10. [Résultats d'évaluation — LogiStore](#10-résultats-dévaluation--logistore)
11. [Références Bibliographiques](#11-références-bibliographiques)

---

## 1. Introduction

Le RAG (*Retrieval-Augmented Generation*) est une architecture d'IA qui combine la recherche documentaire et la génération de texte par un LLM. Introduit par Lewis et al. en 2020, le RAG répond à une limitation fondamentale des LLMs : leur base de connaissances est figée à la date d'entraînement et ne reflète pas les données internes ou récentes d'une organisation.

**Principe général** : plutôt que de faire confiance uniquement au LLM pour répondre, on lui fournit des documents pertinents récupérés en temps réel depuis une base documentaire. Le LLM synthétise alors sa réponse à partir de ce contexte.

**Pourquoi le RAG pour LogiStore ?**

| Besoin LogiStore | Apport du RAG |
|---|---|
| Recherche dans 200K+ tickets SAV | Indexation et retrieval rapide et pertinent |
| Réponses contextualisées aux agents | LLM guidé par les vrais tickets passés |
| Traçabilité des sources | Chaque réponse cite les tickets sources |
| Pas de réentraînement du LLM | Mise à jour des données sans coût de fine-tuning |
| Contrôle des hallucinations | Le LLM reste ancré dans les documents récupérés |

---

## 2. Architectures RAG

### 2.1 RAG Naïf (Naive RAG)

Le RAG naïf est le pipeline le plus simple : la requête utilisateur est encodée en vecteur, les documents les plus proches sont récupérés, et l'ensemble est transmis au LLM pour génération.

**Flux de données** :
[Requête utilisateur]
│
▼
[Encodage en embedding]
│
▼
[Recherche vectorielle dans l'index]
│
▼
[Top-K documents récupérés]
│
▼
[Prompt = requête + contexte documents]
│
▼
[LLM → Génération de la réponse]


**Limites du Naive RAG** :
- Sensible à la qualité du chunking
- Pas de reformulation de requête
- Pas de filtrage ou de re-ranking
- Risque de contexte bruité si les chunks sont trop longs ou hétérogènes

---

### 2.2 RAG Avancé (Advanced RAG)

Le RAG avancé introduit des étapes supplémentaires avant et après la récupération pour améliorer la pertinence.

#### Techniques Pre-Retrieval

| Technique | Description | Avantage |
|---|---|---|
| **Query Rewriting** | Reformulation automatique de la requête par un LLM | Améliore le rappel |
| **HyDE** | Le LLM génère un document hypothétique utilisé comme requête | Meilleur alignement sémantique |
| **Query Expansion** | Ajout de synonymes et termes liés | Augmente la couverture |
| **RAG-Fusion** | Génère plusieurs reformulations + RRF | Robustesse accrue |

#### Techniques Post-Retrieval

| Technique | Description | Avantage |
|---|---|---|
| **Cross-Encoder Re-ranking** | Réévaluation fine des K résultats par un modèle dédié | Précision accrue |
| **Compression contextuelle** | Réduction des chunks pour ne garder que les passages pertinents | Réduit le bruit dans le prompt |
| **Multi-hop RAG** | Chaîne de retrievals successifs | Questions multi-étapes |
| **Self-RAG** | Le modèle décide lui-même quand et quoi récupérer | Adaptabilité dynamique |

---

### 2.3 RAG Modulaire (Modular RAG)

Le RAG modulaire découple chaque composant du pipeline pour permettre une orchestration flexible, avec des modules interchangeables.

**Composants modulaires clés** :
- **Routing** : sélectionner dynamiquement la source de données selon la requête
- **Agents et Tool-use** : le LLM peut décider d'appeler des outils externes
- **Orchestration** : LangChain ou LlamaIndex pour chaîner les modules
- **Mémoire conversationnelle** : conservation du contexte multi-tours

**Frameworks principaux** :

| Framework | Points forts | Cas d'usage |
|---|---|---|
| [LangChain](https://docs.langchain.com) | Écosystème riche, intégrations nombreuses | Pipelines complexes, agents |
| [LlamaIndex](https://docs.llamaindex.ai) | Optimisé pour l'indexation documentaire | RAG orienté connaissance |
| [Haystack](https://haystack.deepset.ai) | Orienté production et search | Moteurs de recherche IA |

---

### 2.4 GraphRAG

GraphRAG, introduit par Microsoft Research (2024), enrichit le RAG classique avec un graphe de connaissances extrait des documents sources.

**Principe** :
1. Les documents sont analysés pour extraire des entités et relations
2. Ces entités forment un graphe de connaissances
3. Lors du retrieval, on parcourt le graphe pour récupérer des contextes sémantiquement liés

**Pertinence pour LogiStore** : adapté aux requêtes complexes du type *"Quels produits sont associés aux tickets de livraison retardée pour les clients premium ?"*

> **Note :** GraphRAG est hors périmètre du MVP. À envisager dans une trajectoire d'extension.

---

### 2.5 Comparatif des architectures

| Critère | Naive RAG | Advanced RAG | Modular RAG | GraphRAG |
|---|---|---|---|---|
| Complexité d'implémentation | Faible | Moyenne | Élevée | Très élevée |
| Qualité de retrieval | Correcte | Bonne | Excellente | Excellente |
| Coût API (tokens) | Faible | Moyen | Élevé | Élevé |
| Traçabilité des sources | ✅ | ✅ | ✅ | ✅ |
| Requêtes complexes multi-entités | ❌ | Partiel | Partiel | ✅ |
| Recommandé pour MVP LogiStore | ✅ | ✅ (hybride) | ⚠️ Trajectoire | ⚠️ Trajectoire |

---

## 3. Stratégies de Chunking

Le découpage des documents en chunks est l'une des décisions les plus critiques dans un pipeline RAG. Un mauvais chunking dégrade irrémédiablement la qualité du retrieval.

### Stratégies disponibles

| Stratégie | Description | Avantages | Inconvénients |
|---|---|---|---|
| **Fixed-size** | Chunks de N tokens fixes | Simple, reproductible | Peut couper une idée en deux |
| **Sliding Window** | Chevauchement entre chunks | Préserve la continuité | Redondance, index plus volumineux |
| **Par structure** | Découpage par titre, paragraphe | Respecte la logique du document | Dépend du formatage source |
| **Sémantique** | Découpage selon la cohérence sémantique | Chunks cohérents | Plus coûteux |
| **Par ticket** | Un chunk = un ticket complet | Idéal pour les tickets courts | Limité si ticket très long |

### Choix retenu pour LogiStore

Les tickets sont des documents courts et structurés. La stratégie **1 ticket = 1 chunk** a été retenue :

```python
chunk = {
    "id": ticket_id,
    "text": f"{subject} | {body} | {answer}",
    "metadata": {
        "language": language,
        "type": type,
        "priority": priority,
        "queue": queue,
    }
}
```

- **Chunk size** : médiane < 512 tokens — pas de découpage nécessaire
- **Justification** : les tickets constituent des unités sémantiques naturelles ; séparer sujet/corps/réponse créerait des chunks orphelins

---

## 4. Modèles d'Embeddings

### 4.1 Principes fondamentaux

Les embeddings transforment un texte en vecteur dense dans un espace sémantique. Des textes sémantiquement similaires ont des vecteurs proches (similarité cosinus élevée).

**Propriétés clés** :
- **Dimension** : 384 à 4096 — plus de dimensions = plus précis mais plus coûteux en stockage
- **Contexte maximal** : longueur maximale en tokens encodable
- **Multilinguisme** : crucial pour LogiStore (5 langues : EN/FR/DE/ES/PT)
- **Score MTEB** : benchmark de référence pour comparer les modèles sur des tâches de retrieval

---

### 4.2 Comparatif des modèles (MTEB 2025-2026)

| Modèle | MTEB Score | Contexte max | Dimensions | Coût / 1M tokens | Self-host | Licence | Multilingual |
|---|---|---|---|---|---|---|---|
| **Qwen3-Embedding-8B** | 70.58 | 32 000 | 7 168 | Gratuit | ✅ | Apache 2.0 | ✅ |
| **NV-Embed-v2** (NVIDIA) | 69.32 | 32 768 | 4 096 | Gratuit | ✅ | CC-BY-NC-4.0 | ⚠️ |
| **Gemini embedding-001** | 68.32 | 2 048 | 3 072 | $0.15 | ❌ | Propriétaire | ✅ |
| **voyage-3-large** | ~67+ | 32 000 | 2 048 | $0.06 | ❌ | Propriétaire | ✅ |
| **Cohere embed-v4** | 65.20 | 128 000 | 1 024 | $0.10 | VPC | Propriétaire | ✅ |
| **text-embedding-3-large** (OpenAI) | 64.60 | 8 192 | 3 072 | $0.13 | ❌ | Propriétaire | ✅ |
| **BAAI/bge-m3** | 63.00 | 8 192 | 1 024 | Gratuit | ✅ | MIT | ✅ (100+ langues) |
| **text-embedding-3-small** (OpenAI) | 62.30 | 8 192 | 1 536 | $0.02 | ❌ | Propriétaire | ✅ |
| **paraphrase-multilingual-MiniLM-L12-v2** | ~57 | 512 | 384 | Gratuit | ✅ | Apache 2.0 | ✅ (50+ langues) |
| **all-MiniLM-L6-v2** | 56.30 | 512 | 384 | Gratuit | ✅ | Apache 2.0 | ❌ |

> **Sources** : [Prem AI — Best Embedding Models for RAG 2026](https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/) · [HuggingFace MTEB](https://huggingface.co/spaces/mteb/leaderboard)

---

### 4.3 Choix retenu pour LogiStore

**`paraphrase-multilingual-MiniLM-L12-v2`** — self-hosted, dim=384, Apache 2.0.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embedding = model.encode("Problème de livraison commande #12345")
# → vecteur de dimension 384
```

**Justification** :
- Couvre les 5 langues du projet (EN/FR/DE/ES/PT) sans API externe
- Aucun coût d'indexation pour 4 000 tickets
- Dimension 384 compatible avec le mapping HNSW d'OpenSearch

**Trajectoire** : migrer vers `BAAI/bge-m3` (dim=1024, MIT) si les besoins de précision augmentent.

---

## 5. Recherche Hybride

La recherche hybride combine BM25 (lexicale) et recherche vectorielle pour obtenir un meilleur rappel et une meilleure précision que chacune seule.

---

### 5.1 BM25 — Recherche lexicale

BM25 (Best Match 25) pondère les termes selon leur fréquence dans le document (TF) et leur rareté dans le corpus (IDF).

**Forces** :
- Exact match : numéros de commande, codes produits, termes techniques
- Robuste et rapide, sans GPU
- Natif dans OpenSearch (`bm25` scorer)
- Interprétable

**Limites** :
- Insensible à la sémantique : "envoi retardé" ≠ "livraison en retard" pour BM25

---

### 5.2 Recherche vectorielle

Encode la requête et les documents en vecteurs denses, puis trouve les plus proches voisins par similarité cosinus.

**Forces** :
- Compréhension sémantique : "délai d'expédition" ≈ "livraison tardive"
- Robustesse aux fautes de frappe et paraphrases
- Multilingue avec les bons modèles

**Limites** :
- Ne retrouve pas les termes exacts rares (numéros de commande, codes)

---

### 5.3 Fusion hybride — Reciprocal Rank Fusion (RRF)

Le RRF fusionne les listes de résultats BM25 et vectoriel en un classement unifié, sans nécessiter de calibration des scores bruts.

**Formule RRF** :

$$\text{RRF\_score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Avec `k = 60` (valeur empirique recommandée, Cormack 2009).

**Implémentation** :

```python
def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    scores = {}
    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

### 5.4 Décisions techniques — LogiStore

| Décision | Choix | Raison |
|---|---|---|
| Fusion | RRF k=60 | Pas de calibration de score nécessaire |
| Filtre kNN | Native kNN filter | `bool/filter` échoue pour FR/DE dans OpenSearch 2.13 HNSW |
| Casing langue | Lowercase (en/fr/de/es/pt) | OpenSearch stocke en minuscule, `.lower()` dans toutes les fonctions |

---

### 5.5 Alternatives avancées

| Technique | Principe | Cas d'usage |
|---|---|---|
| **SPLADE** | Représentations sparses apprises | Hybridation sans BM25 classique |
| **ColBERT** | Late interaction token-level | Haute précision, coûteux |
| **Cross-Encoder Re-ranking** | Réévaluation des top-K par un modèle plus lourd | Amélioration post-retrieval |

---

## 6. Moteurs d'Indexation

### 6.1 OpenSearch

[OpenSearch](https://opensearch.org/docs/) est un moteur de recherche open source (fork Apache 2.0, maintenu par AWS). Il supporte nativement BM25 et la recherche vectorielle via le plugin `knn`.

**Mapping hybride BM25 + k-NN utilisé dans LogiStore** :

```json
{
  "settings": { "index": { "knn": true } },
  "mappings": {
    "properties": {
      "text":      { "type": "text", "analyzer": "standard" },
      "language":  { "type": "keyword" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "lucene"
        }
      }
    }
  }
}
```

---

### 6.2 Qdrant

[Qdrant](https://qdrant.tech/documentation/) est un moteur vectoriel spécialisé, optimisé pour la recherche par similarité. Supporte le filtrage avancé par metadata, le sparse vectoriel (SPLADE), et gRPC.

---

### 6.3 Tableau comparatif

| Critère | OpenSearch 2.13 | Qdrant 1.9 |
|---|---|---|
| **Licence** | Apache 2.0 | Apache 2.0 |
| **BM25 natif** | ✅ | ⚠️ Via sparse vectors |
| **Recherche vectorielle** | ✅ Plugin k-NN (HNSW) | ✅ Natif (HNSW) |
| **Recherche hybride** | ✅ BM25 + k-NN | ✅ Dense + Sparse |
| **Filtrage par metadata** | ✅ | ✅ |
| **Dashboard de visualisation** | ✅ OpenSearch Dashboards | ❌ (API only) |
| **Consommation RAM (1 nœud)** | 512 MB min | 256 MB min |
| **Verdict pour LogiStore MVP** | ✅ **Retenu** | ✅ Option trajectoire |

> **Recommandation** : OpenSearch est retenu car il offre BM25 natif + vectoriel dans un seul service. Qdrant sera envisagé si des besoins spécifiques en filtrage vectoriel avancé émergent.

---

## 7. Évaluation des Systèmes RAG

### 7.1 Métriques d'Information Retrieval (IR)

#### MRR (Mean Reciprocal Rank)

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

#### NDCG@K

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

#### Recall@K

$$\text{Recall@K} = \frac{|\text{documents pertinents dans top-K}|}{|\text{total documents pertinents}|}$$

---

### 7.2 Métriques de génération (LLM-as-Judge)

| Métrique | Définition | Cible recommandée |
|---|---|---|
| **Faithfulness** | Les affirmations sont-elles ancrées dans les documents récupérés ? | > 3.5/5 |
| **Answer Relevancy** | La réponse répond-elle bien à la question posée ? | > 3.5/5 |
| **Context Precision** | Les chunks récupérés sont-ils tous utiles ? | > 3.0/5 |
| **Hallucination Rate** | Proportion d'affirmations inventées par le LLM | < 20% |

---

### 7.3 Framework RAGAS

[RAGAS](https://github.com/explodinggradients/ragas) est le framework de référence pour l'évaluation automatisée des pipelines RAG, sans annotation humaine.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
```

**Publication de référence** : Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. [EACL 2024](https://aclanthology.org/2024.eacl-demo.16/).

---

### 7.4 LLM-as-a-Judge — Choix LogiStore

| Modèle | Rôle | Raison |
|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b:free` | Génération RAG | Modèle puissant, gratuit sur OpenRouter |
| `openai/gpt-oss-20b:free` | Évaluation (LLM-as-Judge) | Suit strictement les instructions JSON ; Nemotron est un modèle "thinking" qui ignore les formats imposés |

**Prompt LLM-as-Judge utilisé** :

```python
FAITHFULNESS_PROMPT = """You are an evaluator. Reply ONLY with a single JSON object on one line.

Does this answer contain only information from the context?
CONTEXT: {context}
ANSWER: {answer}

Reply: {"score": X, "reason": "brief explanation"}
X: 0=hallucinated, 3=partially faithful, 5=fully faithful"""
```

---

## 8. Frameworks d'Orchestration

| Framework | Version stable | Stars GitHub | Cas d'usage principal |
|---|---|---|---|
| [LangChain](https://docs.langchain.com) | 0.3.x | 90K+ | Pipelines complexes, agents, tool-use |
| [LlamaIndex](https://docs.llamaindex.ai) | 0.10.x | 35K+ | RAG optimisé, indexation documentaire |
| [Haystack](https://haystack.deepset.ai) | 2.x | 17K+ | Moteurs de recherche IA, production |
| [DSPy](https://dspy.ai) | 2.x | 18K+ | Optimisation automatique des prompts |
| [RAGAS](https://github.com/explodinggradients/ragas) | 0.2.x | 7K+ | Évaluation RAG uniquement |

> **Choix LogiStore** : Python pur (modules maison) pour maîtriser le pipeline. LangChain ou LlamaIndex envisagés uniquement si des besoins d'agents ou de routing dynamique apparaissent.

---

## 9. Recommandations pour le MVP LogiStore

| Composant | Choix retenu | Justification |
|---|---|---|
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` | 50+ langues, dim=384, local/gratuit |
| **Moteur d'indexation** | OpenSearch 2.13 | BM25 + vectoriel natif, un seul service |
| **Recherche hybride** | BM25 + Cosine + RRF k=60 | Meilleur des deux mondes, sans calibration |
| **LLM génération** | `nvidia/nemotron-3-super-120b-a12b:free` | Disponible gratuitement sur OpenRouter |
| **LLM évaluation** | `openai/gpt-oss-20b:free` | Suit les instructions JSON (Nemotron ne convient pas) |
| **Chunking** | 1 ticket = 1 chunk | Unités sémantiques naturelles, médiane < 512 tokens |
| **Évaluation** | LLM-as-Judge (GPT-OSS-20B) | Sans annotation humaine, automatisable |
| **Orchestration** | Python pur | Maîtrise complète, pas de dépendance lourde |

**Trajectoire d'extension** :
MVP (J1-J10) Extension (J+)
────────────────── ──────────────────────────────
Naive RAG → Advanced RAG (HyDE, re-ranking)
OpenSearch → Qdrant (si perf vectorielle critique)
MiniLM-L12 (dim=384) → BGE-M3 self-hosted (dim=1024)
Streamlit → Interface métier intégrée au SI
Tickets SAV → Autres sources (PIM, GED, ERP)
RAG classique → GraphRAG (questions multi-entités)


---

## 10. Résultats d'évaluation — LogiStore

Évaluation réalisée sur 50 requêtes générées automatiquement (10 par langue) avec `deepseek/deepseek-v4-flash:free` puis nettoyées avec un prompt strict.

### Retrieval (BM25 + kNN HNSW, RRF k=60)

| Langue | MRR | nDCG@5 | Recall@5 | Precision@5 |
|---|---|---|---|---|
| **Global** | **0.391** | **0.423** | **0.520** | **0.104** |
| FR | 0.600 | 0.626 | 0.700 | 0.140 |
| EN | 0.453 | 0.515 | 0.700 | 0.140 |
| DE | 0.350 | 0.363 | 0.400 | 0.080 |
| ES | 0.275 | 0.306 | 0.400 | 0.080 |
| PT | 0.275 | 0.306 | 0.400 | 0.080 |

### RAG (Nemotron 120B génération · GPT-OSS-20B juge · 50 requêtes)

| Métrique | Score /5 | Interprétation |
|---|---|---|
| **Faithfulness** | 0.57 | Nemotron extrapole au-delà du contexte — axe d'amélioration prioritaire |
| **Answer Relevancy** | 2.37 | Réponses partiellement pertinentes |
| **Context Precision** | 3.06 | ✅ Le retrieval hybride ramène du contexte utile |

### Axes d'amélioration identifiés

1. **Faithfulness (0.57/5)** — Ajouter une instruction système *"réponds UNIQUEMENT à partir du contexte fourni"* et réduire `max_tokens` (512 → 256)
2. **PT/ES sous-performants (MRR=0.275)** — Envisager un re-ranker multilingue (cross-encoder) ou un modèle d'embedding spécialisé
3. **DE retrieval moyen (MRR=0.350)** — Augmenter le poids BM25 dans RRF pour les langues morphologiquement riches
4. **Context Precision** — Améliorer avec query expansion (HyDE ou RAG-Fusion)

---

## 11. Références Bibliographiques

### Articles fondateurs

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. [https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)
- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. EACL 2024. [https://aclanthology.org/2024.eacl-demo.16/](https://aclanthology.org/2024.eacl-demo.16/)
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.

### Techniques avancées

- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach*. Microsoft Research. [https://arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130)
- Chen, J. et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity*. [https://arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216)
- Shi, W. et al. (2024). *A Comprehensive Survey of RAG*. [https://arxiv.org/abs/2410.12837](https://arxiv.org/abs/2410.12837)

### Ressources pratiques

| Ressource | URL |
|---|---|
| MTEB Leaderboard | [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |
| Best Embedding Models 2026 — Prem AI | [https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/](https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/) |
| OpenSearch Documentation | [https://opensearch.org/docs/](https://opensearch.org/docs/) |
| Qdrant Documentation | [https://qdrant.tech/documentation/](https://qdrant.tech/documentation/) |
| RAGAS GitHub | [https://github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas) |
| LangChain Documentation | [https://docs.langchain.com](https://docs.langchain.com) |
| LlamaIndex Documentation | [https://docs.llamaindex.ai](https://docs.llamaindex.ai) |
| Hybrid Search — Weaviate | [https://weaviate.io/blog/hybrid-search-explained](https://weaviate.io/blog/hybrid-search-explained) |
| GraphRAG — Microsoft | [https://microsoft.github.io/graphrag/](https://microsoft.github.io/graphrag/) |
| Advanced RAG Techniques — Pinecone | [https://www.pinecone.io/learn/advanced-rag-techniques/](https://www.pinecone.io/learn/advanced-rag-techniques/) |