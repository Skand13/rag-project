# Veille Technologique — Systèmes RAG (Retrieval-Augmented Generation)

**Projet** : RAG-time — LogiStore  

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Architectures RAG](#2-architectures-rag)
   - 2.1 [RAG Naïf (Naive RAG)](#21-rag-naïf-naive-rag)
   - 2.2 [RAG Avancé (Advanced RAG)](#22-rag-avancé-advanced-rag)
   - 2.3 [RAG Modulaire (Modular RAG)](#23-rag-modulaire-modular-rag)
   - 2.4 [GraphRAG](#24-graphrag)
   - 2.5 [Comparatif des architectures](#25-comparatif-des-architectures)
3. [Stratégies de Chunking](#3-stratégies-de-chunking)
4. [Modèles d'Embeddings](#4-modèles-dembeddings)
   - 4.1 [Principes fondamentaux](#41-principes-fondamentaux)
   - 4.2 [Comparatif des modèles (MTEB 2025-2026)](#42-comparatif-des-modèles-mteb-2025-2026)
   - 4.3 [Recommandation pour LogiStore](#43-recommandation-pour-logistore)
5. [Recherche Hybride](#5-recherche-hybride)
   - 5.1 [BM25 — Recherche lexicale](#51-bm25--recherche-lexicale)
   - 5.2 [Recherche vectorielle](#52-recherche-vectorielle)
   - 5.3 [Fusion hybride — Reciprocal Rank Fusion (RRF)](#53-fusion-hybride--reciprocal-rank-fusion-rrf)
   - 5.4 [Alternatives avancées](#54-alternatives-avancées)
6. [Moteurs d'Indexation](#6-moteurs-dindexation)
   - 6.1 [OpenSearch](#61-opensearch)
   - 6.2 [Qdrant](#62-qdrant)
   - 6.3 [Tableau comparatif](#63-tableau-comparatif)
7. [Évaluation des Systèmes RAG](#7-évaluation-des-systèmes-rag)
   - 7.1 [Métriques d'Information Retrieval (IR)](#71-métriques-dinformation-retrieval-ir)
   - 7.2 [Métriques de génération (LLM)](#72-métriques-de-génération-llm)
   - 7.3 [Framework RAGAS](#73-framework-ragas)
   - 7.4 [LLM-as-a-Judge](#74-llm-as-a-judge)
8. [Frameworks d'Orchestration](#8-frameworks-dorchestration)
9. [Recommandations pour le MVP LogiStore](#9-recommandations-pour-le-mvp-logistore)
10. [Références Bibliographiques](#10-références-bibliographiques)

---

## 1. Introduction

Le RAG (*Retrieval-Augmented Generation*) est une architecture d'IA qui combine la recherche documentaire et la génération de texte par un LLM (Large Language Model). Introduit par Lewis et al. en 2020, le RAG répond à une limitation fondamentale des LLMs : leur base de connaissances est figée à la date d'entraînement et ne reflète pas les données internes ou récentes d'une organisation.

**Principe général** : plutôt que de faire confiance uniquement au LLM pour répondre, on lui fournit des documents pertinents récupérés en temps réel depuis une base documentaire. Le LLM synthétise alors sa réponse à partir de ce contexte.

![Architecture RAG générale — NVIDIA](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/retrieval-augmented-generation/_jcr_content/root/responsivegrid/nv_container_1290749/nv_container/nv_image.coreimg.100.1070.jpeg/1773694917823/genai-rag-diagram-1.jpeg)
*Source : [NVIDIA — RAG Diagram](https://www.nvidia.com/en-us/glossary/retrieval-augmented-generation/)*

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

![Pipeline RAG Naïf vs Avancé](https://unable-actionable-car.media.strapiapp.com/Naive_RAG_vs_Advanced_RAG_92a2a6ca72.png)
*Source : [Meilisearch — Naive RAG vs Advanced RAG](https://www.meilisearch.com/blog/naive-rag-vs-advanced-rag)*

**Flux de données** :

```
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
```

**Limites du Naive RAG** :
- Sensible à la qualité du chunking (si les chunks sont mal découpés, la récupération est mauvaise)
- Pas de reformulation de requête : une requête mal formulée donne de mauvais résultats
- Pas de filtrage ou de re-ranking : les K premiers résultats ne sont pas nécessairement les plus pertinents
- Risque de contexte bruité si les chunks sont trop longs ou hétérogènes

---

### 2.2 RAG Avancé (Advanced RAG)

Le RAG avancé introduit des étapes supplémentaires avant et après la récupération pour améliorer la pertinence.

![Évolution Naive → Advanced → Modular RAG](https://www.marktechpost.com/wp-content/uploads/2024/04/Screenshot-2024-04-01-at-12.44.59-PM.png)
*Source : [MarkTechPost — Evolution of RAGs](https://www.marktechpost.com/2024/04/01/evolution-of-rags-naive-rag-advanced-rag-and-modular-rag/)*

#### Techniques Pre-Retrieval

| Technique | Description | Avantage |
|---|---|---|
| **Query Rewriting** | Reformulation automatique de la requête par un LLM | Améliore le rappel |
| **HyDE** (Hypothetical Document Embeddings) | Le LLM génère un document hypothétique, qu'on utilise comme requête | Meilleur alignement sémantique |
| **Query Expansion** | Ajout de synonymes et termes liés | Augmente la couverture |
| **RAG-Fusion** | Génère plusieurs reformulations + RRF | Robustesse accrue |

#### Techniques Post-Retrieval

| Technique | Description | Avantage |
|---|---|---|
| **Cross-Encoder Re-ranking** | Réévaluation fine des K résultats par un modèle dédié | Précision accrue |
| **Compression contextuelle** | Réduction des chunks pour ne garder que les passages pertinents | Réduit le bruit dans le prompt |
| **Multi-hop RAG** | Chaîne de retrievals successifs pour répondre à des questions complexes | Questions multi-étapes |
| **Self-RAG** | Le modèle décide lui-même quand et quoi récupérer | Adaptabilité dynamique |

---

### 2.3 RAG Modulaire (Modular RAG)

Le RAG modulaire découple chaque composant du pipeline pour permettre une orchestration flexible, avec des modules interchangeables.

![Advanced RAG Architecture](https://d3lkc3n5th01x7.cloudfront.net/wp-content/uploads/2024/08/26051537/Advanced-RAG.png)
*Source : [LeewayHertz — Advanced RAG Architecture](https://www.leewayhertz.com/advanced-rag/)*

**Composants modulaires clés** :

- **Routing** : sélectionner dynamiquement la source de données selon la requête (base vectorielle, SQL, API externe)
- **Agents et Tool-use** : le LLM peut décider d'appeler des outils (calculatrice, API, BDD)
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

![GraphRAG — Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/06/GraphRAG-knowledge-graph_Fig1.png)
*Source : [Microsoft Research — GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/)*

**Principe** :
1. Les documents sont analysés pour extraire des entités et relations (ex : `client_123 → problème → livraison_retardée`)
2. Ces entités forment un graphe de connaissances
3. Lors du retrieval, on parcourt le graphe pour récupérer des contextes sémantiquement liés, pas seulement des chunks proches

**Avantage pour LogiStore** : pertinent pour des requêtes complexes du type *"Quels produits sont associés aux tickets de livraison retardée pour les clients premium ?"*, qui nécessitent de relier plusieurs entités.

> **Note :** GraphRAG est une option avancée hors périmètre du MVP. À envisager dans une trajectoire d'extension.

---

### 2.5 Comparatif des architectures

| Critère | Naive RAG | Advanced RAG | Modular RAG | GraphRAG |
|---|---|---|---|---|
| Complexité d'implémentation | Faible | Moyenne | Élevée | Très élevée |
| Qualité de retrieval | Correcte | Bonne | Excellente | Excellente |
| Coût API (tokens) | Faible | Moyen | Élevé | Élevé |
| Traçabilité des sources | Oui | Oui | Oui | Oui |
| Requêtes complexes multi-entités | Non | Partiel | Partiel | Oui |
| Recommandé pour MVP LogiStore | ✅ | ✅ (hybride) | ⚠️ Trajectoire | ⚠️ Trajectoire |

---

## 3. Stratégies de Chunking

Le découpage des documents en chunks est l'une des décisions les plus critiques dans un pipeline RAG. Un mauvais chunking dégrade irrémédiablement la qualité du retrieval, quelle que soit la sophistication du reste du pipeline.

![Stratégies de chunking pour RAG](https://cdn.prod.website-files.com/636e9a9a8d334e3450b08cc9/66f6015d895effddb4773bcf_66f600e60553b0f3c1bbc1ab_Chunking-In-RAG.webp)
*Source : [multimodal.dev — How to Chunk Documents for RAG](https://multimodal.dev/posts/rag-chunking)*

### Stratégies disponibles

| Stratégie | Description | Avantages | Inconvénients |
|---|---|---|---|
| **Fixed-size** | Chunks de N tokens fixes | Simple, reproductible | Peut couper une idée en deux |
| **Sliding Window** | Chevauchement entre chunks (overlap) | Préserve la continuité | Redondance, index plus volumineux |
| **Par structure** | Découpage par titre, paragraphe, section | Respecte la logique du document | Dépend de la qualité du formatage source |
| **Sémantique** | Découpage selon la cohérence sémantique | Chunks cohérents | Plus coûteux (nécessite un LLM ou embeddings) |
| **Par ticket** | Un chunk = un ticket complet (ou un champ) | Idéal pour les tickets courts | Limité si le ticket est très long |
| **Par champ** | Chunk = titre + description + résolution | Enrichissement metadata | Complexité de prétraitement |

### Recommandation pour les tickets SAV de LogiStore

Les tickets du dataset Kaggle sont des documents courts et structurés. La stratégie recommandée est **hybride** :

```python
# Stratégie recommandée pour les tickets SAV
chunk = {
    "id": ticket_id,
    "text": f"Sujet: {subject}\n\nDescription: {description}\n\nRésolution: {resolution}",
    "metadata": {
        "ticket_id": ticket_id,
        "category": category,
        "product": product,
        "priority": priority,
        "date": created_at
    }
}
```

- **Chunk size** : 256-512 tokens (adapté aux tickets courts)
- **Overlap** : 50 tokens pour les tickets longs
- **Metadata** : toujours inclure catégorie, produit, priorité pour filtrage

> **Règle d'or** : tester au moins 3 stratégies de chunking différentes et mesurer Precision@5 sur un jeu de requêtes de test avant de fixer la stratégie finale.

---

## 4. Modèles d'Embeddings

### 4.1 Principes fondamentaux

Les embeddings transforment un texte en vecteur dense dans un espace sémantique. Des textes sémantiquement similaires ont des vecteurs proches (distance cosinus faible).

![Espace vectoriel des embeddings](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Word_embedding_illustration.svg/1280px-Word_embedding_illustration.svg.png)
*Source : [Wikipedia — Word Embedding Illustration](https://en.wikipedia.org/wiki/Word_embedding)*

**Propriétés clés à évaluer** :

- **Dimension** : 384 à 4096 dimensions selon le modèle — plus de dimensions = plus précis mais plus coûteux en stockage
- **Contexte maximal** : longueur maximale en tokens que le modèle peut encoder (512 à 32 000 tokens)
- **Multilinguisme** : crucial pour LogiStore si les tickets peuvent être en français
- **Score MTEB** : benchmark de référence pour comparer les modèles sur des tâches de retrieval

---

### 4.2 Comparatif des modèles (MTEB 2025-2026)

Données issues du [MTEB Leaderboard Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) et du [classement Prem AI (mars 2026)](https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/).

| Modèle | MTEB Score | Contexte max | Dimensions | Coût / 1M tokens | Self-host | Licence | Multilingual |
|---|---|---|---|---|---|---|---|
| **Qwen3-Embedding-8B** | 70.58 | 32 000 | 7 168 | Gratuit | ✅ | Apache 2.0 | ✅ |
| **Gemini embedding-001** | 68.32 | 2 048 | 3 072 | $0.15 | ❌ | Propriétaire | ✅ |
| **NV-Embed-v2** (NVIDIA) | 69.32 | 32 768 | 4 096 | Gratuit | ✅ | CC-BY-NC-4.0 | ⚠️ |
| **text-embedding-3-large** (OpenAI) | 64.60 | 8 192 | 3 072 | $0.13 | ❌ | Propriétaire | ✅ |
| **text-embedding-3-small** (OpenAI) | 62.30 | 8 192 | 1 536 | $0.02 | ❌ | Propriétaire | ✅ |
| **voyage-3-large** | ~67+ | 32 000 | 2 048 | $0.06 | ❌ | Propriétaire | ✅ |
| **BAAI/bge-m3** | 63.00 | 8 192 | 1 024 | Gratuit | ✅ | MIT | ✅ (100+ langues) |
| **Cohere embed-v4** | 65.20 | 128 000 | 1 024 | $0.10 | VPC | Propriétaire | ✅ |
| **Jina embeddings-v3** | ~62+ | 8 192 | 1 024 | $0.018 | ✅ | CC-BY-NC-4.0 | ✅ |
| **all-MiniLM-L6-v2** | 56.30 | 512 | 384 | Gratuit | ✅ | Apache 2.0 | ❌ |
| **stella_en_1.5B_v5** | ~64+ | 8 192 | 1 024 | Gratuit | ✅ | MIT | ❌ |

> **Sources** : [Prem AI — Best Embedding Models for RAG 2026](https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/) · [Modal — MTEB Leaderboard](https://modal.com/blog/mteb-leaderboard-article) · [HuggingFace MTEB](https://huggingface.co/spaces/mteb/leaderboard)

---

### 4.3 Recommandation pour LogiStore

Le projet utilise **OpenRouter** comme fournisseur API. Deux options principales sont recommandées :

#### Option A — API managée via OpenRouter (recommandée pour le MVP)

```python
# Appel embedding via OpenRouter
import openai

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

response = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=["Problème de livraison commande #12345"]
)
embedding = response.data[0].embedding  # Vecteur 1536 dimensions
```

**Justification** : `text-embedding-3-small` offre un excellent rapport qualité/coût ($0.02/1M tokens), supporte le multilinguisme, et est directement accessible via OpenRouter. Pour un MVP à 200K tickets, le coût total d'indexation est estimé à **< 0.50 USD**.

#### Option B — Self-hosted BAAI/bge-m3 (recommandée pour production)

```bash
# Lancement du service d'embedding local (Docker)
docker run -p 8080:80 \
  -e MODEL_ID=BAAI/bge-m3 \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
```

**Justification** : BGE-M3 est multilingue (français inclus), génère simultanément des représentations denses ET sparses (utile pour la recherche hybride), et ne dépend d'aucune API externe. Score MTEB de 63.0 avec licence MIT.

---

## 5. Recherche Hybride

La recherche hybride combine la recherche lexicale (BM25) et la recherche vectorielle pour obtenir un meilleur rappel et une meilleure précision que chacune des deux approches seules.

![Recherche hybride BM25 + Vectoriel](https://www.myscale.com/docs/assets/img/hybrid_search_explain.5ba51866.png)
*Source : [MyScale — Hybrid Search Explain](https://myscale.com/docs/en/hybrid-search/)*

---

### 5.1 BM25 — Recherche lexicale

BM25 (Best Match 25) est l'algorithme de recherche full-text le plus utilisé. Il pondère les termes selon leur fréquence dans le document (TF) et leur rareté dans le corpus (IDF).

**Formule BM25** :

$$\text{score}(d, q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

Avec :
- `f(qi, d)` : fréquence du terme qi dans le document d
- `|d|` : longueur du document
- `avgdl` : longueur moyenne des documents
- `k1 ≈ 1.2`, `b ≈ 0.75` : paramètres de saturation

**Forces de BM25** :
- Exact match : idéal pour les numéros de commande, codes produits, termes techniques
- Robuste et rapide, sans GPU
- Natif dans OpenSearch (`bm25` scorer)
- Interprétable : on comprend pourquoi un document est remonté

**Limites** :
- Insensible à la sémantique : "envoi retardé" ≠ "livraison en retard" pour BM25
- Pas de compréhension contextuelle

---

### 5.2 Recherche vectorielle

La recherche vectorielle encode la requête et les documents en vecteurs denses, puis trouve les plus proches voisins par similarité cosinus.

**Forces** :
- Compréhension sémantique : "délai d'expédition" ≈ "livraison tardive"
- Robustesse aux fautes de frappe et paraphrases
- Multilingue avec les bons modèles

**Limites** :
- Nécessite un GPU pour la génération des embeddings à grande échelle
- Ne retrouve pas les termes exacts rares (numéros de commande, codes)

---

### 5.3 Fusion hybride — Reciprocal Rank Fusion (RRF)

Le RRF fusionne les listes de résultats BM25 et vectoriel en un classement unifié, sans nécessiter de calibration des scores bruts.

![Reciprocal Rank Fusion — Weaviate](https://weaviate.io/assets/images/reciprocal-rank-41eb42456436a9e44ae302611711b691.png)
*Source : [Weaviate — A Web Developer's Guide to Hybrid Search](https://weaviate.io/blog/hybrid-search-explained)*

**Formule RRF** :

$$\text{RRF\_score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Avec :
- `R` : ensemble des listes de ranking (BM25 + vectoriel)
- `rank_r(d)` : rang du document d dans la liste r
- `k = 60` : constante de lissage (valeur empirique recommandée)

**Exemple pratique** :

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

### 5.4 Alternatives avancées

| Technique | Principe | Cas d'usage |
|---|---|---|
| **SPLADE** | Représentations sparses apprises par un modèle | Hybridation sans BM25 classique |
| **ColBERT** | Late interaction token-level entre query et document | Haute précision, coûteux |
| **Reranking Cross-Encoder** | Réévaluation des top-K résultats par un modèle plus lourd | Amélioration post-retrieval |

**Reranking avec un Cross-Encoder** (optionnel pour LogiStore) :

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, doc["text"]) for doc in top_k_results]
scores = reranker.predict(pairs)
reranked = sorted(zip(top_k_results, scores), key=lambda x: x[1], reverse=True)
```

---

## 6. Moteurs d'Indexation

### 6.1 OpenSearch

[OpenSearch](https://opensearch.org/docs/) est un moteur de recherche open source dérivé d'Elasticsearch (fork Apache 2.0, maintenu par AWS). Il supporte nativement BM25 et le recherche vectorielle via le plugin `knn`.

**Installation via Docker** :

```yaml
# docker-compose.yml
services:
  opensearch:
    image: opensearchproject/opensearch:2.13.0
    environment:
      - discovery.type=single-node
      - DISABLE_SECURITY_PLUGIN=true
      - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - opensearch-data:/usr/share/opensearch/data
```

**Mapping hybride BM25 + k-NN** :

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 100
    }
  },
  "mappings": {
    "properties": {
      "text": { "type": "text", "analyzer": "standard" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "lucene"
        }
      },
      "category": { "type": "keyword" },
      "product": { "type": "keyword" },
      "priority": { "type": "keyword" },
      "date": { "type": "date" }
    }
  }
}
```

---

### 6.2 Qdrant

[Qdrant](https://qdrant.tech/documentation/) est un moteur vectoriel spécialisé, optimisé pour la recherche par similarité. Il supporte le filtrage avancé par metadata, le sparse vectoriel (SPLADE), et gRPC pour de hautes performances.

**Installation via Docker** :

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.9.0
    ports:
      - "6333:6333"
      - "6334:6334"  # gRPC
    volumes:
      - qdrant-data:/qdrant/storage
```

**Création de collection** :

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="logistor_tickets",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)
```

---

### 6.3 Tableau comparatif

| Critère | OpenSearch 2.13 | Qdrant 1.9 |
|---|---|---|
| **Licence** | Apache 2.0 | Apache 2.0 |
| **BM25 natif** | ✅ Oui | ⚠️ Via sparse vectors |
| **Recherche vectorielle** | ✅ Plugin k-NN (HNSW) | ✅ Natif (HNSW, filtrage avancé) |
| **Recherche hybride** | ✅ BM25 + k-NN | ✅ Dense + Sparse (SPLADE) |
| **Filtrage par metadata** | ✅ Queries complexes | ✅ Filtres intégrés à la requête vectorielle |
| **Interface** | REST + Dashboards OpenSearch | REST + gRPC + SDK Python |
| **Scalabilité** | Excellente (cluster multi-nœuds) | Bonne (cluster en v1.7+) |
| **Consommation RAM (1 nœud)** | 512 MB min recommandé | 256 MB min |
| **Écosystème IA** | Intégré AWS, LangChain | LangChain, LlamaIndex, Haystack |
| **Dashboard de visualisation** | ✅ OpenSearch Dashboards | ❌ (API only, ou Qdrant Web UI) |
| **Recommandé pour** | Hybride BM25 + vectoriel | Vectoriel pur ou sparse+dense |
| **Verdict pour LogiStore MVP** | ✅ **Recommandé** | ✅ Option solide |

> **Recommandation** : pour le MVP LogiStore, **OpenSearch** est privilégié car il offre le BM25 natif (essentiel pour les recherches par numéros de tickets et termes exacts) combiné à la recherche vectorielle, le tout dans un seul service. Qdrant sera envisagé si des besoins spécifiques en filtrage vectoriel avancé émergent.

---

## 7. Évaluation des Systèmes RAG

L'évaluation est une composante critique souvent négligée. Un système RAG qui "semble bien marcher" peut avoir un taux d'hallucination inacceptable ou rater 40% des documents pertinents.

---

### 7.1 Métriques d'Information Retrieval (IR)

Ces métriques évaluent la qualité du **retrieval** (étape de récupération des documents).

#### Precision@K

Proportion de documents pertinents parmi les K documents récupérés.

$$\text{Precision@K} = \frac{|\text{documents pertinents dans top-K}|}{K}$$

**Exemple** : sur 5 documents récupérés (K=5), 3 sont pertinents → Precision@5 = 0.60

#### Recall@K

Proportion de documents pertinents récupérés parmi tous les documents pertinents existants.

$$\text{Recall@K} = \frac{|\text{documents pertinents dans top-K}|}{|\text{total documents pertinents}|}$$

#### MRR (Mean Reciprocal Rank)

Mesure la position du premier document pertinent dans la liste.

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

**Exemple** : si le 1er document pertinent est en position 2 → score = 1/2 = 0.5

#### NDCG@K (Normalized Discounted Cumulative Gain)

Prend en compte l'ordre des résultats et leur degré de pertinence (gradué).

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

**Implémentation** :

```python
def ndcg_at_k(retrieved_docs, relevant_docs, k):
    dcg = sum(
        (1 / math.log2(rank + 2))
        for rank, doc in enumerate(retrieved_docs[:k])
        if doc in relevant_docs
    )
    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant_docs), k)))
    return dcg / idcg if idcg > 0 else 0
```

---

### 7.2 Métriques de génération (LLM)

Ces métriques évaluent la qualité de la **réponse générée** par le LLM.

| Métrique | Définition | Cible |
|---|---|---|
| **Faithfulness** | Les affirmations de la réponse sont-elles ancrées dans les documents récupérés ? | > 85% |
| **Answer Relevancy** | La réponse répond-elle bien à la question posée ? | > 80% |
| **Context Precision** | Les chunks récupérés sont-ils tous utiles pour répondre ? | > 75% |
| **Context Recall** | Les chunks nécessaires pour répondre ont-ils tous été récupérés ? | > 70% |
| **Hallucination Rate** | Proportion d'affirmations inventées par le LLM | < 5% |
| **ROUGE-L** | Chevauchement de sous-séquences avec une réponse de référence | > 0.40 |
| **BERTScore** | Similarité sémantique avec une réponse de référence | > 0.80 |

---

### 7.3 Framework RAGAS

[RAGAS](https://github.com/explodinggradients/ragas) est le framework de référence pour l'évaluation automatisée des pipelines RAG, sans annotation humaine.

![RAGAS Metrics Cheat Sheet](https://safjan.com/images/ragas_metrics_cheat_sheet/RAGAS_metrics_cheat_sheet_v1.jpg)
*Source : [Krystian Safjan's Blog — RAGAS Metrics Cheat Sheet](https://safjan.com/ragas-metrics-cheat-sheet/)*

**Installation et usage** :

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Données d'évaluation
data = {
    "question": ["Pourquoi ma commande #12345 est en retard ?"],
    "answer": ["Votre commande est retardée en raison d'un problème logistique."],
    "contexts": [["Ticket #12345 : problème logistique détecté le 10/04/2026"]],
    "ground_truth": ["La commande est retardée pour raison logistique"]
}

dataset = Dataset.from_dict(data)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(results)
```

**Publication de référence** : Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. [ACL Anthology 2024](https://aclanthology.org/2024.eacl-demo.16/).

---

### 7.4 LLM-as-a-Judge

Utiliser un LLM plus puissant (ex : GPT-4o) pour noter automatiquement les réponses générées par le système RAG. Cette technique remplace ou complète les annotations humaines.

**Prompt template LLM-as-a-Judge** :

```python
JUDGE_PROMPT = """
Tu es un expert en évaluation de systèmes RAG. Évalue la réponse suivante.

Question : {question}
Contexte récupéré : {context}
Réponse générée : {answer}

Évalue sur 3 critères (score 1-5) :
1. Fidélité (la réponse est-elle ancrée dans le contexte ?)
2. Pertinence (la réponse répond-elle à la question ?)
3. Complétude (la réponse est-elle exhaustive ?)

Réponds en JSON : {{"fidelite": X, "pertinence": X, "completude": X, "justification": "..."}}
"""
```

> **Source** : [Thinkscoop — RAG Evaluation Metrics 2023-2024](https://thinkscoopinc.com/blog/rag-evaluation-metrics)

---

## 8. Frameworks d'Orchestration

| Framework | Version stable | Stars GitHub | Cas d'usage principal |
|---|---|---|---|
| [LangChain](https://docs.langchain.com) | 0.3.x | 90K+ | Pipelines complexes, agents, tool-use |
| [LlamaIndex](https://docs.llamaindex.ai) | 0.10.x | 35K+ | RAG optimisé, indexation documentaire |
| [Haystack](https://haystack.deepset.ai) | 2.x | 17K+ | Moteurs de recherche IA, production |
| [DSPy](https://dspy.ai) | 2.x | 18K+ | Optimisation automatique des prompts |
| [RAGAS](https://github.com/explodinggradients/ragas) | 0.2.x | 7K+ | Évaluation RAG uniquement |

> **Recommandation pour LogiStore MVP** : démarrer sans framework d'orchestration (code Python pur) pour maîtriser le pipeline. Intégrer LangChain ou LlamaIndex uniquement si des besoins d'agents ou de routing dynamique apparaissent.

---

## 9. Recommandations pour le MVP LogiStore

Synthèse des choix technologiques recommandés pour le MVP, basés sur cette veille :

| Composant | Choix recommandé | Justification |
|---|---|---|
| **Embeddings** | `text-embedding-3-small` via OpenRouter | Coût minimal, multilingue, intégration directe |
| **Fallback self-hosted** | `BAAI/bge-m3` | Open source, hybride dense+sparse, français |
| **Moteur d'indexation** | **OpenSearch 2.13** | BM25 + vectoriel natif, un seul service |
| **Recherche hybride** | **BM25 + Cosine + RRF** | Meilleur des deux mondes, implémentation simple |
| **LLM** | `openai/gpt-4o-mini` via OpenRouter | Rapport qualité/coût excellent pour SAV |
| **Chunking** | Par ticket (sujet + description + résolution) | Adapté aux tickets courts, metadata riche |
| **Évaluation** | RAGAS + LLM-as-a-Judge | Sans annotation humaine, automatisable |
| **Orchestration** | Python pur (modules maison) | Maîtrise complète, pas de dépendance lourde |

**Trajectoire d'extension** :

```
MVP (J1-J10)          Extension (J+)
─────────────         ──────────────────────────
Naive RAG          →  Advanced RAG (HyDE, re-ranking)
OpenSearch         →  Qdrant (si perf vectorielle critique)
API OpenRouter     →  BGE-M3 self-hosted (réduction coûts)
Streamlit          →  Interface métier intégrée au SI
Tickets SAV        →  Autres sources (PIM, GED, ERP)
RAG classique      →  GraphRAG (questions multi-entités)
```

---

## 10. Références Bibliographiques

### Articles fondateurs

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. [https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)

- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. EACL 2024. [https://aclanthology.org/2024.eacl-demo.16/](https://aclanthology.org/2024.eacl-demo.16/)

### Techniques avancées

- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. Microsoft Research. [https://arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130)

- Chen, J. et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. [https://arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216)

- Shi, W. et al. (2024). *A Comprehensive Survey of Retrieval-Augmented Generation (RAG)*. [https://arxiv.org/abs/2410.12837](https://arxiv.org/abs/2410.12837)

- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.

### Ressources pratiques

| Ressource | URL |
|---|---|
| MTEB Leaderboard (Hugging Face) | [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |
| Best Embedding Models 2026 — Prem AI | [https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/](https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/) |
| LangChain Documentation | [https://docs.langchain.com](https://docs.langchain.com) |
| LlamaIndex Documentation | [https://docs.llamaindex.ai](https://docs.llamaindex.ai) |
| OpenSearch Documentation | [https://opensearch.org/docs/](https://opensearch.org/docs/) |
| Qdrant Documentation | [https://qdrant.tech/documentation/](https://qdrant.tech/documentation/) |
| RAGAS GitHub | [https://github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas) |
| Advanced RAG Techniques — Pinecone | [https://www.pinecone.io/learn/advanced-rag-techniques/](https://www.pinecone.io/learn/advanced-rag-techniques/) |
| Hybrid Search — Weaviate | [https://weaviate.io/blog/hybrid-search-explained](https://weaviate.io/blog/hybrid-search-explained) |
| GraphRAG — Microsoft | [https://microsoft.github.io/graphrag/](https://microsoft.github.io/graphrag/) |
| RAG Evaluation Metrics — Thinkscoop | [https://thinkscoopinc.com/blog/rag-evaluation-metrics](https://thinkscoopinc.com/blog/rag-evaluation-metrics) |

