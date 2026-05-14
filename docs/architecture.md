### Architecture complète du système RAG

```mermaid
flowchart TD

    %% =====================
    %% SOURCES
    %% =====================
    subgraph Sources["Sources de données"]
        A1["Kaggle Dataset<br/>Tickets SAV"]
        A2["GED / SharePoint"]
        A3["API Zendesk (futur)"]
    end

    %% =====================
    %% INGESTION
    %% =====================
    subgraph Ingestion["Pipeline d'ingestion"]
        B1["Loader<br/>CSV / JSON / API"]
        B2["Préprocesseur<br/>Nettoyage, normalisation"]
        B3["Chunker<br/>Segmentation des tickets"]
        B4["Embedder<br/>OpenRouter API"]
    end

    %% =====================
    %% STORAGE
    %% =====================
    subgraph Storage["Stockage & Indexation"]
        C1[("OpenSearch<br/>BM25 + kNN")]
        C2[("Qdrant<br/>Vectoriel")]
        C3[("MinIO / S3<br/>Data Lake")]
    end

    %% =====================
    %% RETRIEVAL
    %% =====================
    subgraph Retrieval["Moteur de recherche"]
        D1["BM25 Search"]
        D2["Vector Search"]
        D3["Hybrid Fusion<br/>RRF"]
        D4["Re-Ranker<br/>Cross-Encoder"]
    end

    %% =====================
    %% LLM
    %% =====================
    subgraph LLM["Couche LLM"]
        E1["Query Rewriter"]
        E2["Context Summarizer"]
        E3["Answer Generator<br/>OpenRouter API"]
    end

    %% =====================
    %% FRONTEND
    %% =====================
    subgraph Frontend["Frontend (Streamlit)"]
        F1["Barre de recherche"]
        F2["Filtres"]
        F3["Résultats + scores"]
        F4["Chunks contextuels"]
    end

    %% =====================
    %% FLOW INGESTION
    %% =====================
    A1 --> B1
    A2 --> B1
    A3 --> B1

    B1 --> B2 --> B3 --> B4
    B2 --> C3
    B4 --> C1
    B4 --> C2

    %% =====================
    %% FLOW QUERY
    %% =====================
    F1 --> E1
    E1 --> D1
    E1 --> D2

    D1 --> D3
    D2 --> D3

    D3 --> D4
    D4 --> E2
    E2 --> E3

    %% =====================
    %% OUTPUT
    %% =====================
    E3 --> F3
    D4 --> F3
    F3 --> F4
    
````

    
### Choix technologiques justifiés
Moteur d'indexation : OpenSearch (recommandé pour MVP)

Justification :

    - BM25 natif, mature, éprouvé en production

    - Plugin k-NN pour la recherche vectorielle (HNSW)

    - Dashboards OpenSearch pour monitoring des index

    - Déploiement Docker simple, compatible avec une migration vers AWS OpenSearch Service

    - Faiblesse : consommation mémoire plus élevée que Qdrant

**Alternative Qdrant:** privilégier si la recherche vectorielle pure est prioritaire, ou si la scalabilité sur gros volumes vectoriels est critique.




Modèles d'embeddings (via OpenRouter)

| Modèle                        | Dimensions | Avantages                      | Coût            |
| ----------------------------- | ---------- | ------------------------------ | --------------- |
| openai/text-embedding-3-small | 1536       | Polyvalent, rapide, économique | $0.02/1M tokens |
| openai/text-embedding-3-large | 3072       | Meilleure qualité              | $0.13/1M tokens |
| BAAI/bge-m3 (local)           | 1024       | Multilingue, gratuit, hybride  | Compute local   |

Recommandation MVP: text-embedding-3-small pour maîtriser les coûts.
LLM (via OpenRouter)
;


| Modèle                        | Contexte | Usage recommandé                   |
| ----------------------------- | -------- | ---------------------------------- |
| openai/gpt-4o-mini            | 128K     | Synthèse, réponse SAV (économique) |
| openai/gpt-4o                 | 128K     | Évaluation LLM-as-a-Judge          |
| anthropic/claude-3-haiku      | 200K     | Alternative économique             |
| mistralai/mistral-7b-instruct | 32K      | Option open source via OpenRouter  |



Flux de données détaillé

[Source CSV Kaggle]
       │
       ▼
[Loader] ──► lire le CSV ligne par ligne, typage des colonnes
       │
       ▼
[Préprocesseur]
  - Suppression des doublons
  - Normalisation des dates (ISO 8601)
  - Nettoyage du texte (HTML, caractères spéciaux)
  - Anonymisation/pseudonymisation (noms, emails, téléphones)
       │
       ▼
[Chunker]
  - Stratégie : un chunk = un ticket complet (si < 512 tokens)
  - Fallback : fenêtre glissante (512 tokens, overlap 64)
  - Métadonnées : ticket_id, category, product, date, status
       │
       ▼
[Embedder] ──► appel API OpenRouter (batch de 100)
  - Génère un vecteur par chunk
  - Stocke (chunk_id, vecteur, métadonnées) dans Qdrant/OpenSearch
       │
       ▼
[Index BM25] ──► indexation full-text dans OpenSearch
[Index vectoriel] ──► indexation dans OpenSearch k-NN ou Qdrant
       │
       ▼
[Query Pipeline]
  - Requête utilisateur → nettoyage → embedding
  - BM25 search → top-K résultats (K=20)
  - Vector search → top-K résultats (K=20)
  - RRF fusion → top-10
  - Re-ranking optionnel (Cross-Encoder)
       │
       ▼
[LLM Layer optionnel]
  - Contexte = top-3 chunks
  - Prompt : "Voici des tickets similaires : [...]. Génère une réponse suggérée."
       │
       ▼
[Frontend Streamlit]
  - Affichage des résultats avec score, catégorie, date
  - Explication du ranking (score BM25, score vectoriel, score final)