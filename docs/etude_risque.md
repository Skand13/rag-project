# Étude de Risque — Projet RAG-time LogiStore

**Version :** 1.0  
**Date :** Juillet 2025  
**Auteurs :** Équipe Projet RAG-time
**Destinataires :** Direction Informatique, Responsable SAV, DPO LogiStore  
**Classification :** Usage interne — Confidentiel

---

## 1. Introduction

### 1.1 Périmètre de l'analyse

La présente étude de risque porte sur le système **RAG-time LogiStore**, un moteur de recherche augmenté par génération (Retrieval-Augmented Generation) développé en mode MVP sur une période de **10 jours par une équipe de 3 développeurs**. Ce système est destiné aux **30 agents du Service Après-Vente (SAV)** de LogiStore, ETI logistique d'environ 500 employés, pour leur permettre d'interroger en langage naturel une base de **3 999 tickets SAV historiques**, indexés en **5 langues** (anglais, français, allemand, espagnol, portugais).

**Composants techniques couverts par l'analyse :**

| Composant | Technologie | Rôle |
|---|---|---|
| Moteur de recherche | OpenSearch 2.13 (BM25 + kNN HNSW) | Indexation et retrieval hybride |
| Modèle d'embedding | `paraphrase-multilingual-MiniLM-L12-v2` (dim=384) | Vectorisation locale des tickets |
| LLM de génération | Nemotron 120B via OpenRouter | Synthèse des réponses |
| Interface utilisateur | Streamlit (Python) | Front-end agents SAV |
| CI/CD | GitHub Actions (ruff + pytest) | Automatisation tests/lint |
| Modèle d'évaluation | `gpt-oss-20b:free` (OpenRouter) | Métriques RAGAS |

Le périmètre exclut les systèmes amont de création de tickets (CRM/ERP), les infrastructures réseau LogiStore non directement liées au projet, et les postes de travail des agents.

### 1.2 Méthodologie

L'analyse est conduite selon une approche inspirée d'**EBIOS Risk Manager** (ANSSI, version 2018), adaptée au contexte d'un système d'IA générative. Elle est complétée par les référentiels suivants :

- **OWASP Top 10 for LLM Applications** (version 1.1) pour les risques spécifiques aux LLM
- **ENISA — AI Threat Landscape** pour les risques liés aux systèmes d'apprentissage automatique
- **RGPD (Règlement (UE) 2016/679)** pour les aspects conformité données personnelles
- **ISO/IEC 27001:2022** pour le cadre général de la sécurité de l'information

La cotation des risques suit l'échelle suivante :

| Niveau | Probabilité (P) | Impact (I) |
|---|---|---|
| 1 | Improbable (< 10% sur 12 mois) | Mineur (perturbation localisée, < 1 jour) |
| 2 | Peu probable (10–30%) | Significatif (perturbation multi-équipes, quelques jours) |
| 3 | Probable (30–60%) | Grave (arrêt de service, impact client, semaine) |
| 4 | Très probable (> 60%) | Critique (perte de données, sanctions réglementaires) |

La **criticité** est calculée comme le produit **P × I**. Les risques avec criticité ≥ 9 sont considérés **critiques** et font l'objet d'une analyse détaillée.

### 1.3 Niveau de criticité du système

RAG-time LogiStore est classé **niveau de sensibilité MOYEN** pour les critères de disponibilité et d'intégrité, et **ÉLEVÉ** pour la confidentialité, en raison de la présence probable de données à caractère personnel dans les tickets SAV (noms de clients, adresses de livraison, numéros de commandes). Le système n'est pas qualifié OIV/OSE, mais est soumis aux obligations RGPD en tant que traitement de données personnelles de clients.

---

## 2. Tableau des Risques

| ID | Catégorie | Description | P | I | Criticité | Mesures de mitigation |
|---|---|---|---|---|---|---|
| R01 | IA — Hallucination | Nemotron 120B génère des réponses non ancrées dans les tickets (Faithfulness = 0.57/5) | 4 | 3 | **12** | Affichage systématique des sources, garde-fou de confiance, prompt engineering |
| R02 | IA — Biais linguistique | MRR=0.275 pour ES/PT : mauvaise qualité de retrieval pour 2 langues sur 5 | 4 | 2 | **8** | Fine-tuning ou modèle d'embedding dédié, tests par langue |
| R03 | IA — Dérive modèle | Changement de comportement de Nemotron 120B lors de mises à jour OpenRouter sans contrôle de version | 3 | 3 | **9** | Pinning de version API, benchmarks de non-régression |
| R04 | IA — Surconfiance utilisateur | Les agents SAV font confiance aux réponses sans vérifier les sources (Answer Relevancy = 2.37/5) | 3 | 3 | **9** | Formation utilisateurs, interface avec score de confiance visible |
| R05 | Sécurité — Fuite clé API | Clé OpenRouter committée sur GitHub (incident avéré, révoquée) | 4 | 4 | **16** | Rotation obligatoire, secrets manager, scan automatique commits |
| R06 | Sécurité — Prompt injection | Un utilisateur malveillant insère des instructions dans sa requête pour détourner le LLM | 2 | 3 | **6** | Sanitisation des entrées, monitoring des requêtes atypiques |
| R07 | Conformité — RGPD | Les tickets SAV indexés contiennent potentiellement des données personnelles clients | 4 | 4 | **16** | PIA, anonymisation, contrôle d'accès strict, durées de conservation |
| R08 | Conformité — Rétention données | Absence de politique de purge des tickets dans OpenSearch | 3 | 3 | **9** | Politique TTL sur l'index, processus de suppression sur demande |
| R09 | Opérationnel — Disponibilité OpenSearch | OpenSearch sur Docker local sans haute disponibilité ni backup validé | 3 | 3 | **9** | Cluster multi-nœuds, snapshots automatiques, RTO/RPO définis |
| R10 | Opérationnel — Rate limiting OpenRouter | Dépassement des quotas OpenRouter en charge normale ou pic SAV | 3 | 2 | **6** | Gestion des erreurs 429, fallback local, monitoring quotas |
| R11 | Opérationnel — Montée en charge | Streamlit non adapté à une utilisation simultanée par 30 agents | 3 | 2 | **6** | Passage à FastAPI+frontend, load testing, pooling connexions |
| R12 | Opérationnel — Qualité retrieval | Context Precision = 3.06/5, Recall@5 = 0.520 : résultats insuffisants pour des décisions fiables | 3 | 3 | **9** | Amélioration chunking, reranking, expansion de requêtes |
| R13 | Technique — Bug OpenSearch 2.13 | Bug natif sur filtre kNN avec bool/filter nécessitant un contournement non documenté | 3 | 2 | **6** | Documentation workaround, veille correctifs, tests de régression |
| R14 | Technique — Dette technique MVP | Code MVP (10 jours, 3 développeurs) avec couverture de tests partielle | 3 | 3 | **9** | Plan de remboursement dette technique, code review, couverture > 80% |
| R15 | Dépendance fournisseur | Dépendance totale à OpenRouter pour le LLM (Nemotron 120B + modèle d'évaluation) | 3 | 3 | **9** | Stratégie multi-cloud LLM, évaluation déploiement local |
| R16 | Sécurité — Contrôle d'accès | Streamlit sans authentification native robuste : accès potentiellement non restreint | 2 | 3 | **6** | Intégration SSO/LDAP LogiStore, middleware d'authentification |

---

## 3. Analyse Détaillée des Risques Critiques (Criticité ≥ 9)

### R01 — Hallucination du LLM Nemotron 120B (Criticité : 12)

**Contexte technique :** Le score de **Faithfulness mesuré à 0.57/5** lors de l'évaluation RAGAS est particulièrement préoccupant. Ce score, calculé via le modèle `gpt-oss-20b:free` (seul modèle retournant du JSON structuré utilisable pour l'évaluation), indique que Nemotron 120B produit régulièrement des affirmations non étayées par les chunks récupérés. Ce comportement est structurel : avec un chunking « 1 ticket = 1 chunk », le contexte fourni au LLM est parfois insuffisant pour répondre à des questions complexes. Le modèle « comble les trous » avec ses connaissances paramètriques, générant des réponses plausibles mais incorrectes.

**Impact métier :** Un agent SAV recevant une réponse hallucinée sur la procédure de retour d'un produit ou sur l'historique d'une commande peut transmettre des informations erronées au client final. Dans un contexte logistique, cela peut entraîner des litiges, des remboursements injustifiés, ou une dégradation de la satisfaction client mesurable.

**Mesures de mitigation :**
1. **Affichage obligatoire des sources** : chaque réponse doit présenter les 3 tickets sources avec leur ID et un extrait textuel, permettant à l'agent de vérifier.
2. **Score de confiance visible** : exposer le score de faithfulness calculé en temps réel (via un LLM-as-judge léger) dans l'interface Streamlit.
3. **Prompt engineering strict** : instruction système explicite du type *"Réponds uniquement en te basant sur les tickets fournis. Si l'information n'est pas présente, dis-le explicitement."*
4. **Seuil de refus** : si le score de faithfulness estimé est inférieur à un seuil (ex. 0.4), afficher un avertissement *"Réponse incertaine — consulter les tickets directement"*.
5. **Reranking des chunks** : un cross-encoder léger post-retrieval améliorerait le Context Precision (3.06/5) et réduirait mécaniquement les hallucinations par amélioration du contexte fourni.

---

### R05 — Fuite de Clé API OpenRouter sur GitHub (Criticité : 16)

**Contexte technique :** Un **incident de sécurité avéré** a eu lieu durant le développement : une clé API OpenRouter a été committée dans le dépôt GitHub, puis révoquée. Ce type d'incident, classé **LLM05 – Supply Chain Vulnerability** dans l'OWASP Top 10 LLM, expose l'organisation à plusieurs vecteurs de risque :

- **Usage frauduleux de la clé** pendant la fenêtre d'exposition (entre le commit et la révocation) : facturation non maîtrisée, usage d'OpenRouter pour des activités malveillantes imputables à LogiStore.
- **Exfiltration de prompts** : si la clé est utilisée par un attaquant, les tickets SAV transmis en contexte au LLM sont potentiellement exposés (données personnelles clients).
- **Persistance dans l'historique Git** : même après révocation, la clé reste dans l'historique des commits et doit être expurgée (git-filter-repo ou BFG Repo Cleaner).

**Mesures de mitigation :**
1. **Expurgation immédiate de l'historique Git** via `git-filter-repo --path-glob '*.env' --invert-paths` ou BFG Repo Cleaner sur tous les commits contenant la clé.
2. **Secrets manager** : migration vers un vault (HashiCorp Vault, AWS Secrets Manager, ou variables d'environnement chiffrées GitHub Actions) — **aucune clé en clair dans le code**.
3. **Scan automatique** : intégrer `truffleHog` ou `gitleaks` dans la pipeline GitHub Actions pour bloquer tout commit contenant des secrets.
4. **Rotation préventive** : politique de rotation des clés OpenRouter tous les 90 jours, documentée.
5. **Audit de l'usage passé** : vérifier les logs OpenRouter pour la période d'exposition afin de détecter tout usage non autorisé.

---

### R07 — Non-conformité RGPD (Criticité : 16)

**Contexte technique :** Les **3 999 tickets SAV indexés dans OpenSearch** sont susceptibles de contenir des données à caractère personnel au sens de l'article 4(1) du RGPD : noms de clients, adresses postales (livraison/facturation), numéros de téléphone, adresses e-mail, parfois des informations sur des situations personnelles (produits médicaux, etc.). Ces données sont :

- **Vectorisées** par `paraphrase-multilingual-MiniLM-L12-v2` et stockées dans les index kNN d'OpenSearch.
- **Transmises en contexte** à Nemotron 120B via l'API OpenRouter (infrastructure hors UE potentielle).
- **Accessibles aux 30 agents SAV** sans mécanisme de contrôle fin des accès.

**Risques réglementaires :** Absence de registre de traitement documentant ce traitement, absence de PIA (Privacy Impact Assessment) obligatoire pour un traitement automatisé à grande échelle (Art. 35 RGPD), transfert potentiel de données hors UE sans garanties adéquates (OpenRouter/Nemotron).

**Mesures de mitigation :**
1. **PIA immédiat** : le DPO LogiStore doit conduire une analyse d'impact sur la protection des données avant mise en production.
2. **Anonymisation/pseudonymisation** : pipeline de prétraitement masquant les PII (noms, e-mails, téléphones, adresses) avant indexation dans OpenSearch, via des outils comme `presidio` de Microsoft.
3. **Vérification des garanties OpenRouter** : s'assurer que le fournisseur offre des clauses contractuelles types (CCT) ou est hébergé en UE, documenter dans le registre de traitement.
4. **Contrôle d'accès basé sur les rôles (RBAC)** : implémenter des index-level security dans OpenSearch 2.13 pour restreindre l'accès aux tickets par équipe/région.
5. **Droit à l'effacement** : implémenter un endpoint permettant la suppression d'un ticket de l'index sur demande (Art. 17 RGPD).

---

### R09 — Indisponibilité d'OpenSearch (Criticité : 9)

**Contexte technique :** OpenSearch 2.13 est déployé sur **infrastructure Docker locale** (vraisemblablement un seul nœud, configuration MVP). Cette architecture présente plusieurs points de défaillance unique (SPOF) :

- Absence de cluster multi-nœuds : tout redémarrage ou crash du conteneur rend le service indisponible.
- Pas de snapshot automatique validé : en cas de corruption de l'index ou de perte du volume Docker, les **3 999 tickets et leurs vecteurs (384 dimensions × 3 999 vecteurs)** seraient perdus et nécessiteraient une ré-indexation complète.
- La fusion hybride BM25+kNN via RRF (k=60) est computationnellement plus intensive qu'une recherche BM25 seule : un spike de charge pourrait allonger les temps de réponse au-delà de l'acceptable pour les agents SAV.

**Mesures de mitigation :**
1. **Cluster OpenSearch minimum 2 nœuds** (1 master + 1 data) avec réplication de shard à facteur 1.
2. **Snapshots automatiques quotidiens** vers un stockage objet (MinIO local ou S3), avec procédure de restauration testée et documentée.
3. **Définition du RTO/RPO** : objectif de reprise (RTO < 2h, RPO < 24h) validé avec la direction SAV.
4. **Health check dans le CI/CD** : sonde de disponibilité OpenSearch intégrée aux GitHub Actions.

---

### R14 — Dette Technique du MVP (Criticité : 9)

**Contexte technique :** Le MVP a été réalisé en **10 jours par 3 développeurs**, une cadence qui implique structurellement des compromis sur la qualité du code : couverture de tests incomplète (seul `pytest` + `ruff` lint dans la pipeline CI), absence probable de tests d'intégration end-to-end, documentation technique minimale. La présence du **bug de filtre kNN** (workaround `bool/filter` OpenSearch 2.13) et le choix du modèle d'évaluation par élimination (`gpt-oss-20b:free` seul à retourner du JSON propre) révèlent une fragilité de la base de code.

**Mesures de mitigation :**
1. **Sprint de remboursement de dette** : 2 semaines dédiées post-MVP pour refactoring, documentation et tests.
2. **Objectif de couverture de tests > 80%** sur les modules critiques (retrieval, fusion RRF, appel LLM).
3. **Tests de contrat API** : mock de l'API OpenRouter pour tester les comportements en cas d'erreur (429, 500, timeout).
4. **Documentation du workaround kNN** : fiche technique maintenue à jour sur la version OpenSearch et les correctifs attendus.

---

## 4. Risques Liés à l'IA Spécifiquement

### 4.1 Hallucination et Extrapolation (R01)

Traité en détail dans la section 3. La **Faithfulness de 0.57/5** place RAG-time dans une zone rouge pour la fiabilité des réponses générées. Le modèle Nemotron 120B, non spécifiquement entraîné sur des données logistiques, a tendance à extrapoler à partir de ses connaissances générales plutôt que de se limiter strictement aux tickets récupérés. Ce comportement est accentué par la faible précision du contexte récupéré (Context Precision = 3.06/5).

### 4.2 Biais Multilingue (R02)

Le modèle d'embedding `paraphrase-multilingual-MiniLM-L12-v2` présente des performances hétérogènes selon les langues. Le **MRR = 0.275 pour l'espagnol et le portugais** (contre 0.391 en moyenne globale) traduit une représentation vectorielle moins fidèle pour ces langues, probablement sous-représentées dans les données d'entraînement du modèle par rapport au français et à l'anglais. Pour des agents SAV traitant des dossiers ibéro-américains, ce biais constitue un risque opérationnel direct.

**Mitigation :** Évaluer `paraphrase-multilingual-mpnet-base-v2` (dim=768) pour ES/PT, ou envisager un modèle dédié Espagnol/Portugais via HuggingFace. Constituer un dataset de test ES/PT plus représentatif.

### 4.3 Dérive du Modèle (R03)

L'API OpenRouter n'offre pas de garantie de stabilité des versions modèles à long terme. Une mise à jour silencieuse de Nemotron 120B (ou son retrait du catalogue) pourrait modifier substantiellement les comportements du système sans que l'équipe en soit alertée. Ce risque est classé **LLM03 – Training Data Poisoning / Model Drift** dans le vocabulaire OWASP.

**Mitigation :** Implémenter un benchmark de non-régression automatique (exécuté hebdomadairement via GitHub Actions) sur un ensemble de 50 requêtes de référence avec réponses attendues. Alerter si le score de faithfulness ou MRR dérive de plus de 10%.

### 4.4 Dépendance Fournisseur — OpenRouter (R15)

L'ensemble de la couche génération (Nemotron 120B) et d'évaluation (`gpt-oss-20b:free`) repose sur **un seul fournisseur tiers**, OpenRouter. Ce single point of failure commercial présente des risques de : changement tarifaire, suspension de compte, disparition du service, changement de politique d'utilisation (ex. interdiction de traiter des données clients). Le budget opérationnel du système est entièrement dépendant des tarifs OpenRouter.

**Mitigation :** Identifier un LLM de repli déployable localement (Ollama + Mistral 7B/Llama 3.1 8B) pour les fonctions de base. Documenter la procédure de bascule. Négocier des conditions contractuelles avec OpenRouter incluant un SLA et des clauses de portabilité.

---

## 5. Risques de Sécurité et Conformité

### 5.1 RGPD et Données Personnelles (R07)

Traité en détail dans la section 3. Points clés à retenir :
- Les tickets SAV sont présumés contenir des PII jusqu'à analyse contraire.
- L'envoi de contexte contenant des PII vers l'API OpenRouter constitue un **transfert de données** soumis à l'article 46 RGPD si le traitement a lieu hors UE.
- La durée de conservation dans OpenSearch n'est pas bornée (R08).

### 5.2 Gestion des Clés API (R05)

Traité en détail dans la section 3. L'incident avéré impose une **revue complète de la gestion des secrets** avant toute mise en production. La pipeline CI/CD GitHub Actions doit inclure une étape de scan de secrets (`gitleaks` ou `truffleHog`) bloquante.

### 5.3 Contrôle d'Accès à l'Interface (R16)

L'interface Streamlit, dans sa configuration par défaut, ne propose pas d'authentification robuste. Si elle est déployée sur le réseau interne LogiStore sans restriction d'accès IP, **n'importe quel employé** pourrait interroger la base de tickets SAV, y compris des tickets contenant des informations sensibles de clients ou des données commerciales confidentielles.

**Mitigation :** Déployer derrière un reverse proxy (nginx) avec authentification basique en urgence, puis intégrer le SSO/LDAP de LogiStore via `streamlit-authenticator` ou une solution middleware OAuth2 (Authelia, Keycloak).

### 5.4 Exposition des Logs (R06 transverse)

Les logs de l'application Streamlit et d'OpenSearch sont susceptibles de contenir des requêtes utilisateurs incluant des références à des clients. Une politique de rétention et d'accès aux logs doit être définie.

---

## 6. Risques Opérationnels

### 6.1 Disponibilité OpenSearch (R09)

Traité en section 3. L'impact sur les agents SAV en cas d'indisponibilité est direct : impossibilité de rechercher dans l'historique des tickets pendant la durée de la panne.

### 6.2 Rate Limiting et Quotas OpenRouter (R10)

Avec 30 agents SAV pouvant soumettre simultanément des requêtes, chaque requête consomme des tokens proportionnels à la taille des chunks récupérés (médiane < 512 tokens par ticket × top-5 chunks = ~2 500 tokens de contexte) plus la réponse générée (~500 tokens). En pointe, cela représente **jusqu'à 90 000 tokens/minute**, pouvant dépasser les quotas du tier OpenRouter utilisé.

**Mitigation :**
- Implémenter une gestion des erreurs HTTP 429 avec retry exponentiel (backoff).
- Mettre en place un cache de réponses (Redis ou cache in-memory) pour les requêtes identiques ou très proches.
- Monitorer les quotas via l'API de facturation OpenRouter et alerter à 80% du quota mensuel.

### 6.3 Scalabilité Streamlit (R11)

Streamlit est un framework de prototypage conçu pour des démonstrations mono-utilisateur ou à faible concurrence. Avec 30 agents simultanés, les sessions Python sont distinctes mais le serveur Streamlit peut devenir un goulot d'étranglement. De plus, le modèle d'embedding (`paraphrase-multilingual-MiniLM-L12-v2`) est chargé en mémoire : si plusieurs instances sont nécessaires, la charge mémoire se multiplie.

**Mitigation :** Pour la mise en production, migrer vers une architecture **FastAPI (backend REST) + interface légère (React ou Vue.js)**, avec le modèle d'embedding exposé comme microservice (ex. via `sentence-transformers` avec un serveur ONNX ou `infinity-embedding`).

### 6.4 Qualité du Retrieval et Confiance des Résultats (R12)

Un **Recall@5 de 0.520** signifie que pour 48% des requêtes, le ticket le plus pertinent n'est pas dans les 5 premiers résultats. Un **nDCG@5 de 0.423** indique un classement imparfait des résultats pertinents. Ces métriques, mesurées sur l'ensemble du corpus, masquent des disparités importantes par langue (ES/PT sous-performants). À terme, des décisions SAV basées sur des résultats de retrieval de mauvaise qualité peuvent conduire à des réponses clients incorrectes.

---

## 7. Matrice de Risques

La matrice ci-dessous représente les risques selon leur **Probabilité** (axe horizontal, 1–4) et leur **Impact** (axe vertical, 1–4). Les identifiants dans chaque cellule correspondent aux risques du tableau de la section 2.

```
         │  P=1     │  P=2     │  P=3            │  P=4
─────────┼──────────┼──────────┼─────────────────┼──────────────────
 I=4     │          │          │                 │  R05, R07
─────────┼──────────┼──────────┼─────────────────┼──────────────────
 I=3     │          │  R06,R16 │  R03,R04,R08,   │  R01
         │          │          │  R09,R12,R14,R15│
─────────┼──────────┼──────────┼─────────────────┼──────────────────
 I=2     │          │          │  R10,R11,R13    │  R02
─────────┼──────────┼──────────┼─────────────────┼──────────────────
 I=1     │          │          │                 │
```

**Légende de couleurs (zones de criticité) :**

| Zone | Criticité | Couleur | Action |
|---|---|---|---|
| Zone rouge | ≥ 12 | 🔴 Critique | Traitement immédiat |
| Zone orange | 9–11 | 🟠 Élevé | Traitement planifié sous 30 jours |
| Zone jaune | 6–8 | 🟡 Modéré | Traitement planifié sous 90 jours |
| Zone verte | ≤ 5 | 🟢 Faible | Surveillance |

**Risques critiques (zone rouge) :** R05 (16), R07 (16), R01 (12)  
**Risques élevés (zone orange) :** R03, R04, R08, R09, R12, R14, R15 (criticité 9)

---

## 8. Plan de Traitement des Risques

### 8.1 Actions immédiates (avant toute mise en production — Délai : 0–15 jours)

| Priorité | ID | Action | Responsable | Ressources |
|---|---|---|---|---|
| 🔴 P1 | R05 | Expurger la clé API de l'historique Git ; intégrer `gitleaks` dans CI/CD ; migrer vers variables d'environnement chiffrées | Dev Lead | 1 jour |
| 🔴 P1 | R07 | Lancer le PIA avec le DPO ; inventaire des PII dans les tickets ; blocage de la mise en production jusqu'à validation | DPO + Dev | 2 semaines |
| 🔴 P1 | R16 | Mettre en place l'authentification sur Streamlit (authentification basique ou SSO) avant tout déploiement sur réseau interne | Dev + DSI | 2 jours |

### 8.2 Actions à court terme (sous 30 jours)

| Priorité | ID | Action | Responsable | Ressources |
|---|---|---|---|---|
| 🟠 P2 | R01 | Implémenter affichage des sources et score de confiance dans l'UI ; renforcer le prompt système | Dev | 3 jours |
| 🟠 P2 | R09 | Déployer un cluster OpenSearch 2 nœuds avec snapshots automatiques | Infra/Ops | 1 semaine |
| 🟠 P2 | R03 | Mettre en place un benchmark de non-régression hebdomadaire (GitHub Actions) | Dev | 2 jours |
| 🟠 P2 | R14 | Sprint de remboursement de dette technique : couverture tests > 80%, documentation | Équipe Dev | 2 semaines |
| 🟠 P2 | R08 | Définir et implémenter une politique de TTL (durée de conservation) sur les index OpenSearch | Dev + DPO | 3 jours |

### 8.3 Actions à moyen terme (30–90 jours)

| Priorité | ID | Action | Responsable | Ressources |
|---|---|---|---|---|
| 🟡 P3 | R02 | Évaluer et déployer un modèle d'embedding amélioré pour ES/PT ; mesurer gain MRR | Dev/ML | 2 semaines |
| 🟡 P3 | R12 | Implémenter un reranker cross-encoder post-retrieval ; évaluer gain nDCG@5 | Dev/ML | 3 semaines |
| 🟡 P3 | R15 | Déployer un LLM local de repli (Ollama + Mistral 7B) ; documenter procédure de bascule | Dev + Infra | 1 semaine |
| 🟡 P3 | R11 | Migrer vers architecture FastAPI + frontend léger pour la mise en production | Dev | 3–4 semaines |
| 🟡 P3 | R10 | Implémenter cache Redis pour les réponses ; gestionnaire de quotas OpenRouter | Dev | 1 semaine |
| 🟡 P3 | R04 | Organiser sessions de formation agents SAV sur les limites du système IA | Chef de projet + RH | 1 jour |

### 8.4 Suivi et gouvernance

- **Comité de suivi des risques** : réunion mensuelle impliquant le Dev Lead, le DPO, le Responsable SAV et la DSI.
- **Tableau de bord risques** : indicateurs clés (Faithfulness, MRR par langue, disponibilité OpenSearch, quotas OpenRouter) intégrés dans un dashboard de monitoring (Grafana ou équivalent).
- **Révision annuelle** de la présente étude de risque, ou à chaque évolution majeure de la stack technique.
- **Registre des incidents** : tout incident de sécurité (ex. nouvelle fuite de clé, accès non autorisé) doit être consigné et déclencher une révision ciblée de l'étude.

---

## Annexe A — Glossaire

| Terme | Définition |
|---|---|
| RAG | Retrieval-Augmented Generation — combinaison de recherche documentaire et de génération par LLM |
| MRR | Mean Reciprocal Rank — métrique de qualité du classement des résultats de recherche |
| nDCG@5 | Normalized Discounted Cumulative Gain at 5 — mesure la qualité du classement des 5 premiers résultats |
| Faithfulness | Score RAGAS mesurant l'ancrage de la réponse dans les documents sources |
| BM25 | Algorithme de recherche lexicale probabiliste (Best Match 25) |
| kNN HNSW | k-Nearest Neighbors avec index Hierarchical Navigable Small World — recherche vectorielle |
| RRF | Reciprocal Rank Fusion (Cormack 2009) — fusion des scores BM25 et kNN |
| PIA | Privacy Impact Assessment — analyse d'impact sur la protection des données (Art. 35 RGPD) |
| SPOF | Single Point of Failure — point unique de défaillance dans une architecture |
| TTL | Time-To-Live — durée de vie d'une donnée avant suppression automatique |

---

## Annexe B — Références

- ANSSI. (2018). *EBIOS Risk Manager — La méthode*. [https://cyber.gouv.fr/securisation/analyse-des-risques/methode-ebios-rm/](https://cyber.gouv.fr/securisation/analyse-des-risques/methode-ebios-rm/)
- OWASP. (2023). *OWASP Top 10 for Large Language Model Applications v1.1*. [https://owasp.org/www-project-top-10-for-large-language-model-applications/](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- ENISA. (2020). *Artificial Intelligence Cybersecurity Challenges*. European Union Agency for Cybersecurity.
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal rank fusion outperforms condorcet and individual rank learning methods*. SIGIR '09.
- Règlement (UE) 2016/679 du Parlement européen et du Conseil (RGPD).
- OpenSearch Documentation. (2024). *k-NN Search — Known Limitations v2.13*. [https://opensearch.org/docs/latest/search-plugins/knn/](https://opensearch.org/docs/latest/search-plugins/knn/)
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.

---

*Document généré le $(date). Version 1.0 — À réviser avant mise en production.*
