# Knowledge Graph Construction from Unstructured Text (NLP & LLMs)

Proof of Concept : pipeline qui transforme des documents non structurés
(PDF / TXT / DOCX) en un Knowledge Graph interrogeable dans Neo4j, en
combinant du NLP classique (NER, dépendances syntaxiques) et de l'extraction
de relations assistée par LLM.

**Statut :** en cours — Jalon 1/8 (setup & architecture) terminé.

## Pipeline

```
documents bruts → prétraitement + NER → extraction de relations (NLP + LLM)
                → knowledge graph (Neo4j) → requêtes Cypher / visualisation
```

## Structure du projet

```
kg-extraction-pipeline/
├── data/                  # données brutes et prétraitées (gitignored)
├── notebooks/             # exploration, prototypage
├── src/
│   ├── preprocessing/     # nettoyage, tokenisation, segmentation
│   ├── ner/               # reconnaissance d'entités nommées
│   ├── relation_extraction/  # extraction de relations (NLP + LLM)
│   ├── graph/              # construction et requêtage Neo4j
│   ├── llm/                # prompts, appels LLM, RAG
│   ├── utils/               # logger, exceptions, helpers
│   └── main.py             # point d'entrée
├── tests/                  # tests unitaires (pytest)
├── reports/                # sorties, logs, visualisations générées
├── docs/                   # diagrammes, documentation technique
├── config/
│   └── settings.py         # configuration centralisée (pydantic-settings)
├── requirements.txt
├── pyproject.toml           # packaging + config des outils (bonus)
├── Makefile                 # commandes standardisées (bonus)
└── .gitignore
```

## Installation

```bash
git clone <url-du-repo>
cd kg-extraction-pipeline
python -m venv .venv
source .venv/bin/activate       # Windows : .venv\Scripts\activate
cp .env.example .env             # puis renseigne Neo4j / clé API si besoin
make install
make test
make run
```

## Roadmap

- [x] Jalon 1 — Setup & Architecture
- [ ] Jalon 2 — Ingestion & Prétraitement
- [ ] Jalon 3 — NER
- [ ] Jalon 4 — Extraction de relations
- [ ] Jalon 5 — Knowledge Graph (Neo4j)
- [ ] Jalon 6 — Requêtage & Visualisation
- [ ] Jalon 7 — Qualité & Tests
- [ ] Jalon 8 — Livrables finaux

## Stack technique

Python · spaCy · Hugging Face Transformers · OpenAI API / Llama 3 · Neo4j ·
Cypher · Pandas · NetworkX
