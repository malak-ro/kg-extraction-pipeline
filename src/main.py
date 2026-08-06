"""
Point d'entrée du pipeline.

Jalon 1 : smoke test (config + logging + dossiers).
Jalon 2 : récupère un corpus d'abstracts arXiv (si absent localement),
recharge chaque fichier via le DocumentLoaderRegistry, puis nettoie et
segmente chaque document en phrases (data/processed/).
Jalon 3 : extrait les entités de chaque phrase avec GLiNER (retenu après
comparaison face à spaCy) et enrichit data/processed/*.json.
Jalon 4 : extrait les relations entre ces entités avec un LLM (retenu
après comparaison face à une approche syntaxique) et enrichit encore
data/processed/*.json. L'orchestration continuera de grandir ici, jalon
après jalon.
"""
from __future__ import annotations

from config.settings import settings
from src.ner.pipeline import annotate_document, is_already_annotated, save_annotated
from src.preprocessing.arxiv_client import ArxivClient
from src.preprocessing.corpus_builder import save_corpus
from src.preprocessing.loaders import DocumentLoaderRegistry
from src.preprocessing.pipeline import process_document, save_processed
from src.relation_extraction.pipeline import (
    annotate_document_with_relations,
    is_already_annotated_with_relations,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Thématique NER/NLP, cohérente avec l'article partagé par l'utilisateur
ARXIV_QUERY = 'cat:cs.CL AND abs:"named entity"'
ARXIV_MAX_RESULTS = 50


def _ensure_corpus() -> None:
    """Récupère le corpus arXiv une seule fois (idempotent) : si des fichiers
    arxiv_*.txt existent déjà dans data/raw/, on ne rappelle pas l'API —
    à la fois par politesse envers arXiv et parce que les résultats ne
    changent qu'une fois par jour (cf. doc de l'API)."""
    already_fetched = any(settings.raw_data_dir.glob("arxiv_*.txt"))
    if already_fetched:
        logger.info("Corpus arXiv déjà présent localement — pas de nouvel appel API.")
        return

    logger.info("Aucun corpus local détecté — récupération depuis l'API arXiv...")
    client = ArxivClient()
    papers = client.search(query=ARXIV_QUERY, max_results=ARXIV_MAX_RESULTS)
    save_corpus(papers, settings.raw_data_dir)
    logger.info("%d papiers arXiv récupérés et sauvegardés.", len(papers))


def _ensure_entities() -> None:
    """Annote chaque document avec les entités GLiNER, une seule fois
    (idempotent) : si le premier fichier a déjà un champ
    "entities_by_sentence", on considère tout le corpus déjà annoté."""
    processed_files = sorted(settings.processed_data_dir.glob("arxiv_*.json"))
    if not processed_files:
        logger.warning("Aucun document dans data/processed/ — rien à annoter.")
        return

    if is_already_annotated(processed_files[0]):
        logger.info("Entités déjà extraites localement — pas de nouveau passage NER.")
        return

    # Import local : GLiNER (+ torch) est une dépendance lourde, inutile de
    # la charger si l'idempotence ci-dessus permet de sauter cette étape.
    from src.ner.gliner_extractor import GlinerEntityExtractor

    logger.info("Extraction des entités (GLiNER) sur %d documents...", len(processed_files))
    extractor = GlinerEntityExtractor()
    total_entities = 0
    for path in processed_files:
        sentence_entities = annotate_document(path, extractor)
        save_annotated(path, sentence_entities)
        total_entities += sum(len(se.entities) for se in sentence_entities)

    logger.info("%d entités extraites au total -> data/processed/", total_entities)


def _ensure_relations() -> None:
    """Extrait les relations entre entités avec le LLM (Groq), une seule
    fois (idempotent, même logique). Prévient explicitement du temps
    d'exécution : contrairement au NER, chaque phrase à 2+ entités déclenche
    un vrai appel réseau, cadencé volontairement pour respecter le palier
    gratuit Groq (~30 req/min) — plusieurs minutes sur tout le corpus."""
    processed_files = sorted(settings.processed_data_dir.glob("arxiv_*.json"))
    if not processed_files:
        logger.warning("Aucun document dans data/processed/ — rien à annoter.")
        return

    if is_already_annotated_with_relations(processed_files[0]):
        logger.info("Relations déjà extraites localement — pas de nouveau passage.")
        return

    from src.relation_extraction.llm_extractor import LLMRelationExtractor

    logger.info(
        "Extraction des relations (LLM via Groq) sur %d documents — "
        "compte plusieurs minutes, rythme volontairement limité pour "
        "respecter le palier gratuit Groq (~30 requêtes/minute).",
        len(processed_files),
    )
    extractor = LLMRelationExtractor()
    for i, path in enumerate(processed_files, start=1):
        annotate_document_with_relations(path, extractor)
        if i % 5 == 0 or i == len(processed_files):
            logger.info("  ... %d/%d documents traités", i, len(processed_files))

    logger.info("Extraction de relations terminée -> data/processed/")


def main() -> None:
    logger.info("Démarrage du pipeline (environnement=%s)", settings.environment)

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

    _ensure_corpus()

    registry = DocumentLoaderRegistry()
    txt_files = sorted(settings.raw_data_dir.glob("*.txt"))
    documents = [registry.load(path) for path in txt_files]
    logger.info("%d documents chargés depuis data/raw/ via DocumentLoaderRegistry.", len(documents))

    total_sentences = 0
    for document in documents:
        processed = process_document(document)
        save_processed(processed, settings.processed_data_dir)
        total_sentences += processed.sentence_count

    avg = total_sentences / len(documents) if documents else 0
    logger.info(
        "%d documents nettoyés et segmentés (%d phrases au total, %.1f en moyenne/doc) -> data/processed/",
        len(documents), total_sentences, avg,
    )

    _ensure_entities()
    _ensure_relations()

    logger.info("Jalon 4 (extraction de relations) opérationnel.")
    logger.info("Prochaine étape : Jalon 5 — construction du Knowledge Graph (Neo4j).")


if __name__ == "__main__":
    main()
