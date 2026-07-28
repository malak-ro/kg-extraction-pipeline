"""
Point d'entrée du pipeline.

Jalon 1 : smoke test (config + logging + dossiers).
Jalon 2 : récupère un corpus d'abstracts arXiv (si absent localement),
puis recharge chaque fichier via le DocumentLoaderRegistry — la même
abstraction qui lira aussi les PDF/DOCX que l'utilisateur ajoutera plus tard.
L'orchestration continuera de grandir ici, jalon après jalon.
"""
from __future__ import annotations

from config.settings import settings
from src.preprocessing.arxiv_client import ArxivClient
from src.preprocessing.corpus_builder import save_corpus
from src.preprocessing.loaders import DocumentLoaderRegistry
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


def main() -> None:
    logger.info("Démarrage du pipeline (environnement=%s)", settings.environment)

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

    _ensure_corpus()

    registry = DocumentLoaderRegistry()
    txt_files = sorted(settings.raw_data_dir.glob("*.txt"))
    documents = [registry.load(path) for path in txt_files]
    logger.info("%d documents chargés depuis data/raw/ via DocumentLoaderRegistry.", len(documents))

    logger.info("Jalon 2 (ingestion) opérationnel.")
    logger.info("Prochaine étape : nettoyage/segmentation, puis Jalon 3 — NER.")


if __name__ == "__main__":
    main()
