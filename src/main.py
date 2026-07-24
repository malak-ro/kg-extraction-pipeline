"""
Point d'entrée du pipeline.

Pour l'instant (Jalon 1), ce fichier ne fait qu'un smoke test : vérifier que
la config se charge, que le logging fonctionne, et que l'arborescence data/
existe. L'orchestration réelle (ingestion → NER → relations → graphe) sera
ajoutée ici jalon après jalon, sans jamais réécrire ce qui existe déjà.
"""
from __future__ import annotations

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Démarrage du pipeline (environnement=%s)", settings.environment)

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dossiers data/raw et data/processed prêts : %s", settings.data_dir)

    logger.info("Jalon 1 (setup & architecture) opérationnel.")
    logger.info("Prochaine étape : Jalon 2 — ingestion des documents.")


if __name__ == "__main__":
    main()
