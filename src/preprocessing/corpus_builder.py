"""
Persistance des ArxivPaper sur disque.

Séparé de ArxivClient (fetching + parsing) : ce module ne connaît rien du
réseau ou du XML, juste "comment écrire un ArxivPaper sur disque"
(Single Responsibility — deux raisons de changer, deux classes/modules).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.preprocessing.arxiv_client import ArxivPaper
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_corpus(papers: list[ArxivPaper], output_dir: Path) -> Path:
    """Écrit un fichier .txt par papier (relu ensuite par TxtLoader via le
    DocumentLoaderRegistry du Jalon 2 sous-étape 1) + un unique JSON de
    métadonnées (auteurs, date, catégorie...) qui servira à enrichir le
    Knowledge Graph avec des relations qui n'ont pas besoin de NER
    (ex: "Papier X écrit par Auteur Y" est déjà structuré, pas la peine
    d'extraire ça d'un texte).

    Retourne le chemin du fichier de métadonnées.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict] = {}

    for paper in papers:
        # Certains anciens ids arXiv contiennent un "/" (ex: "hep-ex/0307015"),
        # invalide dans un nom de fichier -> on le remplace.
        safe_id = paper.arxiv_id.replace("/", "_")
        txt_path = output_dir / f"arxiv_{safe_id}.txt"
        txt_path.write_text(paper.to_text(), encoding="utf-8")

        metadata[paper.arxiv_id] = {
            "title": paper.title,
            "authors": paper.authors,
            "published": paper.published,
            "primary_category": paper.primary_category,
            "pdf_url": paper.pdf_url,
            "txt_file": txt_path.name,
        }

    metadata_path = output_dir / "arxiv_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("%d papiers sauvegardés dans %s", len(papers), output_dir)
    return metadata_path
