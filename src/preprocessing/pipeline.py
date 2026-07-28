"""
Assemble nettoyage -> segmentation en phrases pour un LoadedDocument, et
sait le sauvegarder dans data/processed/.

main.py reste un simple point d'entrée qui appelle ces fonctions — la
logique métier vit ici, pas dans main.py (Clean Architecture : le point
d'entrée orchestre, il n'implémente pas).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.preprocessing.cleaning import clean_text
from src.preprocessing.loaders import LoadedDocument
from src.preprocessing.sentence_splitter import split_sentences


@dataclass
class ProcessedDocument:
    """Résultat du nettoyage + segmentation, prêt pour le NER (Jalon 3)."""

    source_path: str
    doc_format: str
    sentences: list[str]

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


def process_document(document: LoadedDocument) -> ProcessedDocument:
    """Nettoie le texte puis le découpe en phrases."""
    cleaned = clean_text(document.text)
    sentences = split_sentences(cleaned)
    return ProcessedDocument(
        source_path=str(document.source_path),
        doc_format=document.doc_format,
        sentences=sentences,
    )


def save_processed(document: ProcessedDocument, output_dir: Path) -> Path:
    """Sauvegarde un ProcessedDocument en JSON, même nom de fichier que la
    source (juste l'extension qui change) — facile de relier les deux."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(document.source_path).stem
    out_path = output_dir / f"{stem}.json"
    out_path.write_text(
        json.dumps(asdict(document), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path
