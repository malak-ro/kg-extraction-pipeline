"""
Applique un EntityExtractor sur les documents déjà nettoyés/segmentés
(data/processed/*.json, sortie du Jalon 2) et enrichit chaque fichier
avec les entités trouvées, phrase par phrase.

Volontairement séparé de src/preprocessing/pipeline.py : le prétraitement
(Jalon 2) ne doit rien savoir du NER (Jalon 3) — chaque module ne connaît
que ce qui le concerne. Ici on LIT la sortie du Jalon 2 comme une donnée
d'entrée ordinaire, sans dépendre de son code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ner.base import EntityExtractor
from src.ner.entity import Entity


@dataclass
class SentenceEntities:
    sentence: str
    entities: list[Entity]


def annotate_document(
    processed_json_path: Path, extractor: EntityExtractor
) -> list[SentenceEntities]:
    """Lit un fichier data/processed/*.json et extrait les entités de
    chaque phrase qu'il contient."""
    data = json.loads(processed_json_path.read_text(encoding="utf-8"))
    return [
        SentenceEntities(sentence=sentence, entities=extractor.extract(sentence))
        for sentence in data["sentences"]
    ]


def save_annotated(processed_json_path: Path, sentence_entities: list[SentenceEntities]) -> None:
    """Ajoute un champ "entities_by_sentence" au fichier JSON existant,
    sans toucher aux champs déjà écrits par le Jalon 2 (sentences,
    doc_format, source_path) — un ENRICHISSEMENT, pas un remplacement."""
    data = json.loads(processed_json_path.read_text(encoding="utf-8"))
    data["entities_by_sentence"] = [asdict(se) for se in sentence_entities]
    processed_json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def is_already_annotated(processed_json_path: Path) -> bool:
    """Sert à l'idempotence : ne pas relancer le NER sur tout le corpus
    à chaque exécution de main.py si c'est déjà fait."""
    data = json.loads(processed_json_path.read_text(encoding="utf-8"))
    return "entities_by_sentence" in data
