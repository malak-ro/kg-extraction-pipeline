"""
Applique le LLMRelationExtractor (retenu après comparaison, sous-étape 4)
sur les documents déjà annotés d'entités (data/processed/*.json, sortie
du Jalon 3) — enrichit chaque entrée de entities_by_sentence avec un
champ "relations", même principe d'enrichissement en place que le NER
avait appliqué à la sortie du Jalon 2.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.ner.entity import Entity
from src.relation_extraction.base import RelationExtractor

# Calibré sur les limites réelles du palier gratuit Groq pour gpt-oss-20b
# (30 requêtes/minute) : une marge de sécurité confortable (~24/minute),
# pas la limite exacte.
RATE_LIMIT_SECONDS = 2.5


def _entity_from_dict(d: dict) -> Entity:
    return Entity(text=d["text"], label=d["label"], start_char=d["start_char"], end_char=d["end_char"])


def annotate_document_with_relations(processed_json_path: Path, extractor: RelationExtractor) -> None:
    """Lit un fichier data/processed/*.json déjà annoté d'entités, extrait
    les relations phrase par phrase, et réécrit le fichier enrichi.

    Ne respecte le rythme (time.sleep) qu'après les phrases qui ont
    réellement déclenché un appel API (2+ entités) — pas la peine de
    ralentir sur les phrases à 0 ou 1 entité, qui ne coûtent aucune requête."""
    data = json.loads(processed_json_path.read_text(encoding="utf-8"))

    for entry in data.get("entities_by_sentence", []):
        entities = [_entity_from_dict(e) for e in entry["entities"]]
        relations = extractor.extract(entry["sentence"], entities)
        entry["relations"] = [
            {"subject": r.subject.text, "predicate": r.predicate, "object": r.obj.text}
            for r in relations
        ]
        if len(entities) >= 2:
            time.sleep(RATE_LIMIT_SECONDS)

    processed_json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def is_already_annotated_with_relations(processed_json_path: Path) -> bool:
    """Sert à l'idempotence, même logique qu'au Jalon 3 : ne pas relancer
    ~250 appels API à chaque exécution de main.py si c'est déjà fait."""
    data = json.loads(processed_json_path.read_text(encoding="utf-8"))
    entries = data.get("entities_by_sentence", [])
    return bool(entries) and all("relations" in e for e in entries)
