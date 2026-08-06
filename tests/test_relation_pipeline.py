import json
from pathlib import Path
from unittest.mock import patch

from src.ner.entity import Entity
from src.relation_extraction.base import RelationExtractor
from src.relation_extraction.pipeline import (
    annotate_document_with_relations,
    is_already_annotated_with_relations,
)
from src.relation_extraction.relation import Relation


class _FixedExtractor(RelationExtractor):
    """Retourne une relation fixe si au moins 2 entités sont fournies,
    sinon aucune — suffisant pour vérifier le câblage, pas la qualité
    d'extraction (déjà couverte par test_relation_llm.py)."""

    def extract(self, sentence: str, entities: list[Entity]) -> list[Relation]:
        if len(entities) < 2:
            return []
        return [Relation(subject=entities[0], predicate="related_to", obj=entities[1], sentence=sentence)]


def _write_annotated_json(path: Path, entries: list[dict]) -> None:
    path.write_text(
        json.dumps({"source_path": "x", "doc_format": "txt", "entities_by_sentence": entries}),
        encoding="utf-8",
    )


def _entity_dict(text: str, label: str) -> dict:
    return {"text": text, "label": label, "start_char": 0, "end_char": len(text)}


@patch("src.relation_extraction.pipeline.time.sleep")
def test_annotate_document_adds_relations_field(mock_sleep, tmp_path: Path) -> None:
    path = tmp_path / "arxiv_test.json"
    _write_annotated_json(
        path,
        [{"sentence": "BERT trained on Wikipedia.",
          "entities": [_entity_dict("BERT", "model"), _entity_dict("Wikipedia", "dataset")]}],
    )

    annotate_document_with_relations(path, _FixedExtractor())

    data = json.loads(path.read_text(encoding="utf-8"))
    relations = data["entities_by_sentence"][0]["relations"]
    assert relations == [{"subject": "BERT", "predicate": "related_to", "object": "Wikipedia"}]


@patch("src.relation_extraction.pipeline.time.sleep")
def test_annotate_document_preserves_existing_entities_field(mock_sleep, tmp_path: Path) -> None:
    path = tmp_path / "arxiv_test.json"
    _write_annotated_json(
        path,
        [{"sentence": "BERT trained on Wikipedia.",
          "entities": [_entity_dict("BERT", "model"), _entity_dict("Wikipedia", "dataset")]}],
    )

    annotate_document_with_relations(path, _FixedExtractor())

    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["entities_by_sentence"][0]["entities"]) == 2  # inchangé


@patch("src.relation_extraction.pipeline.time.sleep")
def test_rate_limit_only_applied_after_sentences_with_two_or_more_entities(mock_sleep, tmp_path: Path) -> None:
    """0 ou 1 entité -> aucun appel API réel (voir LLMRelationExtractor),
    donc pas la peine d'attendre non plus."""
    path = tmp_path / "arxiv_test.json"
    _write_annotated_json(
        path,
        [
            {"sentence": "No entities here.", "entities": []},
            {"sentence": "Just BERT.", "entities": [_entity_dict("BERT", "model")]},
            {"sentence": "BERT trained on Wikipedia.",
             "entities": [_entity_dict("BERT", "model"), _entity_dict("Wikipedia", "dataset")]},
        ],
    )

    annotate_document_with_relations(path, _FixedExtractor())

    assert mock_sleep.call_count == 1  # seule la 3e phrase avait 2+ entités


def test_is_already_annotated_with_relations_reflects_current_state(tmp_path: Path) -> None:
    path = tmp_path / "arxiv_test.json"
    _write_annotated_json(
        path,
        [{"sentence": "BERT trained on Wikipedia.",
          "entities": [_entity_dict("BERT", "model"), _entity_dict("Wikipedia", "dataset")]}],
    )
    assert is_already_annotated_with_relations(path) is False

    with patch("src.relation_extraction.pipeline.time.sleep"):
        annotate_document_with_relations(path, _FixedExtractor())

    assert is_already_annotated_with_relations(path) is True
