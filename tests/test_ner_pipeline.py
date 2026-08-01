import json
from pathlib import Path

from src.ner.base import EntityExtractor
from src.ner.entity import Entity
from src.ner.pipeline import annotate_document, is_already_annotated, save_annotated


class _FixedExtractor(EntityExtractor):
    """Retourne toujours la même entité, peu importe le texte — suffisant
    pour vérifier le CÂBLAGE (lecture -> extraction -> écriture), pas la
    qualité d'extraction (déjà couverte par test_ner_gliner.py)."""

    def extract(self, text: str) -> list[Entity]:
        return [Entity(text="BERT", label="model", start_char=0, end_char=4)]


def _write_processed_json(path: Path, sentences: list[str]) -> None:
    path.write_text(
        json.dumps(
            {"source_path": "data/raw/arxiv_test.txt", "doc_format": "txt", "sentences": sentences},
            indent=2,
        ),
        encoding="utf-8",
    )


def test_annotate_document_extracts_entities_for_each_sentence(tmp_path: Path) -> None:
    path = tmp_path / "arxiv_test.json"
    _write_processed_json(path, ["BERT is a model.", "It was released in 2018."])

    result = annotate_document(path, _FixedExtractor())

    assert len(result) == 2
    assert result[0].sentence == "BERT is a model."
    assert result[0].entities == [Entity(text="BERT", label="model", start_char=0, end_char=4)]


def test_save_annotated_adds_field_without_removing_existing_ones(tmp_path: Path) -> None:
    path = tmp_path / "arxiv_test.json"
    _write_processed_json(path, ["BERT is a model."])

    sentence_entities = annotate_document(path, _FixedExtractor())
    save_annotated(path, sentence_entities)

    data = json.loads(path.read_text(encoding="utf-8"))
    # Champs du Jalon 2 toujours présents, intacts
    assert data["doc_format"] == "txt"
    assert data["sentences"] == ["BERT is a model."]
    # Nouveau champ du Jalon 3
    assert data["entities_by_sentence"][0]["entities"][0]["text"] == "BERT"
    assert data["entities_by_sentence"][0]["entities"][0]["label"] == "model"


def test_is_already_annotated_reflects_current_state(tmp_path: Path) -> None:
    path = tmp_path / "arxiv_test.json"
    _write_processed_json(path, ["BERT is a model."])

    assert is_already_annotated(path) is False

    save_annotated(path, annotate_document(path, _FixedExtractor()))

    assert is_already_annotated(path) is True
