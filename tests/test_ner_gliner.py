"""
Le vrai modèle GLiNER n'est jamais chargé dans ces tests : ni accès
réseau vers Hugging Face, ni téléchargement (~centaines de Mo), ni temps
d'inférence réel. On simule (mock) `predict_entities()` pour vérifier
UNIQUEMENT notre logique : conversion dict -> Entity, transmission des
labels/seuil, court-circuit sur texte vide. La vérification avec le
vrai modèle se fait une fois, à la main, sur ta machine (voir les
instructions d'intégration).
"""
from unittest.mock import MagicMock, patch

import pytest

import src.ner.gliner_extractor as gliner_extractor_module
from src.ner.entity import Entity
from src.ner.gliner_extractor import DEFAULT_LABELS, GlinerEntityExtractor


@pytest.fixture(autouse=True)
def _reset_model_cache():
    """Le cache module-level (_model) doit être remis à zéro entre
    chaque test — sinon un test pourrait silencieusement réutiliser le
    mock chargé par un test précédent au lieu d'appeler from_pretrained()
    lui-même, et fausser les assertions sur les appels du mock."""
    gliner_extractor_module._model = None
    yield
    gliner_extractor_module._model = None


@patch("src.ner.gliner_extractor.GLiNER.from_pretrained")
def test_extract_converts_gliner_dicts_to_entities(mock_from_pretrained) -> None:
    mock_model = MagicMock()
    mock_model.predict_entities.return_value = [
        {"text": "BERT", "label": "model", "start": 0, "end": 4, "score": 0.98},
        {"text": "CoNLL-2003", "label": "dataset", "start": 20, "end": 30, "score": 0.87},
    ]
    mock_from_pretrained.return_value = mock_model

    extractor = GlinerEntityExtractor()
    entities = extractor.extract("BERT is trained and tested on CoNLL-2003.")

    assert entities == [
        Entity(text="BERT", label="model", start_char=0, end_char=4),
        Entity(text="CoNLL-2003", label="dataset", start_char=20, end_char=30),
    ]


@patch("src.ner.gliner_extractor.GLiNER.from_pretrained")
def test_extract_passes_custom_labels_and_threshold(mock_from_pretrained) -> None:
    mock_model = MagicMock()
    mock_model.predict_entities.return_value = []
    mock_from_pretrained.return_value = mock_model

    extractor = GlinerEntityExtractor(labels=["method"], threshold=0.7)
    extractor.extract("some text")

    mock_model.predict_entities.assert_called_once_with("some text", ["method"], threshold=0.7)


@patch("src.ner.gliner_extractor.GLiNER.from_pretrained")
def test_extract_uses_default_labels_when_none_given(mock_from_pretrained) -> None:
    mock_model = MagicMock()
    mock_model.predict_entities.return_value = []
    mock_from_pretrained.return_value = mock_model

    GlinerEntityExtractor().extract("some text")

    mock_model.predict_entities.assert_called_once_with("some text", DEFAULT_LABELS, threshold=0.5)


@patch("src.ner.gliner_extractor.GLiNER.from_pretrained")
def test_extract_returns_empty_list_for_empty_text_without_loading_model(
    mock_from_pretrained,
) -> None:
    """Court-circuite avant même de charger le modèle — évite un
    chargement (et un éventuel téléchargement) inutile sur du texte vide."""
    extractor = GlinerEntityExtractor()
    assert extractor.extract("") == []
    assert extractor.extract("   ") == []
    mock_from_pretrained.assert_not_called()


def test_default_labels_match_our_kg_domain() -> None:
    assert "model" in DEFAULT_LABELS
    assert "dataset" in DEFAULT_LABELS
