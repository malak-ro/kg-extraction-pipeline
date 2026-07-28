from src.ner.spacy_extractor import SpacyEntityExtractor


def test_extract_returns_empty_list_for_empty_text() -> None:
    extractor = SpacyEntityExtractor()
    assert extractor.extract("") == []
    assert extractor.extract("   ") == []


def test_extract_recognizes_full_person_names() -> None:
    extractor = SpacyEntityExtractor()
    entities = extractor.extract(
        "Yann LeCun and Geoffrey Hinton are pioneers of deep learning."
    )
    labels = {(e.text, e.label) for e in entities}
    assert ("Yann LeCun", "PERSON") in labels
    assert ("Geoffrey Hinton", "PERSON") in labels


def test_extract_recognizes_dates_and_locations() -> None:
    extractor = SpacyEntityExtractor()
    entities = extractor.extract("The dataset was released in San Francisco in 2018.")
    labels = {(e.text, e.label) for e in entities}
    assert ("San Francisco", "GPE") in labels
    assert ("2018", "DATE") in labels


def test_entity_char_offsets_match_source_text() -> None:
    """Vérifie que start_char/end_char pointent bien vers le bon texte —
    important pour plus tard (extraction de relations basée sur la
    position des entités, surlignage dans une UI)."""
    text = "The dataset was released in San Francisco in 2018."
    extractor = SpacyEntityExtractor()
    entities = extractor.extract(text)
    assert len(entities) > 0
    for entity in entities:
        assert text[entity.start_char : entity.end_char] == entity.text


def test_known_limitation_generic_labels_misclassify_domain_terms() -> None:
    """Test de caractérisation (pas une exigence) : documente une limite
    réelle et observée du modèle générique sur notre domaine — "BERT" est
    bien détecté comme entité, mais étiqueté ORG faute d'une catégorie
    adaptée. C'est exactement pourquoi la sous-étape suivante introduit
    GliNER avec des labels personnalisés (MODEL, DATASET...). Si ce test
    casse un jour (mise à jour du modèle qui corrige ce cas), tant mieux —
    ça voudra dire que cet argument n'est plus valable et qu'il faut
    reconsidérer le choix."""
    extractor = SpacyEntityExtractor()
    entities = extractor.extract(
        "Devlin et al. introduced BERT at Google AI Language in 2018."
    )
    labels = {e.text: e.label for e in entities}
    assert labels.get("BERT") == "ORG"
