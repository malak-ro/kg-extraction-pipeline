from src.ner.entity import Entity
from src.relation_extraction.dependency_extractor import DependencyRelationExtractor


def _e(text: str, label: str, sentence: str) -> Entity:
    start = sentence.index(text)
    return Entity(text=text, label=label, start_char=start, end_char=start + len(text))


def test_extract_returns_empty_list_with_fewer_than_two_entities() -> None:
    extractor = DependencyRelationExtractor()
    assert extractor.extract("BERT is a model.", []) == []
    sentence = "BERT is a model."
    assert extractor.extract(sentence, [_e("BERT", "model", sentence)]) == []


def test_extract_finds_direct_object_relation_active_voice() -> None:
    sentence = "Devlin introduced BERT."
    entities = [_e("Devlin", "person", sentence), _e("BERT", "model", sentence)]
    results = DependencyRelationExtractor().extract(sentence, entities)

    assert len(results) == 1
    assert results[0].subject.text == "Devlin"
    assert results[0].predicate == "introduce"
    assert results[0].obj.text == "BERT"


def test_extract_finds_prepositional_relation_passive_voice_on() -> None:
    """Cas trouvé fréquent dans nos abstracts : "X was trained/evaluated ON Y"."""
    sentence = "BERT was trained on Wikipedia."
    entities = [_e("BERT", "model", sentence), _e("Wikipedia", "dataset", sentence)]
    results = DependencyRelationExtractor().extract(sentence, entities)

    assert len(results) == 1
    assert results[0].predicate == "train_on"


def test_extract_finds_prepositional_relation_passive_voice_by() -> None:
    """Cas vérifié empiriquement : spaCy étiquette le "by" passif comme
    dépendance "agent", pas "prep" — les deux doivent être couvertes."""
    sentence = "GLiNER was proposed by Zaratiana."
    entities = [_e("GLiNER", "model", sentence), _e("Zaratiana", "person", sentence)]
    results = DependencyRelationExtractor().extract(sentence, entities)

    assert len(results) == 1
    assert results[0].predicate == "propose_by"


def test_extract_returns_empty_when_entities_share_no_verb() -> None:
    """Deux entités présentes dans la phrase mais sans verbe les reliant
    directement (ici, sujets coordonnés) -> pas de relation forcée."""
    sentence = "BERT and GLiNER are both popular in NLP research."
    entities = [_e("BERT", "model", sentence), _e("GLiNER", "model", sentence)]
    results = DependencyRelationExtractor().extract(sentence, entities)
    assert results == []


def test_extract_does_not_relate_an_entity_to_itself() -> None:
    """Si un même texte apparaît deux fois et matche par erreur la même
    entité des deux côtés, on ne doit jamais produire (X)-[...]->(X)."""
    sentence = "BERT was trained on Wikipedia."
    entities = [_e("BERT", "model", sentence), _e("Wikipedia", "dataset", sentence)]
    results = DependencyRelationExtractor().extract(sentence, entities)
    for r in results:
        assert r.subject is not r.obj
