import json
from unittest.mock import MagicMock

import pytest

from src.ner.entity import Entity
from src.relation_extraction.llm_extractor import LLMRelationExtractor, LlmRelationExtractionError


def _fake_client(response_text: str) -> MagicMock:
    client = MagicMock()
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    client.chat.completions.create.return_value = completion
    return client


BERT = Entity(text="BERT", label="model", start_char=0, end_char=4)
WIKI = Entity(text="Wikipedia", label="dataset", start_char=21, end_char=30)
SENTENCE = "BERT was trained on Wikipedia."


def test_extract_converts_valid_json_response_to_relations() -> None:
    response = json.dumps([{"subject": "BERT", "predicate": "trained on", "object": "Wikipedia"}])
    extractor = LLMRelationExtractor(client=_fake_client(response))

    relations = extractor.extract(SENTENCE, [BERT, WIKI])

    assert len(relations) == 1
    assert relations[0].subject is BERT
    assert relations[0].obj is WIKI
    assert relations[0].predicate == "trained on"


def test_extract_discards_hallucinated_entities_not_in_input() -> None:
    response = json.dumps([{"subject": "BERT", "predicate": "compared to", "object": "GPT-4"}])
    extractor = LLMRelationExtractor(client=_fake_client(response))

    relations = extractor.extract(SENTENCE, [BERT, WIKI])

    assert relations == []


def test_extract_discards_self_relations() -> None:
    response = json.dumps([{"subject": "BERT", "predicate": "is", "object": "BERT"}])
    extractor = LLMRelationExtractor(client=_fake_client(response))

    relations = extractor.extract(SENTENCE, [BERT, WIKI])

    assert relations == []


def test_extract_returns_empty_list_on_malformed_json() -> None:
    extractor = LLMRelationExtractor(client=_fake_client("this is not json at all"))

    relations = extractor.extract(SENTENCE, [BERT, WIKI])

    assert relations == []


def test_extract_returns_empty_list_with_fewer_than_two_entities_without_calling_api() -> None:
    client = _fake_client("[]")
    extractor = LLMRelationExtractor(client=client)

    assert extractor.extract(SENTENCE, [BERT]) == []
    client.chat.completions.create.assert_not_called()


def test_extract_wraps_api_errors() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = ConnectionError("réseau indisponible")
    extractor = LLMRelationExtractor(client=client)

    with pytest.raises(LlmRelationExtractionError):
        extractor.extract(SENTENCE, [BERT, WIKI])


def test_extract_passes_low_reasoning_effort_and_token_cap() -> None:
    """Vérifie le correctif : reasoning_effort bas + max_completion_tokens
    explicite, pour éviter les réponses vides observées sur le vrai run
    (raisonnement interne de gpt-oss-20b épuisant le budget)."""
    client = _fake_client("[]")
    LLMRelationExtractor(client=client).extract(SENTENCE, [BERT, WIKI])

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_completion_tokens"] >= 500
