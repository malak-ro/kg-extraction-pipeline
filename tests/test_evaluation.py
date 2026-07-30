from src.ner.entity import Entity
from src.ner.evaluation import EvalResult, evaluate, normalize_spacy_label


def _e(text: str, label: str) -> Entity:
    """Petit raccourci : les offsets ne sont pas utilisés par evaluate(),
    donc 0/0 suffit pour ces tests."""
    return Entity(text=text, label=label, start_char=0, end_char=0)


def test_perfect_match_gives_perfect_scores() -> None:
    gold = [_e("BERT", "model"), _e("Devlin", "person")]
    predicted = [_e("BERT", "model"), _e("Devlin", "person")]
    result = evaluate(predicted, gold)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_false_positive_reduces_precision_only() -> None:
    gold = [_e("BERT", "model")]
    predicted = [_e("BERT", "model"), _e("Paris", "organization")]  # "Paris" en trop, mauvais label
    result = evaluate(predicted, gold)
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.precision == 0.5
    assert result.recall == 1.0  # tout le gold a bien été retrouvé


def test_false_negative_reduces_recall_only() -> None:
    gold = [_e("BERT", "model"), _e("GLiNER", "model")]
    predicted = [_e("BERT", "model")]  # "GLiNER" manqué
    result = evaluate(predicted, gold)
    assert result.false_negatives == 1
    assert result.recall == 0.5
    assert result.precision == 1.0  # tout ce qui a été prédit était correct


def test_wrong_label_counts_as_both_fp_and_fn() -> None:
    """"BERT" prédit comme "organization" au lieu de "model" : ni un vrai
    positif (mauvais label), ni juste ignoré — un faux positif (mauvaise
    prédiction) ET un faux négatif (le vrai "BERT"/"model" n'a pas été
    trouvé)."""
    gold = [_e("BERT", "model")]
    predicted = [_e("BERT", "organization")]
    result = evaluate(predicted, gold)
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_empty_predicted_gives_zero_precision_and_recall() -> None:
    result = evaluate([], [_e("BERT", "model")])
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_duplicate_gold_entity_consumed_only_once() -> None:
    gold = [_e("F1", "metric"), _e("F1", "metric")]  # doublon volontaire
    predicted = [_e("F1", "metric")]
    result = evaluate(predicted, gold)
    assert result.true_positives == 1
    assert result.false_negatives == 1  # le 2e "F1" du gold reste non trouvé


def test_eval_result_addition_sums_all_fields() -> None:
    a = EvalResult(true_positives=2, false_positives=1, false_negatives=0)
    b = EvalResult(true_positives=1, false_positives=0, false_negatives=3)
    total = a + b
    assert total.true_positives == 3
    assert total.false_positives == 1
    assert total.false_negatives == 3


def test_normalize_spacy_label_maps_known_labels() -> None:
    assert normalize_spacy_label("PERSON") == "person"
    assert normalize_spacy_label("ORG") == "organization"


def test_normalize_spacy_label_passes_through_unknown_labels() -> None:
    """GPE/DATE/FAC/CARDINAL n'ont pas d'équivalent dans notre
    vocabulaire — ils doivent rester tels quels (et donc ne jamais
    matcher un label gold comme "dataset" ou "model")."""
    assert normalize_spacy_label("GPE") == "GPE"
    assert normalize_spacy_label("CARDINAL") == "CARDINAL"
