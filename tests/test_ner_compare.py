"""
compare() est testée avec des extracteurs FACTICES (implémentant juste
EntityExtractor), jamais les vrais spaCy/GLiNER — exactement le bénéfice
concret du Strategy pattern posé à la sous-étape 1 : n'importe quoi qui
respecte l'interface peut remplacer un vrai modèle pour tester
l'orchestration en isolation, sans dépendance, sans téléchargement, sans
lenteur.
"""
from src.ner.base import EntityExtractor
from src.ner.compare import compare
from src.ner.entity import Entity
from src.ner.gold_standard import GOLD_EXAMPLES


class _PerfectExtractor(EntityExtractor):
    """Retourne toujours exactement les entités gold de l'exemple en
    cours — sert à vérifier qu'un extracteur "parfait" obtient bien
    precision = recall = 1.0 sur l'ensemble du gold standard."""

    def extract(self, text: str) -> list[Entity]:
        for example in GOLD_EXAMPLES:
            if example.text == text:
                return example.gold_entities
        return []


class _EmptyExtractor(EntityExtractor):
    def extract(self, text: str) -> list[Entity]:
        return []


def test_perfect_extractor_scores_1_0_across_all_gold_examples() -> None:
    totals = compare({"perfect": _PerfectExtractor()})
    result = totals["perfect"]
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_empty_extractor_scores_zero() -> None:
    totals = compare({"empty": _EmptyExtractor()})
    result = totals["empty"]
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.false_negatives > 0  # tout le gold standard est manqué


def test_compare_accumulates_across_all_three_gold_examples() -> None:
    """Le nombre total d'entités gold dans les 3 exemples de départ est
    connu (5 + 1 + 4 = 10) — sert de garde-fou si quelqu'un modifie
    gold_standard.py sans s'en rendre compte."""
    totals = compare({"empty": _EmptyExtractor()})
    assert totals["empty"].false_negatives == 10


def test_spacy_label_normalization_is_only_applied_to_the_spacy_entry() -> None:
    """Une entité étiquetée "PERSON" (label spaCy brut) ne doit matcher un
    gold "person" QUE si elle passe par la clé "spacy" (normalisée) —
    pas si elle passe par une autre clé (non normalisée)."""
    raw_person = [Entity(text="Devlin", label="PERSON", start_char=0, end_char=6)]

    class _RawPersonExtractor(EntityExtractor):
        def extract(self, text: str) -> list[Entity]:
            return raw_person

    totals = compare({"spacy": _RawPersonExtractor(), "other": _RawPersonExtractor()})

    # "spacy" : normalisé PERSON -> person, doit matcher le "Devlin"/"person" du gold
    assert totals["spacy"].true_positives >= 1
    # "other" : pas normalisé, "PERSON" (majuscules) ne matche jamais "person"
    assert totals["other"].true_positives == 0
