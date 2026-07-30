"""
Calcul precision/recall/F1 pour comparer les prédictions d'un extracteur
au gold standard annoté à la main.

Évaluation "exacte" (texte + label doivent correspondre exactement) —
comme dans l'étude que ton encadrant a partagée (section 8.1,
"Exact Evaluation"), pas d'évaluation "relâchée" (partial credit).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ner.entity import Entity

# spaCy utilise des labels génériques différents des nôtres (PERSON/ORG
# vs person/organization). On ne normalise QUE les deux catégories qui se
# recoupent vraiment, pour permettre une comparaison honnête sur ce
# terrain commun. Les autres labels spaCy (GPE, DATE, FAC, CARDINAL...)
# n'ont pas d'équivalent dans notre vocabulaire (model/dataset/method/
# metric) : ils ne matcheront jamais un gold — résultat attendu, pas un
# bug de l'évaluation (c'est exactement la limite qu'on documente).
SPACY_LABEL_MAP = {"PERSON": "person", "ORG": "organization"}


def normalize_spacy_label(label: str) -> str:
    return SPACY_LABEL_MAP.get(label, label)


@dataclass
class EvalResult:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __add__(self, other: "EvalResult") -> "EvalResult":
        """Permet d'additionner les résultats de plusieurs phrases avec
        `sum(...)` ou `+=` — pratique pour cumuler sur tout le gold standard."""
        return EvalResult(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )


def _entity_key(entity: Entity) -> tuple[str, str]:
    """Deux entités matchent si texte ET label sont identiques — pas de
    tolérance sur les positions de caractères, le couple (texte, label)
    suffit à identifier une entité de façon fiable ici."""
    return (entity.text, entity.label)


def evaluate(predicted: list[Entity], gold: list[Entity]) -> EvalResult:
    """Compare une liste d'entités prédites à la liste gold. Chaque
    entité gold n'est "consommée" qu'une fois, pour qu'une seule bonne
    prédiction ne compte pas plusieurs fois si le gold a un doublon."""
    remaining_gold = [_entity_key(e) for e in gold]
    true_positives = 0

    for entity in predicted:
        key = _entity_key(entity)
        if key in remaining_gold:
            remaining_gold.remove(key)
            true_positives += 1

    false_positives = len(predicted) - true_positives
    false_negatives = len(gold) - true_positives
    return EvalResult(true_positives, false_positives, false_negatives)
