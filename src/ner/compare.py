"""
Compare spaCy et GliNER sur le gold standard et affiche un rapport.

Contrairement aux modules précédents, ce script fait tourner les VRAIS
modèles (pas de mock ici — c'est justement le but). Lance-le sur ta
machine : `python -m src.ner.compare`.
"""
from __future__ import annotations

from src.ner.base import EntityExtractor
from src.ner.entity import Entity
from src.ner.evaluation import EvalResult, evaluate, normalize_spacy_label
from src.ner.gold_standard import GOLD_EXAMPLES


def _normalize_spacy_entities(entities: list[Entity]) -> list[Entity]:
    return [
        Entity(text=e.text, label=normalize_spacy_label(e.label), start_char=e.start_char, end_char=e.end_char)
        for e in entities
    ]


def compare(extractors: dict[str, EntityExtractor]) -> dict[str, EvalResult]:
    """Additionne TP/FP/FN sur tous les exemples gold, par extracteur.
    `extractors` est un dict injecté (pas construit ici) : c'est ce qui
    rend cette fonction testable sans les vrais modèles — voir
    tests/test_ner_compare.py, avec des extracteurs factices."""
    totals = {name: EvalResult(0, 0, 0) for name in extractors}

    for example in GOLD_EXAMPLES:
        for name, extractor in extractors.items():
            predicted = extractor.extract(example.text)
            if name == "spacy":
                predicted = _normalize_spacy_entities(predicted)
            totals[name] = totals[name] + evaluate(predicted, example.gold_entities)

    return totals


def print_report(totals: dict[str, EvalResult]) -> None:
    print(f"{'Extracteur':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for name, result in totals.items():
        print(f"{name:<10} {result.precision:>10.2f} {result.recall:>10.2f} {result.f1:>10.2f}")


if __name__ == "__main__":
    # Imports ici, pas en haut du fichier : évite de charger spaCy/GLiNER
    # (et donc potentiellement de télécharger le modèle GLiNER) juste en
    # important ce module pour ses fonctions utilitaires (comme le font
    # les tests).
    from src.ner.gliner_extractor import GlinerEntityExtractor
    from src.ner.spacy_extractor import SpacyEntityExtractor

    results = compare({"spacy": SpacyEntityExtractor(), "gliner": GlinerEntityExtractor()})
    print_report(results)
