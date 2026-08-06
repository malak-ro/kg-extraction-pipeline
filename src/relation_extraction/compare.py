"""
Compare les deux extracteurs de relations sur un petit jeu de phrases,
choisies pour inclure des cas où l'approche syntaxique (sous-étape 1) est
connue pour échouer (coordination, ordre inhabituel, comparaison) — voir
les docstrings de dependency_extractor.py.

Contrairement à la comparaison chiffrée du NER (Jalon 3), pas de
precision/recall/F1 ici : construire un gold standard de relations fiable
demanderait beaucoup plus d'annotation que pour des entités (il faut
spécifier les bonnes PAIRES, pas juste des spans). Une comparaison
qualitative sur des cas volontairement difficiles est proportionnée à ce
stade — un gold standard chiffré reste possible à ajouter plus tard si
besoin d'un chiffre précis.

Lance ce script sur ta machine : `python -m src.relation_extraction.compare`
(fait un vrai appel réseau à Groq pour la partie LLM).
"""
from __future__ import annotations

from src.ner.entity import Entity


def _entity(text: str, label: str, sentence: str) -> Entity:
    start = sentence.index(text)
    return Entity(text=text, label=label, start_char=start, end_char=start + len(text))


# (phrase, [(texte_entite, label), ...])
TEST_CASES: list[tuple[str, list[tuple[str, str]]]] = [
    # Cas "facile" : sert de repère, les deux approches devraient réussir.
    ("BERT was trained on Wikipedia.", [("BERT", "model"), ("Wikipedia", "dataset")]),
    # Coordination : l'approche syntaxique ne relie que la première entité.
    (
        "GLiNER and spaCy were both evaluated on the WNUT2017 dataset.",
        [("GLiNER", "model"), ("spaCy", "model"), ("WNUT2017", "dataset")],
    ),
    # Proposition en tête de phrase (ordre inhabituel).
    (
        "Introduced by Devlin et al. in 2018, BERT quickly became the standard approach.",
        [("Devlin", "person"), ("BERT", "model")],
    ),
    # Structure de comparaison.
    (
        "Compared to spaCy, GLiNER achieves higher accuracy on domain-specific entities.",
        [("spaCy", "model"), ("GLiNER", "model")],
    ),
]


def run_comparison() -> None:
    # Imports locaux : évite de charger spaCy/le client LLM juste pour
    # importer TEST_CASES ailleurs (ex: dans un test).
    from src.relation_extraction.dependency_extractor import DependencyRelationExtractor
    from src.relation_extraction.llm_extractor import LLMRelationExtractor

    dependency = DependencyRelationExtractor()
    llm = LLMRelationExtractor()

    for sentence, entity_specs in TEST_CASES:
        entities = [_entity(text, label, sentence) for text, label in entity_specs]
        print(f"\n{sentence}")
        print("  dependency:", dependency.extract(sentence, entities) or "(aucune relation)")
        print("  llm       :", llm.extract(sentence, entities) or "(aucune relation)")


if __name__ == "__main__":
    run_comparison()
