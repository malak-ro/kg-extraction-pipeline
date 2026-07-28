"""
Extraction d'entités via spaCy (en_core_web_sm) — labels génériques issus
du jeu d'entraînement OntoNotes (PERSON, ORG, GPE, DATE, CARDINAL...).

Important, vérifié empiriquement (voir tests) : ce modèle n'a jamais vu de
labels comme "MODEL" ou "DATASET" — impossible de les lui demander. Il
reste utile pour ce qu'il capture réellement dans nos abstracts (noms
d'auteurs, organisations, dates), mais force les entités propres à la
recherche IA dans des catégories génériques inadaptées (ex: "BERT" détecté
mais étiqueté ORG, faute de mieux). Pour des labels personnalisés adaptés
à notre domaine, direction GliNER en zero-shot — sous-étape suivante.
"""
from __future__ import annotations

import spacy
from spacy.language import Language

from src.ner.base import EntityExtractor
from src.ner.entity import Entity

_MODEL_NAME = "en_core_web_sm"
_nlp: Language | None = None


def _get_nlp() -> Language:
    """Même pattern de cache que sentence_splitter.py — mais ici on garde
    le composant 'ner' actif (contrairement à la segmentation de phrases,
    qui le désactivait car inutile pour son propre besoin)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(_MODEL_NAME, disable=["lemmatizer"])
        except OSError as exc:
            raise RuntimeError(
                f"Modèle spaCy '{_MODEL_NAME}' introuvable. Installe-le avec : "
                f"python -m spacy download {_MODEL_NAME}"
            ) from exc
    return _nlp


class SpacyEntityExtractor(EntityExtractor):
    """Labels génériques OntoNotes : PERSON, ORG, GPE, DATE, CARDINAL..."""

    def extract(self, text: str) -> list[Entity]:
        if not text.strip():
            return []
        doc = _get_nlp()(text)
        return [
            Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
            )
            for ent in doc.ents
        ]
