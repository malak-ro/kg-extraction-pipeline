"""
Représentation uniforme d'une entité extraite, peu importe le moteur NER
utilisé (spaCy, GliNER, LLM...) — même logique que LoadedDocument au
Jalon 2 : le reste du pipeline (relations, graphe) ne travaillera que sur
cet objet, jamais sur les structures internes propres à un moteur donné.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Entity:
    text: str        # texte exact de l'entité, ex: "BERT"
    label: str       # type détecté, ex: "ORG", "PERSON", ou un label personnalisé
    start_char: int  # position de début dans le texte source
    end_char: int     # position de fin (exclusive)

    def __repr__(self) -> str:
        return f"Entity({self.text!r}, {self.label})"
