"""
Représentation uniforme d'une relation extraite entre deux entités, peu
importe l'approche utilisée (règles syntaxiques, LLM...) — même logique
qu'Entity au Jalon 3.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ner.entity import Entity


@dataclass
class Relation:
    subject: Entity
    predicate: str
    obj: Entity  # "object" est un mot réservé Python, d'où "obj"
    sentence: str  # phrase source — utile pour tracer/vérifier une relation

    def __repr__(self) -> str:
        return f"({self.subject.text}) -[{self.predicate}]-> ({self.obj.text})"
