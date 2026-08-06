"""
Interface commune aux extracteurs de relations (Strategy pattern, même
logique qu'EntityExtractor au Jalon 3).

Différence de signature importante avec EntityExtractor : ici on prend en
entrée la phrase ET les entités déjà connues (extraites au Jalon 3), pas
juste le texte brut — l'extraction de relations opère sur des PAIRES
d'entités déjà identifiées, elle ne les redécouvre pas.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.ner.entity import Entity
from src.relation_extraction.relation import Relation


class RelationExtractor(ABC):
    @abstractmethod
    def extract(self, sentence: str, entities: list[Entity]) -> list[Relation]:
        """Retourne les relations trouvées entre les entités de `entities`
        au sein de `sentence`."""
