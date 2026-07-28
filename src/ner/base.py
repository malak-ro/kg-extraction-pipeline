"""
Interface commune à tous les moteurs d'extraction d'entités (Strategy
pattern).

Pourquoi une interface plutôt qu'une fonction avec un paramètre
backend="spacy" ? On va comparer plusieurs moteurs dans ce Jalon (spaCy,
GliNER, potentiellement un LLM), avec des besoins de configuration très
différents — GliNER a besoin d'une liste de labels personnalisés, spaCy
n'en a pas. Chaque implémentation garde sa propre logique
d'initialisation, mais expose le même contrat `extract()`.

Différence avec DocumentLoaderRegistry (Jalon 2) : le registry choisissait
automatiquement le bon loader selon l'extension du fichier (dispatch
automatique). Ici c'est nous qui choisissons délibérément quel
extracteur utiliser — ou on les fait tourner tous en parallèle pour les
comparer. Pas d'auto-détection : deux situations différentes, deux
patterns différents.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.ner.entity import Entity


class EntityExtractor(ABC):
    """Contrat commun : n'importe quel moteur NER sait répondre à
    'donne-moi les entités de ce texte'."""

    @abstractmethod
    def extract(self, text: str) -> list[Entity]:
        """Retourne la liste des entités détectées dans `text`."""
