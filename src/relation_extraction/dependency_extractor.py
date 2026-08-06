"""
Extraction de relations par analyse syntaxique (dependency parsing) —
l'approche "NLP classique", sans LLM ni entraînement.

Principe : pour chaque verbe de la phrase, on regarde ses enfants
syntaxiques directs — sujet (nsubj/nsubjpass) et complément (dobj, ou
préposition + pobj). Si le sujet ET le complément correspondent chacun à
une entité déjà connue (extraites au Jalon 3), on émet une relation.

Point vérifié empiriquement, pas supposé : en voix passive ("X was
proposed by Y"), spaCy étiquette "by" avec la dépendance "agent", pas
"prep" — les deux doivent être vérifiées pour couvrir les tournures
actives ET passives, très fréquentes dans les abstracts scientifiques.

Limite assumée (pas un oubli) : seuls les liens syntaxiques DIRECTS sont
couverts. Une phrase avec coordination ("X fait A et fait B"), où le 2e
verbe hérite du sujet du 1er via une relation "conj", n'est pas gérée —
et le parser lui-même se trompe parfois sur des tournures inhabituelles
(ex: "et al."). Exactement le genre de cas où l'approche LLM
(sous-étape 2) devrait mieux se comporter, sans règle à écrire à la main
pour chaque construction syntaxique possible.
"""
from __future__ import annotations

import spacy
from spacy.language import Language
from spacy.tokens import Token

from src.ner.entity import Entity
from src.relation_extraction.base import RelationExtractor
from src.relation_extraction.relation import Relation

_MODEL_NAME = "en_core_web_sm"
_nlp: Language | None = None

_SUBJECT_DEPS = {"nsubj", "nsubjpass"}
_OBJECT_DEPS = {"dobj", "attr", "oprd"}
_PREP_DEPS = {"prep", "agent"}  # "agent" = le "by" d'une tournure passive


def _get_nlp() -> Language:
    """Même pattern de cache que les modules précédents. Le NER interne
    de spaCy reste désactivé : on utilise les entités DÉJÀ extraites par
    GLiNER au Jalon 3, pas celles que spaCy trouverait lui-même. Le
    lemmatizer, en revanche, DOIT rester actif — contrairement à
    sentence_splitter.py qui n'en avait pas besoin, ce module l'utilise
    pour construire le prédicat (token.lemma_)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(_MODEL_NAME, disable=["ner"])
        except OSError as exc:
            raise RuntimeError(
                f"Modèle spaCy '{_MODEL_NAME}' introuvable. Installe-le avec : "
                f"python -m spacy download {_MODEL_NAME}"
            ) from exc
    return _nlp


def _subtree_span_chars(token: Token) -> tuple[int, int]:
    """Étend un token à tout son sous-arbre syntaxique — ex: le nom tête
    "benchmark" s'étend à "the CoNLL-2003 benchmark" — pour mieux
    chevaucher des entités multi-mots dont ce token n'est que la tête."""
    subtree = list(token.subtree)
    return min(t.idx for t in subtree), max(t.idx + len(t.text) for t in subtree)


def _match_entity(token: Token, entities_by_char: dict[tuple[int, int], Entity]) -> Entity | None:
    """Fait correspondre un span syntaxique à une entité déjà connue par
    CHEVAUCHEMENT de positions de caractères (pas égalité stricte de
    texte, plus fiable en cas de répétition d'un mot dans la phrase)."""
    start, end = _subtree_span_chars(token)
    for (e_start, e_end), entity in entities_by_char.items():
        if start <= e_end and end >= e_start:
            return entity
    return None


class DependencyRelationExtractor(RelationExtractor):
    def extract(self, sentence: str, entities: list[Entity]) -> list[Relation]:
        if len(entities) < 2:
            return []

        doc = _get_nlp()(sentence)
        entities_by_char = {(e.start_char, e.end_char): e for e in entities}
        relations: list[Relation] = []

        for token in doc:
            if token.pos_ != "VERB":
                continue

            subject_token = next((c for c in token.children if c.dep_ in _SUBJECT_DEPS), None)
            if subject_token is None:
                continue
            subject_entity = _match_entity(subject_token, entities_by_char)
            if subject_entity is None:
                continue

            # Objet direct : "introduced BERT"
            object_token = next((c for c in token.children if c.dep_ in _OBJECT_DEPS), None)
            if object_token is not None:
                object_entity = _match_entity(object_token, entities_by_char)
                if object_entity is not None and object_entity is not subject_entity:
                    relations.append(Relation(subject_entity, token.lemma_, object_entity, sentence))

            # Complément prépositionnel : "evaluated ON X", "proposed BY Y"
            for prep in (c for c in token.children if c.dep_ in _PREP_DEPS):
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue
                object_entity = _match_entity(pobj, entities_by_char)
                if object_entity is not None and object_entity is not subject_entity:
                    predicate = f"{token.lemma_}_{prep.text}"
                    relations.append(Relation(subject_entity, predicate, object_entity, sentence))

        return relations
