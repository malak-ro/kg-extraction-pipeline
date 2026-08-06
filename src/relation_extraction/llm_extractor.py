"""
Extraction de relations via un LLM, en passant par Groq (gratuit, sans
carte bancaire, API compatible OpenAI — cf. config/settings.py).

Pourquoi en complément de l'approche syntaxique (sous-étape 1) ?
Un LLM ne dépend pas d'un patron syntaxique précis : il gère mieux les
formulations complexes, coordonnées, ou implicites qu'un arbre de
dépendances. Le prix à payer : un appel réseau (latence, coût potentiel),
et un risque d'hallucination — qu'on limite en validant que chaque
sujet/objet retourné correspond EXACTEMENT à une entité qu'on a fournie
au modèle, jamais une entité qu'il aurait inventée.

Correctif appliqué après un run réel sur tout le corpus : gpt-oss-20b est
un modèle "raisonneur" (chain-of-thought interne avant la réponse finale).
Sur ~250 appels, 18 sont revenus avec un contenu vide — le raisonnement
interne avait épuisé le budget de tokens avant de produire la réponse
(comportement documenté sur le forum communautaire de Groq). Notre tâche
(extraire un triplet simple, entités déjà données) n'a besoin d'aucun
raisonnement profond, d'où reasoning_effort="low" + un budget de tokens
généreux en filet de sécurité.
"""
from __future__ import annotations

import json

from openai import OpenAI

from config.settings import settings
from src.ner.entity import Entity
from src.relation_extraction.base import RelationExtractor
from src.relation_extraction.relation import Relation
from src.utils.exceptions import PipelineError
from src.utils.logger import get_logger

logger = get_logger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Modèle ouvert hébergé par Groq (le nom contient "openai/" mais ce n'est
# PAS l'API OpenAI — voir la conversation sur le choix de Groq).
MODEL_NAME = "openai/gpt-oss-20b"

_SYSTEM_PROMPT = (
    "You are a precise relation extraction system for a research knowledge "
    "graph. Given a sentence and a list of entities already identified in "
    "it, find relations explicitly stated or clearly implied between pairs "
    "of these entities. Use ONLY the exact entity texts provided — never "
    "invent new entities. Respond with ONLY a JSON array, no other text, "
    "no markdown formatting."
)


class LlmRelationExtractionError(PipelineError):
    """Levée quand l'appel au LLM échoue (réseau, clé invalide...)."""


class LLMRelationExtractor(RelationExtractor):
    def __init__(self, client: OpenAI | None = None, model: str = MODEL_NAME) -> None:
        """`client` est injectable : c'est ce qui permet de tester cette
        classe avec un client factice, sans jamais appeler le vrai réseau
        (voir tests/test_relation_llm.py)."""
        self._client = client or OpenAI(base_url=GROQ_BASE_URL, api_key=settings.groq_api_key)
        self._model = model

    def extract(self, sentence: str, entities: list[Entity]) -> list[Relation]:
        if len(entities) < 2:
            return []

        entity_by_text = {e.text: e for e in entities}

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(sentence, entities)},
                ],
                temperature=0,
                # Tâche simple (triplet à partir d'entités déjà données) :
                # pas besoin d'un raisonnement poussé, et ça évite que le
                # raisonnement interne épuise le budget avant la réponse.
                reasoning_effort="low",
                max_completion_tokens=1000,
            )
        except Exception as exc:  # le client OpenAI lève plusieurs types selon l'erreur
            raise LlmRelationExtractionError(f"Appel au LLM échoué : {exc}") from exc

        raw = response.choices[0].message.content
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Réponse LLM non-JSON, ignorée : %r", raw)
            return []

        return self._validate_and_convert(parsed, entity_by_text, sentence)

    def _build_prompt(self, sentence: str, entities: list[Entity]) -> str:
        entity_list = ", ".join(f'"{e.text}" ({e.label})' for e in entities)
        return (
            f'Sentence: "{sentence}"\n'
            f"Entities: {entity_list}\n\n"
            'Return a JSON array of relations, each as '
            '{"subject": ..., "predicate": ..., "object": ...} '
            "(subject and object must exactly match one of the entity texts "
            "above). Return [] if no relation exists."
        )

    def _validate_and_convert(
        self, parsed: object, entity_by_text: dict[str, Entity], sentence: str
    ) -> list[Relation]:
        """Rejette silencieusement toute ligne malformée ou toute entité
        qui ne correspond pas exactement à celles fournies — c'est le
        principal garde-fou contre l'hallucination du LLM."""
        if not isinstance(parsed, list):
            return []

        relations = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            subject = entity_by_text.get(item.get("subject"))
            obj = entity_by_text.get(item.get("object"))
            predicate = item.get("predicate")
            if subject is None or obj is None or not predicate or subject is obj:
                continue
            relations.append(Relation(subject=subject, predicate=predicate, obj=obj, sentence=sentence))
        return relations
