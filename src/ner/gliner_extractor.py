"""
Extraction d'entités via GLiNER en zero-shot.

Contrairement à spaCy (labels figés par son jeu d'entraînement), GLiNER
accepte n'importe quelle liste de labels au moment de l'appel, sans
réentraînement — exactement la réponse à la limite documentée dans
spacy_extractor.py. On peut donc lui demander directement "model",
"dataset", "method", "metric" : les catégories qui comptent pour NOTRE
graphe de connaissances, pas des catégories génériques.

Modèle choisi : gliner_medium-v2.1 (licence Apache 2.0). Volontairement
PAS "gliner_large" tout court (l'ancienne version sans suffixe -v2.1,
sous licence CC BY-NC — restrictive pour un usage professionnel/portfolio).
"large-v2.1" reste disponible en changeant juste MODEL_NAME si tu veux
troquer un peu de vitesse contre un peu de précision.

Différence importante avec spaCy côté installation : pas de commande
séparée type `spacy download` — le modèle se télécharge automatiquement
au premier appel de `from_pretrained()` (mécanisme Hugging Face standard,
mis en cache localement ensuite). En contrepartie, GLiNER est un modèle
Transformer : torch (PyTorch) est requis, et le téléchargement est
nettement plus gros que ce qu'on a installé jusqu'ici.
"""
from __future__ import annotations

from gliner import GLiNER

from src.ner.base import EntityExtractor
from src.ner.entity import Entity

MODEL_NAME = "urchade/gliner_medium-v2.1"

# Labels choisis pour NOTRE graphe (recherche IA) — pas les catégories
# génériques de spaCy. GLiNER fonctionne mieux avec des labels en
# minuscules ou "Title Case" (recommandation officielle du projet).
DEFAULT_LABELS = ["model", "dataset", "method", "metric", "person", "organization"]

_model: GLiNER | None = None


def _get_model() -> GLiNER:
    """Même pattern de cache que spaCy (_nlp) — mais ici le premier appel
    télécharge aussi les poids du modèle depuis Hugging Face s'ils ne sont
    pas déjà en cache local."""
    global _model
    if _model is None:
        _model = GLiNER.from_pretrained(MODEL_NAME)
    return _model


class GlinerEntityExtractor(EntityExtractor):
    """Zero-shot : les labels sont fournis à la construction (pas figés
    dans le modèle comme avec spaCy), ce qui permet de comparer
    facilement différents jeux de labels sans changer de modèle."""

    def __init__(self, labels: list[str] | None = None, threshold: float = 0.5) -> None:
        self.labels = labels if labels is not None else DEFAULT_LABELS
        self.threshold = threshold

    def extract(self, text: str) -> list[Entity]:
        if not text.strip():
            return []
        raw_entities = _get_model().predict_entities(text, self.labels, threshold=self.threshold)
        return [
            Entity(
                text=e["text"],
                label=e["label"],
                start_char=e["start"],
                end_char=e["end"],
            )
            for e in raw_entities
        ]
