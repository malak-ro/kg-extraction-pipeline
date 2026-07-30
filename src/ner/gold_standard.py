"""
Exemples annotés à la main pour évaluer objectivement les extracteurs NER.

Seulement 3 exemples de départ ici, volontairement — le but n'est pas que
je fabrique un gold standard à ta place (l'intérêt pédagogique est
justement dans le jugement humain sur les frontières d'entités), mais de
poser le format et l'outillage. Ajoute 10-15 phrases de TON vrai corpus
(data/processed/*.json) avec tes propres annotations avant de tirer une
vraie conclusion — 3 exemples suffisent pour vérifier que le code
fonctionne, pas pour comparer sérieusement deux modèles.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ner.entity import Entity


@dataclass
class GoldExample:
    text: str
    gold_entities: list[Entity]

    @classmethod
    def from_spans(cls, text: str, spans: list[tuple[str, str]]) -> "GoldExample":
        """Calcule les offsets via str.index() plutôt qu'à la main (évite
        les erreurs de calcul manuel). Limite connue : si un même texte
        d'entité apparaît deux fois dans la phrase, seule la première
        occurrence est prise — non géré ici, à garder en tête si tu
        ajoutes des phrases avec des répétitions."""
        entities = []
        for span_text, label in spans:
            start = text.index(span_text)
            entities.append(
                Entity(text=span_text, label=label, start_char=start, end_char=start + len(span_text))
            )
        return cls(text=text, gold_entities=entities)


# Le premier exemple reprend la phrase EXACTE qu'on vient de tester en
# conditions réelles avec GliNER — les annotations ci-dessous sont MON
# jugement (ex: "Devlin" seul, pas "Devlin et al.") : à toi de me dire si
# tu es d'accord ou si tu annoterais différemment.
GOLD_EXAMPLES = [
    GoldExample.from_spans(
        "Devlin et al. introduced BERT at Google AI Language in 2018. "
        "The model was evaluated on the CoNLL-2003 benchmark and reached 92.3 F1.",
        [
            ("Devlin", "person"),
            ("BERT", "model"),
            ("Google AI Language", "organization"),
            ("CoNLL-2003", "dataset"),
            ("F1", "metric"),
        ],
    ),
    GoldExample.from_spans(
        "The authors used fine-tuning to adapt the pretrained model for the downstream task.",
        [("fine-tuning", "method")],
    ),
    GoldExample.from_spans(
        "GLiNER was proposed by Zaratiana et al. and evaluated on the WNUT2017 dataset "
        "using the F1 score.",
        [
            ("GLiNER", "model"),
            ("Zaratiana", "person"),
            ("WNUT2017", "dataset"),
            ("F1", "metric"),
        ],
    ),
]
