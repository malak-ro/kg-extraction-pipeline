"""
Segmentation en phrases via spaCy.

Pourquoi en_core_web_sm plutôt que spacy.blank("en") + sentencizer (qui
évite de télécharger un modèle) ? Testé empiriquement sur des phrases
académiques types : le sentencizer basique coupe à tort après "et al."
(extrêmement fréquent dans une bibliographie scientifique), alors que
en_core_web_sm — qui utilise le parser de dépendances plutôt que la seule
ponctuation — le gère correctement. Exemple observé :

    blank + sentencizer -> "Devlin et al." | "introduced BERT in 2018."  (faux)
    en_core_web_sm       -> "Devlin et al. introduced BERT in 2018."     (correct)

Ce téléchargement n'est pas perdu : c'est le même modèle qu'on réutilisera
pour le NER au Jalon 3.
"""
from __future__ import annotations

import spacy
from spacy.language import Language

_MODEL_NAME = "en_core_web_sm"
_nlp: Language | None = None


def _get_nlp() -> Language:
    """Charge le modèle une seule fois et le réutilise (même logique de
    cache que get_settings() au Jalon 1). `disable=["ner", "lemmatizer"]`
    accélère le chargement/traitement : on n'a besoin que du tagger et du
    parser pour les frontières de phrase — vérifié que ça ne change pas
    un seul résultat de segmentation."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(_MODEL_NAME, disable=["ner", "lemmatizer"])
        except OSError as exc:
            raise RuntimeError(
                f"Modèle spaCy '{_MODEL_NAME}' introuvable. Installe-le avec : "
                f"python -m spacy download {_MODEL_NAME}"
            ) from exc
    return _nlp


def split_sentences(text: str) -> list[str]:
    """Découpe un texte (déjà nettoyé via cleaning.clean_text) en phrases.
    Retourne une liste vide pour un texte vide plutôt que de planter."""
    if not text.strip():
        return []
    doc = _get_nlp()(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
