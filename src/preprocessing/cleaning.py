"""
Nettoyage de texte, appliqué avant la segmentation en phrases.

Fonctions pures (pas de classe) : contrairement aux loaders (qui avaient
besoin de polymorphisme entre formats) ou à ArxivClient (qui garde une
session HTTP), le nettoyage est une simple transformation texte -> texte.
Pas besoin d'une classe pour ça — une classe ici ajouterait de la
cérémonie sans bénéfice (cf. YAGNI).
"""
from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")

# Quelques artefacts LaTeX qui fuient parfois dans les abstracts arXiv
# (ex: observé dans un vrai résumé de la doc officielle arXiv : "0.30 \pm 0.04").
# Non exhaustif — un vrai parseur LaTeX serait disproportionné ici ; on
# corrige juste les cas les plus fréquents.
_LATEX_ARTIFACTS = {
    r"\pm": "±",
    r"\times": "×",
    r"\sim": "~",
}


def normalize_unicode(text: str) -> str:
    """Normalise les variantes Unicode (ex: guillemets typographiques,
    ligatures) vers une forme canonique (NFKC)."""
    return unicodedata.normalize("NFKC", text)


def strip_common_latex_artifacts(text: str) -> str:
    """Remplace quelques échappements LaTeX courants par leur équivalent
    Unicode lisible. Pas un parseur LaTeX complet — voir docstring du module."""
    for pattern, replacement in _LATEX_ARTIFACTS.items():
        text = text.replace(pattern, replacement)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse tabs/retours à la ligne/espaces multiples en un seul espace.
    Important pour le texte extrait de PDF, où les sauts de ligne tombent
    n'importe où dans une phrase."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Pipeline de nettoyage complet, dans cet ordre précis : normaliser
    l'unicode et le LaTeX avant de recollapser les espaces, sinon un
    remplacement pourrait réintroduire des espaces multiples."""
    text = normalize_unicode(text)
    text = strip_common_latex_artifacts(text)
    text = normalize_whitespace(text)
    return text
