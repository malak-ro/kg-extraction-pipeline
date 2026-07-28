"""
Client pour l'API arXiv (métadonnées uniquement — pas les PDF complets).

Pourquoi l'API officielle plutôt que scraper les pages HTML de recherche ?
Stable (schéma documenté et versionné), autorisé par les conditions
d'utilisation d'arXiv, et retourne du XML structuré au lieu de HTML à parser
à l'aveugle.

Pourquoi requests + xml.etree.ElementTree plutôt que la librairie `feedparser`
(recommandée dans la doc officielle arXiv) ?
feedparser est très bien, mais ajoute une dépendance et masque le
fonctionnement réel du flux Atom. Ici on garde le contrôle total avec une
seule dépendance (requests) — le prix à payer est de déclarer nous-mêmes les
namespaces XML, ce qu'on fait une seule fois ci-dessous.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urlencode

import requests

from src.utils.exceptions import ArxivApiError
from src.utils.logger import get_logger

logger = get_logger(__name__)

API_URL = "http://export.arxiv.org/api/query"

# Les 3 namespaces XML définis par l'API arXiv (cf. doc officielle) : Atom pour
# les champs standards (title, summary, author...), arxiv pour les extensions
# propres à arXiv (primary_category, comment, doi...).
NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Exigé par les conditions d'utilisation de l'API arXiv : au moins 3 secondes
# entre deux appels successifs.
RATE_LIMIT_SECONDS = 3


@dataclass
class ArxivPaper:
    """Un article arXiv, réduit aux champs qui nous intéressent pour le KG."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str  # date ISO 8601 telle que retournée par l'API
    primary_category: str
    pdf_url: str

    def to_text(self) -> str:
        """Contenu textuel utilisé pour le fichier .txt (titre + abstract) —
        c'est ce que TxtLoader relira ensuite, et ce sur quoi NER/RE
        travailleront aux jalons suivants.

        On force un point final après le titre s'il n'en a pas déjà un :
        repéré en testant bout en bout que sans ça, la segmentation en
        phrases fusionne le titre avec la première phrase de l'abstract
        (aucune ponctuation entre les deux sinon)."""
        title = self.title if self.title.endswith((".", "!", "?")) else f"{self.title}."
        return f"{title}\n\n{self.abstract}"


class ArxivClient:
    """Interroge l'API arXiv et retourne des ArxivPaper.
    Ne touche jamais au système de fichiers — juste HTTP + parsing XML
    (Single Responsibility). La persistance sur disque est le rôle de
    corpus_builder.py."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "kg-extraction-pipeline/0.1 (internship project)"}
        )

    def search(self, query: str, max_results: int = 50) -> list[ArxivPaper]:
        """Exécute une recherche et retourne la liste des papiers trouvés.
        Fait un seul appel HTTP (max_results <= 2000, largement suffisant ici)."""
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{API_URL}?{urlencode(params)}"
        logger.info("Requête arXiv : %s", url)

        response = self._session.get(url, timeout=30)
        response.raise_for_status()
        time.sleep(RATE_LIMIT_SECONDS)  # politesse envers l'API (cf. conditions d'utilisation)

        return self._parse(response.text)

    def _parse(self, atom_xml: str) -> list[ArxivPaper]:
        """Parse un flux Atom complet en liste d'ArxivPaper.
        Séparée de `search()` pour pouvoir être testée hors-ligne avec un XML
        déjà capturé, sans appel réseau (voir tests/test_arxiv_client.py)."""
        root = ET.fromstring(atom_xml)
        papers = []
        for entry in root.findall("atom:entry", NAMESPACES):
            papers.append(self._parse_entry(entry))
        return papers

    def _parse_entry(self, entry: ET.Element) -> ArxivPaper:
        title_text = entry.findtext("atom:title", namespaces=NAMESPACES, default="").strip()

        # L'API retourne les erreurs comme un <entry> unique avec title="Error"
        if title_text == "Error":
            message = entry.findtext("atom:summary", namespaces=NAMESPACES, default="?")
            raise ArxivApiError(f"L'API arXiv a renvoyé une erreur : {message.strip()}")

        raw_id = entry.findtext("atom:id", namespaces=NAMESPACES, default="").strip()
        # rsplit("/", 1) casserait les anciens ids qui contiennent eux-mêmes un
        # "/" (ex: ".../abs/hep-ex/0307015" -> il faut garder "hep-ex/0307015",
        # pas juste "0307015") — on retire donc le préfixe fixe plutôt que de
        # découper sur le dernier "/".
        arxiv_id = raw_id.removeprefix("https://arxiv.org/abs/").removeprefix(
            "http://arxiv.org/abs/"
        )

        abstract = " ".join(
            entry.findtext("atom:summary", namespaces=NAMESPACES, default="").split()
        )
        published = entry.findtext("atom:published", namespaces=NAMESPACES, default="").strip()

        authors = [
            name_el.text.strip()
            for name_el in entry.findall("atom:author/atom:name", NAMESPACES)
            if name_el.text
        ]

        primary_category_el = entry.find("arxiv:primary_category", NAMESPACES)
        primary_category = (
            primary_category_el.get("term", "unknown")
            if primary_category_el is not None
            else "unknown"
        )

        pdf_url = ""
        for link in entry.findall("atom:link", NAMESPACES):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=" ".join(title_text.split()),
            abstract=abstract,
            authors=authors,
            published=published,
            primary_category=primary_category,
            pdf_url=pdf_url,
        )
