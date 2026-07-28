from pathlib import Path

import pytest

from src.preprocessing.arxiv_client import ArxivClient
from src.utils.exceptions import ArxivApiError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client() -> ArxivClient:
    return ArxivClient()


@pytest.fixture
def sample_xml() -> str:
    return (FIXTURES / "arxiv_sample_response.xml").read_text(encoding="utf-8")


@pytest.fixture
def error_xml() -> str:
    return (FIXTURES / "arxiv_error_response.xml").read_text(encoding="utf-8")


def test_parse_returns_one_paper_per_entry(client, sample_xml) -> None:
    papers = client._parse(sample_xml)
    assert len(papers) == 2


def test_parse_extracts_id_without_url_prefix(client, sample_xml) -> None:
    papers = client._parse(sample_xml)
    # id source: "http://arxiv.org/abs/hep-ex/0307015" -> on garde "hep-ex/0307015"
    assert papers[0].arxiv_id == "hep-ex/0307015"


def test_parse_collapses_multiline_title_and_summary(client, sample_xml) -> None:
    papers = client._parse(sample_xml)
    # Le titre source est éclaté sur 2 lignes dans le XML -> une seule ligne, espaces normalisés
    assert papers[0].title == "Multi-Electron Production at High Transverse Momenta in ep Collisions at HERA"
    assert "\n" not in papers[0].abstract


def test_parse_handles_single_author(client, sample_xml) -> None:
    papers = client._parse(sample_xml)
    assert papers[0].authors == ["H1 Collaboration"]


def test_parse_handles_multiple_authors(client, sample_xml) -> None:
    papers = client._parse(sample_xml)
    assert papers[1].authors == ["Jane Doe", "John Smith"]


def test_parse_extracts_primary_category(client, sample_xml) -> None:
    papers = client._parse(sample_xml)
    assert papers[0].primary_category == "hep-ex"
    assert papers[1].primary_category == "cs.CL"


def test_parse_extracts_pdf_url(client, sample_xml) -> None:
    papers = client._parse(sample_xml)
    assert papers[0].pdf_url == "http://arxiv.org/pdf/hep-ex/0307015v1"


def test_parse_raises_on_api_error_response(client, error_xml) -> None:
    with pytest.raises(ArxivApiError):
        client._parse(error_xml)


def test_to_text_combines_title_and_abstract(client, sample_xml) -> None:
    paper = client._parse(sample_xml)[1]
    text = paper.to_text()
    assert text.startswith("A Toy Example Paper for Testing Purposes")
    assert "fixture abstract" in text
