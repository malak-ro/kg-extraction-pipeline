from pathlib import Path

import pytest

from src.preprocessing.loaders import (
    DocumentLoaderRegistry,
    DocxLoader,
    PdfLoader,
    TxtLoader,
)
from src.utils.exceptions import DocumentLoadError

FIXTURES = Path(__file__).parent / "fixtures"


def test_txt_loader_reads_content() -> None:
    doc = TxtLoader().load(FIXTURES / "sample.txt")
    assert doc.doc_format == "txt"
    assert "Transformer" in doc.text
    assert doc.char_count > 0


def test_pdf_loader_extracts_text() -> None:
    doc = PdfLoader().load(FIXTURES / "sample.pdf")
    assert doc.doc_format == "pdf"
    assert "GPT-3" in doc.text
    assert "OpenAI" in doc.text


def test_docx_loader_extracts_paragraphs() -> None:
    doc = DocxLoader().load(FIXTURES / "sample.docx")
    assert doc.doc_format == "docx"
    assert "BERT" in doc.text
    assert "Google AI Language" in doc.text


def test_supports_only_matches_its_own_extension() -> None:
    txt_loader = TxtLoader()
    assert txt_loader.supports(Path("a.txt")) is True
    assert txt_loader.supports(Path("a.pdf")) is False


def test_registry_dispatches_to_the_right_loader() -> None:
    """Le test le plus important : on ne choisit PAS le loader nous-même,
    le registry doit le déduire automatiquement de l'extension."""
    registry = DocumentLoaderRegistry()

    txt_doc = registry.load(FIXTURES / "sample.txt")
    pdf_doc = registry.load(FIXTURES / "sample.pdf")
    docx_doc = registry.load(FIXTURES / "sample.docx")

    assert txt_doc.doc_format == "txt"
    assert pdf_doc.doc_format == "pdf"
    assert docx_doc.doc_format == "docx"


def test_registry_raises_on_unsupported_format() -> None:
    registry = DocumentLoaderRegistry()
    with pytest.raises(DocumentLoadError):
        registry.load(FIXTURES / "sample.epub")


def test_missing_file_raises_document_load_error_not_a_raw_oserror() -> None:
    """On veut toujours DocumentLoadError, jamais une FileNotFoundError brute
    qui obligerait le code appelant à connaître les détails d'implémentation."""
    with pytest.raises(DocumentLoadError):
        TxtLoader().load(FIXTURES / "ce_fichier_n_existe_pas.txt")
