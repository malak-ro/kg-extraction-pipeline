import json
from pathlib import Path

from src.preprocessing.arxiv_client import ArxivPaper
from src.preprocessing.corpus_builder import save_corpus


def _make_paper(arxiv_id: str = "2401.10825v3") -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title="Recent Advances in Named Entity Recognition",
        abstract="A survey of NER approaches.",
        authors=["Imed Keraghel", "Stanislas Morbieu", "Mohamed Nadif"],
        published="2024-01-19T18:59:11Z",
        primary_category="cs.CL",
        pdf_url="http://arxiv.org/pdf/2401.10825v3",
    )


def test_save_corpus_writes_one_txt_file_per_paper(tmp_path: Path) -> None:
    save_corpus([_make_paper()], tmp_path)
    txt_files = list(tmp_path.glob("arxiv_*.txt"))
    assert len(txt_files) == 1
    assert txt_files[0].name == "arxiv_2401.10825v3.txt"


def test_save_corpus_txt_content_matches_to_text(tmp_path: Path) -> None:
    paper = _make_paper()
    save_corpus([paper], tmp_path)
    content = (tmp_path / "arxiv_2401.10825v3.txt").read_text(encoding="utf-8")
    assert content == paper.to_text()


def test_save_corpus_sanitizes_slash_in_old_style_ids(tmp_path: Path) -> None:
    """Les anciens ids arXiv type 'hep-ex/0307015' contiennent un '/', invalide
    dans un nom de fichier — on vérifie qu'il est bien remplacé."""
    save_corpus([_make_paper(arxiv_id="hep-ex/0307015")], tmp_path)
    assert (tmp_path / "arxiv_hep-ex_0307015.txt").exists()


def test_save_corpus_writes_valid_metadata_json(tmp_path: Path) -> None:
    paper = _make_paper()
    metadata_path = save_corpus([paper], tmp_path)

    assert metadata_path.name == "arxiv_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert paper.arxiv_id in metadata
    entry = metadata[paper.arxiv_id]
    assert entry["title"] == paper.title
    assert entry["authors"] == paper.authors
    assert entry["txt_file"] == "arxiv_2401.10825v3.txt"
