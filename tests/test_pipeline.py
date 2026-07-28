import json
from pathlib import Path

from src.preprocessing.loaders import LoadedDocument
from src.preprocessing.pipeline import process_document, save_processed


def _make_document() -> LoadedDocument:
    return LoadedDocument(
        source_path=Path("data/raw/arxiv_test.txt"),
        text="BERT was introduced in 2018.\n\nIt outperformed prior work by 3.2 points.",
        doc_format="txt",
    )


def test_process_document_cleans_and_splits_sentences() -> None:
    processed = process_document(_make_document())
    assert processed.doc_format == "txt"
    assert processed.sentence_count == 2
    assert processed.sentences[0] == "BERT was introduced in 2018."


def test_save_processed_writes_json_with_same_stem(tmp_path: Path) -> None:
    processed = process_document(_make_document())
    out_path = save_processed(processed, tmp_path)

    assert out_path.name == "arxiv_test.json"
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["sentences"] == processed.sentences
    assert data["doc_format"] == "txt"
