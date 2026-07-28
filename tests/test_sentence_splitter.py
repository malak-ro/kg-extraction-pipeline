from src.preprocessing.sentence_splitter import split_sentences


def test_split_sentences_returns_empty_list_for_empty_text() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_splits_on_simple_boundaries() -> None:
    text = "BERT was introduced in 2018. It changed NLP significantly."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "BERT was introduced in 2018."
    assert sentences[1] == "It changed NLP significantly."


def test_split_sentences_does_not_break_on_et_al() -> None:
    """La raison même du choix de en_core_web_sm plutôt que le sentencizer
    basique (voir docstring de sentence_splitter.py) — on vérifie que ça
    tient dans le temps, pas juste au moment où je l'ai testé à la main."""
    text = "Devlin et al. introduced BERT in 2018. It outperformed prior work."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "Devlin et al. introduced BERT in 2018."


def test_split_sentences_does_not_break_on_decimal_numbers() -> None:
    text = "The model reached 92.3 points on the benchmark. That was a new record."
    sentences = split_sentences(text)
    assert len(sentences) == 2
