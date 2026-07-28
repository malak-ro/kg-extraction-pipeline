from src.preprocessing.cleaning import (
    clean_text,
    normalize_unicode,
    normalize_whitespace,
    strip_common_latex_artifacts,
)


def test_normalize_whitespace_collapses_tabs_and_newlines() -> None:
    messy = "Hello \n\n  world\t\tthis   is   a   test  "
    assert normalize_whitespace(messy) == "Hello world this is a test"


def test_normalize_unicode_fixes_ligatures() -> None:
    # "ﬁ" (U+FB01, ligature courante dans le texte extrait de PDF) -> "fi"
    assert normalize_unicode("ef\ufb01cient") == "efficient"


def test_strip_common_latex_artifacts_replaces_pm() -> None:
    # Cas réel observé dans un abstract arXiv authentique : "0.30 \pm 0.04"
    text = r"expectations of 0.30 \pm 0.04 and 0.23 \pm 0.04"
    result = strip_common_latex_artifacts(text)
    assert result == "expectations of 0.30 ± 0.04 and 0.23 ± 0.04"


def test_clean_text_combines_all_steps_on_realistic_messy_input() -> None:
    # Extrait authentique (doc officielle arXiv), avec retours à la ligne
    # au milieu de mots et un artefact LaTeX.
    messy = (
        "three di-electron events and\n"
        "three tri-electron events are observed, compared to Standard Model "
        "expectations\nof 0.30 \\pm 0.04 and 0.23 \\pm 0.04, respectively.\n"
    )
    result = clean_text(messy)
    assert "\n" not in result
    assert "±" in result
    assert "  " not in result
