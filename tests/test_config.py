"""
Tests de config/settings.py.

On passe `_env_file=None` partout : ça ignore un éventuel .env local pour que
les tests soient déterministes (même résultat sur ta machine et en CI/CD),
et qu'on ne dépende jamais d'un fichier .env qui pourrait ne pas exister.
"""
import pytest

from config.settings import Settings


def test_default_environment_is_dev() -> None:
    s = Settings(_env_file=None)
    assert s.environment == "dev"


def test_neo4j_uri_has_a_sane_default() -> None:
    s = Settings(_env_file=None)
    assert s.neo4j_uri.startswith("bolt://")


def test_paths_are_resolved_under_project_root() -> None:
    s = Settings(_env_file=None)
    assert s.raw_data_dir.parent == s.data_dir
    assert s.processed_data_dir.parent == s.data_dir


def test_invalid_environment_is_rejected() -> None:
    """pydantic doit refuser une valeur hors du Literal["dev","test","prod"]."""
    with pytest.raises(Exception):
        Settings(_env_file=None, environment="staging")
