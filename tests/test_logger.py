import logging

from src.utils.logger import get_logger


def test_get_logger_returns_a_configured_logger() -> None:
    logger = get_logger("test.logger.basic")
    assert isinstance(logger, logging.Logger)
    assert logger.level > 0  # 0 = NOTSET ; on a explicitement défini un niveau
    assert len(logger.handlers) == 2  # console + fichier


def test_get_logger_is_idempotent() -> None:
    """Appeler get_logger deux fois pour le même nom ne doit pas dupliquer
    les handlers (sinon chaque message serait loggé plusieurs fois)."""
    logger1 = get_logger("test.logger.idempotent")
    logger2 = get_logger("test.logger.idempotent")
    assert logger1 is logger2
    assert len(logger1.handlers) == 2
