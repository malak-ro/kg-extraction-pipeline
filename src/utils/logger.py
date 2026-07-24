"""
Configuration centralisée du logging.

Pourquoi pas print() ?
print() ne peut pas être filtré par niveau de gravité, redirigé vers un
fichier en prod, ou coupé pendant les tests — logging fait les trois sans
toucher au code appelant.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from config.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré, scopé à `name` (utilise __name__ à
    l'appel, convention standard : ça préfixe les logs avec le module d'origine).

    Idempotent : appeler cette fonction deux fois pour le même `name` ne
    duplique pas les handlers (sinon chaque log apparaîtrait 2x, 3x, ...).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path(settings.data_dir).parent / "reports" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False  # évite les doublons via le root logger
    return logger
