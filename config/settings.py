"""
Configuration centralisée de l'application.

Pourquoi pydantic-settings plutôt que des `os.getenv()` éparpillés ?
- Fail fast : une config manquante ou invalide lève une erreur claire au
  démarrage, pas un KeyError cryptique trois couches plus bas dans le pipeline.
- Typé : ton IDE auto-complète `settings.neo4j_uri` ; un dict ne le permet pas.
- Source unique de vérité : variables d'environnement, fichier .env et valeurs
  par défaut sont fusionnées via un seul objet validé, au lieu d'être lues à
  plusieurs endroits différents du code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Path(__file__) pointe vers CE fichier (config/settings.py), donc .parent.parent
# est la racine du projet — ça marche quel que soit le répertoire depuis lequel
# on lance le script (contrairement à un chemin relatif du type "./data").
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Schéma de configuration. Chaque champ = une variable d'environnement
    possible (en MAJUSCULES dans .env), avec une valeur par défaut sûre."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # ignore les variables d'env non déclarées ici
    )

    # --- Environnement ---
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Neo4j (utilisé à partir du Jalon 5) ---
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="changeme")

    # --- LLM (utilisé à partir du Jalon 4) ---
    openai_api_key: str | None = None

    # --- Chemins (Path, pas str, pour pouvoir appeler .mkdir()/.exists() direct) ---
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"


@lru_cache
def get_settings() -> Settings:
    """Singleton mis en cache : parsé une seule fois, réutilisé partout.
    `lru_cache` sur une fonction sans argument = un cache d'une seule entrée."""
    return Settings()


settings = get_settings()
