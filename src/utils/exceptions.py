"""
Hiérarchie d'exceptions personnalisées.

Pourquoi ne pas juste lever `ValueError` partout ?
Un appelant qui fait `except PipelineError` attrape toutes les erreurs que
CE système peut produire, sans avaler par accident une ValueError sans
rapport levée par pandas, spaCy ou requests trois couches plus bas — ce qui
masquerait un vrai bug.
"""


class PipelineError(Exception):
    """Classe de base pour toutes les erreurs levées par ce projet."""


class DocumentLoadError(PipelineError):
    """Levée quand un document (PDF/TXT/DOCX) ne peut pas être lu ou parsé."""


class ExtractionError(PipelineError):
    """Levée quand le NER ou l'extraction de relations échoue sur un texte."""


class GraphWriteError(PipelineError):
    """Levée quand l'écriture dans Neo4j échoue."""


class ConfigurationError(PipelineError):
    """Levée quand une configuration requise est manquante ou invalide."""
