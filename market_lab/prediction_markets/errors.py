from __future__ import annotations


class PredictionMarketError(Exception):
    error_code = "PM0_ERROR"
    exit_code = 3

    def __init__(self, message: str, *, path: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.details = details or {}


class UsageError(PredictionMarketError):
    error_code = "PM0_USAGE"
    exit_code = 2


class SchemaError(PredictionMarketError):
    error_code = "PM0_SCHEMA"
    exit_code = 3


class ConflictError(PredictionMarketError):
    error_code = "PM0_CONFLICT"
    exit_code = 4


class IntegrityError(PredictionMarketError):
    error_code = "PM0_INTEGRITY"
    exit_code = 4


class PathEscapeError(PredictionMarketError):
    error_code = "PM0_PATH_ESCAPE"
    exit_code = 4


class NotFoundError(PredictionMarketError):
    error_code = "PM0_NOT_FOUND"
    exit_code = 5


class CaptureError(PredictionMarketError):
    error_code = "PM0_CAPTURE"
    exit_code = 5


class PaperError(PredictionMarketError):
    error_code = "PM1_ERROR"
    exit_code = 3


class PaperIntegrityError(PaperError):
    error_code = "PM1_INTEGRITY"
    exit_code = 4


class PaperNotSyntheticError(PaperError):
    error_code = "PM1_NOT_SYNTHETIC"
    exit_code = 3


class PaperNotAdmissibleError(PaperError):
    error_code = "PM1_NOT_ADMISSIBLE"
    exit_code = 3
