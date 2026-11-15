"""Domain exceptions. Later milestones map these to HTTP status codes."""


class FinCombError(Exception):
    """Base class for all application errors."""


class DataSourceError(FinCombError):
    """A market data source failed or returned unusable data."""


class SymbolNotFoundError(DataSourceError):
    """The source has no data for the requested symbol and date range."""


class ValidationError(FinCombError):
    """Input data failed validation."""
