"""Domain exceptions. Later milestones map these to HTTP status codes."""


class FinCombError(Exception):
    """Base class for all application errors."""


class DataSourceError(FinCombError):
    """A market data source failed or returned unusable data."""


class SymbolNotFoundError(DataSourceError):
    """The source has no data for the requested symbol and date range."""


class DataSourceAuthError(DataSourceError):
    """Missing or invalid API credentials for a data source (retrying won't help)."""


class ValidationError(FinCombError):
    """Input data failed validation."""
