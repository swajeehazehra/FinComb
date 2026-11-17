"""Domain exceptions. Later milestones map these to HTTP status codes."""


class FinCombError(Exception):
    """Base class for all application errors."""


class DataSourceError(FinCombError):
    """A market data source failed or returned unusable data."""


class SymbolNotFoundError(DataSourceError):
    """The source has no data for the requested symbol and date range."""


class DataSourceAuthError(DataSourceError):
    """Missing or invalid API credentials for a data source (retrying won't help)."""


class DataOwnershipError(FinCombError):
    """A symbol's stored history comes from a different (or several) data source(s)."""


class ValidationError(FinCombError):
    """Input data failed validation."""


class InsufficientDataError(FinCombError):
    """Not enough observations to compute a statistic."""


class MissingDataError(FinCombError):
    """A gap in the middle of a return series where compounding needs every period."""
