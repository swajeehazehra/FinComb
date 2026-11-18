from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fincomb.api.schemas.instruments import SYMBOL_PATTERN
from fincomb.db.models import AssetType, RunStatus, Severity
from fincomb.ingestion.pipeline import IngestionSummary
from fincomb.services.ingestion_service import normalise_symbols

# One request downloads synchronously; keep it small enough to finish in seconds.
MAX_INGEST_SYMBOLS = 20
DEFAULT_START = date(2015, 1, 1)  # same default as `fincomb ingest`


class IngestionRequest(BaseModel):
    symbols: list[str] = Field(
        min_length=1, max_length=MAX_INGEST_SYMBOLS, examples=[["AAPL", "SPY"]]
    )
    start: date | None = Field(
        default=None,
        description="update: default 2015-01-01. refresh: default each symbol's first stored date.",
    )
    end: date | None = Field(
        default=None, description="Default: today, capped to the last completed session"
    )
    source: Literal["yfinance", "tiingo", "csv"] | None = Field(
        default=None,
        description="Default: the source that already owns each symbol, or "
        "DEFAULT_DATA_SOURCE for new symbols. In refresh mode, switches ownership.",
    )
    mode: Literal["update", "refresh"] = Field(
        default="update",
        description="update = add new bars (like `fincomb ingest`); refresh = replace "
        "the whole stored history from one source (like `fincomb refresh`).",
    )
    asset_type: AssetType | None = None

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, v: list[str]) -> list[str]:
        out = normalise_symbols(v)
        bad = [s for s in out if not re.match(SYMBOL_PATTERN, s)]
        if bad:
            raise ValueError(f"invalid ticker(s): {', '.join(bad)}")
        if not out:
            raise ValueError("give at least one ticker")
        return out

    @model_validator(mode="after")
    def _dates(self) -> IngestionRequest:
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self


class SymbolResultOut(BaseModel):
    symbol: str
    inserted: int
    updated: int
    rejected: int
    warnings: int
    replaced: bool
    notes: list[str]
    error: str | None


class RunSummaryOut(BaseModel):
    run_id: int
    source: str
    status: RunStatus
    end_capped_to: date | None = Field(
        description="Set when the requested end was after the last completed session"
    )
    inserted: int
    updated: int
    rejected: int
    results: list[SymbolResultOut]

    @classmethod
    def from_summary(cls, s: IngestionSummary) -> RunSummaryOut:
        return cls(
            run_id=s.run_id,
            source=s.source,
            status=s.status,
            end_capped_to=s.end_capped_to,
            inserted=s.inserted,
            updated=s.updated,
            rejected=s.rejected,
            results=[SymbolResultOut(**vars(r)) for r in s.results],
        )


class IngestionResponse(BaseModel):
    """One run per data source used (symbols keep their owning source)."""

    runs: list[RunSummaryOut]


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    symbols: list[str]
    start_date: date
    end_date: date
    started_at: datetime
    finished_at: datetime | None
    status: RunStatus
    rows_inserted: int
    rows_updated: int
    rows_rejected: int
    error_message: str | None


class IssueOut(BaseModel):
    symbol: str | None
    bar_date: date | None
    rule: str
    severity: Severity
    detail: str | None


class RunDetail(RunOut):
    issues: list[IssueOut]


class AuditRowOut(BaseModel):
    symbol: str
    owner: str | None
    sources: list[str]
    bars: int
    jumps: int = Field(description="Daily adjusted moves above the extreme-return threshold")
    status: str = Field(
        examples=["OK", "MIXED SOURCES", "NO OWNER", "OWNER MISMATCH", "SUSPICIOUS JUMPS"]
    )


class AuditResponse(BaseModel):
    ok: bool = Field(description="True when every symbol has status OK")
    rows: list[AuditRowOut]
