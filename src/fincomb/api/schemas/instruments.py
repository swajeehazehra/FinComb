from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fincomb.db.models import AssetType

SYMBOL_PATTERN = r"^[A-Za-z0-9.\-^=]{1,20}$"


class InstrumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    name: str | None
    asset_type: AssetType
    exchange: str | None
    currency: str
    is_active: bool
    data_source: str | None = Field(description="The one source that owns this symbol's history")
    created_at: datetime


class Coverage(BaseModel):
    first_date: date | None
    last_date: date | None
    bars: int


class InstrumentDetail(InstrumentOut):
    coverage: Coverage


class InstrumentCreate(BaseModel):
    symbol: str = Field(pattern=SYMBOL_PATTERN, examples=["AAPL"])
    asset_type: AssetType = AssetType.STOCK
    name: str | None = Field(default=None, max_length=200)

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class InstrumentUpdate(BaseModel):
    asset_type: AssetType
