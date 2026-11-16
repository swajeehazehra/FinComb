"""Typed application settings, read from environment variables and `.env`.

Nothing secret is hardcoded: the database password must come from the
environment. Missing or malformed values fail fast at startup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    postgres_user: str = "fincomb"
    postgres_password: SecretStr = Field(..., description="Required, no default on purpose")
    postgres_db: str = "fincomb"
    postgres_test_db: str = "fincomb_test"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    log_level: str = "INFO"
    log_json: bool = False

    default_data_source: Literal["yfinance", "tiingo", "csv"] = "yfinance"
    tiingo_api_key: SecretStr | None = None  # only needed for --source tiingo
    csv_data_dir: Path = Path("tests/fixtures")

    # Data-quality thresholds
    extreme_return_threshold: float = 0.40  # |daily return| above this is flagged
    stale_price_run: int = 5  # identical closes in a row before flagging

    def database_url(self, *, test: bool = False) -> URL:
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_test_db if test else self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
