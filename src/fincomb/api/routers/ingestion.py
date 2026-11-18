from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query

from fincomb.api.deps import FactoryDep, SessionDep, SettingsDep
from fincomb.api.errors import not_found
from fincomb.api.schemas.common import ERROR_RESPONSES
from fincomb.api.schemas.ingestion import (
    DEFAULT_START,
    AuditResponse,
    AuditRowOut,
    IngestionRequest,
    IngestionResponse,
    IssueOut,
    RunDetail,
    RunOut,
    RunSummaryOut,
)
from fincomb.db.repositories import IngestionRunRepository
from fincomb.ingestion.audit import audit_data
from fincomb.services import ingestion_service

router = APIRouter(tags=["ingestion"], responses=ERROR_RESPONSES)


@router.post("/ingestion/runs", response_model=IngestionResponse, status_code=201)
def create_run(body: IngestionRequest, factory: FactoryDep, settings: SettingsDep):
    """Download, validate and store daily bars (synchronous).

    Same rules as the CLI: each symbol keeps its owning source unless `source` is
    given, the end date is capped to the last completed session, and one bad symbol
    doesn't stop the others. Per-symbol failures (for example a source-ownership
    conflict) are reported in `results[].error` and the run's `status`.
    """
    mode = "replace" if body.mode == "refresh" else "update"
    start = body.start if body.start or mode == "replace" else DEFAULT_START
    summaries = ingestion_service.ingest(
        factory,
        settings,
        body.symbols,
        start,
        body.end or date.today(),
        source=body.source,
        mode=mode,
        asset_type=body.asset_type,
    )
    return IngestionResponse(runs=[RunSummaryOut.from_summary(s) for s in summaries])


@router.get("/ingestion/runs", response_model=list[RunOut])
def list_runs(session: SessionDep, limit: Annotated[int, Query(ge=1, le=200)] = 20):
    """Most recent ingestion runs first."""
    return IngestionRunRepository(session).recent(limit)


@router.get("/ingestion/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: int, session: SessionDep):
    """One run with every data-quality issue it recorded."""
    repo = IngestionRunRepository(session)
    run = repo.get(run_id)
    if run is None:
        raise not_found("ingestion run", run_id)
    issues = [
        IssueOut(
            symbol=symbol,
            bar_date=issue.bar_date,
            rule=issue.rule,
            severity=issue.severity,
            detail=issue.detail,
        )
        for issue, symbol in repo.issues_with_symbols(run_id)
    ]
    return RunDetail(**RunOut.model_validate(run).model_dump(), issues=issues)


@router.get("/data/audit", response_model=AuditResponse)
def audit(session: SessionDep, settings: SettingsDep):
    """Same checks as `fincomb check-data`: one source per symbol, an owner, no
    suspicious jumps."""
    rows = [AuditRowOut(**vars(r)) for r in audit_data(session, settings.extreme_return_threshold)]
    return AuditResponse(ok=all(r.status == "OK" for r in rows), rows=rows)
