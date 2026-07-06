"""
API routes for LLM benchmark operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated

from fastapi import Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .benchmark.db import get_run_db, list_runs_db
from .benchmark.engine import (
    BenchmarkConfig,
    RunStatus,
    cancel_run,
    create_run,
    get_active_run,
    list_active_runs,
    start_run,
)
from .benchmark.report import generate_report_bytes
from .benchmark.stats import generate_csv_bytes, generate_json_bytes, generate_summary
from .core import Dependency, create_router
from .models import (
    BenchmarkConfigIn,
    BenchmarkResultItem,
    BenchmarkResultsOut,
    BenchmarkRunOut,
    EndpointOut,
)

logger = logging.getLogger(__name__)

_io_pool = ThreadPoolExecutor(max_workers=8)

benchmark_router = create_router()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_obo_token(
    token: Annotated[str | None, Header(alias="X-Forwarded-Access-Token")] = None,
) -> str:
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Forwarded-Access-Token header for user authentication",
        )
    return token


def _run_to_out(run) -> BenchmarkRunOut:
    return BenchmarkRunOut(
        id=run.id,
        status=run.status.value,
        progress=run.progress,
        progress_message=run.progress_message,
        created_at=run.created_at,
        completed_at=run.completed_at,
        error=run.error,
        endpoint_names=run.config.endpoint_names if run.config else [],
        result_count=len(run.results),
    )


def _get_results_for_download(run_id: str, pool) -> tuple[list, dict]:
    """Return (results, meta) from active run or database. Raises HTTPException on failure."""
    run = get_active_run(run_id)
    if run:
        if not run.results:
            raise HTTPException(status_code=400, detail="No results available")
        return run.results, run.results_meta

    if pool is not None:
        try:
            db_run = get_run_db(pool, run_id)
            if db_run:
                if not db_run["results"]:
                    raise HTTPException(status_code=400, detail="No results available")
                return db_run["results"], db_run["results_meta"] or {}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Failed to get run %s from database", run_id)

    raise HTTPException(status_code=404, detail="Benchmark run not found")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _has_provisioned_throughput(entity) -> bool:
    return (
        getattr(entity, "min_provisioned_throughput", None) is not None
        or getattr(entity, "max_provisioned_throughput", None) is not None
    )


def _classify_endpoint(detailed_ep) -> str:
    """Classify a serving endpoint using its config.

    Real Databricks FMAPI (pay-per-token) endpoints have a ``foundation_model``
    whose ``name`` lives under the ``system.ai.`` catalog prefix, or an
    ``entity_name`` with the same prefix.  Provisioned-throughput variants set
    ``min_provisioned_throughput`` and/or ``max_provisioned_throughput``.
    """
    config = getattr(detailed_ep, "config", None)
    entities = getattr(config, "served_entities", None) if config else None
    if not entities:
        entities = getattr(config, "served_models", None) if config else None
    if not entities:
        return "CUSTOM"

    entity = entities[0]

    if getattr(entity, "external_model", None) is not None:
        return "EXTERNAL_MODEL"

    is_system_ai = False

    fm = getattr(entity, "foundation_model", None)
    if fm is not None:
        fm_name = getattr(fm, "name", "") or ""
        if fm_name.startswith("system.ai."):
            is_system_ai = True

    if not is_system_ai:
        entity_name = getattr(entity, "entity_name", "") or ""
        if entity_name.startswith("system.ai."):
            is_system_ai = True

    if is_system_ai:
        if _has_provisioned_throughput(entity):
            return "PROVISIONED_THROUGHPUT"
        model_name = (
            getattr(entity, "name", "") or getattr(entity, "entity_name", "") or ""
        )
        model_name = model_name.removeprefix("system.ai.")
        if not model_name.startswith("databricks-"):
            return "PROVISIONED_THROUGHPUT"
        return "PAY_PER_TOKEN"

    if _has_provisioned_throughput(entity):
        return "PROVISIONED_THROUGHPUT"

    return "CUSTOM"


def _endpoint_to_out(ep) -> EndpointOut | None:
    """Build an EndpointOut from a ServingEndpoint object (from list or get)."""
    if not ep.name:
        return None

    model_name = None
    model_type = None
    task = None
    scale_to_zero = False

    if ep.config and ep.config.served_entities:
        entity = ep.config.served_entities[0]
        model_name = getattr(entity, "name", None) or getattr(
            entity, "entity_name", None
        )
        model_type = getattr(entity, "entity_version", None)
        scale_to_zero = bool(getattr(entity, "scale_to_zero_enabled", False))

    if ep.task:
        task = str(ep.task)

    if ep.state and ep.state.ready:
        state_str = ep.state.ready.value if hasattr(ep.state.ready, "value") else str(ep.state.ready)
    elif ep.state and ep.state.config_update:
        state_str = ep.state.config_update.value if hasattr(ep.state.config_update, "value") else str(ep.state.config_update)
    else:
        state_str = "READY"

    return EndpointOut(
        name=ep.name or "",
        state=state_str,
        model_name=model_name,
        model_type=model_type,
        creator=getattr(ep, "creator", None),
        task=task,
        endpoint_type=_classify_endpoint(ep),
        scale_to_zero_enabled=scale_to_zero,
    )


def _enrich_via_get(user_ws, name: str) -> tuple[str, str, bool]:
    """Fetch full endpoint detail via .get() and return (name, endpoint_type, scale_to_zero)."""
    try:
        ep = user_ws.serving_endpoints.get(name)
        endpoint_type = _classify_endpoint(ep)
        scale_to_zero = False
        config = getattr(ep, "config", None)
        entities = getattr(config, "served_entities", None) if config else None
        if entities:
            scale_to_zero = bool(getattr(entities[0], "scale_to_zero_enabled", False))
        return name, endpoint_type, scale_to_zero
    except Exception:
        logger.warning("Failed to fetch detail for endpoint %s", name)
        return name, "CUSTOM", False


@benchmark_router.get(
    "/endpoints",
    response_model=list[EndpointOut],
    operation_id="listEndpoints",
)
def list_endpoints(user_ws: Dependency.UserClient):
    """List available LLM serving endpoints in the workspace."""
    try:
        endpoints: list[EndpointOut] = []
        for ep in user_ws.serving_endpoints.list():
            out = _endpoint_to_out(ep)
            if out is not None:
                endpoints.append(out)

        futures = {
            _io_pool.submit(_enrich_via_get, user_ws, e.name): e.name
            for e in endpoints
        }
        enriched: dict[str, tuple[str, bool]] = {}
        for future in as_completed(futures):
            name, endpoint_type, scale_to_zero = future.result()
            enriched[name] = (endpoint_type, scale_to_zero)

        for ep_out in endpoints:
            if ep_out.name in enriched:
                ep_out.endpoint_type = enriched[ep_out.name][0]
                ep_out.scale_to_zero_enabled = enriched[ep_out.name][1]

        endpoints.sort(key=lambda e: e.name)
        return endpoints
    except Exception as exc:
        logger.exception("Failed to list serving endpoints")
        raise HTTPException(status_code=502, detail=f"Failed to list endpoints: {exc}")


# ---------------------------------------------------------------------------
# Benchmark CRUD
# ---------------------------------------------------------------------------


@benchmark_router.get(
    "/benchmarks",
    response_model=list[BenchmarkRunOut],
    operation_id="listBenchmarks",
)
def list_benchmark_runs(pool: Dependency.Pool):
    """List benchmark runs from in-memory active runs + database history."""
    active = [_run_to_out(r) for r in list_active_runs()]
    active_ids = {r.id for r in active}

    if pool is not None:
        try:
            db_rows = list_runs_db(pool)
            for row in db_rows:
                if row["id"] not in active_ids:
                    active.append(
                        BenchmarkRunOut(
                            id=row["id"],
                            status=row["status"],
                            progress=row["progress"],
                            progress_message=row["progress_message"],
                            created_at=row["created_at"],
                            completed_at=row["completed_at"],
                            error=row["error"],
                            endpoint_names=row["endpoint_names"],
                            result_count=row["result_count"],
                        )
                    )
        except Exception:
            logger.exception("Failed to list runs from database")

    active.sort(key=lambda r: r.created_at, reverse=True)
    return active


@benchmark_router.post(
    "/benchmarks",
    response_model=BenchmarkRunOut,
    operation_id="startBenchmark",
)
async def start_benchmark(
    body: BenchmarkConfigIn,
    ws: Dependency.Client,
    pool: Dependency.Pool,
    token: Annotated[str | None, Header(alias="X-Forwarded-Access-Token")] = None,
):
    """Start a new benchmark run."""
    obo_token = _get_obo_token(token)

    workspace_host = str(ws.config.host or "")
    if not workspace_host:
        raise HTTPException(status_code=500, detail="Cannot determine workspace host")

    config = BenchmarkConfig(
        endpoint_names=body.endpoint_names,
        workspace_host=workspace_host,
        api_token=obo_token,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        qps_list=body.qps_list,
        parallel_workers=body.parallel_workers,
        requests_per_worker=body.requests_per_worker,
        timeout=body.timeout,
        max_retries=body.max_retries,
        cache_hit_rate=body.cache_hit_rate,
    )

    run = create_run(config)
    start_run(run, pool=pool)
    return _run_to_out(run)


@benchmark_router.get(
    "/benchmarks/{run_id}",
    response_model=BenchmarkResultsOut,
    operation_id="getBenchmark",
)
def get_benchmark(run_id: str, pool: Dependency.Pool):
    """Get benchmark run status and results."""
    run = get_active_run(run_id)
    if run:
        summary = None
        if run.results:
            data = {**run.results_meta, "results": run.results}
            summary = generate_summary(data)
        return BenchmarkResultsOut(
            id=run.id,
            status=run.status.value,
            results=[BenchmarkResultItem(**r) for r in run.results],
            summary=summary,
            meta=run.results_meta if run.results_meta else None,
        )

    if pool is not None:
        try:
            db_run = get_run_db(pool, run_id)
            if db_run:
                summary = None
                if db_run["results"]:
                    data = {**(db_run["results_meta"] or {}), "results": db_run["results"]}
                    summary = generate_summary(data)
                return BenchmarkResultsOut(
                    id=db_run["id"],
                    status=db_run["status"],
                    results=[BenchmarkResultItem(**r) for r in db_run["results"]],
                    summary=summary,
                    meta=db_run["results_meta"],
                )
        except Exception:
            logger.exception("Failed to get run %s from database", run_id)

    raise HTTPException(status_code=404, detail="Benchmark run not found")


@benchmark_router.post(
    "/benchmarks/{run_id}/cancel",
    response_model=BenchmarkRunOut,
    operation_id="cancelBenchmark",
)
def cancel_benchmark(run_id: str):
    """Cancel a running benchmark."""
    run = get_active_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found or already finished")
    if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
        raise HTTPException(status_code=400, detail="Benchmark is not running")
    cancel_run(run)
    return _run_to_out(run)


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


@benchmark_router.get(
    "/benchmarks/{run_id}/stream",
    operation_id="streamBenchmark",
)
async def stream_benchmark(run_id: str):
    """SSE stream of benchmark progress."""
    run = get_active_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(run.progress_queue.get(), timeout=30)
                yield {"event": "progress", "data": json.dumps(event)}
                if event.get("type") == "done":
                    return
            except asyncio.TimeoutError:
                # Send keepalive
                yield {
                    "event": "ping",
                    "data": json.dumps(
                        {"type": "ping", "status": run.status.value, "progress": run.progress}
                    ),
                }
                if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
                    return

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


@benchmark_router.get(
    "/benchmarks/{run_id}/pdf",
    operation_id="downloadPdf",
)
def download_pdf(run_id: str, pool: Dependency.Pool):
    """Download PDF report for a completed benchmark."""
    results, meta = _get_results_for_download(run_id, pool)

    pdf_bytes = generate_report_bytes(results, meta)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="benchmark_report_{run_id[:8]}.pdf"'},
    )


@benchmark_router.get(
    "/benchmarks/{run_id}/csv",
    operation_id="downloadCsv",
)
def download_csv(run_id: str, pool: Dependency.Pool):
    """Download CSV summary for a completed benchmark."""
    results, meta = _get_results_for_download(run_id, pool)

    data = {**meta, "results": results}
    csv_bytes = generate_csv_bytes(data)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="benchmark_summary_{run_id[:8]}.csv"'},
    )


@benchmark_router.get(
    "/benchmarks/{run_id}/json",
    operation_id="downloadJson",
)
def download_json(run_id: str, pool: Dependency.Pool):
    """Download raw JSON results for a completed benchmark."""
    results, meta = _get_results_for_download(run_id, pool)

    data = {**meta, "results": results}
    json_bytes = generate_json_bytes(data)
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="benchmark_results_{run_id[:8]}.json"'},
    )
