"""
Postgres persistence layer for benchmark runs and results.

Uses psycopg with OAuth token rotation for Lakebase Autoscaling.
When deployed on Databricks Apps, the connection pool generates fresh
OAuth tokens via the Databricks SDK. In local dev (no PGHOST), this
module is not used -- the app falls back to in-memory only.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_SCHEMA = "benchmark"

_CREATE_TABLES = f"""
CREATE SCHEMA IF NOT EXISTS {_SCHEMA};

CREATE TABLE IF NOT EXISTS {_SCHEMA}.benchmark_runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    progress REAL NOT NULL DEFAULT 0,
    progress_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT,
    endpoint_names JSONB NOT NULL DEFAULT '[]',
    config JSONB,
    results_meta JSONB
);

CREATE TABLE IF NOT EXISTS {_SCHEMA}.benchmark_results (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES {_SCHEMA}.benchmark_runs(id) ON DELETE CASCADE,
    endpoint_name TEXT NOT NULL,
    num_workers INT NOT NULL,
    median_latency REAL,
    p95_latency REAL,
    throughput REAL,
    avg_input_tokens REAL,
    avg_output_tokens REAL,
    successful_requests INT NOT NULL,
    failed_requests INT NOT NULL,
    total_requests INT NOT NULL,
    elapsed_time REAL NOT NULL,
    qps REAL NOT NULL DEFAULT 0,
    output_tokens INT NOT NULL DEFAULT 0,
    input_tokens INT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_benchmark_results_run_id
    ON {_SCHEMA}.benchmark_results(run_id);
"""


def is_postgres_configured() -> bool:
    return bool(os.environ.get("PGHOST"))


def create_pool(workspace_client: Any) -> ConnectionPool[Any]:
    """Build a psycopg ConnectionPool with OAuth token rotation."""

    import uuid as _uuid

    # Lakebase Autoscaling injects LAKEBASE_ENDPOINT (endpoint resource path);
    # the legacy Provisioned tier injects INSTANCE_NAME instead.
    endpoint = os.environ.get("LAKEBASE_ENDPOINT", "")
    instance_name = os.environ.get("INSTANCE_NAME", "")
    username = os.environ.get("PGUSER", "")
    host = os.environ.get("PGHOST", "")
    port = os.environ.get("PGPORT", "5432")
    database = os.environ.get("PGDATABASE", "")
    sslmode = os.environ.get("PGSSLMODE", "require")

    class OAuthConnection(psycopg.Connection):
        @classmethod
        def connect(cls, conninfo="", **kwargs):
            if endpoint:
                credential = workspace_client.postgres.generate_database_credential(
                    endpoint
                )
            else:
                credential = workspace_client.database.generate_database_credential(
                    request_id=str(_uuid.uuid4()),
                    instance_names=[instance_name],
                )
            kwargs["password"] = credential.token
            return super().connect(conninfo, **kwargs)

    conninfo = (
        f"dbname={database} user={username} host={host} "
        f"port={port} sslmode={sslmode}"
    )

    pool = ConnectionPool(
        conninfo=conninfo,
        connection_class=OAuthConnection,
        min_size=1,
        max_size=5,
        # Lakebase Autoscaling suspends the compute when idle, which kills
        # pooled connections (psycopg AdminShutdown on next use). Validate on
        # checkout, and retire idle connections before the suspend timeout.
        check=ConnectionPool.check_connection,
        max_idle=240,
        open=True,
    )
    return pool


def init_tables(pool: ConnectionPool) -> None:
    """Create tables if they don't exist."""
    with pool.connection() as conn:
        conn.execute(_CREATE_TABLES)
        conn.commit()
    logger.info("Benchmark database tables initialized")


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


def save_run(
    pool: ConnectionPool,
    *,
    run_id: str,
    status: str,
    progress: float,
    progress_message: str,
    created_at: str,
    completed_at: str | None,
    error: str | None,
    endpoint_names: list[str],
    config: dict[str, Any] | None,
    results_meta: dict[str, Any] | None,
) -> None:
    """Insert or update a benchmark run."""
    with pool.connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {_SCHEMA}.benchmark_runs
                (id, status, progress, progress_message, created_at,
                 completed_at, error, endpoint_names, config, results_meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                progress = EXCLUDED.progress,
                progress_message = EXCLUDED.progress_message,
                completed_at = EXCLUDED.completed_at,
                error = EXCLUDED.error,
                results_meta = EXCLUDED.results_meta
            """,
            (
                run_id,
                status,
                progress,
                progress_message,
                created_at,
                completed_at,
                error,
                json.dumps(endpoint_names),
                json.dumps(config) if config else None,
                json.dumps(results_meta) if results_meta else None,
            ),
        )
        conn.commit()


def save_results(
    pool: ConnectionPool,
    run_id: str,
    results: list[dict[str, Any]],
) -> None:
    """Batch-insert benchmark result rows for a run."""
    if not results:
        return
    with pool.connection() as conn:
        conn.execute(
            f"DELETE FROM {_SCHEMA}.benchmark_results WHERE run_id = %s", (run_id,)
        )
        with conn.cursor() as cur:
            for r in results:
                cur.execute(
                    f"""
                    INSERT INTO {_SCHEMA}.benchmark_results
                        (run_id, endpoint_name, num_workers, median_latency,
                         p95_latency, throughput, avg_input_tokens,
                         avg_output_tokens, successful_requests,
                         failed_requests, total_requests, elapsed_time,
                         qps, output_tokens, input_tokens)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        run_id,
                        r.get("endpoint_name", ""),
                        r.get("num_workers", 0),
                        r.get("median_latency"),
                        r.get("p95_latency"),
                        r.get("throughput"),
                        r.get("avg_input_tokens"),
                        r.get("avg_output_tokens"),
                        r.get("successful_requests", 0),
                        r.get("failed_requests", 0),
                        r.get("total_requests", 0),
                        r.get("elapsed_time", 0),
                        r.get("qps", 0),
                        r.get("output_tokens", 0),
                        r.get("input_tokens", 0),
                    ),
                )
        conn.commit()


def list_runs_db(pool: ConnectionPool) -> list[dict[str, Any]]:
    """Return all runs from the database, newest first."""
    with pool.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, status, progress, progress_message, created_at,
                   completed_at, error, endpoint_names, results_meta,
                   (SELECT COUNT(*) FROM {_SCHEMA}.benchmark_results br
                    WHERE br.run_id = r.id) AS result_count
            FROM {_SCHEMA}.benchmark_runs r
            ORDER BY created_at DESC
            """
        ).fetchall()

    return [
        {
            "id": row[0],
            "status": row[1],
            "progress": row[2],
            "progress_message": row[3],
            "created_at": row[4],
            "completed_at": row[5],
            "error": row[6],
            "endpoint_names": json.loads(row[7]) if isinstance(row[7], str) else (row[7] or []),
            "result_count": row[9],
            "results_meta": json.loads(row[8]) if isinstance(row[8], str) else row[8],
        }
        for row in rows
    ]


def get_run_db(
    pool: ConnectionPool, run_id: str
) -> dict[str, Any] | None:
    """Fetch a single run with its results from the database."""
    with pool.connection() as conn:
        row = conn.execute(
            f"""
            SELECT id, status, progress, progress_message, created_at,
                   completed_at, error, endpoint_names, config, results_meta
            FROM {_SCHEMA}.benchmark_runs WHERE id = %s
            """,
            (run_id,),
        ).fetchone()

        if not row:
            return None

        result_rows = conn.execute(
            f"""
            SELECT endpoint_name, num_workers, median_latency, p95_latency,
                   throughput, avg_input_tokens, avg_output_tokens,
                   successful_requests, failed_requests, total_requests,
                   elapsed_time, qps, output_tokens, input_tokens
            FROM {_SCHEMA}.benchmark_results WHERE run_id = %s
            """,
            (run_id,),
        ).fetchall()

    results = [
        {
            "endpoint_name": r[0],
            "num_workers": r[1],
            "median_latency": r[2],
            "p95_latency": r[3],
            "throughput": r[4],
            "avg_input_tokens": r[5],
            "avg_output_tokens": r[6],
            "successful_requests": r[7],
            "failed_requests": r[8],
            "total_requests": r[9],
            "elapsed_time": r[10],
            "qps": r[11],
            "output_tokens": r[12],
            "input_tokens": r[13],
        }
        for r in result_rows
    ]

    return {
        "id": row[0],
        "status": row[1],
        "progress": row[2],
        "progress_message": row[3],
        "created_at": row[4],
        "completed_at": row[5],
        "error": row[6],
        "endpoint_names": json.loads(row[7]) if isinstance(row[7], str) else (row[7] or []),
        "config": json.loads(row[8]) if isinstance(row[8], str) else row[8],
        "results_meta": json.loads(row[9]) if isinstance(row[9], str) else row[9],
        "results": results,
    }
