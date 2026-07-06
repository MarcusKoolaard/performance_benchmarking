"""
Benchmark engine for LLM serving endpoints.

Refactored from the CLI scripts (benchmark_llm.py, compare_multi_endpoints.py)
to run inside the app runtime with progress callbacks, cancellation, and
in-memory run management.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BenchmarkConfig:
    endpoint_names: list[str]
    workspace_host: str
    api_token: str
    input_tokens: list[int] = field(default_factory=lambda: [1000])
    output_tokens: list[int] = field(default_factory=lambda: [200, 500, 1000])
    qps_list: list[float] = field(default_factory=lambda: [0.5, 1.0])
    parallel_workers: list[int] = field(default_factory=lambda: [4, 6])
    requests_per_worker: int = 5
    timeout: int = 300
    max_retries: int = 3
    cache_hit_rate: float = 0.0


@dataclass
class BenchmarkRun:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: BenchmarkConfig | None = None
    status: RunStatus = RunStatus.PENDING
    progress: float = 0.0
    progress_message: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    results_meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    error: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    progress_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=500)
    )
    _task: asyncio.Task[None] | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Run Manager (in-memory store for active runs in this process)
# ---------------------------------------------------------------------------

_active_runs: dict[str, BenchmarkRun] = {}


def get_active_run(run_id: str) -> BenchmarkRun | None:
    return _active_runs.get(run_id)


def list_active_runs() -> list[BenchmarkRun]:
    return sorted(_active_runs.values(), key=lambda r: r.created_at, reverse=True)


def create_run(config: BenchmarkConfig) -> BenchmarkRun:
    run = BenchmarkRun(config=config)
    _active_runs[run.id] = run
    return run


def remove_active_run(run_id: str) -> None:
    """Remove a finished run from the in-memory dict (after persisting to DB)."""
    _active_runs.pop(run_id, None)


# ---------------------------------------------------------------------------
# Single-endpoint benchmark worker
# ---------------------------------------------------------------------------


def _build_request_payload(
    in_tokens: int, out_tokens: int, cacheable: bool = False
) -> dict:
    overhead_tokens = 50
    user_content_tokens = max(1, in_tokens - overhead_tokens)
    if cacheable:
        # Identical payload — after the first request, providers/engines with
        # prefix caching serve this without a full prefill.
        prefix = ""
    else:
        # A unique nonce at the START of the user content breaks exact-prefix
        # matching (a trailing nonce would not), forcing a full prefill.
        prefix = uuid.uuid4().hex + " "
        user_content_tokens = max(1, user_content_tokens - 20)
    repeat_count = max(1, user_content_tokens // 2)
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You must generate exactly {out_tokens} tokens in your response. "
                    "Write a detailed explanation that uses precisely this number of tokens."
                ),
            },
            {"role": "user", "content": prefix + "Hello! " * repeat_count},
        ],
        "max_tokens": out_tokens,
    }


async def _worker(
    *,
    endpoint_name: str,
    display_name: str,
    endpoint_url: str,
    session: aiohttp.ClientSession,
    worker_index: int,
    num_requests: int,
    in_tokens: int,
    out_tokens: int,
    qps: float,
    max_retries: int,
    cache_hit_rate: float,
    cancel_event: asyncio.Event,
    latencies: list[tuple[int, int, float]],
    failed_counter: list[int],
    emit: Any,
    on_request_done: Any | None = None,
) -> None:
    """Single async worker that sends requests to an endpoint."""
    await asyncio.sleep(0.1 * worker_index)

    for i in range(num_requests):
        if cancel_event.is_set():
            return

        if i > 0:
            await asyncio.sleep(1.0 / qps)

        cacheable = random.random() * 100.0 < cache_hit_rate
        json_data = json.dumps(
            _build_request_payload(in_tokens, out_tokens, cacheable=cacheable)
        )

        request_start = time.time()
        retry_count = 0
        success = False

        while not success and retry_count < max_retries:
            if cancel_event.is_set():
                return
            try:
                async with session.post(endpoint_url, data=json_data) as response:
                    if response.ok:
                        result = await response.json(content_type=None)
                        usage = result.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        if prompt_tokens is None or completion_tokens is None:
                            raise ValueError("Missing usage token counts in response")
                        latency = time.time() - request_start
                        latencies.append((prompt_tokens, completion_tokens, latency))
                        success = True
                        await emit(
                            f"[{display_name}] Worker {worker_index}: "
                            f"Request {i + 1}/{num_requests} completed in {latency:.2f}s"
                        )
                    else:
                        retry_count += 1
                        await emit(
                            f"[{display_name}] Worker {worker_index}: "
                            f"Request {i + 1} failed (HTTP {response.status}), "
                            f"retry {retry_count}/{max_retries}"
                        )
                        if retry_count < max_retries:
                            await asyncio.sleep(retry_count)
            except asyncio.TimeoutError:
                retry_count += 1
                elapsed = time.time() - request_start
                await emit(
                    f"[{display_name}] Worker {worker_index}: "
                    f"Request {i + 1} timed out after {elapsed:.1f}s, "
                    f"retry {retry_count}/{max_retries}"
                )
                if retry_count < max_retries:
                    await asyncio.sleep(retry_count)
            except Exception as exc:
                retry_count += 1
                await emit(
                    f"[{display_name}] Worker {worker_index}: "
                    f"Request {i + 1} error ({type(exc).__name__}), "
                    f"retry {retry_count}/{max_retries}"
                )
                if retry_count < max_retries:
                    await asyncio.sleep(retry_count)

        if not success:
            failed_counter[0] += 1
            await emit(
                f"[{display_name}] Worker {worker_index}: "
                f"Request {i + 1} FAILED after {max_retries} retries"
            )

        if on_request_done is not None:
            await on_request_done()


async def _run_single_endpoint(
    *,
    endpoint_name: str,
    display_name: str,
    workspace_host: str,
    api_token: str,
    num_workers: int,
    requests_per_worker: int,
    in_tokens: int,
    out_tokens: int,
    qps: float,
    timeout: int,
    max_retries: int,
    cache_hit_rate: float,
    cancel_event: asyncio.Event,
    emit: Any,
    on_request_done: Any | None = None,
) -> dict[str, Any] | None:
    """Run benchmark for a single endpoint with given parameters."""
    host = workspace_host.rstrip("/")
    if host.endswith("/api/2.0"):
        host = host[: -len("/api/2.0")]
    endpoint_url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    latencies: list[tuple[int, int, float]] = []
    failed_counter = [0]

    client_timeout = aiohttp.ClientTimeout(total=timeout)
    start_time = time.time()

    async with aiohttp.ClientSession(timeout=client_timeout, headers=headers) as session:
        tasks = [
            asyncio.create_task(
                _worker(
                    endpoint_name=endpoint_name,
                    display_name=display_name,
                    endpoint_url=endpoint_url,
                    session=session,
                    worker_index=idx,
                    num_requests=requests_per_worker,
                    in_tokens=in_tokens,
                    out_tokens=out_tokens,
                    qps=qps,
                    max_retries=max_retries,
                    cache_hit_rate=cache_hit_rate,
                    cancel_event=cancel_event,
                    latencies=latencies,
                    failed_counter=failed_counter,
                    emit=emit,
                    on_request_done=on_request_done,
                )
            )
            for idx in range(num_workers)
        ]
        await asyncio.gather(*tasks)

    elapsed_time = time.time() - start_time

    total_reqs = num_workers * requests_per_worker

    if not latencies:
        return {
            "endpoint_name": display_name,
            "num_workers": num_workers,
            "median_latency": None,
            "p95_latency": None,
            "throughput": None,
            "avg_input_tokens": None,
            "avg_output_tokens": None,
            "successful_requests": 0,
            "failed_requests": failed_counter[0],
            "total_requests": total_reqs,
            "elapsed_time": elapsed_time,
        }

    avg_input_tokens = statistics.mean([inp for inp, _, _ in latencies])
    avg_output_tokens = statistics.mean([outp for _, outp, _ in latencies])
    latency_vals = [lat for _, _, lat in latencies]
    median_latency = statistics.median(latency_vals)

    sorted_lats = sorted(latency_vals)
    p95_index = max(0, math.ceil(len(sorted_lats) * 0.95) - 1)
    p95_latency = sorted_lats[p95_index] if sorted_lats else median_latency

    total_tokens = sum(inp + outp for inp, outp, _ in latencies)
    throughput = total_tokens / elapsed_time if elapsed_time > 0 else 0

    return {
        "endpoint_name": display_name,
        "num_workers": num_workers,
        "median_latency": median_latency,
        "p95_latency": p95_latency,
        "throughput": throughput,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "successful_requests": len(latencies),
        "failed_requests": failed_counter[0],
        "total_requests": total_reqs,
        "elapsed_time": elapsed_time,
    }


# ---------------------------------------------------------------------------
# Full benchmark suite runner
# ---------------------------------------------------------------------------


async def run_benchmark_suite(run: BenchmarkRun, pool: Any | None = None) -> None:
    """Execute the full benchmark suite for a run. Meant to be run as a background task."""
    config = run.config
    assert config is not None

    run.status = RunStatus.RUNNING
    run.progress = 0.0

    async def emit(msg: str) -> None:
        run.progress_message = msg
        try:
            run.progress_queue.put_nowait(
                {"type": "log", "message": msg, "progress": run.progress}
            )
        except asyncio.QueueFull:
            pass

    try:
        all_results: list[dict[str, Any]] = []

        num_endpoints = len(config.endpoint_names)
        total_combos = (
            len(config.qps_list)
            * len(config.parallel_workers)
            * len(config.output_tokens)
            * len(config.input_tokens)
        )

        total_requests = sum(
            nw * config.requests_per_worker * num_endpoints
            for _ in config.qps_list
            for nw in config.parallel_workers
            for _ in config.output_tokens
            for _ in config.input_tokens
        )
        completed_requests = [0]

        async def on_request_done() -> None:
            completed_requests[0] += 1
            run.progress = min(completed_requests[0] / max(total_requests, 1) * 100, 99.9)

        combo_index = 0

        for qps in config.qps_list:
            for num_workers in config.parallel_workers:
                for out_tokens in config.output_tokens:
                    for in_tokens in config.input_tokens:
                        if run.cancel_event.is_set():
                            run.status = RunStatus.CANCELLED
                            run.completed_at = datetime.now().isoformat()
                            await emit("Benchmark cancelled by user")
                            _send_done(run)
                            asyncio.create_task(_persist_and_cleanup(run, pool))
                            return

                        combo_index += 1

                        await emit(
                            f"Config {combo_index}/{total_combos}: "
                            f"QPS={qps} | Workers={num_workers} | "
                            f"InTokens={in_tokens} | OutTokens={out_tokens}"
                        )

                        endpoint_tasks = []
                        for ep_name in config.endpoint_names:
                            endpoint_tasks.append(
                                _run_single_endpoint(
                                    endpoint_name=ep_name,
                                    display_name=ep_name,
                                    workspace_host=config.workspace_host,
                                    api_token=config.api_token,
                                    num_workers=num_workers,
                                    requests_per_worker=config.requests_per_worker,
                                    in_tokens=in_tokens,
                                    out_tokens=out_tokens,
                                    qps=qps,
                                    timeout=config.timeout,
                                    max_retries=config.max_retries,
                                    cache_hit_rate=config.cache_hit_rate,
                                    cancel_event=run.cancel_event,
                                    emit=emit,
                                    on_request_done=on_request_done,
                                )
                            )

                        results = await asyncio.gather(*endpoint_tasks)

                        for result in results:
                            if result is not None:
                                result["qps"] = qps
                                result["output_tokens"] = out_tokens
                                result["input_tokens"] = in_tokens
                                all_results.append(result)

                        await asyncio.sleep(1)

        run.progress = 100.0
        run.results = all_results
        run.results_meta = {
            "timestamp": datetime.now().isoformat(),
            "endpoints": config.endpoint_names,
            "configuration": {
                "qps_list": config.qps_list,
                "parallel_workers": config.parallel_workers,
                "output_tokens": config.output_tokens,
                "input_tokens": config.input_tokens,
                "requests_per_worker": config.requests_per_worker,
                "cache_hit_rate": config.cache_hit_rate,
            },
        }
        run.status = RunStatus.COMPLETED
        run.completed_at = datetime.now().isoformat()
        await emit(
            f"Benchmark complete! {len(all_results)} results across "
            f"{total_combos} configurations."
        )
        _send_done(run)
        asyncio.create_task(_persist_and_cleanup(run, pool))

    except Exception as exc:
        run.status = RunStatus.FAILED
        run.error = str(exc)
        run.completed_at = datetime.now().isoformat()
        try:
            run.progress_queue.put_nowait(
                {"type": "error", "message": str(exc), "progress": run.progress}
            )
        except asyncio.QueueFull:
            pass
        _send_done(run)
        asyncio.create_task(_persist_and_cleanup(run, pool))


async def _persist_and_cleanup(run: BenchmarkRun, pool: Any | None) -> None:
    """Persist to DB in a background thread, then evict from memory after a delay."""
    import logging

    _log = logging.getLogger(__name__)
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, _persist_run, run, pool
        )
    except Exception:
        _log.exception("Background persist failed for run %s", run.id)
    # Keep the run in memory long enough for the DB to be readable on
    # subsequent queries, then evict so it doesn't accumulate forever.
    await asyncio.sleep(120)
    remove_active_run(run.id)


def _persist_run(run: BenchmarkRun, pool: Any | None) -> None:
    """Persist a finished run to Postgres (if pool is available)."""
    if pool is not None:
        try:
            from .db import save_run, save_results

            config_dict = None
            if run.config:
                config_dict = {
                    "endpoint_names": run.config.endpoint_names,
                    "input_tokens": run.config.input_tokens,
                    "output_tokens": run.config.output_tokens,
                    "qps_list": run.config.qps_list,
                    "parallel_workers": run.config.parallel_workers,
                    "requests_per_worker": run.config.requests_per_worker,
                    "timeout": run.config.timeout,
                    "max_retries": run.config.max_retries,
                }
            save_run(
                pool,
                run_id=run.id,
                status=run.status.value,
                progress=run.progress,
                progress_message=run.progress_message,
                created_at=run.created_at,
                completed_at=run.completed_at,
                error=run.error,
                endpoint_names=run.config.endpoint_names if run.config else [],
                config=config_dict,
                results_meta=run.results_meta if run.results_meta else None,
            )
            save_results(pool, run.id, run.results)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Failed to persist run %s to database", run.id
            )


def _send_done(run: BenchmarkRun) -> None:
    """Put a sentinel on the queue so SSE consumers know the run finished."""
    try:
        run.progress_queue.put_nowait(
            {
                "type": "done",
                "status": run.status.value,
                "progress": run.progress,
                "message": run.progress_message,
            }
        )
    except asyncio.QueueFull:
        pass


def start_run(run: BenchmarkRun, pool: Any | None = None) -> None:
    """Launch the benchmark suite as a background asyncio task."""
    loop = asyncio.get_running_loop()
    run._task = loop.create_task(run_benchmark_suite(run, pool=pool))


def cancel_run(run: BenchmarkRun) -> None:
    """Signal a run to cancel."""
    run.cancel_event.set()
