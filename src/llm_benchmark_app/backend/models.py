from __future__ import annotations

from pydantic import BaseModel, Field
from .. import __version__


class VersionOut(BaseModel):
    version: str

    @classmethod
    def from_metadata(cls):
        return cls(version=__version__)


# ---------------------------------------------------------------------------
# Serving Endpoint models
# ---------------------------------------------------------------------------


class EndpointOut(BaseModel):
    name: str
    state: str = ""
    model_name: str | None = None
    model_type: str | None = None
    creator: str | None = None
    task: str | None = None
    endpoint_type: str = "CUSTOM"
    scale_to_zero_enabled: bool = False


# ---------------------------------------------------------------------------
# Benchmark models
# ---------------------------------------------------------------------------


class BenchmarkConfigIn(BaseModel):
    endpoint_names: list[str] = Field(..., min_length=1, max_length=4)
    input_tokens: list[int] = Field(default=[500])
    output_tokens: list[int] = Field(default=[200])
    qps_list: list[float] = Field(default=[1.0])
    parallel_workers: list[int] = Field(default=[2])
    requests_per_worker: int = Field(default=5, ge=1, le=10)
    timeout: int = Field(default=300, ge=30, le=3600)
    max_retries: int = Field(default=3, ge=0, le=10)
    cache_hit_rate: float = Field(default=0.0, ge=0, le=100)


class BenchmarkRunOut(BaseModel):
    id: str
    status: str
    progress: float = 0.0
    progress_message: str = ""
    created_at: str
    completed_at: str | None = None
    error: str | None = None
    endpoint_names: list[str] = []
    result_count: int = 0


class BenchmarkResultItem(BaseModel):
    endpoint_name: str
    num_workers: int
    median_latency: float | None = None
    p95_latency: float | None = None
    throughput: float | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    successful_requests: int
    failed_requests: int
    total_requests: int
    elapsed_time: float
    qps: float = 0
    output_tokens: int = 0
    input_tokens: int = 0


class BenchmarkResultsOut(BaseModel):
    id: str
    status: str
    results: list[BenchmarkResultItem] = []
    summary: dict | None = None
    meta: dict | None = None
