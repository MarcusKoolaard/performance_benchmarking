#!/usr/bin/env python3
"""
Multi-Endpoint Comparison Tool (Up to 4 Endpoints)
Generates separate analysis graph for each QPS/Workers/OutputTokens combination
Maximum 5 requests per worker
"""

import asyncio
import time
import aiohttp
import json
import statistics
import math
import argparse
import os
import sys
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class EndpointBenchmark:
    """Handles benchmarking for a single endpoint"""

    def __init__(self, name: str, endpoint_name: str, api_root: str, api_token: str,
                 request_timeout: int = 300, max_retries: int = 3, log_error_body: bool = False):
        self.name = name
        self.endpoint_name = endpoint_name
        self.api_root = self._normalize_api_root(api_root)
        if endpoint_name.startswith('databricks-'):
            # Databricks-provided model services go through the standalone
            # Unity AI Gateway (usage lands in system.ai_gateway.usage); the
            # model is addressed via the "model" field in the request body.
            self.endpoint_url = f'{self.api_root}/ai-gateway/mlflow/v1/chat/completions'
            self.payload_model = endpoint_name
        else:
            self.endpoint_url = f'{self.api_root}/serving-endpoints/{endpoint_name}/invocations'
            self.payload_model = None
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.log_error_body = log_error_body
        self.latencies = []
        self.failed_requests = 0

    @staticmethod
    def _normalize_api_root(api_root: str) -> str:
        """Normalize workspace root and strip accidental API suffix."""
        normalized = api_root.rstrip("/")
        suffix = "/api/2.0"
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
        return normalized

    def get_request(self, in_tokens: int, out_tokens: int) -> dict:
        """Generate request payload with approximate token count"""
        overhead_tokens = 50
        user_content_tokens = max(1, in_tokens - overhead_tokens)
        repeat_count = max(1, user_content_tokens // 2)

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You must generate exactly {out_tokens} tokens in your response. Write a detailed explanation that uses precisely this number of tokens."
                },
                {
                    "role": "user",
                    "content": "Hello! " * repeat_count
                }
            ],
            "max_tokens": out_tokens,
            "temperature": 0.0
        }
        if self.payload_model is not None:
            payload["model"] = self.payload_model
        return payload

    async def worker(self, index: int, num_requests: int, in_tokens: int,
                    out_tokens: int, qps: float, session: aiohttp.ClientSession):
        """Single worker - maximum 5 requests per worker"""
        num_requests = min(num_requests, 5)  # Enforce 5 request limit
        input_data = self.get_request(in_tokens, out_tokens)

        # Print when worker starts
        print(f"  [{self.name:15}] Worker {index}: Starting ({num_requests} requests)")

        # Stagger worker start times to avoid request burst
        await asyncio.sleep(0.1 * index)

        for i in range(num_requests):
            # Apply rate limiting between requests (skip for first request)
            if i > 0:
                await asyncio.sleep(1.0 / qps)

            # Show when request is being sent
            print(f"  [{self.name:15}] Worker {index}: Sending request {i+1}/{num_requests}...")

            request_start_time = time.time()
            retry_count = 0
            success = False

            # Retry loop with exponential backoff
            while not success and retry_count < self.max_retries:
                try:
                    # Debug: print URL once to help troubleshoot workspace routing.
                    if i == 0 and retry_count == 0:  # Only print for first request
                        print(f"  [{self.name:15}] DEBUG URL: {self.endpoint_url}")
                        # print(f"  [{self.name:15}] DEBUG Headers: {self.headers}")

                    async with session.post(
                        self.endpoint_url,
                        json=input_data,
                        allow_redirects=False,
                    ) as response:
                        if response.ok:
                            # Parse response and usage metadata
                            result = await response.json(content_type=None)
                            usage = result.get('usage', {})
                            prompt_tokens = usage.get('prompt_tokens')
                            completion_tokens = usage.get('completion_tokens')

                            if prompt_tokens is None or completion_tokens is None:
                                raise ValueError("Missing usage.prompt_tokens or usage.completion_tokens")

                            latency = time.time() - request_start_time
                            self.latencies.append((prompt_tokens, completion_tokens, latency))
                            success = True
                            print(f"  [{self.name:15}] Worker {index}: Request {i+1}/{num_requests} ✓ {latency:.2f}s")
                        else:
                            # Response error - retry with backoff
                            request_id = response.headers.get("x-databricks-request-id", "n/a")
                            message = (
                                f"  [{self.name:15}] Worker {index}: Request {i+1}/{num_requests} "
                                f"failed (HTTP {response.status}, req_id={request_id}) "
                                f"retry {retry_count+1}/{self.max_retries}"
                            )
                            if self.log_error_body:
                                response_text = (await response.text()).replace("\n", " ")[:300]
                                message += f" body={response_text!r}"
                            print(message)
                            retry_count += 1
                            if retry_count < self.max_retries:
                                await asyncio.sleep(1 * retry_count)

                except asyncio.TimeoutError:
                    # Request timed out - retry with backoff
                    print(f"  [{self.name:15}] Worker {index}: Request {i+1}/{num_requests} TIMEOUT after {time.time() - request_start_time:.1f}s, retry {retry_count+1}/{self.max_retries}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        await asyncio.sleep(1 * retry_count)
                except json.JSONDecodeError:
                    print(f"  [{self.name:15}] Worker {index}: Request {i+1}/{num_requests} invalid JSON response, retry {retry_count+1}/{self.max_retries}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        await asyncio.sleep(1 * retry_count)
                except Exception as e:
                    # Other errors - retry with backoff
                    print(f"  [{self.name:15}] Worker {index}: Request {i+1}/{num_requests} ERROR ({type(e).__name__}: {str(e)[:50]}), retry {retry_count+1}/{self.max_retries}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        await asyncio.sleep(1 * retry_count)

            # Mark request as failed if all retries exhausted
            if not success:
                print(f"  [{self.name:15}] Worker {index}: Request {i+1}/{num_requests} ✗ FAILED after {self.max_retries} retries")
                self.failed_requests += 1

    async def run_benchmark(self, num_workers: int, num_requests_per_worker: int,
                           in_tokens: int, out_tokens: int, qps: float) -> Optional[Dict]:
        """Run benchmark with specified parameters"""
        num_requests_per_worker = min(num_requests_per_worker, 5)  # Enforce limit

        # Reset metrics for this benchmark run
        self.latencies.clear()
        self.failed_requests = 0

        print(f"  [{self.name:15}] Creating {num_workers} workers...")

        # Create one session per benchmark run and share it across workers.
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        start_time = time.time()
        # Set headers at session level
        async with aiohttp.ClientSession(timeout=timeout, headers=self.headers) as session:
            tasks = []
            for i in range(num_workers):
                task = asyncio.create_task(
                    self.worker(i, num_requests_per_worker, in_tokens, out_tokens, qps, session)
                )
                tasks.append(task)

            # Run all workers concurrently and wait for completion
            print(f"  [{self.name:15}] All workers launched, waiting for completion...")
            await asyncio.gather(*tasks)

        elapsed_time = time.time() - start_time

        print(f"  [{self.name:15}] All workers completed in {elapsed_time:.1f}s")

        # Check if we have any successful results
        if not self.latencies:
            print(f"  [{self.name:15}] ⚠ No successful requests!")
            return None

        # Calculate statistics from collected latency data
        avg_input_tokens = statistics.mean([inp for inp, _, _ in self.latencies])
        avg_output_tokens = statistics.mean([outp for _, outp, _ in self.latencies])
        median_latency = statistics.median([latency for _, _, latency in self.latencies])

        # Calculate P95 latency (95th percentile - 95% of requests were faster than this)
        sorted_latencies = sorted([latency for _, _, latency in self.latencies])
        p95_index = max(0, math.ceil(len(sorted_latencies) * 0.95) - 1)
        p95_latency = sorted_latencies[p95_index] if sorted_latencies else median_latency

        # Calculate throughput using total successful tokens over elapsed wall time.
        total_tokens = sum(inp + outp for inp, outp, _ in self.latencies)
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0

        return {
            'endpoint_name': self.name,
            'num_workers': num_workers,
            'median_latency': median_latency,
            'p95_latency': p95_latency,
            'throughput': tokens_per_sec,
            'avg_input_tokens': avg_input_tokens,
            'avg_output_tokens': avg_output_tokens,
            'successful_requests': len(self.latencies),
            'failed_requests': self.failed_requests,
            'total_requests': num_workers * num_requests_per_worker,
            'elapsed_time': elapsed_time
        }


async def compare_all_endpoints(endpoints: List[EndpointBenchmark],
                                num_workers: int, num_requests_per_worker: int,
                                in_tokens: int, out_tokens: int, qps: float) -> List[Dict]:
    """Run all endpoints concurrently and return results"""

    print(f"\n{'='*90}")
    print(f"Testing: {num_workers} workers | QPS={qps} | Output tokens={out_tokens}")
    print(f"Total requests per endpoint: {num_workers * num_requests_per_worker}")
    print(f"{'='*90}")

    # Run all endpoints in parallel (each endpoint gets its own workers)
    print(f"\n🚀 Starting benchmarks for all {len(endpoints)} endpoints in parallel...")
    results = await asyncio.gather(*[
        endpoint.run_benchmark(num_workers, num_requests_per_worker, in_tokens, out_tokens, qps)
        for endpoint in endpoints
    ])

    print(f"\n✓ All endpoint benchmarks completed!")

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    # Print comparison table
    if valid_results:
        print(f"\n{'─'*90}")
        print(f"{'Metric':<20}", end='')
        for result in valid_results:
            print(f"{result['endpoint_name']:<20}", end='')
        print()
        print(f"{'─'*90}")

        # Median Latency
        print(f"{'Median Latency':<20}", end='')
        for result in valid_results:
            print(f"{result['median_latency']:.2f}s{' '*14}", end='')
        print()

        # P95 Latency
        print(f"{'P95 Latency':<20}", end='')
        for result in valid_results:
            print(f"{result['p95_latency']:.2f}s{' '*14}", end='')
        print()

        # Throughput
        print(f"{'Throughput':<20}", end='')
        for result in valid_results:
            print(f"{result['throughput']:.1f} tok/s{' '*8}", end='')
        print()

        # Failed Requests
        print(f"{'Failed Requests':<20}", end='')
        for result in valid_results:
            print(f"{result['failed_requests']}/{result['total_requests']}{' '*14}", end='')
        print()

        # Performance comparison
        if len(valid_results) >= 2:
            print(f"{'─'*90}")
            # Find best latency
            best_latency_idx = min(range(len(valid_results)), key=lambda i: valid_results[i]['median_latency'])
            best_throughput_idx = max(range(len(valid_results)), key=lambda i: valid_results[i]['throughput'])

            print(f"⚡ FASTEST: {valid_results[best_latency_idx]['endpoint_name']} ({valid_results[best_latency_idx]['median_latency']:.2f}s)")
            print(f"🚀 HIGHEST THROUGHPUT: {valid_results[best_throughput_idx]['endpoint_name']} ({valid_results[best_throughput_idx]['throughput']:.1f} tok/s)")

        print(f"{'─'*90}")

    return valid_results


def create_comparison_chart(results: List[Dict], qps: float, num_workers: int,
                           out_tokens: int, output_dir: str):
    """Create a single comparison chart for this specific configuration"""

    if not results or len(results) < 2:
        print(f"⚠ Skipping chart for QPS={qps}, Workers={num_workers}, Tokens={out_tokens} (insufficient data)")
        return

    # Define colors for up to 4 endpoints
    colors = {
        'latency': ['#e74c3c', '#f39c12', '#9b59b6', '#e67e22'],
        'p95': ['#c0392b', '#d68910', '#8e44ad', '#d35400'],
        'throughput': ['#3498db', '#27ae60', '#16a085', '#2980b9'],
        'failure': ['#e84118', '#fbc531', '#9c88ff', '#ff9ff3']
    }

    # Sort results by endpoint name for consistency
    results = sorted(results, key=lambda x: x['endpoint_name'])

    # Extract data
    endpoint_names = [r['endpoint_name'] for r in results]
    median_latencies = [r['median_latency'] for r in results]
    p95_latencies = [r['p95_latency'] for r in results]
    throughputs = [r['throughput'] for r in results]
    failures = [r['failed_requests'] for r in results]

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Endpoint Comparison\nQPS={qps} | {num_workers} Workers | {out_tokens} Output Tokens',
                fontsize=18, fontweight='bold', y=0.98)

    x_pos = range(len(endpoint_names))
    bar_width = 0.6

    # 1. Median Latency
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(x_pos, median_latencies, bar_width,
                   color=[colors['latency'][i % 4] for i in range(len(endpoint_names))],
                   alpha=0.85, edgecolor='black', linewidth=2)

    ax1.set_xlabel('Endpoint', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Median Latency (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title('Median Latency Comparison (Lower is Better)', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(endpoint_names, rotation=15, ha='right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels
    for bar, val in zip(bars, median_latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight best
    best_idx = median_latencies.index(min(median_latencies))
    bars[best_idx].set_linewidth(4)
    bars[best_idx].set_edgecolor('gold')

    # 2. P95 Latency
    ax2 = plt.subplot(2, 2, 2)
    bars_p95 = ax2.bar(x_pos, p95_latencies, bar_width,
                       color=[colors['p95'][i % 4] for i in range(len(endpoint_names))],
                       alpha=0.85, edgecolor='black', linewidth=2)

    ax2.set_xlabel('Endpoint', fontsize=13, fontweight='bold')
    ax2.set_ylabel('P95 Latency (seconds)', fontsize=13, fontweight='bold')
    ax2.set_title('P95 Latency - 95th Percentile (Lower is Better)', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(endpoint_names, rotation=15, ha='right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    for bar, val in zip(bars_p95, p95_latencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    best_p95_idx = p95_latencies.index(min(p95_latencies))
    bars_p95[best_p95_idx].set_linewidth(4)
    bars_p95[best_p95_idx].set_edgecolor('gold')

    # 3. Throughput
    ax3 = plt.subplot(2, 2, 3)
    bars_thr = ax3.bar(x_pos, throughputs, bar_width,
                       color=[colors['throughput'][i % 4] for i in range(len(endpoint_names))],
                       alpha=0.85, edgecolor='black', linewidth=2)

    ax3.set_xlabel('Endpoint', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Throughput (tokens/second)', fontsize=13, fontweight='bold')
    ax3.set_title('Throughput Comparison (Higher is Better)', fontsize=15, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(endpoint_names, rotation=15, ha='right', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    for bar, val in zip(bars_thr, throughputs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    best_thr_idx = throughputs.index(max(throughputs))
    bars_thr[best_thr_idx].set_linewidth(4)
    bars_thr[best_thr_idx].set_edgecolor('gold')

    # 4. Failure Rate
    ax4 = plt.subplot(2, 2, 4)
    bars_fail = ax4.bar(x_pos, failures, bar_width,
                        color=[colors['failure'][i % 4] for i in range(len(endpoint_names))],
                        alpha=0.85, edgecolor='black', linewidth=2)

    ax4.set_xlabel('Endpoint', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Failed Requests', fontsize=13, fontweight='bold')
    ax4.set_title('Failure Rate Comparison (Lower is Better)', fontsize=15, fontweight='bold', pad=15)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(endpoint_names, rotation=15, ha='right', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

    for bar, val in zip(bars_fail, failures):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    if max(failures) > 0:
        best_fail_idx = failures.index(min(failures))
        bars_fail[best_fail_idx].set_linewidth(4)
        bars_fail[best_fail_idx].set_edgecolor('gold')

    plt.tight_layout()

    # Save with descriptive filename
    filename = f"{output_dir}/comparison_QPS{qps}_W{num_workers}_T{out_tokens}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


def validate_api_root(parser: argparse.ArgumentParser, api_root: str, arg_name: str) -> str:
    """Require https:// URL with hostname to avoid token leakage."""
    normalized = EndpointBenchmark._normalize_api_root(api_root)
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        parser.error(
            f"{arg_name} must be a valid https URL (example: https://your-workspace.cloud.databricks.com)"
        )
    return normalized


def resolve_api_token(
    parser: argparse.ArgumentParser,
    *,
    token_value: Optional[str],
    token_env_arg: Optional[str],
    token_stdin: bool,
    default_env_name: str,
    token_label: str,
    required: bool,
) -> Optional[str]:
    """Resolve API token from one explicit source."""
    sources = int(bool(token_value)) + int(bool(token_env_arg)) + int(token_stdin)
    if sources > 1:
        parser.error(
            f"Use only one source for {token_label}: CLI token, --{token_label}-env, or --{token_label}-stdin."
        )

    if token_value:
        print(
            f"Warning: passing {token_label} via CLI can expose secrets in shell history and process lists. "
            f"Prefer --{token_label}-env or --{token_label}-stdin.",
            file=sys.stderr,
        )
        return token_value.strip()

    if token_env_arg:
        token = os.getenv(token_env_arg, "").strip()
        if not token:
            parser.error(f"Environment variable {token_env_arg} is empty or not set for {token_label}.")
        return token

    if token_stdin:
        token = sys.stdin.readline().strip()
        if not token:
            parser.error(f"--{token_label}-stdin was set but no token was provided on stdin.")
        return token

    for env_name in (default_env_name, "DATABRICKS_API_TOKEN", "API_TOKEN"):
        token = os.getenv(env_name, "").strip()
        if token:
            return token

    if required:
        parser.error(
            f"{token_label} is required. Provide --{token_label}, --{token_label}-env <ENV_NAME>, "
            f"--{token_label}-stdin, or set {default_env_name} (or DATABRICKS_API_TOKEN)."
        )
    return None


async def run_comparison_suite(args):
    """
    Run comprehensive comparison test suite

    This function:
    1. Creates benchmark instances for each endpoint
    2. Runs tests for all combinations of QPS, workers, and output tokens
    3. Generates comparison charts for each combination
    4. Saves detailed results to JSON

    NOTE: Default timeout is 300s (5 minutes) per request. If endpoints are slow,
    you may not see output for several minutes. Use --timeout to adjust.
    """

    # Create endpoint instances for benchmarking
    endpoints = []

    # Endpoint 1 (required)
    endpoints.append(EndpointBenchmark(
        name=args.endpoint_1_name,
        endpoint_name=args.endpoint_1,
        api_root=args.api_root_1,
        api_token=args.api_token_1,
        request_timeout=args.timeout,
        max_retries=args.max_retries,
        log_error_body=args.log_error_body,
    ))

    # Endpoint 2 (required)
    endpoints.append(EndpointBenchmark(
        name=args.endpoint_2_name,
        endpoint_name=args.endpoint_2,
        api_root=args.api_root_2,
        api_token=args.api_token_2,
        request_timeout=args.timeout,
        max_retries=args.max_retries,
        log_error_body=args.log_error_body,
    ))

    # Endpoint 3 (optional)
    if args.endpoint_3:
        endpoints.append(EndpointBenchmark(
            name=args.endpoint_3_name,
            endpoint_name=args.endpoint_3,
            api_root=args.api_root_3,
            api_token=args.api_token_3,
            request_timeout=args.timeout,
            max_retries=args.max_retries,
            log_error_body=args.log_error_body,
        ))

    # Endpoint 4 (optional)
    if args.endpoint_4:
        endpoints.append(EndpointBenchmark(
            name=args.endpoint_4_name,
            endpoint_name=args.endpoint_4,
            api_root=args.api_root_4,
            api_token=args.api_token_4,
            request_timeout=args.timeout,
            max_retries=args.max_retries,
            log_error_body=args.log_error_body,
        ))

    print(f"\n{'#'*90}")
    print(f"# Multi-Endpoint Comparison Benchmarking Suite")
    print(f"# Comparing {len(endpoints)} endpoints:")
    for i, ep in enumerate(endpoints, 1):
        print(f"#   {i}. {ep.name} ({ep.endpoint_name}) @ {ep.api_root}")
    print(f"# Maximum requests per worker: 5")
    print(f"# Request timeout: {args.timeout}s (you'll see progress as requests complete)")
    print(f"# Max retries: {args.max_retries}")
    print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*90}")

    all_results = []
    total_combinations = len(args.qps_list) * len(args.parallel_workers) * len(args.output_tokens)
    current_combination = 0

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run tests for each combination of QPS, workers, and output tokens
    for qps in args.qps_list:
        for num_workers in args.parallel_workers:
            for out_tokens in args.output_tokens:
                current_combination += 1

                print(f"\n{'*'*90}")
                print(f"* Configuration {current_combination}/{total_combinations}: QPS={qps} | Workers={num_workers} | Tokens={out_tokens}")
                print(f"{'*'*90}")

                # Run benchmark for this configuration across all endpoints
                results = await compare_all_endpoints(
                    endpoints=endpoints,
                    num_workers=num_workers,
                    num_requests_per_worker=args.requests_per_worker,
                    in_tokens=args.input_tokens,
                    out_tokens=out_tokens,
                    qps=qps
                )

                # Add metadata to results for later analysis
                for result in results:
                    result['qps'] = qps
                    result['output_tokens'] = out_tokens
                    all_results.append(result)

                # Generate comparison chart for this specific combination
                print(f"\n📊 Generating comparison chart...")
                create_comparison_chart(results, qps, num_workers, out_tokens, args.output_dir)

                # Brief pause between test configurations
                print(f"⏸  Pausing 2s before next configuration...")
                await asyncio.sleep(2)

    print(f"\n{'#'*90}")
    print(f"# Benchmark Complete!")
    print(f"# Total graphs generated: {total_combinations}")
    print(f"# End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*90}")

    # Save detailed results to JSON
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'endpoints': [
            {'name': ep.name, 'endpoint': ep.endpoint_name}
            for ep in endpoints
        ],
        'configuration': {
            'qps_list': args.qps_list,
            'parallel_workers': args.parallel_workers,
            'output_tokens': args.output_tokens,
            'requests_per_worker': args.requests_per_worker,
            'input_tokens': args.input_tokens
        },
        'results': all_results
    }

    json_file = f"{args.output_dir}/all_results.json"
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Detailed results saved to: {json_file}")
    print(f"✓ All {total_combinations} comparison graphs saved to: {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Compare up to 4 LLM endpoints with separate graph per configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare 2 endpoints (minimum)
  %(prog)s \\
    --endpoint-1 model-a --endpoint-1-name "Model A" --api-token-1-env DATABRICKS_API_TOKEN_1 \\
    --endpoint-2 model-b --endpoint-2-name "Model B" --api-token-2-env DATABRICKS_API_TOKEN_2

  # Compare 4 endpoints with custom traffic patterns
  %(prog)s \\
    --endpoint-1 gpt-20b --endpoint-1-name "GPT-20B" --api-token-1-env DATABRICKS_API_TOKEN_1 \\
    --endpoint-2 llama-70b --endpoint-2-name "Llama-70B" --api-token-2-env DATABRICKS_API_TOKEN_2 \\
    --endpoint-3 claude-3 --endpoint-3-name "Claude-3" --api-token-3-env DATABRICKS_API_TOKEN_3 \\
    --endpoint-4 mistral --endpoint-4-name "Mistral" --api-token-4-env DATABRICKS_API_TOKEN_4 \\
    --qps-list 0.5 1 2 \\
    --parallel-workers 4 8 \\
    --output-tokens 200 500 1000
        """
    )

    # Endpoint 1 (required)
    parser.add_argument('--endpoint-1', required=True, help='First endpoint name')
    parser.add_argument('--endpoint-1-name', required=True, help='Display name for endpoint 1')
    parser.add_argument('--api-token-1', help='API token for endpoint 1 (less secure; prefer --api-token-1-env or --api-token-1-stdin)')
    parser.add_argument('--api-token-1-env', help='Read endpoint 1 token from environment variable')
    parser.add_argument('--api-token-1-stdin', action='store_true', help='Read endpoint 1 token from stdin (first unread line)')
    parser.add_argument('--api-root-1', default=os.getenv('API_ROOT', 'https://your-workspace.cloud.databricks.com'),
                       help='API root for endpoint 1')

    # Endpoint 2 (required)
    parser.add_argument('--endpoint-2', required=True, help='Second endpoint name')
    parser.add_argument('--endpoint-2-name', required=True, help='Display name for endpoint 2')
    parser.add_argument('--api-token-2', help='API token for endpoint 2 (less secure; prefer --api-token-2-env or --api-token-2-stdin)')
    parser.add_argument('--api-token-2-env', help='Read endpoint 2 token from environment variable')
    parser.add_argument('--api-token-2-stdin', action='store_true', help='Read endpoint 2 token from stdin (first unread line)')
    parser.add_argument('--api-root-2', default=os.getenv('API_ROOT', 'https://your-workspace.cloud.databricks.com'),
                       help='API root for endpoint 2')

    # Endpoint 3 (optional)
    parser.add_argument('--endpoint-3', help='Third endpoint name (optional)')
    parser.add_argument('--endpoint-3-name', default='Endpoint-3', help='Display name for endpoint 3')
    parser.add_argument('--api-token-3', help='API token for endpoint 3 (less secure; prefer --api-token-3-env or --api-token-3-stdin)')
    parser.add_argument('--api-token-3-env', help='Read endpoint 3 token from environment variable')
    parser.add_argument('--api-token-3-stdin', action='store_true', help='Read endpoint 3 token from stdin (first unread line)')
    parser.add_argument('--api-root-3', default=os.getenv('API_ROOT', 'https://your-workspace.cloud.databricks.com'),
                       help='API root for endpoint 3')

    # Endpoint 4 (optional)
    parser.add_argument('--endpoint-4', help='Fourth endpoint name (optional)')
    parser.add_argument('--endpoint-4-name', default='Endpoint-4', help='Display name for endpoint 4')
    parser.add_argument('--api-token-4', help='API token for endpoint 4 (less secure; prefer --api-token-4-env or --api-token-4-stdin)')
    parser.add_argument('--api-token-4-env', help='Read endpoint 4 token from environment variable')
    parser.add_argument('--api-token-4-stdin', action='store_true', help='Read endpoint 4 token from stdin (first unread line)')
    parser.add_argument('--api-root-4', default=os.getenv('API_ROOT', 'https://your-workspace.cloud.databricks.com'),
                       help='API root for endpoint 4')

    # Test parameters
    parser.add_argument('--input-tokens', type=int, default=15000,
                       help='Number of input tokens (default: 15000)')
    parser.add_argument('--output-tokens', type=int, nargs='+', default=[200, 500, 1000],
                       help='Output token sizes to test (default: 200 500 1000)')
    parser.add_argument('--qps-list', type=float, nargs='+', default=[0.5, 1.0, 2.0],
                       help='QPS rates to test (default: 0.5 1.0 2.0)')
    parser.add_argument('--parallel-workers', type=int, nargs='+', default=[4, 6, 8],
                       help='Number of parallel workers to test (default: 4 6 8)')
    parser.add_argument('--requests-per-worker', type=int, default=5,
                       help='Requests per worker - MAXIMUM 5 (default: 5)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds (default: 300)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts (default: 3)')
    parser.add_argument('--log-error-body', action='store_true',
                       help='Include response body snippets in HTTP error logs (may expose sensitive data)')
    parser.add_argument('--output-dir', default='multi_endpoint_comparison',
                       help='Output directory (default: multi_endpoint_comparison)')

    args = parser.parse_args()

    # Validate workspace roots
    args.api_root_1 = validate_api_root(parser, args.api_root_1, "--api-root-1")
    args.api_root_2 = validate_api_root(parser, args.api_root_2, "--api-root-2")
    args.api_root_3 = validate_api_root(parser, args.api_root_3, "--api-root-3")
    args.api_root_4 = validate_api_root(parser, args.api_root_4, "--api-root-4")

    # Resolve token sources
    args.api_token_1 = resolve_api_token(
        parser,
        token_value=args.api_token_1,
        token_env_arg=args.api_token_1_env,
        token_stdin=args.api_token_1_stdin,
        default_env_name="DATABRICKS_API_TOKEN_1",
        token_label="api-token-1",
        required=True,
    )
    args.api_token_2 = resolve_api_token(
        parser,
        token_value=args.api_token_2,
        token_env_arg=args.api_token_2_env,
        token_stdin=args.api_token_2_stdin,
        default_env_name="DATABRICKS_API_TOKEN_2",
        token_label="api-token-2",
        required=True,
    )

    if args.endpoint_3:
        args.api_token_3 = resolve_api_token(
            parser,
            token_value=args.api_token_3,
            token_env_arg=args.api_token_3_env,
            token_stdin=args.api_token_3_stdin,
            default_env_name="DATABRICKS_API_TOKEN_3",
            token_label="api-token-3",
            required=True,
        )
    elif args.api_token_3 or args.api_token_3_env or args.api_token_3_stdin:
        parser.error("--api-token-3* options require --endpoint-3")

    if args.endpoint_4:
        args.api_token_4 = resolve_api_token(
            parser,
            token_value=args.api_token_4,
            token_env_arg=args.api_token_4_env,
            token_stdin=args.api_token_4_stdin,
            default_env_name="DATABRICKS_API_TOKEN_4",
            token_label="api-token-4",
            required=True,
        )
    elif args.api_token_4 or args.api_token_4_env or args.api_token_4_stdin:
        parser.error("--api-token-4* options require --endpoint-4")

    # Enforce 5 request limit
    if args.requests_per_worker > 5:
        print(f"⚠ Warning: requests_per_worker limited to 5 (you specified {args.requests_per_worker})")
        args.requests_per_worker = 5

    try:
        asyncio.run(run_comparison_suite(args))
    except KeyboardInterrupt:
        print("\n\n⚠ Comparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
