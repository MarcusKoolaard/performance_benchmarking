#!/usr/bin/env python3
"""
LLM Endpoint Benchmarking Tool
Run load tests against Databricks LLM serving endpoints
"""

import asyncio
import time
import aiohttp
import json
import statistics
import argparse
import sys
import os
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class LLMBenchmark:
    def __init__(self, endpoint_name: str, api_root: str, api_token: str,
                 request_timeout: int = 300, max_retries: int = 3):
        self.endpoint_name = endpoint_name
        self.api_root = api_root.rstrip('/')
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
        self.latencies = []
        self.failed_requests = 0

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
                    out_tokens: int, qps: float):
        """Single worker that processes requests with error handling"""
        input_data = self.get_request(in_tokens, out_tokens)

        # Offset threads slightly to avoid thundering herd
        await asyncio.sleep(0.1 * index)

        for i in range(num_requests):
            # Rate limiting: wait between requests based on QPS
            if i > 0:
                await asyncio.sleep(1.0 / qps)

            request_start_time = time.time()
            retry_count = 0
            success = False

            while not success and retry_count < self.max_retries:
                try:
                    timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(self.endpoint_url, headers=self.headers,
                                               json=input_data, allow_redirects=False) as response:
                            if response.ok:
                                chunks = []
                                async for chunk, _ in response.content.iter_chunks():
                                    chunks.append(chunk)

                                latency = time.time() - request_start_time
                                result = json.loads(b''.join(chunks))
                                self.latencies.append((
                                    result['usage']['prompt_tokens'],
                                    result['usage']['completion_tokens'],
                                    latency
                                ))
                                success = True
                                print(f"  ✓ Worker {index}: Request {i+1}/{num_requests} completed in {latency:.2f}s")
                            else:
                                retry_count += 1
                                print(f"  ✗ Worker {index}: Request {i+1} failed (HTTP {response.status}), retry {retry_count}/{self.max_retries}")
                                if retry_count < self.max_retries:
                                    await asyncio.sleep(1 * retry_count)

                except asyncio.TimeoutError:
                    retry_count += 1
                    print(f"  ⏱ Worker {index}: Request {i+1} timed out, retry {retry_count}/{self.max_retries}")
                    if retry_count < self.max_retries:
                        await asyncio.sleep(1 * retry_count)
                except Exception as e:
                    retry_count += 1
                    print(f"  ⚠ Worker {index}: Request {i+1} error: {e}, retry {retry_count}/{self.max_retries}")
                    if retry_count < self.max_retries:
                        await asyncio.sleep(1 * retry_count)

            if not success:
                self.failed_requests += 1
                print(f"  ✗ Worker {index}: Request {i+1} failed after {self.max_retries} retries")

    async def run_benchmark(self, num_workers: int, num_requests_per_worker: int,
                           in_tokens: int, out_tokens: int, qps: float) -> Optional[Dict]:
        """Run benchmark with specified parameters"""
        print(f"\n{'='*70}")
        print(f"Benchmark: {num_workers} workers | {num_requests_per_worker} req/worker | "
              f"{out_tokens} out tokens | QPS={qps}")
        print(f"{'='*70}")

        self.latencies.clear()
        self.failed_requests = 0

        start_time = time.time()

        # Create and run workers
        tasks = []
        for i in range(num_workers):
            task = asyncio.create_task(
                self.worker(i, num_requests_per_worker, in_tokens, out_tokens, qps)
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time

        if not self.latencies:
            print(f"\n⚠ No successful requests!")
            return None

        # Compute statistics
        avg_input_tokens = statistics.mean([inp for inp, _, _ in self.latencies])
        avg_output_tokens = statistics.mean([outp for _, outp, _ in self.latencies])
        latency_values = [latency for _, _, latency in self.latencies]
        median_latency = statistics.median(latency_values)
        p95_latency = statistics.quantiles(latency_values, n=20)[18] if len(latency_values) > 1 else median_latency
        tokens_per_sec = (avg_input_tokens + avg_output_tokens) * num_workers / median_latency

        print(f"\n{'─'*70}")
        print(f"Results:")
        print(f"  Input tokens:    {avg_input_tokens:.0f}")
        print(f"  Output tokens:   {avg_output_tokens:.0f}")
        print(f"  Median latency:  {median_latency:.2f}s")
        print(f"  P95 latency:     {p95_latency:.2f}s")
        print(f"  Throughput:      {tokens_per_sec:.1f} tokens/sec")
        print(f"  Failed requests: {self.failed_requests}/{num_workers * num_requests_per_worker}")
        print(f"  Total time:      {elapsed_time:.2f}s")
        print(f"{'─'*70}")

        return {
            'endpoint_name': self.endpoint_name,
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


async def run_test_suite(benchmark: LLMBenchmark, args):
    """Run comprehensive test suite"""
    all_results = []

    print(f"\n{'#'*70}")
    print(f"# LLM Endpoint Benchmarking Suite")
    print(f"# Endpoint: {benchmark.endpoint_name}")
    print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    for qps in args.qps_list:
        for in_tokens in args.input_tokens:
            for out_tokens in args.output_tokens:
                for num_workers in args.parallel_workers:
                    result = await benchmark.run_benchmark(
                        num_workers=num_workers,
                        num_requests_per_worker=args.requests_per_worker,
                        in_tokens=in_tokens,
                        out_tokens=out_tokens,
                        qps=qps
                    )

                    if result:
                        result['qps'] = qps
                        result['input_tokens'] = in_tokens
                        result['output_tokens'] = out_tokens
                        all_results.append(result)

                    # Brief pause between tests
                    await asyncio.sleep(2)

    print(f"\n{'#'*70}")
    print(f"# Test Suite Complete!")
    print(f"# End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    # Print summary table
    print_summary_table(all_results)

    # Save results and generate visualizations
    if args.output_dir or args.output_file:
        output_dir = args.output_dir or os.path.dirname(args.output_file) or 'benchmark_results'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate charts
        print(f"\n{'='*70}")
        print("Generating performance visualizations...")
        print(f"{'='*70}\n")

        create_performance_charts(all_results, benchmark.endpoint_name, output_dir)
        create_summary_chart(all_results, benchmark.endpoint_name, output_dir)

        # Save results JSON
        json_file = args.output_file if args.output_file else f"{output_dir}/results.json"
        save_results(all_results, json_file, benchmark.endpoint_name)

        print(f"\n{'='*70}")
        print(f"All files saved to: {output_dir}/")
        print(f"{'='*70}\n")


def print_summary_table(results: List[Dict]):
    """Print formatted summary table"""
    if not results:
        return

    print(f"\n{'='*115}")
    print(f"SUMMARY TABLE")
    print(f"{'='*115}")
    print(f"{'QPS':<8} {'In Tokens':<12} {'Out Tokens':<12} {'Workers':<10} {'Latency (s)':<15} "
          f"{'Throughput':<20} {'Failed':<10}")
    print(f"{'-'*115}")

    for r in results:
        in_tok = int(r.get('input_tokens', r.get('avg_input_tokens', 0)))
        print(f"{r['qps']:<8} {in_tok:<12} {int(r['output_tokens']):<12} {r['num_workers']:<10} "
              f"{r['median_latency']:<15.2f} {r['throughput']:<20.1f} "
              f"{r['failed_requests']}/{r['total_requests']:<10}")

    print(f"{'='*115}\n")


def save_results(results: List[Dict], filename: str, endpoint_name: str):
    """Save results to JSON file"""
    output = {
        'timestamp': datetime.now().isoformat(),
        'endpoints': [endpoint_name],
        'results': results
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to {filename}")


def create_performance_charts(results: List[Dict], endpoint_name: str, output_dir: str):
    """Create performance charts for single endpoint benchmarks"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results:
        print("⚠ No results to visualize")
        return

    # Group results by worker count
    results_by_workers = {}
    for result in results:
        workers = result['num_workers']
        if workers not in results_by_workers:
            results_by_workers[workers] = []
        results_by_workers[workers].append(result)

    # Create charts for each worker configuration
    for num_workers, worker_results in results_by_workers.items():
        # Sort by QPS and output tokens
        worker_results_sorted = sorted(
            worker_results,
            key=lambda x: (x.get('qps', 0), x.get('output_tokens', 0))
        )

        # Create labels
        labels = [
            f"QPS={r.get('qps', 'N/A')}\n{r.get('output_tokens', 0)}tok"
            for r in worker_results_sorted
        ]
        x_pos = range(len(labels))

        # Extract metrics
        median_latencies = [r['median_latency'] for r in worker_results_sorted]
        p95_latencies = [r.get('p95_latency', r['median_latency']) for r in worker_results_sorted]
        throughputs = [r['throughput'] for r in worker_results_sorted]
        failed_requests = [r['failed_requests'] for r in worker_results_sorted]

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'{endpoint_name} - {num_workers} Parallel Workers',
                    fontsize=18, fontweight='bold', y=0.995)

        # 1. Median Latency
        ax1 = plt.subplot(2, 2, 1)
        bars1 = ax1.bar(x_pos, median_latencies, color='#e74c3c', alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Traffic Condition', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Median Latency (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Median Latency', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # 2. P95 Latency
        ax2 = plt.subplot(2, 2, 2)
        bars2 = ax2.bar(x_pos, p95_latencies, color='#c0392b', alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Traffic Condition', fontsize=12, fontweight='bold')
        ax2.set_ylabel('P95 Latency (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('P95 Latency (95th Percentile)', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # 3. Throughput
        ax3 = plt.subplot(2, 2, 3)
        bars3 = ax3.bar(x_pos, throughputs, color='#3498db', alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Traffic Condition', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Throughput (tokens/second)', fontsize=12, fontweight='bold')
        ax3.set_title('Throughput', fontsize=14, fontweight='bold', pad=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # 4. Failed Requests
        ax4 = plt.subplot(2, 2, 4)
        bars4 = ax4.bar(x_pos, failed_requests, color='#e84118', alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Traffic Condition', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Failed Requests', fontsize=12, fontweight='bold')
        ax4.set_title('Failure Rate', fontsize=14, fontweight='bold', pad=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')

        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        plt.tight_layout()

        # Save figure
        filename = f"{output_dir}/performance_{num_workers}workers.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
        plt.close()


def create_summary_chart(results: List[Dict], endpoint_name: str, output_dir: str):
    """Create overall summary chart showing trends across all worker configurations"""

    if not results:
        return

    # Group by worker count
    data_by_workers = {}
    for result in results:
        workers = result['num_workers']
        if workers not in data_by_workers:
            data_by_workers[workers] = {
                'latencies': [],
                'throughputs': [],
                'p95_latencies': []
            }
        data_by_workers[workers]['latencies'].append(result['median_latency'])
        data_by_workers[workers]['throughputs'].append(result['throughput'])
        p95 = result.get('p95_latency', result['median_latency'])
        data_by_workers[workers]['p95_latencies'].append(p95)

    # Calculate averages
    workers_list = sorted(data_by_workers.keys())
    avg_latencies = [statistics.mean(data_by_workers[w]['latencies']) for w in workers_list]
    avg_throughputs = [statistics.mean(data_by_workers[w]['throughputs']) for w in workers_list]
    avg_p95 = [statistics.mean(data_by_workers[w]['p95_latencies']) for w in workers_list]

    # Create summary figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{endpoint_name} - Overall Performance Summary',
                fontsize=16, fontweight='bold')

    # Latency trend
    ax1.plot(workers_list, avg_latencies, marker='o', linewidth=3, markersize=10,
            label='Median Latency', color='#e74c3c')
    ax1.plot(workers_list, avg_p95, marker='s', linewidth=3, markersize=10,
            label='P95 Latency', color='#c0392b', linestyle='--')
    ax1.set_xlabel('Parallel Workers', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency Scaling', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    for i, (w, lat, p95) in enumerate(zip(workers_list, avg_latencies, avg_p95)):
        ax1.annotate(f'{lat:.2f}s', (w, lat), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9, color='#e74c3c', fontweight='bold')
        ax1.annotate(f'{p95:.2f}s', (w, p95), textcoords="offset points",
                    xytext=(0,-15), ha='center', fontsize=9, color='#c0392b', fontweight='bold')

    # Throughput trend
    ax2.plot(workers_list, avg_throughputs, marker='o', linewidth=3, markersize=10,
            label='Throughput', color='#3498db')
    ax2.set_xlabel('Parallel Workers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Throughput (tokens/second)', fontsize=12, fontweight='bold')
    ax2.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    for i, (w, thr) in enumerate(zip(workers_list, avg_throughputs)):
        ax2.annotate(f'{int(thr)}', (w, thr), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9, color='#3498db', fontweight='bold')

    plt.tight_layout()

    filename = f"{output_dir}/summary_overall.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


def normalize_api_root(api_root: str) -> str:
    """Normalize workspace root and strip accidental API suffix."""
    normalized = api_root.rstrip("/")
    suffix = "/api/2.0"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized


def validate_api_root(parser: argparse.ArgumentParser, api_root: str) -> str:
    """Require https:// URL with hostname to avoid token leakage."""
    normalized = normalize_api_root(api_root)
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        parser.error("--api-root must be a valid https URL (example: https://your-workspace.cloud.databricks.com)")
    return normalized


def resolve_api_token(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Resolve API token from one explicit source."""
    sources = int(bool(args.api_token)) + int(bool(args.api_token_env)) + int(args.api_token_stdin)
    if sources > 1:
        parser.error("Use only one token source: --api-token OR --api-token-env OR --api-token-stdin")

    if args.api_token:
        print(
            "Warning: passing tokens via --api-token can expose secrets in shell history and process lists. "
            "Prefer --api-token-env or --api-token-stdin.",
            file=sys.stderr,
        )
        token = args.api_token.strip()
    elif args.api_token_env:
        token = os.getenv(args.api_token_env, "").strip()
        if not token:
            parser.error(f"Environment variable {args.api_token_env} is empty or not set.")
    elif args.api_token_stdin:
        token = sys.stdin.readline().strip()
        if not token:
            parser.error("--api-token-stdin was set but no token was provided on stdin.")
    else:
        token = os.getenv("DATABRICKS_API_TOKEN", "").strip()
        if not token:
            token = os.getenv("API_TOKEN", "").strip()
        if not token:
            parser.error(
                "API token is required. Provide --api-token, --api-token-env <ENV_NAME>, "
                "--api-token-stdin, or set DATABRICKS_API_TOKEN."
            )
    return token


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark LLM endpoint performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default settings
  %(prog)s --endpoint gpt-oss-20b-load-testing --api-token-env DATABRICKS_API_TOKEN

  # Custom test with specific parameters
  %(prog)s --endpoint my-llm --api-token-env DATABRICKS_API_TOKEN --input-tokens 500 1000 --output-tokens 200 500 --qps-list 1 2 --parallel-workers 4 8

  # Read token from stdin (safer than command-line token)
  printf '%s\n' "$DATABRICKS_API_TOKEN" | %(prog)s --endpoint my-llm --api-token-stdin --output-file results.json
        """
    )

    # Required arguments
    parser.add_argument('--endpoint', required=True,
                       help='Name of the Databricks serving endpoint')
    parser.add_argument('--api-token',
                       help='Databricks API token (less secure; prefer --api-token-env or --api-token-stdin)')
    parser.add_argument('--api-token-env',
                       help='Read API token from the named environment variable')
    parser.add_argument('--api-token-stdin', action='store_true',
                       help='Read API token from stdin (first line)')

    # Optional arguments
    parser.add_argument('--api-root', default='https://your-workspace.cloud.databricks.com',
                       help='Databricks API root URL')
    parser.add_argument('--input-tokens', type=int, nargs='+', default=[15000],
                       help='Input token sizes to test (default: 15000)')
    parser.add_argument('--output-tokens', type=int, nargs='+', default=[200, 500, 1000],
                       help='Output token sizes to test (default: 200 500 1000)')
    parser.add_argument('--qps-list', type=float, nargs='+', default=[0.5, 1.0],
                       help='QPS rates to test (default: 0.5 1.0)')
    parser.add_argument('--parallel-workers', type=int, nargs='+', default=[4, 6],
                       help='Number of parallel workers to test (default: 4 6)')
    parser.add_argument('--requests-per-worker', type=int, default=5,
                       help='Number of requests per worker (default: 5)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds (default: 300)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts (default: 3)')
    parser.add_argument('--output-file', '-o',
                       help='Save results to JSON file')
    parser.add_argument('--output-dir',
                       help='Output directory for results and charts (default: benchmark_results)')

    args = parser.parse_args()
    args.api_root = validate_api_root(parser, args.api_root)
    args.api_token = resolve_api_token(parser, args)

    # Create benchmark instance
    benchmark = LLMBenchmark(
        endpoint_name=args.endpoint,
        api_root=args.api_root,
        api_token=args.api_token,
        request_timeout=args.timeout,
        max_retries=args.max_retries
    )

    # Run the test suite
    try:
        asyncio.run(run_test_suite(benchmark, args))
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
