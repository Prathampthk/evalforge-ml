"""Real loopback HTTP load: embedding-inclusive retrieval, closed-loop concurrent clients."""

from __future__ import annotations

import argparse
import json
import asyncio
import os
import secrets
import socket
import subprocess  # nosec B404 - launch only this repository's uvicorn via the current interpreter
import sys
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import importlib.util
import importlib.metadata
import platform

import httpx
import numpy as np
import psutil

from evaluation.experiment import provenance, timestamp, write_json
from evaluation.ranking import evaluate_ranking
from evaluation.retrieval_data import load_dataset


@contextmanager
def running_server(root: Path, encoder: str, workers: int, log: Path, model_cache: str = "artifacts/model-cache"):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = secrets.token_urlsafe(24)
    env = {
        **os.environ,
        "RAG_API_TOKEN": token,
        "EVALFORGE_EMBEDDER": encoder,
        "EVALFORGE_SNAPSHOT_DIR": str(root.resolve()),
        "EVALFORGE_MODEL_CACHE": str(Path(model_cache).resolve()),
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TELEMETRY_EXPORTER": "none",
    }
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    if sys.platform == "darwin" and encoder == "minilm":
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.submodule_search_locations is None:
            raise RuntimeError("PyTorch is not installed")
        # Resolve FAISS and torch against one trusted OpenMP dylib, rather than disabling duplicate checks.
        env["DYLD_LIBRARY_PATH"] = str(Path(next(iter(spec.submodule_search_locations))) / "lib")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        process = subprocess.Popen(  # nosec B603 - fixed module; arguments validated by caller/argparse
            [
                sys.executable,
                "-m",
                "uvicorn",
                "serving.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                str(workers),
                "--no-access-log",
            ],
            env=env,
            stdout=handle,
            stderr=handle,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 90
            with httpx.Client(trust_env=False, timeout=1) as client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited during startup; see {log}")
                    try:
                        response = client.get(url + "/health")
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError(f"Server readiness timeout; see {log}")
            # First health response does not imply every spawned worker has completed startup.
            ready_pids: set[int] = set()

            def probe():
                try:
                    response = httpx.get(url + "/health", timeout=2, trust_env=False, headers={"Connection": "close"})
                    return response.json()["pid"] if response.status_code == 200 else None
                except (httpx.HTTPError, ValueError, KeyError):
                    return None

            with ThreadPoolExecutor(max_workers=min(16, workers * 4)) as pool:
                while len(ready_pids) < workers and time.monotonic() < deadline:
                    ready_pids.update(pid for pid in pool.map(lambda _: probe(), range(workers * 4)) if pid)
                    if len(ready_pids) < workers:
                        time.sleep(0.05)
            if len(ready_pids) != workers:
                raise RuntimeError("Not all configured workers became ready")
            yield url, token, process.pid
        finally:
            children = psutil.Process(process.pid).children(recursive=True) if process.poll() is None else []
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for child in children:
                try:
                    if child.is_running():
                        child.terminate()
                except psutil.NoSuchProcess:
                    pass


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)),
        **{f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)},
    }


def process_rss(pid: int) -> int:
    try:
        parent = psutil.Process(pid)
        return sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


async def load(url: str, token: str, queries: list[tuple[str, str]], concurrency: int, requests: int, pid: int) -> dict:
    headers = {"Authorization": "Bearer " + token}
    memory = []
    stop = asyncio.Event()

    async def monitor():
        while not stop.is_set():
            memory.append(process_rss(pid))
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=url, headers=headers, limits=limits, timeout=30, trust_env=False) as client:
        # Real HTTP warmup excluded from measured requests. No query-result/embedding cache exists.
        await asyncio.gather(
            *[
                client.post("/retrieve", json={"query": queries[i % len(queries)][1], "k": 10})
                for i in range(max(16, concurrency))
            ]
        )
        monitor_task = asyncio.create_task(monitor())
        next_request = 0

        async def worker():
            nonlocal next_request
            while next_request < requests:
                index = next_request
                next_request += 1
                qid, text = queries[index % len(queries)]
                started = time.perf_counter()
                record: dict[str, Any] = {"request": index, "query_id": qid}
                try:
                    response = await client.post("/retrieve", json={"query": text, "k": 10})
                    elapsed = (time.perf_counter() - started) * 1000
                    record.update(client_ms=elapsed, status=response.status_code)
                    if response.status_code == 200:
                        body = response.json()
                        server_ms = float(response.headers["X-EvalForge-Server-Ms"])
                        record.update(
                            stages=body["timings_ms"],
                            server_to_headers_ms=server_ms,
                            framework_and_queue_ms=server_ms - body["timings_ms"]["handler_total"],
                            transport_framework_client_residual_ms=elapsed - server_ms,
                            ranked_ids=[item["id"] for item in body["results"]],
                            snapshot_version=body["snapshot_version"],
                            pid=body["pid"],
                        )
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    record.update(
                        client_ms=(time.perf_counter() - started) * 1000, status=0, error_type=type(exc).__name__
                    )
                records.append(record)

        started = time.perf_counter()
        try:
            await asyncio.gather(*[worker() for _ in range(concurrency)])
        finally:
            duration = time.perf_counter() - started
            stop.set()
            await monitor_task
        control_samples = []
        control_errors = []
        semaphore = asyncio.Semaphore(concurrency)

        async def ping_control():
            async with semaphore:
                start = time.perf_counter()
                try:
                    response = await client.get("/ping")
                    if response.status_code != 200:
                        control_errors.append(response.status_code)
                    else:
                        control_samples.append((time.perf_counter() - start) * 1000)
                except httpx.HTTPError:
                    control_errors.append(0)

        await asyncio.gather(*[ping_control() for _ in range(100)])
    successes = [r for r in records if r["status"] == 200 and "stages" in r]
    return {
        "concurrency": concurrency,
        "requests": requests,
        "successful": len(successes),
        "errors": requests - len(successes),
        "error_rate": (requests - len(successes)) / requests,
        "duration_seconds": duration,
        "completed_requests_per_second": len(records) / duration,
        "successful_requests_per_second": len(successes) / duration,
        "client_end_to_end": percentiles([r["client_ms"] for r in records]),
        "successful_client_end_to_end": percentiles([r["client_ms"] for r in successes]),
        "stages": {
            name: percentiles([r["stages"][name] for r in successes])
            for name in ("embedding", "embedding_wait", "faiss_lookup", "processing", "handler_total")
        },
        "server_to_headers": percentiles([r["server_to_headers_ms"] for r in successes]),
        "framework_and_queue": percentiles(
            [r["server_to_headers_ms"] - r["stages"]["handler_total"] for r in successes]
        ),
        "transport_framework_client_residual": percentiles(
            [r["transport_framework_client_residual_ms"] for r in successes]
        ),
        "post_load_empty_http_control": {
            "latency": percentiles(control_samples),
            "errors": len(control_errors),
            "scope": "same-host HTTP stack control; not a pure network measurement",
        },
        "memory": {
            "sampled_peak_sum_rss_bytes": max(memory, default=0),
            "sample_interval_ms": 50,
            "scope": "server parent plus children summed RSS; shared pages can be double-counted; not client RSS/PSS",
        },
        "worker_pids_observed": sorted({r["pid"] for r in successes}),
        "records": sorted(records, key=lambda r: r["request"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument("--encoder", choices=["minilm", "test-hash"], default="minilm")
    args = parser.parse_args()
    if args.requests < 1 or any(n < 1 for n in args.workers + args.concurrency):
        raise ValueError("Positive load settings required")
    if args.output.exists():
        raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    data = load_dataset(args.manifest)
    queries = list(data.queries.items())
    from serving.snapshots import SnapshotRepository

    current_version = (args.snapshots / "CURRENT").read_text().strip()
    if len(current_version) != 32 or any(c not in "0123456789abcdef" for c in current_version):
        raise ValueError("Invalid snapshot pointer")
    manifest = json.loads((args.snapshots / "versions" / current_version / "manifest.json").read_text())
    snapshot = SnapshotRepository(args.snapshots, manifest["model"]).latest()
    if snapshot is None or {c.id: c.content for c in snapshot.chunks} != data.corpus:
        raise ValueError("Serving snapshot does not match the benchmark corpus")
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": timestamp(),
        "dataset": data.identity(),
        "snapshot_manifest": manifest,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("httpx", "uvicorn", "fastapi", "faiss-cpu", "torch", "sentence-transformers")
            },
        },
        "code": provenance(),
        "scope": (
            "HTTP loopback closed-loop retrieval including text embedding; "
            "excludes ingestion, index build and startup"
        ),
        "limitations": [
            "Residual combines transport, scheduling and client overhead; it is not pure network latency",
            "Loopback single-host burst test, not WAN latency, open-loop saturation or production capacity",
            "Per-process encoder is serialized; embedding_wait measures contention; no LLM call",
            "Raw samples retained; repeated requests are not independent relevance judgments",
        ],
        "results": [],
    }
    for workers in args.workers:
        with running_server(args.snapshots, args.encoder, workers, args.output / f"server-w{workers}.log") as (
            url,
            token,
            pid,
        ):
            with httpx.Client(trust_env=False) as client:
                report["service"] = client.get(url + "/health").json()
            for concurrency in args.concurrency:
                result = asyncio.run(load(url, token, queries, concurrency, args.requests, pid))
                result["workers"] = workers
                # One observed ranking/query for HTTP quality; do not count timing repetitions as new judgments.
                unique = {r["query_id"]: r for r in result["records"] if "ranked_ids" in r}
                quality = [evaluate_ranking(r["ranked_ids"], data.qrels[qid], 10) for qid, r in unique.items()]
                valid = [s for s in quality if s is not None]
                result["quality_query_count"] = len(valid)
                result["quality"] = {m: float(np.mean([q[m] for q in valid])) for m in valid[0]} if valid else None
                report["results"].append(result)
                write_json(args.output / "report.json", report)
                print(
                    workers,
                    concurrency,
                    result["successful_requests_per_second"],
                    result["client_end_to_end"],
                    flush=True,
                )
    report["completed_at"] = timestamp()
    report["status"] = "COMPLETE"
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
