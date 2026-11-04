"""Build/run a disposable retrieval container, verify workers, update CAS and volume restart."""

import argparse
import asyncio
import secrets
import shutil
import socket
import subprocess  # nosec B404 - fixed Docker CLI operations on owned test resources
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from evaluation.serving_checks import check_restart, check_service
from datetime import datetime, timezone
import json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/docker-integration.json"))
    args = parser.parse_args()
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("Docker is unavailable; no container verification result claimed")

    def run(*arguments):
        result = subprocess.run([docker, *arguments], check=True, capture_output=True, text=True)  # nosec B603
        return result.stdout.strip()

    suffix = uuid.uuid4().hex[:10]
    name = "evalforge-check-" + suffix
    volume = "evalforge-volume-" + suffix
    image = "evalforge-retrieval-test:" + suffix
    token = secrets.token_urlsafe(24)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    report: dict[str, Any] = {
        "status": "RUNNING",
        "started_at": timestamp(),
        "encoder": "synthetic-test-word-hash",
        "scope": "actual Docker HTTP/FAISS snapshot integration, not semantic model performance",
    }
    try:
        run("build", "-f", "Dockerfile.retrieval", "-t", image, ".")
        run("volume", "create", volume)
        run(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:8000",
            "--mount",
            f"type=volume,source={volume},target=/data",
            "-e",
            "EVALFORGE_EMBEDDER=test-hash",
            "-e",
            "RAG_API_TOKEN=" + token,
            image,
        )

        def ready():
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                try:
                    if httpx.get(url + "/health", timeout=1, trust_env=False).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
            raise RuntimeError("Container readiness timed out")

        ready()
        report["checks"] = asyncio.run(check_service(url, token))
        run("restart", name)
        ready()
        report["restart"] = asyncio.run(check_restart(url, token, report["checks"]))
        report["status"] = "COMPLETE"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error_type"] = type(exc).__name__
        if isinstance(exc, subprocess.CalledProcessError):
            # Build diagnostics contain no runtime token; never print docker run arguments.
            if exc.cmd[1] == "build":
                print(exc.stderr[-12000:])
        raise
    finally:
        report["completed_at"] = timestamp()
        write_json(args.output, report)
        # Only names created by this invocation are removed; no broad prune commands.
        for command in [("rm", "-f", name), ("volume", "rm", volume), ("image", "rm", image)]:
            subprocess.run([docker, *command], capture_output=True, check=False)  # nosec B603
    print(args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
