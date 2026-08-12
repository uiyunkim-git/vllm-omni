"""Integration fixtures: real router binary (in docker) + host-run mock workers.

The router runs from the image named by ROUTER_IMAGE (default
"vllm-router-check:audit"), which contains the built binary at
/app/target/release/vllm-router. The container uses --network host so the
router can reach mock workers bound to 127.0.0.1 on the host — this is why
these tests are Linux-only.

Everything here skips cleanly (pytest.skip with a reason) when docker or the
image is unavailable, so `./run_tests.sh unit` and even a plain
`./run_tests.sh` stay green on machines without docker.
"""

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

TESTS_DIR = Path(__file__).resolve().parents[1]
ROUTER_IMAGE = os.environ.get("ROUTER_IMAGE", "vllm-router-check:audit")
ROUTER_BIN = "/app/target/release/vllm-router"


def _docker_unavailable_reason() -> str | None:
    if sys.platform != "linux":
        return "router integration tests need --network host (Linux only)"
    if shutil.which("docker") is None:
        return "docker CLI not found"
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=15
        )
    except Exception as e:
        return f"docker info failed: {e}"
    if r.returncode != 0:
        return "docker daemon not reachable"
    r = subprocess.run(
        ["docker", "image", "inspect", ROUTER_IMAGE], capture_output=True, timeout=15
    )
    if r.returncode != 0:
        return (
            f"router image '{ROUTER_IMAGE}' not found - build it "
            "(see tests/README.md) or set ROUTER_IMAGE"
        )
    return None


_SKIP_REASON = _docker_unavailable_reason()


@pytest.fixture(scope="session")
def docker_router_env():
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)
    return ROUTER_IMAGE


def find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http_ok(url: str, timeout_s: float, what: str) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(0.15)
    raise TimeoutError(f"{what} not ready after {timeout_s}s ({url}): {last_err}")


# ---------------------------------------------------------------------------
# Mock worker handle
# ---------------------------------------------------------------------------

class MockWorker:
    def __init__(self, proc: subprocess.Popen, port: int):
        self.proc = proc
        self.port = port
        self.url = f"http://127.0.0.1:{port}"

    def stats(self) -> dict:
        return httpx.get(f"{self.url}/_stats", timeout=5.0).json()

    def reset(self) -> None:
        httpx.post(f"{self.url}/_reset", timeout=5.0).raise_for_status()

    def configure(self, **kwargs) -> None:
        httpx.post(f"{self.url}/_config", json=kwargs, timeout=5.0).raise_for_status()

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture
def mock_worker_factory():
    """Factory fixture: spawn mock workers as host subprocesses."""
    workers: list[MockWorker] = []

    def _spawn(
        model: str = "mock-model",
        delay_ms: int = 0,
        fail_mode: str = "",
        fail_n: int = -1,
        stream_chunks: int = 3,
        interchunk_ms: int = 10,
    ) -> MockWorker:
        port = find_free_port()
        cmd = [
            sys.executable,
            str(TESTS_DIR / "mock_worker.py"),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--model", model,
            "--delay-ms", str(delay_ms),
            "--fail-n", str(fail_n),
            "--stream-chunks", str(stream_chunks),
            "--interchunk-ms", str(interchunk_ms),
        ]
        if fail_mode:
            cmd += ["--fail-mode", fail_mode]
        proc = subprocess.Popen(cmd)
        w = MockWorker(proc, port)
        workers.append(w)
        _wait_http_ok(f"{w.url}/health", 20.0, f"mock worker :{port}")
        return w

    yield _spawn
    for w in workers:
        w.stop()


# ---------------------------------------------------------------------------
# Router-in-docker handle
# ---------------------------------------------------------------------------

class RouterHandle:
    def __init__(self, container: str, port: int, prom_port: int):
        self.container = container
        self.port = port
        self.prom_port = prom_port
        self.url = f"http://127.0.0.1:{port}"
        self.metrics_url = f"http://127.0.0.1:{prom_port}/metrics"

    # --- admin helpers -----------------------------------------------------
    def add_worker(self, worker_url: str) -> None:
        r = httpx.post(
            f"{self.url}/add_worker", params={"url": worker_url}, timeout=30.0
        )
        assert r.status_code == 200, f"add_worker({worker_url}): {r.status_code} {r.text}"

    def remove_worker(self, worker_url: str) -> None:
        r = httpx.post(
            f"{self.url}/remove_worker", params={"url": worker_url}, timeout=10.0
        )
        assert r.status_code == 200, f"remove_worker: {r.status_code} {r.text}"

    def workers(self) -> dict:
        r = httpx.get(f"{self.url}/workers", timeout=10.0)
        assert r.status_code == 200, r.text
        return r.json()

    def list_worker_urls(self) -> list:
        r = httpx.get(f"{self.url}/list_workers", timeout=10.0)
        assert r.status_code == 200, r.text
        return r.json().get("urls", [])

    def metrics(self) -> str:
        return httpx.get(self.metrics_url, timeout=10.0).text

    # --- inference helpers ---------------------------------------------------
    def chat(self, model: str, stream: bool = False, timeout: float = 30.0,
             client: httpx.Client | None = None) -> httpx.Response:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        }
        if client is not None:
            return client.post(f"{self.url}/v1/chat/completions", json=payload,
                               timeout=timeout)
        return httpx.post(f"{self.url}/v1/chat/completions", json=payload,
                          timeout=timeout)

    def logs(self, tail: int = 100) -> str:
        r = subprocess.run(
            ["docker", "logs", "--tail", str(tail), self.container],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout + r.stderr

    def stop(self) -> None:
        subprocess.run(
            ["docker", "rm", "-f", self.container],
            capture_output=True, timeout=30,
        )


@pytest.fixture
def router_factory(docker_router_env):
    """Factory fixture: launch the real router binary in docker (host network)."""
    handles: list[RouterHandle] = []

    def _spawn(policy: str = "least_connections", extra_args: list | None = None,
               worker_urls: list | None = None) -> RouterHandle:
        port = find_free_port()
        prom_port = find_free_port()
        name = f"vllm-router-test-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker", "run", "-d", "--name", name,
            "--network", "host",
            "--entrypoint", ROUTER_BIN,
            docker_router_env,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--prometheus-host", "127.0.0.1",
            "--prometheus-port", str(prom_port),
            "--policy", policy,
            "--worker-startup-timeout-secs", "20",
            "--worker-startup-check-interval", "1",
            "--request-timeout-secs", "60",
        ]
        if worker_urls:
            cmd += ["--worker-urls", *worker_urls]
        if extra_args:
            cmd += [str(a) for a in extra_args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, f"docker run failed: {r.stderr}"
        handle = RouterHandle(name, port, prom_port)
        handles.append(handle)
        try:
            _wait_http_ok(f"{handle.url}/liveness", 30.0, "router")
        except TimeoutError:
            print("---- router logs ----")
            print(handle.logs())
            raise
        return handle

    yield _spawn
    for h in handles:
        h.stop()
