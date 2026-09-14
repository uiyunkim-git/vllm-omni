"""Integration fixtures: central's control plane against a mock worker agent.

The Rust router is gone, so there is no binary to run in docker any more. What
is worth exercising end to end is the path central actually drives:

    central/manager.py  --HTTP-->  worker agent  --starts-->  dynamo instance
                        <--health check on the instance's system port--

`tests/mock_worker.py` implements both halves (agent API + the instance's
system port), so these tests need no docker, no GPU and no external network —
only loopback sockets.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

TESTS_DIR = Path(__file__).resolve().parents[1]


def find_free_port() -> int:
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
        time.sleep(0.1)
    raise TimeoutError(f"{what} not ready after {timeout_s}s ({url}): {last_err}")


class MockAgent:
    """Handle on a mock worker agent subprocess."""

    def __init__(self, proc: subprocess.Popen, port: int, worker_id: str):
        self.proc = proc
        self.port = port
        self.worker_id = worker_id
        self.host = "127.0.0.1"
        self.url = f"http://127.0.0.1:{port}"

    def state(self) -> dict:
        return httpx.get(f"{self.url}/_state", timeout=5.0).json()

    def deploy_requests(self) -> list:
        return self.state()["deploy_requests"]

    def configure(self, **patch) -> None:
        httpx.post(f"{self.url}/_config", json=patch, timeout=5.0).raise_for_status()

    def reset(self) -> None:
        httpx.post(f"{self.url}/_reset", timeout=5.0).raise_for_status()

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture
def mock_agent_factory():
    agents: list = []

    def _spawn(worker_id: str = "mock-worker", port_start: int = 21001) -> MockAgent:
        port = find_free_port()
        cmd = [
            sys.executable, str(TESTS_DIR / "mock_worker.py"),
            "--host", "127.0.0.1", "--port", str(port),
            "--worker-id", worker_id, "--port-start", str(port_start),
        ]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, env=env)
        agent = MockAgent(proc, port, worker_id)
        agents.append(agent)
        _wait_http_ok(f"{agent.url}/_state", 30.0, f"mock worker agent :{port}")
        return agent

    yield _spawn
    for a in agents:
        a.stop()


@pytest.fixture
def mock_agent(mock_agent_factory) -> MockAgent:
    return mock_agent_factory()


@pytest.fixture
def registered_worker(central_manager, mock_agent):
    """An accepted worker in central's DB pointing at the mock agent."""
    central_manager.register_worker(
        mock_agent.worker_id, mock_agent.host, mock_agent.port,
        [{"id": 0, "name": "MockGPU", "utilization": 0,
          "memory_used": 0, "memory_total": 81920}],
    )
    central_manager.accept_worker(mock_agent.worker_id, mock_agent.worker_id)
    return mock_agent
