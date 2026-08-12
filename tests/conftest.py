"""Shared fixtures for the vllm-omni test suite.

The production modules were written to run inside containers and have
module-level side effects, so importing them for unit testing needs care:

* ``central/main.py``
    - does ``from manager import CentralManager`` and instantiates it at import
      time; the real CentralManager opens sqlite at /app/data. We stub the
      ``manager`` module in sys.modules for the duration of the import.
    - mounts ``frontend/static`` relative to CWD, so we chdir into central/
      while executing the module.

* ``worker/manager.py``
    - calls ``os.makedirs("/app/data")`` at import time, which fails on a dev
      host. We patch os.makedirs during the import only.
"""

import contextlib
import importlib.util
import os
import socket
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CENTRAL_DIR = REPO_ROOT / "central"
WORKER_DIR = REPO_ROOT / "worker"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


# ---------------------------------------------------------------------------
# central/main.py under test
# ---------------------------------------------------------------------------

class FakeCentralManager:
    """Minimal stand-in for central/manager.py:CentralManager.

    Only what central/main.py's module level + the pure functions we unit-test
    touch. Deployments/workers are settable per test.
    """

    def __init__(self):
        self.deployments = []
        self.workers = {}

    def load_deployments(self):
        return self.deployments

    def get_workers(self):
        return self.workers

    # Referenced by endpoints we don't unit test; keep harmless no-ops.
    def register_worker(self, *a, **k):
        pass

    async def run_health_checks(self):
        pass

    async def sync_p2c_workers(self):
        pass


def _load_central_main() -> types.ModuleType:
    fake_manager_mod = types.ModuleType("manager")
    fake_manager_mod.CentralManager = FakeCentralManager

    # Keep the module from pointing at the production router by default.
    os.environ.setdefault("ROUTER_METRICS_URL", "http://127.0.0.1:1/metrics")
    os.environ.setdefault("ROUTER_WORKERS_URL", "http://127.0.0.1:1/workers")

    spec = importlib.util.spec_from_file_location(
        "central_main_under_test", CENTRAL_DIR / "main.py"
    )
    module = importlib.util.module_from_spec(spec)

    saved_manager = sys.modules.get("manager")
    sys.modules["manager"] = fake_manager_mod
    sys.modules[spec.name] = module
    cwd = os.getcwd()
    try:
        os.chdir(CENTRAL_DIR)  # Jinja2Templates("frontend") etc. are CWD-relative
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
        if saved_manager is not None:
            sys.modules["manager"] = saved_manager
        else:
            sys.modules.pop("manager", None)
    return module


@pytest.fixture(scope="session")
def central_main():
    """The imported central/main.py module (session-scoped; state reset by
    the autouse fixture below)."""
    return _load_central_main()


@pytest.fixture(autouse=True)
def _reset_central_state(request):
    """Clear central/main.py module-global metric state between tests."""
    yield
    mod = sys.modules.get("central_main_under_test")
    if mod is not None:
        mod._metric_history.clear()
        mod._latest_scrape.clear()
        mod._last_live_instances.clear()
        mod.manager.deployments = []
        mod.manager.workers = {}


# ---------------------------------------------------------------------------
# worker/manager.py under test
# ---------------------------------------------------------------------------

def _load_worker_manager() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "worker_manager_under_test", WORKER_DIR / "manager.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # worker/manager.py runs os.makedirs("/app/data") at import time.
    with mock.patch("os.makedirs"):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def worker_manager_module():
    return _load_worker_manager()
