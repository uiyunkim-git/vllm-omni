"""Shared fixtures for the vllm-omni test suite (Dynamo-native platform).

The production modules run inside containers and have module-level side
effects, so importing them for unit testing needs care:

* ``central/dynamo.py``
    - pure; reads a few env vars at import time. Loaded first and registered as
      ``sys.modules["dynamo"]`` so ``central/main.py`` and ``central/manager.py``
      import the very same object (tests can monkeypatch its constants).

* ``central/db.py``
    - runs ``init_db()`` at import, which does ``os.makedirs("/app/data")`` and
      opens sqlite there. We execute it with ``os.makedirs``/``sqlite3.connect``
      patched, then repoint ``DB_PATH`` at a per-test tmp file (the module reads
      the constant on every ``get_db()`` call, so that is all it takes).

* ``central/manager.py``
    - ``os.makedirs("/app/data")`` at import; needs ``dynamo`` and ``db`` to be
      importable by bare name.

* ``central/main.py``
    - instantiates ``CentralManager`` at import time; we stub the ``manager``
      module with :class:`FakeCentralManager` for the duration of the import,
      and chdir into central/ because Jinja2Templates/StaticFiles are
      CWD-relative.

* ``worker/manager.py``
    - ``os.makedirs("/app/data")`` at import time; patched away.

Nothing here touches docker, etcd, the Dynamo frontend or any network socket.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sqlite3
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CENTRAL_DIR = REPO_ROOT / "central"
WORKER_DIR = REPO_ROOT / "worker"
FIXTURES_DIR = TESTS_DIR / "fixtures"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def read_fixture(name: str) -> str:
    """Contents of tests/fixtures/<name> (Prometheus exposition samples)."""
    return (FIXTURES_DIR / name).read_text()


@pytest.fixture(scope="session")
def frontend_metrics_text() -> str:
    return read_fixture("frontend_metrics.txt")


@pytest.fixture(scope="session")
def instance_metrics_text() -> str:
    return read_fixture("instance_metrics.txt")


# ---------------------------------------------------------------------------
# module loading helpers
# ---------------------------------------------------------------------------

def _exec_module(name: str, path: Path, patches=()) -> types.ModuleType:
    """Import `path` as module `name`, registering it in sys.modules first so
    the production modules' bare `import dynamo` / `import db` resolve to it."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    ctxs = [p for p in patches]
    try:
        for c in ctxs:
            c.__enter__()
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        for c in reversed(ctxs):
            c.__exit__(None, None, None)
    return module


# ---------------------------------------------------------------------------
# central/dynamo.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def central_dynamo() -> types.ModuleType:
    """central/dynamo.py — the single source of truth for the data plane."""
    if "dynamo" in sys.modules and getattr(sys.modules["dynamo"], "__file__", "") == str(
        CENTRAL_DIR / "dynamo.py"
    ):
        return sys.modules["dynamo"]
    return _exec_module("dynamo", CENTRAL_DIR / "dynamo.py")


# ---------------------------------------------------------------------------
# central/db.py + central/manager.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _central_db_module(tmp_path_factory) -> types.ModuleType:
    boot_db = tmp_path_factory.mktemp("central-db-boot") / "omni.db"
    real_connect = sqlite3.connect
    patches = [
        mock.patch("os.makedirs"),
        mock.patch(
            "sqlite3.connect",
            side_effect=lambda _path, **kw: real_connect(str(boot_db), **kw),
        ),
    ]
    return _exec_module("db", CENTRAL_DIR / "db.py", patches)


@pytest.fixture
def central_db(_central_db_module, tmp_path) -> types.ModuleType:
    """central/db.py pointed at a fresh sqlite file for this test."""
    db = _central_db_module
    db.DB_DIR = str(tmp_path)
    db.DB_PATH = str(tmp_path / "omni.db")
    db.init_db()
    return db


@pytest.fixture(scope="session")
def central_manager_module(central_dynamo, _central_db_module) -> types.ModuleType:
    return _exec_module(
        "manager", CENTRAL_DIR / "manager.py", [mock.patch("os.makedirs")]
    )


@pytest.fixture
def central_manager(central_manager_module, central_db):
    """A CentralManager backed by a per-test temp sqlite database."""
    return central_manager_module.CentralManager()


# ---------------------------------------------------------------------------
# central/main.py
# ---------------------------------------------------------------------------

class FakeCentralManager:
    """Stand-in for central/manager.py:CentralManager.

    central/main.py instantiates a manager at import time; the metrics code we
    unit-test only ever calls ``load_deployments()`` (and ``get_workers()`` on
    paths we do not exercise). Deployments are settable per test.
    """

    def __init__(self):
        self.deployments: list = []
        self.workers: dict = {}

    def load_deployments(self):
        return self.deployments

    def get_workers(self):
        return self.workers

    def register_worker(self, *a, **k):
        pass

    async def run_health_checks(self):
        pass

    async def auto_update_loop(self):
        pass


@pytest.fixture(scope="session")
def central_main(central_dynamo) -> types.ModuleType:
    """central/main.py imported with a fake manager (no sqlite, no docker)."""
    fake_manager_mod = types.ModuleType("manager")
    fake_manager_mod.CentralManager = FakeCentralManager
    # main.py imports this constant to authenticate its worker-proxy calls; the
    # fake module has to mirror manager's public surface.
    fake_manager_mod.WORKER_HEADERS = {}

    saved_manager = sys.modules.get("manager")
    sys.modules["manager"] = fake_manager_mod
    cwd = os.getcwd()
    try:
        os.chdir(CENTRAL_DIR)  # Jinja2Templates("frontend") / StaticFiles are CWD-relative
        module = _exec_module("central_main_under_test", CENTRAL_DIR / "main.py")
    finally:
        os.chdir(cwd)
        if saved_manager is not None:
            sys.modules["manager"] = saved_manager
        else:
            sys.modules.pop("manager", None)
    return module


@pytest.fixture(autouse=True)
def _reset_central_main_state():
    """Clear central/main.py module-global metric state between tests."""
    yield
    mod = sys.modules.get("central_main_under_test")
    if mod is not None:
        mod._metric_history.clear()
        mod._latest_scrape.clear()
        mod.manager.deployments = []
        mod.manager.workers = {}


# ---------------------------------------------------------------------------
# worker/manager.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def worker_manager_module() -> types.ModuleType:
    return _exec_module(
        "worker_manager_under_test", WORKER_DIR / "manager.py", [mock.patch("os.makedirs")]
    )


@pytest.fixture(scope="session")
def worker_templates_dir() -> Path:
    return WORKER_DIR / "templates"
