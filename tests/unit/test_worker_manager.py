"""Unit tests for worker/manager.py (WorkerManager) with docker/subprocess mocked.

Covers:
- port allocation: skipping ports used by prior deployments AND ports parsed
  from `docker ps` (including the host = internal + 40000 offset mapping);
- "address already in use" retry loop advancing to the next free port;
- concurrent deploys serialized by _state_lock (distinct ports, both records
  persisted to local_deployments.json);
- stop_replica replica-id reconstruction for worker ids that themselves
  contain '-' (e.g. "kbds-worker-4").
"""

import json
import subprocess as real_subprocess
import threading
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"


# ---------------------------------------------------------------------------
# Fake subprocess plumbing
# ---------------------------------------------------------------------------

class FakeSubprocess:
    """Drop-in for the `subprocess` module inside worker/manager.py.

    Records every command; behaviour per command family is configurable.
    """

    DEVNULL = real_subprocess.DEVNULL
    PIPE = real_subprocess.PIPE
    CompletedProcess = real_subprocess.CompletedProcess

    def __init__(self, docker_ps_output: str = "", compose_up_hook=None):
        self.commands = []
        self.docker_ps_output = docker_ps_output
        # compose_up_hook(cmd, attempt_index) -> CompletedProcess, lets tests
        # simulate port-conflict failures.
        self.compose_up_hook = compose_up_hook
        self._compose_attempts = 0
        self._lock = threading.Lock()

    def _ok(self, cmd):
        return real_subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def run(self, cmd, **kwargs):
        with self._lock:
            self.commands.append(list(cmd))
        if cmd[:2] == ["docker", "compose"] and "up" in cmd:
            with self._lock:
                attempt = self._compose_attempts
                self._compose_attempts += 1
            if self.compose_up_hook is not None:
                return self.compose_up_hook(cmd, attempt)
            return self._ok(cmd)
        # docker image inspect / docker rm -f / openssl / docker compose down
        return self._ok(cmd)

    def check_output(self, cmd, **kwargs):
        with self._lock:
            self.commands.append(list(cmd))
        if cmd[:2] == ["docker", "ps"]:
            return self.docker_ps_output
        return ""


@pytest.fixture
def make_manager(worker_manager_module, tmp_path, monkeypatch):
    """Factory: WorkerManager wired to tmp_path DATA_DIR + FakeSubprocess."""
    wm = worker_manager_module

    def _make(docker_ps_output: str = "", compose_up_hook=None):
        fake = FakeSubprocess(docker_ps_output, compose_up_hook)
        monkeypatch.setattr(wm, "DATA_DIR", str(tmp_path))
        monkeypatch.setattr(wm, "subprocess", fake)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "unit-test-token")
        mgr = wm.WorkerManager()
        # Point at the real production templates (constructor hardcodes
        # /app/templates) so rendering is exercised end-to-end.
        mgr.env = Environment(loader=FileSystemLoader(str(WORKER_DIR / "templates")))
        return mgr, fake, wm

    return _make


def _deploy_req(deploy_id="dep1", replica_id="dep1_w1_0", **overrides):
    req = {
        "deploy_id": deploy_id,
        "replica_id": replica_id,
        "name": "test",
        "model": "org/test-model",
        "served_model_name": "test-model",
        "is_embedding": False,
        "engine": "vllm",
        "gpus": [0],
        "tp": 1,
        "max_len": 4096,
        "gpu_util": 0.9,
        "extra_args": None,
        "vllm_image": None,
    }
    req.update(overrides)
    return req


def _load_state(wm, tmp_path):
    with open(tmp_path / "local_deployments.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------

class TestPortAllocation:
    def test_first_deploy_gets_port_start(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        dep = mgr.deploy_model(_deploy_req())
        assert dep["ports"] == [wm._PORT_START]  # 21001

    def test_skips_ports_of_existing_deployments(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        (tmp_path / "local_deployments.json").write_text(json.dumps([
            {"id": "old", "replica_id": "old_w_0",
             "ports": [wm._PORT_START, wm._PORT_START + 1], "nodes": []},
        ]))
        dep = mgr.deploy_model(_deploy_req())
        assert dep["ports"] == [wm._PORT_START + 2]

    def test_skips_ports_seen_in_docker_ps_including_40000_offset(
        self, make_manager, tmp_path
    ):
        # docker ps shows a container bound to host port 61001 -> internal
        # 21001 must be considered used (61001 - 40000), and one bound
        # directly to 21002 blocks that too.
        ps = (
            "0.0.0.0:61001->8000/tcp, :::61001->8000/tcp\n"
            "0.0.0.0:21002->8000/tcp\n"
        )
        mgr, fake, wm = make_manager(docker_ps_output=ps)
        dep = mgr.deploy_model(_deploy_req())
        assert dep["ports"] == [wm._PORT_START + 2]  # 21003

    def test_docker_ps_failure_is_nonfatal(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()

        def boom(cmd, **kw):
            raise RuntimeError("docker daemon down")

        fake.check_output = boom
        dep = mgr.deploy_model(_deploy_req())
        assert dep["ports"] == [wm._PORT_START]

    def test_address_in_use_retries_on_next_port(self, make_manager, tmp_path):
        def hook(cmd, attempt):
            if attempt == 0:
                return real_subprocess.CompletedProcess(
                    cmd, 1, stdout="", stderr="Error: address already in use"
                )
            return real_subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        mgr, fake, wm = make_manager(compose_up_hook=hook)
        dep = mgr.deploy_model(_deploy_req())
        # nodes[] carries the port actually used after the retry
        assert dep["nodes"][0]["port"] == wm._PORT_START + 1
        # ports[] must record the port ACTUALLY bound after the retry, not the
        # originally allocated one — otherwise future allocations reserve the
        # wrong port (fixed in _deploy_model_locked).
        assert dep["ports"] == [wm._PORT_START + 1]

    def test_non_port_error_fails_immediately_and_cleans_up(
        self, make_manager, tmp_path
    ):
        def hook(cmd, attempt):
            return real_subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="no space left on device"
            )

        mgr, fake, wm = make_manager(compose_up_hook=hook)
        with pytest.raises(RuntimeError, match="docker compose failed"):
            mgr.deploy_model(_deploy_req())
        # only one compose-up attempt, and no record persisted
        ups = [c for c in fake.commands if c[:2] == ["docker", "compose"] and "up" in c]
        assert len(ups) == 1
        assert not (tmp_path / "local_deployments.json").exists()

    def test_missing_image_rejected_before_any_side_effect(
        self, make_manager, tmp_path
    ):
        mgr, fake, wm = make_manager()

        def run(cmd, **kw):
            fake.commands.append(list(cmd))
            if cmd[:3] == ["docker", "image", "inspect"]:
                return real_subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such image")
            return real_subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        fake.run = run
        with pytest.raises(Exception, match="not found on this worker"):
            mgr.deploy_model(_deploy_req(vllm_image="vllm/vllm-openai:v0.99"))
        assert all("compose" not in c for c in fake.commands)


# ---------------------------------------------------------------------------
# Concurrency: _state_lock serialization
# ---------------------------------------------------------------------------

class TestConcurrentDeploys:
    def test_two_concurrent_deploys_get_distinct_ports_and_both_persist(
        self, make_manager, tmp_path
    ):
        import time as _time

        def slow_up(cmd, attempt):
            _time.sleep(0.05)  # widen the race window inside the locked section
            return real_subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        mgr, fake, wm = make_manager(compose_up_hook=slow_up)

        results, errors = [], []

        def deploy(replica):
            try:
                results.append(
                    mgr.deploy_model(_deploy_req(replica_id=f"dep1_w1_{replica}"))
                )
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=deploy, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        ports = sorted(p for r in results for p in r["ports"])
        assert ports == [wm._PORT_START, wm._PORT_START + 1], (
            "concurrent deploys must not double-allocate a port"
        )
        state = _load_state(wm, tmp_path)
        assert len(state) == 2, "second save must not clobber the first record"


# ---------------------------------------------------------------------------
# stop_replica replica-id reconstruction
# ---------------------------------------------------------------------------

class TestStopReplica:
    def _seed(self, tmp_path, replica_id, deploy_id="dep1"):
        (tmp_path / "local_deployments.json").write_text(json.dumps([
            {"id": deploy_id, "replica_id": replica_id,
             "ports": [21001], "nodes": [{"name": f"vllm_{replica_id}", "port": 21001}]},
        ]))
        # stop paths only run `docker compose down` when the compose file exists
        run_dir = tmp_path / f"run_{replica_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "docker-compose.yml").write_text("# seeded by test\n")

    def test_simple_worker_id(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        self._seed(tmp_path, "dep1_worker1_0")
        assert mgr.stop_replica("dep1", "worker1-0") is True
        assert _load_state(wm, tmp_path) == []

    def test_worker_id_containing_dashes(self, make_manager, tmp_path):
        """global_gpu_id 'kbds-worker-4-0' must split as
        wid='kbds-worker-4', gid='0' (rsplit on the LAST dash)."""
        mgr, fake, wm = make_manager()
        self._seed(tmp_path, "dep1_kbds-worker-4_0")
        assert mgr.stop_replica("dep1", "kbds-worker-4-0") is True
        assert _load_state(wm, tmp_path) == []
        # compose down targeted the right project
        downs = [c for c in fake.commands if "down" in c]
        assert any("vllm_dep1_kbds-worker-4_0" in c for c in downs)

    def test_multi_digit_gpu_index(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        self._seed(tmp_path, "dep1_node-a_12")
        assert mgr.stop_replica("dep1", "node-a-12") is True

    def test_unknown_replica_returns_false(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        self._seed(tmp_path, "dep1_worker1_0")
        assert mgr.stop_replica("dep1", "worker1-1") is False
        assert len(_load_state(wm, tmp_path)) == 1

    def test_wrong_deploy_id_returns_false(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        self._seed(tmp_path, "dep1_worker1_0")
        assert mgr.stop_replica("dep2", "worker1-0") is False


# ---------------------------------------------------------------------------
# stop_deployment
# ---------------------------------------------------------------------------

class TestStopDeployment:
    def test_stops_all_replicas_of_a_deploy(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        (tmp_path / "local_deployments.json").write_text(json.dumps([
            {"id": "depA", "replica_id": "depA_w_0", "ports": [21001], "nodes": []},
            {"id": "depA", "replica_id": "depA_w_1", "ports": [21002], "nodes": []},
            {"id": "depB", "replica_id": "depB_w_0", "ports": [21003], "nodes": []},
        ]))
        assert mgr.stop_deployment("depA") is True
        state = _load_state(wm, tmp_path)
        assert [d["id"] for d in state] == ["depB"]

    def test_unknown_deploy_returns_false(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        assert mgr.stop_deployment("ghost") is False
