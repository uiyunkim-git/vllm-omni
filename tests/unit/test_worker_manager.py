"""Unit tests for worker/manager.py (WorkerManager) with docker/subprocess mocked.

Covers the Dynamo-native deploy path:
- engine selection: default "dynamo", legacy "vllm", anything else rejected;
- the four derived ports (system/rpc/response-stream/kv) reaching both the
  rendered compose file and the host firewall rules;
- TLS material generated ONLY for the legacy vLLM engine;
- reapply_host_ports re-asserting firewall rules for dynamo deployments only;
- port allocation (existing records + `docker ps`, incl. the +40000 host
  mapping) and the "address already in use" retry loop;
- concurrent deploys serialized by _state_lock;
- stop_replica replica-id reconstruction for worker ids containing '-'.

Real production jinja templates are rendered; no docker is ever invoked.
"""

import json
import re
import subprocess as real_subprocess
import threading
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

WORKER_DIR = Path(__file__).resolve().parents[2] / "worker"

pytestmark = pytest.mark.unit


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

    # -- helpers ------------------------------------------------------------
    def ran(self, *prefix) -> list:
        return [c for c in self.commands if c[: len(prefix)] == list(prefix)]

    def firewall_ports(self) -> list:
        """Ports of every `iptables -I INPUT ... --dport N` rule we emitted."""
        ports = []
        for cmd in self.commands:
            if "nsenter" not in cmd:
                continue
            ports += [int(p) for p in re.findall(r"--dport (\d+)", cmd[-1])]
        return sorted(set(ports))


@pytest.fixture
def make_manager(worker_manager_module, tmp_path, monkeypatch):
    """Factory: WorkerManager wired to tmp_path DATA_DIR + FakeSubprocess."""
    wm = worker_manager_module

    def _make(docker_ps_output: str = "", compose_up_hook=None):
        fake = FakeSubprocess(docker_ps_output, compose_up_hook)
        monkeypatch.setattr(wm, "DATA_DIR", str(tmp_path))
        monkeypatch.setattr(wm, "subprocess", fake)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "unit-test-token")
        monkeypatch.setenv("WORKER_HOST", "10.0.0.7")
        mgr = wm.WorkerManager()
        # Point at the real production templates (constructor hardcodes
        # /app/templates) so rendering is exercised end-to-end.
        mgr.env = Environment(loader=FileSystemLoader(str(WORKER_DIR / "templates")))
        return mgr, fake, wm

    return _make


def _deploy_req(deploy_id="dep1", replica_id="dep1_w1_0", **overrides):
    """A deploy request shaped exactly like central/manager.py sends one.

    Note there is deliberately no "engine" key: the worker must default to
    dynamo (worker/manager.py DEFAULT_ENGINE).
    """
    req = {
        "deploy_id": deploy_id,
        "replica_id": replica_id,
        "name": "test",
        "model": "openai/gpt-oss-120b",
        "served_model_name": "openai/gpt-oss-120b",
        "is_embedding": False,
        "gpus": [0],
        "tp": 1,
        "max_len": 32768,
        "gpu_util": 0.9,
        "extra_args": None,
        "vllm_image": None,
        # dynamo fields (central/manager.py:_dynamo_fields)
        "advertise_host": "10.0.0.7",
        "etcd_endpoints": "http://143.248.74.105:2379",
        "namespace": "dynamo-openai-gpt-oss-120b",
        "reasoning_parser": "gpt_oss",
        "tool_call_parser": "harmony",
        "block_size": 64,
    }
    req.update(overrides)
    return req


def _legacy_req(**overrides):
    return _deploy_req(engine="vllm", model="org/test-model",
                       served_model_name="test-model", **overrides)


def _load_state(wm, tmp_path):
    with open(tmp_path / "local_deployments.json") as f:
        return json.load(f)


def _compose_doc(tmp_path, replica_id="dep1_w1_0"):
    return yaml.safe_load((tmp_path / f"run_{replica_id}" / "docker-compose.yml").read_text())


def _env_map(doc, node_name):
    out = {}
    for entry in doc["services"][node_name]["environment"]:
        key, _, value = str(entry).partition("=")
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

class TestEngineSelection:
    def test_engine_defaults_to_dynamo(self, make_manager, tmp_path, worker_manager_module):
        assert worker_manager_module.DEFAULT_ENGINE == "dynamo"
        mgr, fake, wm = make_manager()
        dep = mgr.deploy_model(_deploy_req())
        assert dep["nodes"][0]["name"] == "dynamo_dep1_w1_0"
        doc = _compose_doc(tmp_path)
        assert doc["services"]["dynamo_dep1_w1_0"]["entrypoint"] == [
            "python3", "-m", "dynamo.vllm"
        ]

    def test_default_image_is_the_dynamo_runtime(self, make_manager, worker_manager_module):
        mgr, fake, wm = make_manager()
        mgr.deploy_model(_deploy_req())
        (inspect,) = fake.ran("docker", "image", "inspect")
        assert inspect[-1] == worker_manager_module.DYNAMO_IMAGE

    def test_legacy_vllm_engine_still_supported(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        dep = mgr.deploy_model(_legacy_req())
        assert dep["nodes"][0]["name"] == "vllm_dep1_w1_0"
        (inspect,) = fake.ran("docker", "image", "inspect")
        assert inspect[-1] == "vllm/vllm-openai:latest"

    @pytest.mark.parametrize("engine", ["ollama", "sglang", "", "DYNAMO"])
    def test_unknown_engine_rejected_before_any_side_effect(
        self, make_manager, tmp_path, engine
    ):
        mgr, fake, wm = make_manager()
        with pytest.raises(Exception, match="Unsupported engine"):
            mgr.deploy_model(_deploy_req(engine=engine))
        assert fake.commands == []
        assert not (tmp_path / "local_deployments.json").exists()


# ---------------------------------------------------------------------------
# TLS: legacy engine only
# ---------------------------------------------------------------------------

class TestTlsCertificates:
    def test_dynamo_deploy_generates_no_certs(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        mgr.deploy_model(_deploy_req())
        assert fake.ran("openssl") == []
        assert not (tmp_path / "run_dep1_w1_0" / "vllm.key").exists()

    def test_legacy_vllm_deploy_generates_certs(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        mgr.deploy_model(_legacy_req())
        (openssl,) = fake.ran("openssl")
        assert "-keyout" in openssl and "-out" in openssl
        assert openssl[openssl.index("-keyout") + 1].endswith("vllm.key")


# ---------------------------------------------------------------------------
# Dynamo port derivation
# ---------------------------------------------------------------------------

class TestDynamoPortDerivation:
    def test_compose_gets_the_three_listener_ports_and_the_kv_port(
        self, make_manager, tmp_path, worker_manager_module
    ):
        wmod = worker_manager_module
        mgr, fake, wm = make_manager()
        dep = mgr.deploy_model(_deploy_req())
        base = dep["ports"][0]
        env = _env_map(_compose_doc(tmp_path), "dynamo_dep1_w1_0")
        assert env["DYN_SYSTEM_PORT"] == str(base + wmod.DYNAMO_SYSTEM_PORT_OFFSET)
        assert env["DYN_TCP_RPC_PORT"] == str(base + wmod.DYNAMO_RPC_PORT_OFFSET)
        assert env["DYN_TCP_RESPONSE_STREAM_PORT"] == str(base + wmod.DYNAMO_RESP_PORT_OFFSET)
        cmd = [str(c) for c in _compose_doc(tmp_path)["services"]["dynamo_dep1_w1_0"]["command"]]
        kv = json.loads(cmd[cmd.index("--kv-events-config") + 1])
        assert kv["endpoint"] == f"tcp://*:{base + wmod.DYNAMO_KV_PORT_OFFSET}"

    def test_offsets_match_central(self, worker_manager_module, central_dynamo):
        """worker/manager.py and central/dynamo.py must never disagree."""
        wmod = worker_manager_module
        assert wmod.DYNAMO_SYSTEM_PORT_OFFSET == central_dynamo.SYSTEM_PORT_OFFSET
        assert wmod.DYNAMO_RPC_PORT_OFFSET == central_dynamo.RPC_PORT_OFFSET
        assert wmod.DYNAMO_RESP_PORT_OFFSET == central_dynamo.RESP_PORT_OFFSET
        assert wmod.DYNAMO_KV_PORT_OFFSET == central_dynamo.KV_PORT_OFFSET
        assert wmod.HOST_PORT_OFFSET == central_dynamo.VLLM_API_PORT_OFFSET

    def test_firewall_opened_for_all_four_ports(
        self, make_manager, tmp_path, worker_manager_module
    ):
        """Host-network workers get no docker-published port rules, so the
        agent inserts the INPUT ACCEPTs itself."""
        wmod = worker_manager_module
        mgr, fake, wm = make_manager()
        dep = mgr.deploy_model(_deploy_req())
        base = dep["ports"][0]
        assert fake.firewall_ports() == sorted(
            base + off
            for off in (wmod.DYNAMO_SYSTEM_PORT_OFFSET, wmod.DYNAMO_RPC_PORT_OFFSET,
                        wmod.DYNAMO_RESP_PORT_OFFSET, wmod.DYNAMO_KV_PORT_OFFSET)
        )

    def test_legacy_deploy_opens_no_firewall_rules(self, make_manager, tmp_path):
        """Docker publishes the vLLM port itself (FORWARD chain)."""
        mgr, fake, wm = make_manager()
        mgr.deploy_model(_legacy_req())
        assert fake.firewall_ports() == []

    def test_advertise_host_falls_back_to_worker_host_env(
        self, make_manager, tmp_path, monkeypatch
    ):
        mgr, fake, wm = make_manager()
        monkeypatch.setenv("WORKER_HOST", "192.168.1.50")  # after make_manager's default
        req = _deploy_req()
        req.pop("advertise_host")
        mgr.deploy_model(req)
        env = _env_map(_compose_doc(tmp_path), "dynamo_dep1_w1_0")
        assert env["DYN_TCP_RPC_HOST"] == "192.168.1.50"

    def test_namespace_and_etcd_are_passed_through(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        mgr.deploy_model(_deploy_req(namespace="dynamo-my-model",
                                     etcd_endpoints="http://etcd:2379"))
        env = _env_map(_compose_doc(tmp_path), "dynamo_dep1_w1_0")
        assert env["DYN_NAMESPACE"] == "dynamo-my-model"
        assert env["ETCD_ENDPOINTS"] == "http://etcd:2379"

    def test_embedding_deploy_renders_the_pooling_variant(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        mgr.deploy_model(_deploy_req(is_embedding=True))
        cmd = [str(c) for c in _compose_doc(tmp_path)["services"]["dynamo_dep1_w1_0"]["command"]]
        assert "--embedding-worker" in cmd
        assert "--kv-events-config" not in cmd


# ---------------------------------------------------------------------------
# reapply_host_ports (startup: firewall rules do not survive a host reboot)
# ---------------------------------------------------------------------------

class TestReapplyHostPorts:
    def _seed(self, tmp_path, deps):
        (tmp_path / "local_deployments.json").write_text(json.dumps(deps))

    def test_only_dynamo_deployments_are_reapplied(self, make_manager, tmp_path,
                                                   worker_manager_module):
        wmod = worker_manager_module
        mgr, fake, wm = make_manager()
        self._seed(tmp_path, [
            {"id": "a", "replica_id": "a_w_0", "ports": [21001],
             "nodes": [{"name": "dynamo_a_w_0", "port": 21001}]},
            {"id": "b", "replica_id": "b_w_0", "ports": [21002],
             "nodes": [{"name": "vllm_b_w_0", "port": 21002}]},
            {"id": "c", "replica_id": "c_w_0", "ports": [21003], "nodes": []},
        ])
        calls = []
        mgr._ensure_host_ports_open = lambda ports, *a, **k: calls.append(sorted(ports))
        mgr.reapply_host_ports()
        assert calls == [sorted(
            21001 + off
            for off in (wmod.DYNAMO_SYSTEM_PORT_OFFSET, wmod.DYNAMO_RPC_PORT_OFFSET,
                        wmod.DYNAMO_RESP_PORT_OFFSET, wmod.DYNAMO_KV_PORT_OFFSET)
        )]

    def test_multiple_dynamo_replicas_are_batched(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        self._seed(tmp_path, [
            {"id": "a", "replica_id": "a_w_0", "ports": [21001],
             "nodes": [{"name": "dynamo_a_w_0", "port": 21001}]},
            {"id": "a", "replica_id": "a_w_1", "ports": [21002],
             "nodes": [{"name": "dynamo_a_w_1", "port": 21002}]},
        ])
        calls = []
        mgr._ensure_host_ports_open = lambda ports, *a, **k: calls.append(sorted(ports))
        mgr.reapply_host_ports()
        assert len(calls) == 1
        assert len(calls[0]) == 8

    def test_no_dynamo_deployments_means_no_call(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        self._seed(tmp_path, [
            {"id": "b", "replica_id": "b_w_0", "ports": [21002],
             "nodes": [{"name": "vllm_b_w_0", "port": 21002}]},
        ])
        calls = []
        mgr._ensure_host_ports_open = lambda ports, *a, **k: calls.append(ports)
        mgr.reapply_host_ports()
        assert calls == []

    def test_missing_state_file_is_harmless(self, make_manager, tmp_path):
        mgr, fake, wm = make_manager()
        calls = []
        mgr._ensure_host_ports_open = lambda ports, *a, **k: calls.append(ports)
        mgr.reapply_host_ports()
        assert calls == []


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
        # wrong port.
        assert dep["ports"] == [wm._PORT_START + 1]

    def test_retry_rewrites_the_derived_dynamo_ports(self, make_manager, tmp_path):
        """The compose file must be re-rendered with the NEW base port, or the
        instance would advertise ports derived from the abandoned one."""
        def hook(cmd, attempt):
            if attempt == 0:
                return real_subprocess.CompletedProcess(
                    cmd, 1, stdout="", stderr="Error: address already in use"
                )
            return real_subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        mgr, fake, wm = make_manager(compose_up_hook=hook)
        dep = mgr.deploy_model(_deploy_req())
        env = _env_map(_compose_doc(tmp_path), "dynamo_dep1_w1_0")
        assert env["DYN_SYSTEM_PORT"] == str(dep["ports"][0] + wm.DYNAMO_SYSTEM_PORT_OFFSET)

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
            mgr.deploy_model(_deploy_req(vllm_image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:9.9"))
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
            {"id": deploy_id, "replica_id": replica_id, "ports": [21001],
             "nodes": [{"name": f"dynamo_{replica_id}", "port": 21001}]},
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


# ---------------------------------------------------------------------------
# Image listing (both engine families)
# ---------------------------------------------------------------------------

class TestImageListing:
    def test_lists_dynamo_and_vllm_images_only(self, make_manager):
        mgr, fake, wm = make_manager()
        listing = (
            "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2\t20.6GB\t2026-09-01\n"
            "vllm/vllm-openai:latest\t19GB\t2026-08-01\n"
            "alpine:3.20\t7MB\t2026-01-01\n"
        )

        def run(cmd, **kw):
            fake.commands.append(list(cmd))
            if cmd[:2] == ["docker", "images"]:
                return real_subprocess.CompletedProcess(cmd, 0, stdout=listing, stderr="")
            return real_subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        fake.run = run
        images = mgr.list_vllm_images()
        assert [i["name"] for i in images] == [
            "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2",
            "vllm/vllm-openai:latest",
        ]
        assert [i["engine"] for i in images] == ["dynamo", "vllm"]


# ── Reconcile: bring back engine containers that died and stayed dead ────────
# After a host reboot the NVIDIA driver is often not loaded when docker starts
# the engine containers; they exit 128 and docker's `unless-stopped` eventually
# gives up, leaving the gateway with no workers and nobody to fix it.

def _dep_record(replica_id, name, port=21001):
    return {"id": replica_id.split("_")[0], "replica_id": replica_id, "ports": [port],
            "nodes": [{"name": name, "port": port}]}


def _write_compose(tmp_path, replica_id):
    d = tmp_path / f"run_{replica_id}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "docker-compose.yml").write_text("services: {}\n")


class _InspectAware(FakeSubprocess):
    """FakeSubprocess that answers `docker inspect -f {{.State.Running}}`."""

    def __init__(self, running_by_name):
        super().__init__()
        self.running_by_name = running_by_name

    def run(self, cmd, **kwargs):
        if cmd[:2] == ["docker", "inspect"] and "{{.State.Running}}" in cmd:
            self.commands.append(list(cmd))
            name = cmd[-1]
            if name not in self.running_by_name:
                return real_subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such object")
            value = "true" if self.running_by_name[name] else "false"
            return real_subprocess.CompletedProcess(cmd, 0, stdout=value + "\n", stderr="")
        return super().run(cmd, **kwargs)


class TestReconcile:
    def test_restarts_only_the_stopped_container(self, make_manager, tmp_path, monkeypatch,
                                                 worker_manager_module):
        mgr, _fake, wm = make_manager()
        fake = _InspectAware({"dynamo_a": True, "dynamo_b": False})
        monkeypatch.setattr(wm, "subprocess", fake)
        monkeypatch.setattr(mgr, "load_local_deployments",
                            lambda: [_dep_record("a", "dynamo_a"), _dep_record("b", "dynamo_b")])
        _write_compose(tmp_path, "a")
        _write_compose(tmp_path, "b")

        assert mgr.reconcile_local_deployments() == ["b"]

        ups = [c for c in fake.commands if c[:2] == ["docker", "compose"] and "up" in c]
        assert len(ups) == 1
        assert "vllm_b" in ups[0]

    def test_skips_when_the_compose_file_is_gone(self, make_manager, tmp_path, monkeypatch,
                                                 worker_manager_module):
        mgr, _fake, wm = make_manager()
        fake = _InspectAware({"dynamo_c": False})
        monkeypatch.setattr(wm, "subprocess", fake)
        monkeypatch.setattr(mgr, "load_local_deployments", lambda: [_dep_record("c", "dynamo_c")])
        # no compose file written for replica "c"

        assert mgr.reconcile_local_deployments() == []
        assert not [c for c in fake.commands if c[:2] == ["docker", "compose"]]

    def test_missing_container_counts_as_stopped(self, make_manager, tmp_path, monkeypatch,
                                                 worker_manager_module):
        mgr, _fake, wm = make_manager()
        fake = _InspectAware({})  # docker inspect fails: container removed entirely
        monkeypatch.setattr(wm, "subprocess", fake)
        monkeypatch.setattr(mgr, "load_local_deployments", lambda: [_dep_record("d", "dynamo_d")])
        _write_compose(tmp_path, "d")

        assert mgr.reconcile_local_deployments() == ["d"]

    def test_survives_an_unreachable_docker(self, make_manager, monkeypatch, worker_manager_module):
        mgr, _fake, wm = make_manager()

        class Boom(FakeSubprocess):
            def run(self, cmd, **kwargs):
                raise OSError("docker socket gone")

        monkeypatch.setattr(wm, "subprocess", Boom())
        monkeypatch.setattr(mgr, "load_local_deployments", lambda: [_dep_record("e", "dynamo_e")])
        assert mgr.reconcile_local_deployments() == []
