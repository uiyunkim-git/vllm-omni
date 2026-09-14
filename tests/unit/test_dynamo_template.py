"""Rendering tests for worker/templates/dynamo_node.j2.

The template IS the data-plane contract on the worker side: one bad flag and a
`dynamo.vllm` worker either refuses to start or registers itself with ports the
frontend cannot reach. These tests render the production template with jinja2
(the same way worker/manager.py does) and assert on the parsed YAML.

Nothing here runs docker; only text and YAML.
"""

import json

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

pytestmark = pytest.mark.unit

BASE_PORT = 21001
SYSTEM_PORT = BASE_PORT + 10000   # 31001, DYN_SYSTEM_PORT (i16!) health/metrics
RPC_PORT = BASE_PORT + 12000      # 33001, TCP request plane
RESP_PORT = BASE_PORT + 42000     # 63001, TCP response stream
KV_PORT = BASE_PORT + 44000       # 65001, ZMQ KV events

NODE_NAME = "dynamo_ce5877fe_neuron-worker_0"


@pytest.fixture(scope="module")
def render(worker_templates_dir):
    """render(**overrides) -> (compose_text, parsed_yaml)."""
    env = Environment(loader=FileSystemLoader(str(worker_templates_dir)))
    template = env.get_template("dynamo_node.j2")

    def _render(**overrides):
        # Mirrors worker/manager.py:_write_compose exactly.
        ctx = {
            "node_name": NODE_NAME,
            "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2",
            "model_name": "openai/gpt-oss-120b",
            "served_model_name": "openai/gpt-oss-120b",
            "huggingface_token": "hf_unit_test",
            "tensor_parallel_size": 1,
            "gpu_ids": ["0"],
            "port": BASE_PORT,
            "max_model_len": 32768,
            "gpu_memory_util": 0.95,
            "replica_id": "ce5877fe_neuron-worker_0",
            "host_cache_dir": "/home/u/.cache/huggingface",
            "extra_args": [],
            "system_port": SYSTEM_PORT,
            "rpc_port": RPC_PORT,
            "resp_port": RESP_PORT,
            "kv_port": KV_PORT,
            "advertise_host": "10.0.0.7",
            "etcd_endpoints": "http://143.248.74.105:2379",
            "namespace": "dynamo-openai-gpt-oss-120b",
            "reasoning_parser": None,
            "tool_call_parser": None,
            "block_size": 64,
            "is_embedding": False,
        }
        ctx.update(overrides)
        text = template.render(**ctx)
        return text, yaml.safe_load(text)

    return _render


def service(doc: dict) -> dict:
    return doc["services"][NODE_NAME]


def env_map(doc: dict) -> dict:
    out = {}
    for entry in service(doc)["environment"]:
        key, _, value = str(entry).partition("=")
        out[key] = value
    return out


def command(doc: dict) -> list:
    return [str(c) for c in service(doc)["command"]]


def flag_value(cmd: list, flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


# ---------------------------------------------------------------------------
# container shape
# ---------------------------------------------------------------------------

class TestComposeShape:
    def test_renders_valid_yaml_with_one_service(self, render):
        _, doc = render()
        assert list(doc["services"]) == [NODE_NAME]
        assert service(doc)["container_name"] == NODE_NAME

    def test_host_networking(self, render):
        """The advertised ports must be the real host ports — the frontend
        dials them directly, there is no docker port publishing."""
        _, doc = render()
        svc = service(doc)
        assert svc["network_mode"] == "host"
        assert "ports" not in svc, "host networking must not publish ports"

    def test_runs_the_dynamo_vllm_module_not_an_api_server(self, render):
        _, doc = render()
        assert service(doc)["entrypoint"] == ["python3", "-m", "dynamo.vllm"]

    def test_gpu_reservation_uses_the_requested_device_ids(self, render):
        _, doc = render(gpu_ids=["2", "3"])
        devices = service(doc)["deploy"]["resources"]["reservations"]["devices"][0]
        assert devices["device_ids"] == ["2", "3"]
        assert devices["driver"] == "nvidia"

    def test_hf_cache_is_mounted(self, render):
        _, doc = render(host_cache_dir="/data/hf")
        assert service(doc)["volumes"] == ["/data/hf:/root/.cache/huggingface"]

    def test_no_tls_material(self, render):
        """A dynamo worker exposes no HTTPS server; the frontend owns ingress."""
        text, doc = render()
        assert "vllm.crt" not in text and "vllm.key" not in text
        assert "--ssl-keyfile" not in text and "--ssl-certfile" not in text


# ---------------------------------------------------------------------------
# environment: discovery + the three derived listener ports
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_three_derived_ports(self, render):
        _, doc = render()
        env = env_map(doc)
        assert env["DYN_SYSTEM_PORT"] == str(SYSTEM_PORT)
        assert env["DYN_TCP_RPC_PORT"] == str(RPC_PORT)
        assert env["DYN_TCP_RESPONSE_STREAM_PORT"] == str(RESP_PORT)

    def test_system_port_fits_in_an_i16(self, render):
        """Dynamo parses DYN_SYSTEM_PORT as i16 — this is why health/metrics
        cannot live in the +40000 slot the legacy vLLM API used."""
        _, doc = render()
        assert int(env_map(doc)["DYN_SYSTEM_PORT"]) <= 32767

    def test_advertised_hosts(self, render):
        _, doc = render(advertise_host="10.0.0.42")
        env = env_map(doc)
        assert env["DYN_TCP_RPC_HOST"] == "10.0.0.42"
        assert env["DYN_TCP_RESPONSE_STREAM_HOST"] == "10.0.0.42"
        assert env["DYN_EVENT_PLANE_HOST"] == "10.0.0.42"

    def test_etcd_discovery(self, render):
        _, doc = render(etcd_endpoints="http://etcd.local:2379",
                        namespace="dynamo-my-model")
        env = env_map(doc)
        assert env["DYN_DISCOVERY_BACKEND"] == "etcd"
        assert env["ETCD_ENDPOINTS"] == "http://etcd.local:2379"
        assert env["DYN_NAMESPACE"] == "dynamo-my-model"

    def test_deterministic_hashing_for_kv_radix_trees(self, render):
        _, doc = render()
        assert env_map(doc)["PYTHONHASHSEED"] == "0"

    def test_hf_token_passed_when_present(self, render):
        _, doc = render(huggingface_token="hf_abc")
        env = env_map(doc)
        assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_abc"
        assert env["HF_TOKEN"] == "hf_abc"

    def test_hf_token_omitted_when_empty(self, render):
        _, doc = render(huggingface_token="")
        assert "HF_TOKEN" not in env_map(doc)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

class TestCommand:
    def test_model_and_served_name(self, render):
        _, doc = render()
        cmd = command(doc)
        assert flag_value(cmd, "--model") == "openai/gpt-oss-120b"
        assert flag_value(cmd, "--served-model-name") == "openai/gpt-oss-120b"

    def test_served_name_omitted_when_unset(self, render):
        _, doc = render(served_model_name=None)
        assert "--served-model-name" not in command(doc)

    def test_engine_args(self, render):
        _, doc = render(max_model_len=8192, gpu_memory_util=0.8, tensor_parallel_size=4)
        cmd = command(doc)
        assert flag_value(cmd, "--max-model-len") == "8192"
        assert flag_value(cmd, "--gpu-memory-utilization") == "0.8"
        assert flag_value(cmd, "--tensor-parallel-size") == "4"

    def test_tp_flag_omitted_for_a_single_gpu(self, render):
        _, doc = render(tensor_parallel_size=1)
        assert "--tensor-parallel-size" not in command(doc)

    def test_optional_engine_args_omitted_when_unset(self, render):
        _, doc = render(max_model_len=None, gpu_memory_util=None)
        cmd = command(doc)
        assert "--max-model-len" not in cmd
        assert "--gpu-memory-utilization" not in cmd

    def test_extra_args_appended_verbatim(self, render):
        _, doc = render(extra_args=["--enable-prefix-caching", "--max-num-seqs", "256"])
        cmd = command(doc)
        assert cmd[-3:] == ["--enable-prefix-caching", "--max-num-seqs", "256"]

    def test_no_router_mode_flag(self, render):
        """`dynamo.vllm` rejects --router-mode (it is a frontend flag); emitting
        it makes every worker fail to start."""
        text, doc = render()
        assert "--router-mode" not in text
        assert "--router-mode" not in command(doc)

    def test_no_api_server_flags(self, render):
        text, _ = render()
        for flag in ("--api-key", "--port", "--host", "--uvicorn"):
            assert flag not in text


class TestKvEvents:
    def test_kv_events_config_is_json_pointing_at_the_kv_port(self, render):
        _, doc = render()
        cfg = json.loads(flag_value(command(doc), "--kv-events-config"))
        assert cfg["publisher"] == "zmq"
        assert cfg["enable_kv_cache_events"] is True
        assert cfg["endpoint"] == f"tcp://*:{KV_PORT}"
        assert cfg["topic"] == "kv-events"

    def test_block_size_must_match_the_frontend(self, render):
        _, doc = render(block_size=128)
        assert flag_value(command(doc), "--block-size") == "128"


class TestEmbeddingVariant:
    def test_pooling_runner_flags(self, render):
        _, doc = render(is_embedding=True)
        cmd = command(doc)
        assert "--embedding-worker" in cmd
        assert flag_value(cmd, "--runner") == "pooling"

    def test_no_kv_events_or_block_size(self, render):
        """Pooling models publish no KV events and cannot be KV-routed."""
        _, doc = render(is_embedding=True)
        cmd = command(doc)
        assert "--kv-events-config" not in cmd
        assert "--block-size" not in cmd

    def test_generate_variant_has_no_pooling_flags(self, render):
        _, doc = render(is_embedding=False)
        cmd = command(doc)
        assert "--embedding-worker" not in cmd
        assert "--runner" not in cmd

    def test_embedding_variant_is_still_valid_yaml_with_the_ports(self, render):
        _, doc = render(is_embedding=True)
        assert env_map(doc)["DYN_SYSTEM_PORT"] == str(SYSTEM_PORT)


class TestParserFlags:
    def test_both_parsers_rendered(self, render):
        _, doc = render(reasoning_parser="gpt_oss", tool_call_parser="harmony")
        cmd = command(doc)
        assert flag_value(cmd, "--dyn-reasoning-parser") == "gpt_oss"
        assert flag_value(cmd, "--dyn-tool-call-parser") == "harmony"

    def test_parsers_omitted_when_not_inferred(self, render):
        _, doc = render(reasoning_parser=None, tool_call_parser=None)
        cmd = command(doc)
        assert "--dyn-reasoning-parser" not in cmd
        assert "--dyn-tool-call-parser" not in cmd

    def test_reasoning_parser_alone(self, render):
        _, doc = render(reasoning_parser="qwen3", tool_call_parser=None)
        cmd = command(doc)
        assert "--dyn-reasoning-parser" in cmd
        assert "--dyn-tool-call-parser" not in cmd

    def test_vllm_side_parser_flags_are_not_emitted(self, render):
        """The parsers live on the FRONTEND side in Dynamo: only the --dyn-*
        variants are advertised by the worker."""
        text, _ = render(reasoning_parser="gpt_oss", tool_call_parser="harmony")
        assert "- --reasoning-parser" not in text
        assert "- --tool-call-parser" not in text
