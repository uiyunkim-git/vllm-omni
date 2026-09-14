"""Unit tests for central/manager.py deploy plumbing that needs no docker.

Covered:
  * ``_dynamo_fields`` — what central hands the worker agent for engine=dynamo
    (advertise host, etcd endpoint, namespace, parsers, block size, embedding).
  * ``DEPLOY_CONFIG_KEYS`` persistence — a deployment must survive a
    save/load round-trip through sqlite with its engine settings intact, so the
    UI can show and recreate it.
  * engine defaulting to "dynamo".

The database is a per-test temp sqlite file (central/db.py DB_PATH is
monkeypatched by the ``central_db`` fixture in tests/conftest.py).
"""

import pytest

pytestmark = pytest.mark.unit

WORKER = {"id": "neuron-worker", "host": "10.0.0.7", "port": 8085, "status": "active"}


def _dep(**overrides) -> dict:
    dep = {
        "id": "ce5877fe",
        "name": "GPT-OSS 120B (dynamo)",
        "model": "openai/gpt-oss-120b",
        "served_model_name": "openai/gpt-oss-120b",
        "engine": "dynamo",
        "deployment_type": "replicas",
        "status": "running",
        "gpus": ["neuron-worker-0", "neuron-worker-1"],
        "nodes": [
            {"name": "dynamo_ce5877fe_neuron-worker_0", "host": "10.0.0.7",
             "port": 21001, "is_healthy": True},
        ],
    }
    dep.update(overrides)
    return dep


# ---------------------------------------------------------------------------
# _dynamo_fields
# ---------------------------------------------------------------------------

class TestDynamoFields:
    def test_legacy_vllm_engine_gets_no_dynamo_fields(self, central_manager):
        assert central_manager._dynamo_fields(
            {"engine": "vllm", "model": "openai/gpt-oss-120b"}, WORKER
        ) == {}

    def test_engine_defaults_to_dynamo(self, central_manager):
        out = central_manager._dynamo_fields({"model": "openai/gpt-oss-120b"}, WORKER)
        assert out, "a request without an explicit engine must take the dynamo path"

    def test_topology_fields(self, central_manager, central_dynamo):
        out = central_manager._dynamo_fields(
            {"engine": "dynamo", "model": "openai/gpt-oss-120b"}, WORKER
        )
        assert out["advertise_host"] == WORKER["host"]
        assert out["etcd_endpoints"] == central_dynamo.ETCD_ENDPOINT
        assert out["namespace"] == central_dynamo.namespace_for("openai/gpt-oss-120b")

    def test_namespace_follows_the_served_model_name(self, central_manager, central_dynamo):
        out = central_manager._dynamo_fields(
            {"model": "openai/gpt-oss-120b", "served_model_name": "gpt-oss"}, WORKER
        )
        assert out["namespace"] == central_dynamo.namespace_for("gpt-oss")

    def test_replicas_of_one_model_share_the_namespace(self, central_manager):
        req = {"model": "openai/gpt-oss-120b"}
        a = central_manager._dynamo_fields(req, WORKER)
        b = central_manager._dynamo_fields(req, {**WORKER, "host": "10.0.0.9"})
        assert a["namespace"] == b["namespace"]
        assert a["advertise_host"] != b["advertise_host"]

    def test_parsers_inferred_from_the_model_name(self, central_manager):
        out = central_manager._dynamo_fields({"model": "openai/gpt-oss-120b"}, WORKER)
        assert (out["reasoning_parser"], out["tool_call_parser"]) == ("gpt_oss", "harmony")

    def test_explicit_parsers_win(self, central_manager):
        out = central_manager._dynamo_fields(
            {"model": "openai/gpt-oss-120b",
             "reasoning_parser": "custom_r", "tool_call_parser": "custom_t"},
            WORKER,
        )
        assert (out["reasoning_parser"], out["tool_call_parser"]) == ("custom_r", "custom_t")

    def test_unknown_model_family_leaves_parsers_unset(self, central_manager):
        out = central_manager._dynamo_fields(
            {"model": "meta-llama/Llama-3.1-8B-Instruct"}, WORKER
        )
        assert out["reasoning_parser"] is None
        assert out["tool_call_parser"] is None

    def test_block_size_defaults_to_the_frontend_setting(self, central_manager, central_dynamo):
        out = central_manager._dynamo_fields({"model": "m"}, WORKER)
        assert out["block_size"] == central_dynamo.KV_BLOCK_SIZE

    def test_block_size_override(self, central_manager):
        out = central_manager._dynamo_fields({"model": "m", "block_size": 128}, WORKER)
        assert out["block_size"] == 128

    @pytest.mark.parametrize("value,expected", [(True, True), (False, False), (None, False)])
    def test_is_embedding_is_normalised_to_bool(self, central_manager, value, expected):
        out = central_manager._dynamo_fields({"model": "m", "is_embedding": value}, WORKER)
        assert out["is_embedding"] is expected

    def test_field_set_matches_the_worker_deploy_api(self, central_manager):
        """These keys are spread into the worker request; worker/main.py's
        WorkerDeployRequest must accept exactly them."""
        out = central_manager._dynamo_fields({"model": "m"}, WORKER)
        assert set(out) == {
            "advertise_host", "etcd_endpoints", "namespace",
            "reasoning_parser", "tool_call_parser", "block_size", "is_embedding",
        }


# ---------------------------------------------------------------------------
# save_deployments / load_deployments
# ---------------------------------------------------------------------------

class TestDeploymentPersistence:
    def test_empty_database(self, central_manager):
        assert central_manager.load_deployments() == []

    def test_round_trip_of_core_fields(self, central_manager):
        central_manager.save_deployments([_dep()])
        (got,) = central_manager.load_deployments()
        for key in ("id", "name", "model", "served_model_name", "engine",
                    "deployment_type", "status", "gpus", "nodes"):
            assert got[key] == _dep()[key], key

    def test_deploy_config_keys_round_trip(self, central_manager, central_manager_module):
        config = {
            "max_len": 32768,
            "gpu_util": 0.95,
            "extra_args": "--enable-prefix-caching",
            "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2",
            "is_embedding": False,
            "reasoning_parser": "gpt_oss",
            "tool_call_parser": "harmony",
            "block_size": 64,
            "tp": 1,
        }
        assert set(config) == set(central_manager_module.DEPLOY_CONFIG_KEYS)
        central_manager.save_deployments([_dep(config=config, **config)])
        (got,) = central_manager.load_deployments()
        assert got["config"] == config
        # ...and the settings are also spread at the top level, the shape the
        # deploy path and the API both use.
        for key, value in config.items():
            assert got[key] == value, key

    def test_config_collected_from_top_level_keys_when_absent(
        self, central_manager, central_manager_module
    ):
        dep = _dep(max_len=8192, gpu_util=0.8, tp=2)
        dep.pop("config", None)
        central_manager.save_deployments([dep])
        (got,) = central_manager.load_deployments()
        assert got["config"] == {"max_len": 8192, "gpu_util": 0.8, "tp": 2}

    def test_engine_defaults_to_dynamo_on_save(self, central_manager, central_dynamo):
        dep = _dep()
        dep.pop("engine")
        central_manager.save_deployments([dep])
        (got,) = central_manager.load_deployments()
        assert got["engine"] == central_dynamo.DEFAULT_ENGINE == "dynamo"

    def test_legacy_engine_is_preserved(self, central_manager):
        central_manager.save_deployments([_dep(engine="vllm")])
        assert central_manager.load_deployments()[0]["engine"] == "vllm"

    def test_served_model_name_falls_back_to_model(self, central_manager):
        dep = _dep()
        dep.pop("served_model_name")
        central_manager.save_deployments([dep])
        (got,) = central_manager.load_deployments()
        assert got["served_model_name"] == dep["model"]

    def test_save_replaces_the_whole_set(self, central_manager):
        central_manager.save_deployments([_dep(id="a"), _dep(id="b")])
        assert {d["id"] for d in central_manager.load_deployments()} == {"a", "b"}
        central_manager.save_deployments([_dep(id="b")])
        assert [d["id"] for d in central_manager.load_deployments()] == ["b"]

    def test_nodes_survive_with_their_health_flag(self, central_manager):
        central_manager.save_deployments([_dep()])
        (got,) = central_manager.load_deployments()
        assert got["nodes"][0]["port"] == 21001
        assert got["nodes"][0]["is_healthy"] is True

    def test_node_urls_are_derivable_after_a_round_trip(self, central_manager, central_dynamo):
        """The round-tripped record must be enough for the metrics path to
        rebuild the instance URL."""
        central_manager.save_deployments([_dep()])
        (got,) = central_manager.load_deployments()
        assert central_dynamo.node_url(got, got["nodes"][0]) == "http://10.0.0.7:31001"


# ---------------------------------------------------------------------------
# workers table (used by the deploy path to resolve advertise_host)
# ---------------------------------------------------------------------------

class TestWorkerRegistry:
    def test_register_accept_and_list(self, central_manager):
        central_manager.register_worker(
            "neuron-worker", "10.0.0.7", 8085,
            [{"id": 0, "name": "H100", "utilization": 5, "memory_used": 1, "memory_total": 80}],
        )
        workers = central_manager.get_workers()
        assert workers["neuron-worker"]["status"] == "pending"
        assert central_manager.get_all_gpus() == []  # pending workers offer no GPUs

        assert central_manager.accept_worker("neuron-worker", "neuron") is True
        gpus = central_manager.get_all_gpus()
        assert [g["id"] for g in gpus] == ["neuron-worker-0"]
        assert gpus[0]["worker_name"] == "neuron"

    def test_reregistration_updates_the_address(self, central_manager):
        central_manager.register_worker("w", "10.0.0.7", 8085, [])
        central_manager.register_worker("w", "10.0.0.8", 8086, [])
        assert central_manager.get_workers()["w"]["host"] == "10.0.0.8"
        assert central_manager.get_workers()["w"]["port"] == 8086

    def test_delete_worker(self, central_manager):
        central_manager.register_worker("w", "10.0.0.7", 8085, [])
        assert central_manager.delete_worker("w") is True
        assert central_manager.get_workers() == {}
        assert central_manager.delete_worker("w") is False
