"""Central control plane vs. a mock worker agent (loopback only, no docker).

Replaces the old router integration layer: what matters now is that central
hands the worker agent a correct Dynamo deploy request, records the nodes it
gets back, and then finds those nodes healthy on their system port.
"""

import pytest

pytestmark = pytest.mark.integration

MODEL = "openai/gpt-oss-120b"


def _deploy_req(worker_id: str, **overrides) -> dict:
    req = {
        "name": "GPT-OSS 120B (dynamo)",
        "deployment_type": "replicas",
        "model": MODEL,
        "served_model_name": MODEL,
        "gpus": [f"{worker_id}-0"],
        "tp": 1,
        "is_embedding": False,
        "max_len": 32768,
        "gpu_util": 0.95,
        "extra_args": None,
        "image": None,
    }
    req.update(overrides)
    return req


class TestDynamoDeploy:
    async def test_worker_receives_the_dynamo_fields(
        self, central_manager, central_dynamo, registered_worker
    ):
        agent = registered_worker
        dep = await central_manager.deploy_model(_deploy_req(agent.worker_id))

        (sent,) = agent.deploy_requests()
        assert sent["engine"] == "dynamo"
        assert sent["advertise_host"] == agent.host
        assert sent["etcd_endpoints"] == central_dynamo.ETCD_ENDPOINT
        assert sent["namespace"] == central_dynamo.namespace_for(MODEL)
        assert sent["reasoning_parser"] == "gpt_oss"
        assert sent["tool_call_parser"] == "harmony"
        assert sent["block_size"] == central_dynamo.KV_BLOCK_SIZE
        assert sent["is_embedding"] is False
        assert sent["deploy_id"] == dep["id"]
        assert sent["replica_id"] == f"{dep['id']}_{agent.worker_id}_0"

    async def test_deployment_is_persisted_with_its_nodes(
        self, central_manager, registered_worker
    ):
        agent = registered_worker
        dep = await central_manager.deploy_model(_deploy_req(agent.worker_id))
        (stored,) = central_manager.load_deployments()
        assert stored["id"] == dep["id"]
        assert stored["engine"] == "dynamo"
        assert stored["status"] == "starting"
        assert stored["config"]["max_len"] == 32768
        assert stored["config"]["reasoning_parser"] == "gpt_oss"
        (node,) = stored["nodes"]
        assert node["host"] == agent.host
        assert node["is_healthy"] is False

    async def test_health_check_marks_the_instance_ready(
        self, central_manager, central_dynamo, registered_worker
    ):
        agent = registered_worker
        await central_manager.deploy_model(_deploy_req(agent.worker_id))

        await central_manager.run_health_checks()
        (stored,) = central_manager.load_deployments()
        assert stored["nodes"][0]["is_healthy"] is True
        assert stored["status"] == "running"
        # the probe went to the system port, not the base port
        assert central_dynamo.node_api_port(stored, stored["nodes"][0]) == (
            stored["nodes"][0]["port"] + central_dynamo.SYSTEM_PORT_OFFSET
        )

    async def test_instance_metrics_are_scrapable(
        self, central_manager, central_dynamo, registered_worker
    ):
        import httpx

        agent = registered_worker
        agent.configure(requests_total=17, errors_total=2, running=3, waiting=1,
                        kv_cache_usage=0.25)
        await central_manager.deploy_model(_deploy_req(agent.worker_id))
        (stored,) = central_manager.load_deployments()
        url = central_dynamo.node_url(stored, stored["nodes"][0])

        async with httpx.AsyncClient() as client:
            summary = await central_dynamo.scrape_instance(client, url)
        assert summary["requests_total"] == 17     # generate endpoint only
        assert summary["errors_total"] == 2
        assert summary["running"] == 3
        assert summary["waiting"] == 1
        assert summary["kv_cache_usage_pct"] == 25.0

    async def test_not_ready_instance_stays_unhealthy(
        self, central_manager, registered_worker
    ):
        agent = registered_worker
        agent.configure(ready=False)
        await central_manager.deploy_model(_deploy_req(agent.worker_id))
        await central_manager.run_health_checks()
        (stored,) = central_manager.load_deployments()
        assert stored["nodes"][0]["is_healthy"] is False
        assert stored["status"] == "starting"

    async def test_stop_deployment_reaches_the_worker(
        self, central_manager, registered_worker
    ):
        agent = registered_worker
        dep = await central_manager.deploy_model(_deploy_req(agent.worker_id))
        assert await central_manager.stop_deployment(dep["id"]) is True
        assert central_manager.load_deployments() == []
        assert agent.state()["deployments"] == []

    async def test_embedding_deploy_passes_the_flag(
        self, central_manager, registered_worker
    ):
        agent = registered_worker
        await central_manager.deploy_model(
            _deploy_req(agent.worker_id, is_embedding=True,
                        model="Qwen/Qwen3-Embedding-0.6B",
                        served_model_name="Qwen/Qwen3-Embedding-0.6B")
        )
        (sent,) = agent.deploy_requests()
        assert sent["is_embedding"] is True

    async def test_tp_deploy_sends_every_gpu_once(
        self, central_manager, registered_worker
    ):
        agent = registered_worker
        await central_manager.deploy_model(
            _deploy_req(agent.worker_id, deployment_type="tp", tp=2,
                        gpus=[f"{agent.worker_id}-0", f"{agent.worker_id}-1"])
        )
        (sent,) = agent.deploy_requests()
        assert sorted(sent["gpus"]) == [0, 1]
        assert sent["tp"] == 2


class TestLegacyVllmDeploy:
    async def test_no_dynamo_fields_are_sent(self, central_manager, registered_worker):
        agent = registered_worker
        await central_manager.deploy_model(_deploy_req(agent.worker_id, engine="vllm"))
        (sent,) = agent.deploy_requests()
        assert sent["engine"] == "vllm"
        # the worker's pydantic defaults fill these in; central sent nothing
        assert sent["advertise_host"] is None
        assert sent["etcd_endpoints"] is None
        assert sent["namespace"] == "dynamo"


class TestDeployFailure:
    async def test_failed_deploy_records_nothing(
        self, central_manager, registered_worker
    ):
        """The worker rejects the very first replica, so central must persist
        no deployment at all.

        NOTE: this does NOT exercise the partial-deploy rollback path (replica
        N of M failing after 1..N-1 started). That path is currently dead —
        _deploy_model_inner rebinds `touched_wids` to a fresh set instead of
        filling the one deploy_model passes in — so no test asserts it here;
        see the suite notes in tests/README.md.
        """
        agent = registered_worker
        agent.configure(fail_deploy=True)
        with pytest.raises(Exception):
            await central_manager.deploy_model(_deploy_req(agent.worker_id))
        assert central_manager.load_deployments() == []
        assert agent.state()["deployments"] == []
