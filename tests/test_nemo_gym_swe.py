# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import traceback
from typing import Any

import httpx
import pytest

from areal.workflow.nemo_gym_swe import NeMoGymSWEAgent, NeMoGymSWEError


@pytest.fixture
def task() -> dict[str, Any]:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Fix the regression"}],
            "metadata": {
                "instance_id": "repo-123",
                "dataset_name": "R2E-Gym",
                "instance_dict": json.dumps(
                    {"repo": "example/repo", "FAIL_TO_PASS": ["test_regression"]}
                ),
            },
        },
        "benchmark_metadata": {"split": "train", "nested": [1, 2]},
    }


def completed_result(resolved: bool = True) -> dict[str, Any]:
    return {
        "reward": float(resolved),
        "resolved": resolved,
        "instance_config": {"mask_sample": False},
        "agent_timed_out": None,
        "eval_timed_out": None,
        "agent_error_kind": None,
        "response": {
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": []}],
        },
    }


@pytest.mark.asyncio
async def test_run_preserves_task_and_routes_session_headers(task):
    """Full grader metadata and row settings survive without session credentials."""
    task["responses_create_params"].update(
        model="row-model", temperature=0.7, top_p=0.8, max_output_tokens=4096
    )
    original = copy.deepcopy(task)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completed_result())

    agent = NeMoGymSWEAgent("http://sandbox.test/", model="default-model")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reward = await agent.run(
            task,
            base_url="http://proxy.test:8080/",
            api_key="episode-secret",
            http_client=client,
        )
        assert not client.is_closed
    assert reward == 1.0
    assert task == original
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "http://sandbox.test/run"
    assert request.headers["Authorization"] == "Bearer episode-secret"
    assert request.headers["X-AReaL-Proxy-Url"] == "http://proxy.test:8080/"
    assert len(request.headers["X-AReaL-Attempt-Id"]) == 32
    assert json.loads(request.content) == original
    assert b"episode-secret" not in request.content


@pytest.mark.asyncio
async def test_run_missing_generation_settings_fills_defaults(task):
    """Defaults are filled on the submitted copy only."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completed_result())

    agent = NeMoGymSWEAgent("http://sandbox.test", model="qwen", temperature=0.4)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await agent.run(
            task, base_url="http://proxy.test", api_key="session", http_client=client
        )
    params = json.loads(requests[0].content)["responses_create_params"]
    assert params["model"] == "qwen"
    assert params["temperature"] == 0.4
    assert params["top_p"] == 1.0
    assert params["max_output_tokens"] == 32768
    assert "model" not in task["responses_create_params"]


@pytest.mark.asyncio
@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({}, id="ordinary-null-timeouts"),
        pytest.param(
            {"agent_timed_out": False, "eval_timed_out": False},
            id="false-timeouts",
        ),
        pytest.param({"instance_config": {}}, id="no-mask"),
        pytest.param({"instance_config": {"mask_sample": True}}, id="masked"),
        pytest.param({"mask_sample": True}, id="top-level-mask"),
        pytest.param(
            {
                "instance_config": {"mask_sample": True},
                "agent_error_kind": "max_iteration",
            },
            id="max-iteration",
        ),
        pytest.param(
            {
                "instance_config": {"mask_sample": True},
                "agent_error_kind": "context_window",
            },
            id="context-window",
        ),
        pytest.param({"agent_error_kind": "stuck_in_loop"}, id="stuck-in-loop"),
        pytest.param({"agent_error_kind": "other"}, id="other-agent-error"),
        pytest.param({"agent_error_kind": "future-kind"}, id="unknown-agent-error"),
        pytest.param(
            {"instance_config": {"mask_sample": True}, "agent_timed_out": True},
            id="agent-timeout",
        ),
        pytest.param(
            {"instance_config": {"mask_sample": True}, "eval_timed_out": True},
            id="eval-timeout",
        ),
    ],
)
async def test_run_valid_gym_outcome_preserves_reward(task, resolved, changes):
    """Gym outcome metadata does not change the reference trainer's reward."""
    result = completed_result(resolved) | changes
    original = copy.deepcopy(result)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=result)

    agent = NeMoGymSWEAgent("http://sandbox.test", model="qwen")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reward = await agent.run(
            task,
            base_url="http://proxy.test",
            api_key="session",
            http_client=client,
        )
    assert reward == float(resolved)
    assert [request.method for request in requests] == ["POST"]
    assert result == original


@pytest.mark.asyncio
async def test_run_concurrent_attempts_isolate_session_credentials(task):
    """One shared agent and HTTP client can serve independent rollout sessions."""
    requests = []
    both_started = asyncio.Event()

    async def handler(request):
        requests.append(request)
        if len(requests) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return httpx.Response(
            200, json=completed_result(request.headers["Authorization"].endswith("one"))
        )

    agent = NeMoGymSWEAgent("http://sandbox.test", model="qwen")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rewards = await asyncio.gather(
            *(
                agent.run(
                    task,
                    base_url=f"http://proxy-{name}.test",
                    api_key=name,
                    http_client=client,
                )
                for name in ("one", "two")
            )
        )
    assert rewards == [1.0, 0.0]
    assert len({r.headers["X-AReaL-Attempt-Id"] for r in requests}) == 2
    for request in requests:
        key = request.headers["Authorization"].removeprefix("Bearer ")
        assert request.headers["X-AReaL-Proxy-Url"] == f"http://proxy-{key}.test"
    assert "model" not in task["responses_create_params"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"instance_config": None},
        {"instance_config": []},
        {"instance_config": {"mask_sample": "false"}},
        {"instance_config": {"mask_sample": 0}},
        {"instance_config": {"mask_sample": None}},
        {"mask_sample": "false"},
        {"agent_timed_out": "false"},
        {"eval_timed_out": 0},
        {"agent_error_kind": {"secret": "provider-secret"}},
        {"agent_error_kind": False},
        {"agent_error_kind": 1},
        {"agent_error_kind": []},
        {"resolved": None},
        {"resolved": 1},
        {"reward": False},
        {"reward": "1"},
        {"reward": 0},
        {"reward": 0.5},
        {"response": {"output": []}},
        {"response": {"output": "provider-secret"}},
        {"response": {"output": [{}], "status": "in_progress"}},
        {"response": {"output": [{}], "error": "provider-secret"}},
    ],
)
async def test_run_invalid_result_revokes_and_raises(task, changes):
    """Malformed outcomes cannot become zero-reward training data."""
    result = completed_result() | changes
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=result)

    agent = NeMoGymSWEAgent("http://sandbox.test", model="qwen")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NeMoGymSWEError) as exc:
            await agent.run(
                task,
                base_url="http://proxy.test",
                api_key="session-secret",
                http_client=client,
            )
    assert [r.method for r in requests] == ["POST", "DELETE"]
    assert (
        requests[1].url.path == "/attempts/" + requests[0].headers["X-AReaL-Attempt-Id"]
    )
    assert requests[1].headers["Authorization"] == "Bearer session-secret"
    assert "provider-secret" not in str(exc.value)
    assert "session-secret" not in str(exc.value)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf")])
def test_reward_nonfinite_value_raises(reward):
    """Only binary rewards agreeing with the evaluator result are accepted."""
    result = completed_result() | {"reward": reward}
    with pytest.raises(NeMoGymSWEError, match="invalid or inconsistent reward"):
        NeMoGymSWEAgent._reward(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "redirect", "transport", "json", "shape"])
async def test_run_request_failures_are_safe_and_never_retried(task, failure):
    """Response bodies, request headers, and transport exception details stay private."""
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "DELETE":
            raise httpx.ConnectError("cleanup session-secret", request=request)
        if failure == "transport":
            raise httpx.ConnectError("session-secret provider-secret", request=request)
        if failure == "http":
            return httpx.Response(500, text="provider-secret")
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "http://other.test"})
        if failure == "json":
            return httpx.Response(200, text="provider-secret")
        return httpx.Response(200, json=["provider-secret"])

    agent = NeMoGymSWEAgent("http://sandbox.test", model="qwen")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        with pytest.raises(NeMoGymSWEError) as exc:
            await agent.run(
                task,
                base_url="http://proxy.test",
                api_key="session-secret",
                http_client=client,
            )
    formatted = "".join(traceback.format_exception(exc.value))
    assert "session-secret" not in formatted
    assert "provider-secret" not in formatted
    assert [r.method for r in requests] == ["POST", "DELETE"]
    assert all(r.url.host == "sandbox.test" for r in requests)
    if failure == "http":
        assert str(exc.value) == "Runner returned HTTP 500 (Internal Server Error)"
    elif failure == "redirect":
        assert str(exc.value) == "Runner returned HTTP 307 (Temporary Redirect)"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_type", ["wall_clock", "httpx"])
async def test_run_timeout_revokes_attempt_with_bounded_cleanup(task, timeout_type):
    """Both HTTP timeouts and the total deadline revoke forwarding credentials."""
    requests = []
    cleanup_cancelled = asyncio.Event()

    async def handler(request):
        requests.append(request)
        if request.method == "DELETE":
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_cancelled.set()
        if timeout_type == "httpx":
            raise httpx.ReadTimeout("session-secret", request=request)
        await asyncio.Event().wait()

    agent = NeMoGymSWEAgent(
        "http://sandbox.test",
        model="qwen",
        request_timeout_seconds=0.01,
        cancellation_timeout_seconds=0.01,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NeMoGymSWEError, match="timed out"):
            await asyncio.wait_for(
                agent.run(
                    task,
                    base_url="http://proxy.test",
                    api_key="session-secret",
                    http_client=client,
                ),
                timeout=2,
            )
    assert [r.method for r in requests] == ["POST", "DELETE"]
    assert cleanup_cancelled.is_set()


@pytest.mark.asyncio
async def test_run_cancellation_preserves_cancellation_after_revoke(task):
    """Repeated cancellation cannot interrupt the bounded revocation request."""
    run_started, revoke_started, allow_revoke = (asyncio.Event() for _ in range(3))
    requests = []

    async def handler(request):
        requests.append(request)
        if request.method == "POST":
            run_started.set()
            await asyncio.Event().wait()
        revoke_started.set()
        await allow_revoke.wait()
        return httpx.Response(200, json={"revoked": True})

    agent = NeMoGymSWEAgent(
        "http://sandbox.test", model="qwen", cancellation_timeout_seconds=1
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        running = asyncio.create_task(
            agent.run(
                task,
                base_url="http://proxy.test",
                api_key="session",
                http_client=client,
            )
        )
        await asyncio.wait_for(run_started.wait(), timeout=2)
        running.cancel()
        await asyncio.wait_for(revoke_started.wait(), timeout=2)
        running.cancel()
        allow_revoke.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=2)
    assert [r.method for r in requests] == ["POST", "DELETE"]


@pytest.mark.parametrize(
    "settings",
    [
        {"runner_url": "file:///tmp/runner"},
        {"runner_url": "http://secret@runner.test"},
        {"runner_url": "http://runner.test?token=secret"},
        {"runner_url": "http://runner.test\nsecret"},
        {"model": ""},
        {"temperature": float("nan")},
        {"temperature": -1},
        {"top_p": 0},
        {"max_output_tokens": True},
        {"max_output_tokens": 0},
        {"request_timeout_seconds": float("inf")},
        {"cancellation_timeout_seconds": 0},
    ],
)
def test_constructor_invalid_settings_fail_early(settings):
    """Invalid configuration fails before any episode or network request."""
    with pytest.raises(ValueError):
        NeMoGymSWEAgent(
            **({"runner_url": "http://sandbox.test", "model": "qwen"} | settings)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.clear(),
        lambda row: row.update(responses_create_params=[]),
        lambda row: row["responses_create_params"].update(metadata=[]),
        lambda row: row["responses_create_params"]["metadata"].pop("instance_id"),
        lambda row: row["responses_create_params"]["metadata"].update(
            instance_dict="[]"
        ),
        lambda row: row["responses_create_params"]["metadata"].update(
            instance_dict="secret"
        ),
        lambda row: row["responses_create_params"].update(input=None),
        lambda row: row["responses_create_params"].update(top_p=None),
        lambda row: row["responses_create_params"].update(model=None),
        lambda row: row.update(extra=object()),
    ],
)
async def test_run_malformed_task_makes_no_network_request(task, mutation):
    """Dataset errors fail before creating a remote attempt."""
    mutation(task)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(500)

    agent = NeMoGymSWEAgent("http://sandbox.test", model="qwen")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            await agent.run(
                task,
                base_url="http://proxy.test",
                api_key="session",
                http_client=client,
            )
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session",
    [
        {"base_url": "http://proxy.test/v1"},
        {"base_url": "http://user:secret@proxy.test"},
        {"api_key": ""},
        {"api_key": "secret\nheader"},
    ],
)
async def test_run_invalid_session_makes_no_network_request(task, session):
    """Only a proxy root URL and header-safe episode key are accepted."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(500)

    agent = NeMoGymSWEAgent("http://sandbox.test", model="qwen")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            await agent.run(
                task,
                http_client=client,
                **({"base_url": "http://proxy.test", "api_key": "session"} | session),
            )
    assert requests == []
