# SPDX-License-Identifier: Apache-2.0

"""Exercise the sandbox bridge against AReaL's real session proxy on CPU."""

import asyncio
import json
import threading
from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
import torch
from fastapi import FastAPI, Request

from examples.swe.mlperf.runner_bridge import create_app
from tests.experimental.openai.vision_stubs import EOS_ID, StubTokenizer

from areal.api import ModelRequest, ModelResponse, RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import deserialize_interactions
from areal.experimental.openai.proxy.tensor_reference import GroupTensorStoreRegistry
from areal.infra import workflow_context
from areal.infra.processor_cache import ProcessorCacheRegistry
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.workflow.nemo_gym_swe import NeMoGymSWEAgent

_ADMIN_KEY = "test-admin-key"


class _RecordingTokenizer(StubTokenizer):
    """Record per-episode template options at the actual tokenization boundary."""

    def __init__(self):
        self.template_calls: list[tuple[int, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any):
        episode = int(messages[0]["content"].removeprefix("Fix task "))
        self.template_calls.append((episode, kwargs.get("enable_thinking")))
        return super().apply_chat_template(messages, **kwargs)


class _ConcurrentEngine:
    """Require overlapping episodes and return distinguishable inference records."""

    def __init__(self, tokenizer: StubTokenizer, group_size: int = 2):
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.records: dict[tuple[int, int], ModelResponse] = {}
        self.first_turns = asyncio.Event()

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        episode = int(req.metadata["episode"])
        turn = int(req.metadata["turn"])
        version = 10 + episode * 10 + turn
        response = ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=[701 + episode, 711 + turn, EOS_ID],
            output_logprobs=[-0.125 * (episode + 1), -0.25 * (turn + 1), -0.375],
            output_versions=[version, version, version + 1],
            stop_reason="stop",
            tokenizer=self.tokenizer,
        )
        self.records[(episode, turn)] = response
        assert req.gconfig.temperature == 0.7
        assert req.gconfig.top_p == 0.9
        assert req.gconfig.max_new_tokens == 16
        if turn == 0:
            if all((other, 0) in self.records for other in range(self.group_size)):
                self.first_turns.set()
            await asyncio.wait_for(self.first_turns.wait(), timeout=5)
        return response


@pytest.fixture
def _reset_proxy(monkeypatch):
    """Isolate the real proxy's in-memory sessions without mocking capture."""
    monkeypatch.setattr(srv, "_session_cache", {})
    monkeypatch.setattr(srv, "_api_key_to_session", {})
    monkeypatch.setattr(srv, "_session_to_api_key", {})
    monkeypatch.setattr(srv, "_capacity", 0)
    monkeypatch.setattr(srv, "_admin_api_key", _ADMIN_KEY)
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    monkeypatch.setattr(srv, "_last_cleanup_time", 0.0)
    monkeypatch.setattr(srv, "_processor_cache_registry", ProcessorCacheRegistry())
    monkeypatch.setattr(srv, "_group_tensor_store_registry", GroupTensorStoreRegistry())


def _task(episode: int) -> dict[str, Any]:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": f"Fix task {episode}"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 16,
            "metadata": {
                "instance_id": f"task-{episode}",
                "dataset_name": "R2EGym",
                "instance_dict": json.dumps({"instance_id": f"task-{episode}"}),
            },
        }
    }


@pytest.mark.parametrize("context_limit_hit", [False, True])
@pytest.mark.asyncio
async def test_bridge_concurrent_sessions_preserve_inference_records(
    monkeypatch, _reset_proxy, context_limit_hit
):
    """Model calls, rewards and exported token records remain session-local."""
    tokenizer = _RecordingTokenizer()
    engine = _ConcurrentEngine(tokenizer)
    openai_client = ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        chat_template_type="hf",
        api_key="test",
        engine_max_tokens=32768,
    )
    monkeypatch.setattr(srv, "_openai_client", openai_client)
    gym = FastAPI()
    model_requests: list[tuple[int, str, str]] = []
    gym_routes: list[str] = []

    async def record_forwarded_request(request: httpx.Request) -> None:
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == 16
        assert "max_tokens" not in body
        model_requests.append(
            (
                int(body["metadata"]["episode"]),
                body["model"],
                request.headers["authorization"],
            )
        )

    @gym.post("/run")
    async def run_gym(request: Request) -> dict[str, Any]:
        # The native Gym receives an opaque route, never AReaL's session key.
        body = await request.json()
        assert "authorization" not in request.headers
        assert "x-areal-proxy-url" not in request.headers
        params = body["responses_create_params"]
        episode = int(params["metadata"]["instance_id"].removeprefix("task-"))
        gym_routes.append(params["model"])
        messages = deepcopy(params["input"])
        output = []
        for turn in range(2):
            if context_limit_hit and turn == 1:
                messages.append({"role": "user", "content": "x" * 32768})
            completion = await bridge_client.post(
                "/v1/chat/completions",
                json={
                    "model": params["model"],
                    "messages": messages,
                    "temperature": params["temperature"],
                    "top_p": params["top_p"],
                    "max_completion_tokens": None,
                    "chat_template_kwargs": {"enable_thinking": episode == 1},
                    "metadata": {"episode": str(episode), "turn": str(turn)},
                },
            )
            if context_limit_hit and turn == 1:
                assert completion.status_code == 400, completion.text
                error = completion.json()["error"]
                assert error["code"] == "context_length_exceeded"
                assert error["prompt_tokens"] >= error["max_tokens"] == 32768
                assert "context length" in completion.text
                # Pinned Gym signals context exhaustion to OpenHands with this
                # empty completion. It must never become a training interaction.
                output.append(
                    {"role": "assistant", "content": None, "tool_calls": None}
                )
                break
            assert completion.status_code == 200, completion.text
            message = completion.json()["choices"][0]["message"]
            assert message["tool_calls"] is None
            output.append(message)
            messages.append(message)
            messages.append({"role": "user", "content": "Run the next check"})
        return {
            "reward": float(episode == 0 and not context_limit_hit),
            "resolved": episode == 0 and not context_limit_hit,
            "instance_config": {"mask_sample": False},
            "agent_error_kind": "context_window" if context_limit_hit else None,
            "response": {"output": output},
        }

    try:
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=srv.app),
                base_url="http://proxy",
            ) as proxy_client,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=srv.app),
                event_hooks={"request": [record_forwarded_request]},
            ) as model_client,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gym), base_url="http://gym"
            ) as gym_client,
        ):
            bridge = create_app(
                "http://gym",
                concurrency=2,
                episode_timeout=10,
                gym_client=gym_client,
                model_client=model_client,
            )
            async with (
                bridge.router.lifespan_context(bridge),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=bridge),
                    base_url="http://bridge",
                ) as bridge_client,
            ):
                admin_headers = {"Authorization": f"Bearer {_ADMIN_KEY}"}
                sessions = []
                for episode in range(2):
                    grant = await proxy_client.post(
                        "/grant_capacity", headers=admin_headers
                    )
                    assert grant.status_code == 200
                    started = await proxy_client.post(
                        "/rl/start_session",
                        headers=admin_headers,
                        json={"task_id": f"episode-{episode}"},
                    )
                    assert started.status_code == 200
                    sessions.append(started.json())
                assert sessions[0]["api_key"] != sessions[1]["api_key"]

                rejected = await proxy_client.post(
                    "/chat/completions",
                    headers={"Authorization": "Bearer invalid-session"},
                    json={
                        "model": "policy",
                        "messages": _task(0)["responses_create_params"]["input"],
                    },
                )
                assert rejected.status_code == 401
                denied_export = await proxy_client.post(
                    "/export_trajectories",
                    headers={"Authorization": f"Bearer {sessions[0]['api_key']}"},
                    json={"session_id": sessions[1]["session_id"]},
                )
                assert denied_export.status_code == 403

                agent = NeMoGymSWEAgent(runner_url="http://bridge", model="policy")
                tasks = [_task(episode) for episode in range(2)]
                originals = deepcopy(tasks)
                rewards = await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            agent.run(
                                tasks[episode],
                                base_url="http://proxy",
                                api_key=session["api_key"],
                                http_client=bridge_client,
                            )
                            for episode, session in enumerate(sessions)
                        ),
                        return_exceptions=True,
                    ),
                    timeout=10,
                )
                assert rewards == ([0.0, 0.0] if context_limit_hit else [1.0, 0.0])
                turns = 1 if context_limit_hit else 2
                assert len(engine.records) == 2 * turns
                assert tasks == originals
                assert len(set(gym_routes)) == 2
                assert all(route != "policy" for route in gym_routes)
                assert len(model_requests) == 4
                assert sorted(tokenizer.template_calls) == [
                    (0, False),
                    (0, False),
                    (1, True),
                    (1, True),
                ]
                for episode, model, authorization in model_requests:
                    assert model == "policy"
                    assert authorization == f"Bearer {sessions[episode]['api_key']}"

                for episode, session in enumerate(sessions):
                    headers = {"Authorization": f"Bearer {session['api_key']}"}
                    reward_result = await proxy_client.post(
                        "/rl/set_reward",
                        headers=headers,
                        json={"reward": rewards[episode]},
                    )
                    assert reward_result.status_code == 200
                    ended = await proxy_client.post("/rl/end_session", headers=headers)
                    assert ended.status_code == 200
                    assert ended.json()["interaction_count"] == turns
                    exported = await proxy_client.post(
                        "/export_trajectories",
                        headers=admin_headers,
                        json={
                            "session_id": session["session_id"],
                            "style": "individual",
                        },
                    )
                    assert exported.status_code == 200
                    interactions = deserialize_interactions(
                        exported.json()["interactions"]
                    )
                    assert len(interactions) == turns
                    for turn, interaction in enumerate(interactions.values()):
                        expected = engine.records[(episode, turn)]
                        tensors = interaction.to_tensor_dict()
                        expected_tensors = {
                            "input_ids": [
                                expected.input_tokens + expected.output_tokens
                            ],
                            "loss_mask": [
                                [0] * expected.input_len + [1] * expected.output_len
                            ],
                            "logprobs": [
                                [0.0] * expected.input_len + expected.output_logprobs
                            ],
                            "versions": [
                                [-1] * expected.input_len + expected.output_versions
                            ],
                            "rewards": [rewards[episode]],
                        }
                        for name, values in expected_tensors.items():
                            torch.testing.assert_close(
                                tensors[name], torch.tensor(values), rtol=0, atol=0
                            )

                assert not srv._session_cache
                assert not srv._api_key_to_session
                assert not srv._session_to_api_key
                for route in gym_routes:
                    expired = await bridge_client.post(
                        "/v1/chat/completions", json={"model": route, "messages": []}
                    )
                    assert expired.status_code == 404
    finally:
        await openai_client.close()


@pytest.mark.parametrize("rewards", [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
@pytest.mark.asyncio
async def test_masked_gym_result_preserves_group_rewards_and_training_records(rewards):
    """A masked success must finish without rejecting its pending siblings."""
    tokenizer = _RecordingTokenizer()
    engine = _ConcurrentEngine(tokenizer, group_size=4)
    openai_client = ArealOpenAI(
        engine=engine, tokenizer=tokenizer, chat_template_type="hf", api_key="test"
    )
    agent = NeMoGymSWEAgent(runner_url="http://runner", model="policy")
    masked_finished = asyncio.Event()
    completed = []

    class GymWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            episode = workflow_context.get().sample_idx
            cache = InteractionCache()
            task = _task(episode)

            async def run_gym(request):
                if request.method == "DELETE":
                    return httpx.Response(200)
                completion = await openai_client.chat.completions.create(
                    messages=task["responses_create_params"]["input"],
                    temperature=0.7,
                    top_p=0.9,
                    max_completion_tokens=16,
                    metadata={"episode": str(episode), "turn": "0"},
                    areal_cache=cache,
                )
                if episode != 0:
                    await asyncio.wait_for(masked_finished.wait(), timeout=5)
                return httpx.Response(
                    200,
                    json={
                        "reward": rewards[episode],
                        "resolved": bool(rewards[episode]),
                        "instance_config": {"mask_sample": episode == 0},
                        "agent_error_kind": "max_iteration" if episode == 0 else None,
                        "agent_timed_out": None,
                        "eval_timed_out": None,
                        "response": {
                            "output": [completion.choices[0].message.model_dump()]
                        },
                    },
                )

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(run_gym)
            ) as runner_client:
                reward = await agent.run(
                    task,
                    base_url="http://proxy",
                    api_key="test-session",
                    http_client=runner_client,
                )
            completed.append(episode)
            if episode == 0:
                masked_finished.set()
            cache.set_last_reward(reward)
            return cache.export_interactions(style="individual")

    try:
        grouped = GroupedRolloutWorkflow(GymWorkflow(), group_size=4, logger=Mock())
        result = await asyncio.wait_for(grouped.arun_episode(engine, {}), timeout=10)
        assert completed[0] == 0
        assert sorted(completed) == list(range(4))
        assert result is not None
        torch.testing.assert_close(
            result["rewards"], torch.tensor(rewards), rtol=0, atol=0
        )
        torch.testing.assert_close(
            result["begin_of_trajectory"],
            torch.ones(4, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        for episode in range(4):
            expected = engine.records[(episode, 0)]
            expected_tensors = {
                "input_ids": expected.input_tokens + expected.output_tokens,
                "loss_mask": [0] * expected.input_len + [1] * expected.output_len,
                "logprobs": [0.0] * expected.input_len + expected.output_logprobs,
                "versions": [-1] * expected.input_len + expected.output_versions,
            }
            for name, values in expected_tensors.items():
                torch.testing.assert_close(
                    result[name][episode], torch.tensor(values), rtol=0, atol=0
                )
    finally:
        await openai_client.close()
