"""CPU-only contract tests for the standalone Gym/AReaL routing bridge."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager
from copy import deepcopy
from uuid import uuid4

import httpx
import pytest
from aiohttp import web

from examples.swe.mlperf import runner_bridge
from examples.swe.mlperf.runner_bridge import create_app


def _headers(key="episode-secret", attempt_id=None, proxy="http://proxy:9000"):
    return {
        "Authorization": f"Bearer {key}",
        "X-AReaL-Attempt-Id": attempt_id or str(uuid4()),
        "X-AReaL-Proxy-Url": proxy,
    }


def _row(model="trained-model"):
    return {
        "responses_create_params": {
            "model": model,
            "input": [{"role": "user", "content": "Fix the issue"}],
            "temperature": 0.8,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        },
        "instance_id": "test-instance",
        "meta": {"instance_dict": {"hidden_tests": "private-test-data"}},
    }


def _completion(identifier="completion-123"):
    return {
        "id": identifier,
        "model": "trained-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Done"},
                "finish_reason": "stop",
            }
        ],
    }


def _chat(token, **overrides):
    return {
        "model": token,
        "messages": [{"role": "user", "content": "Fix the issue"}],
        **overrides,
    }


@asynccontextmanager
async def _bridge(gym_handler, model_handler, **kwargs):
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(gym_handler)) as gym,
        httpx.AsyncClient(transport=httpx.MockTransport(model_handler)) as model,
    ):
        app = create_app(
            "http://gym:8083", gym_client=gym, model_client=model, **kwargs
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://bridge"
            ) as client,
        ):
            yield client


@pytest.mark.asyncio
async def test_concurrency_above_100_reaches_gym_and_model_without_pool_wait(ip_stack):
    """Owned HTTP pools admit every episode and its model call concurrently."""
    if not ip_stack["ipv4_loopback"]:
        pytest.skip("IPv4 loopback sockets are unavailable")

    concurrency = 128
    model_calls = 0
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def model(request):
        nonlocal model_calls
        model_calls += 1
        if model_calls == concurrency:
            all_started.set()
        await release.wait()
        return web.json_response(_completion())

    async def gym(request):
        body = await request.json()
        token = body["responses_create_params"]["model"]
        response = await client.post("/v1/chat/completions", json=_chat(token))
        assert response.status_code == 200
        return web.json_response({"reward": 0})

    upstream = web.Application()
    upstream.router.add_post("/run", gym)
    upstream.router.add_post("/chat/completions", model)
    runner = web.AppRunner(upstream)
    await runner.setup()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        await web.SockSite(runner, sock, backlog=concurrency * 2).start()
        app = create_app(url, concurrency=concurrency)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://bridge"
            ) as client,
        ):
            requests = [
                asyncio.create_task(
                    client.post("/run", headers=_headers(proxy=url), json=_row())
                )
                for _ in range(concurrency)
            ]
            try:
                await asyncio.wait_for(all_started.wait(), timeout=5)
                assert model_calls == concurrency
            finally:
                release.set()
                results = await asyncio.wait_for(asyncio.gather(*requests), timeout=5)
            assert all(response.status_code == 200 for response in results)
    finally:
        await runner.cleanup()
        sock.close()


@pytest.mark.asyncio
async def test_concurrent_episodes_keep_keys_and_completion_ids_isolated():
    """Each opaque model route maps back to its original model and episode key."""
    seen_gym = []
    seen_proxy = []
    tools = [{"type": "function", "function": {"name": "run", "parameters": {}}}]

    async def model(request):
        body = json.loads(request.content)
        seen_proxy.append((str(request.url), request.headers["authorization"], body))
        await asyncio.sleep(0)
        return httpx.Response(200, json=_completion(f"id-{body['model']}"))

    async def gym(request):
        assert str(request.url) == "http://gym:8083/run"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        seen_gym.append(body)
        token = body["responses_create_params"]["model"]
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": token,
                "messages": [{"role": "user", "content": "Implement"}],
                "tools": tools,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 42,
                "stop": ["END"],
                "seed": 7,
                "stream": False,
                "n": 1,
            },
        )
        assert response.status_code == 200
        return httpx.Response(
            200,
            json={
                "reward": 0,
                "instance_config": {"mask_sample": False},
                "completion": response.json(),
            },
        )

    async with _bridge(gym, model) as client:
        rows = [_row("model-a"), _row("model-b")]
        original = deepcopy(rows)
        results = await asyncio.gather(
            client.post("/run", headers=_headers("key-a"), json=rows[0]),
            client.post("/run", headers=_headers("key-b"), json=rows[1]),
        )

    assert rows == original
    assert all(result.status_code == 200 for result in results)
    assert {r.json()["completion"]["id"] for r in results} == {
        "id-model-a",
        "id-model-b",
    }
    assert len({row["responses_create_params"]["model"] for row in seen_gym}) == 2
    assert all(row["meta"] == original[0]["meta"] for row in seen_gym)
    assert "key-a" not in json.dumps(seen_gym)
    assert "key-b" not in json.dumps([r.json() for r in results])
    assert {(key, body["model"]) for _, key, body in seen_proxy} == {
        ("Bearer key-a", "model-a"),
        ("Bearer key-b", "model-b"),
    }
    for url, _, body in seen_proxy:
        assert url == "http://proxy:9000/chat/completions"
        assert body["tools"] == tools
        assert "max_tokens" not in body
        assert {
            k: body[k]
            for k in ("temperature", "top_p", "max_completion_tokens", "stop", "seed")
        } == {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_completion_tokens": 42,
            "stop": ["END"],
            "seed": 7,
        }


@pytest.mark.asyncio
async def test_cancel_running_attempt_revokes_model_and_holds_slot_until_gym_drains():
    """Cancellation does not admit another job while the old Ray job is alive."""
    gym_started = asyncio.Event()
    model_started = asyncio.Event()
    model_cancelled = asyncio.Event()
    release_gym = asyncio.Event()
    started_models = []
    route_tokens = []

    async def model(request):
        model_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            model_cancelled.set()

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        route_tokens.append(token)
        started_models.append(token)
        gym_started.set()
        if len(started_models) == 1:
            callback = asyncio.create_task(
                client.post("/v1/chat/completions", json=_chat(token))
            )
            await model_started.wait()
            await release_gym.wait()
            assert (await callback).status_code == 410
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(gym, model, concurrency=1) as client:
        headers = _headers()
        first = asyncio.create_task(client.post("/run", headers=headers, json=_row()))
        await asyncio.wait_for(model_started.wait(), 2)
        second = asyncio.create_task(
            client.post("/run", headers=_headers("second-key"), json=_row())
        )
        response = await client.delete(
            f"/attempts/{headers['X-AReaL-Attempt-Id']}", headers=headers
        )
        assert response.json() == {"status": "draining", "sandbox_job_stopped": False}
        await asyncio.wait_for(model_cancelled.wait(), 2)
        late = await client.post(
            "/v1/chat/completions", json={"model": route_tokens[0]}
        )
        assert late.status_code == 404
        await asyncio.sleep(0)
        assert len(started_models) == 1
        assert not first.done()
        release_gym.set()
        assert (await asyncio.wait_for(first, 2)).status_code == 409
        assert (await asyncio.wait_for(second, 2)).status_code == 200
        assert len(started_models) == 2


@pytest.mark.asyncio
async def test_cancel_queued_attempt_never_reaches_gym():
    """An authenticated cancellation removes queued work without starting it."""
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def gym(request):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(gym, lambda _: httpx.Response(200), concurrency=1) as client:
        first = asyncio.create_task(
            client.post("/run", headers=_headers(), json=_row())
        )
        await asyncio.wait_for(started.wait(), 2)
        headers = _headers("queued-key", attempt_id=uuid4().hex)
        second = asyncio.create_task(client.post("/run", headers=headers, json=_row()))
        for _ in range(20):
            if (await client.get("/health")).json()["queued_attempts"]:
                break
            await asyncio.sleep(0)
        path = f"/attempts/{headers['X-AReaL-Attempt-Id']}"
        assert (
            await client.delete(path, headers=_headers("wrong-key"))
        ).status_code == 403
        assert (await client.delete(path, headers=headers)).json()[
            "status"
        ] == "cancelled"
        assert (await asyncio.wait_for(second, 2)).status_code == 409
        release.set()
        assert (await asyncio.wait_for(first, 2)).status_code == 200
        assert calls == 1


@pytest.mark.asyncio
async def test_caller_task_cancellation_keeps_job_draining():
    """Losing the HTTP caller cannot silently release an occupied sandbox slot."""
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def gym(request):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(gym, lambda _: httpx.Response(200), concurrency=1) as client:
        first = asyncio.create_task(
            client.post("/run", headers=_headers(), json=_row())
        )
        await asyncio.wait_for(started.wait(), 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert (await client.get("/health")).json()["active_attempts"] == 1
        second = asyncio.create_task(
            client.post("/run", headers=_headers(), json=_row())
        )
        await asyncio.sleep(0)
        assert calls == 1
        release.set()
        assert (await asyncio.wait_for(second, 2)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_model_http_failure_cannot_be_reported_as_zero_reward(status):
    """Even a Gym success response cannot hide failed model calls."""

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        result = await client.post("/v1/chat/completions", json=_chat(token))
        assert result.status_code == 502
        assert "sensitive-error" not in result.text
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(
        gym,
        lambda _: httpx.Response(
            status, json={"error": "sensitive-error context length"}
        ),
    ) as client:
        response = await client.post("/run", headers=_headers(), json=_row())
    assert response.status_code == 502
    assert "sensitive-error" not in response.text
    assert "reward" not in response.json()


def _context_error(**overrides):
    return {
        "error": {
            "code": "context_length_exceeded",
            "type": "invalid_request_error",
            "param": "messages",
            "message": "sensitive upstream content",
            "prompt_tokens": 32900,
            "max_tokens": 32768,
            **overrides,
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_tokens", [32768, 32900])
@pytest.mark.parametrize("mask_sample", [False, True])
async def test_context_limit_preserves_native_grading_and_mask_without_completion(
    prompt_tokens, mask_sample
):
    """A typed overflow ends generation without changing Gym's grading result."""
    model_calls = 0
    native_result = {
        "reward": int(mask_sample),
        "instance_config": {
            "mask_sample": mask_sample,
            "termination_reason": "context_window_exceeded",
        },
        "completion_ids": ["recorded-completion"],
    }

    def model(request):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return httpx.Response(200, json=_completion("recorded-completion"))
        return httpx.Response(
            400,
            json=_context_error(
                prompt_tokens=prompt_tokens,
                type="sensitive upstream type",
                param="sensitive upstream param",
                private_field="sensitive upstream field",
            ),
        )

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        first = await client.post("/v1/chat/completions", json=_chat(token))
        assert first.status_code == 200
        assert first.json()["id"] == "recorded-completion"
        terminal = await client.post("/v1/chat/completions", json=_chat(token))
        assert terminal.status_code == 400
        assert terminal.json() == {
            "error": {
                "code": "context_length_exceeded",
                "type": "invalid_request_error",
                "param": "messages",
                "message": (
                    f"Prompt has {prompt_tokens} tokens, reaching or exceeding "
                    "the context length limit of 32768 tokens."
                ),
                "prompt_tokens": prompt_tokens,
                "max_tokens": 32768,
            }
        }
        assert "context length" in terminal.json()["error"]["message"]
        assert "sensitive" not in terminal.text
        assert "id" not in terminal.json()
        assert "choices" not in terminal.json()
        return httpx.Response(200, json=native_result)

    async with _bridge(gym, model) as client:
        response = await client.post("/run", headers=_headers(), json=_row())
        assert (await client.get("/health")).json()["active_attempts"] == 0
    assert response.status_code == 200
    assert response.json() == native_result
    assert model_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, _context_error(code="invalid_request_error")),
        (400, _context_error(code=None)),
        (400, _context_error(prompt_tokens=None)),
        (400, _context_error(max_tokens=None)),
        (400, _context_error(prompt_tokens="32900")),
        (400, _context_error(max_tokens="32768")),
        (400, _context_error(prompt_tokens=32900.0)),
        (400, _context_error(max_tokens=32768.0)),
        (400, _context_error(prompt_tokens=True, max_tokens=1)),
        (400, _context_error(max_tokens=True)),
        (400, _context_error(prompt_tokens=-1)),
        (400, _context_error(max_tokens=0)),
        (400, _context_error(max_tokens=-1)),
        (400, _context_error(prompt_tokens=32767)),
        (400, {"error": ["sensitive context length"]}),
        (400, ["sensitive context length"]),
        (429, _context_error()),
        (500, _context_error()),
    ],
)
async def test_unrecognized_context_error_poisons_episode_without_leaking_body(
    status, body, caplog, monkeypatch
):
    """Only the precise context-limit protocol can bypass model failure tracking."""
    # AReaL tests can replace the root logger, detaching pytest's capture handler.
    monkeypatch.setattr(runner_bridge.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(runner_bridge.logger, "propagate", False)
    monkeypatch.setattr(runner_bridge.logger, "disabled", False)

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        response = await client.post("/v1/chat/completions", json=_chat(token))
        assert response.status_code == 502
        assert "sensitive" not in response.text
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(gym, lambda _: httpx.Response(status, json=body)) as client:
        response = await client.post("/run", headers=_headers(), json=_row())
    assert response.status_code == 502
    assert "sensitive" not in response.text
    assert "sensitive" not in caplog.text
    assert f"HTTP {status}" in caplog.text
    assert "reward" not in response.json()


@pytest.mark.asyncio
async def test_non_json_context_error_poisons_episode():
    """A context-length phrase in unstructured error text is not trusted."""

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        response = await client.post("/v1/chat/completions", json=_chat(token))
        assert response.status_code == 502
        assert "sensitive" not in response.text
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(
        gym,
        lambda _: httpx.Response(400, text="sensitive context length"),
    ) as client:
        response = await client.post("/run", headers=_headers(), json=_row())
    assert response.status_code == 502


@pytest.mark.asyncio
async def test_context_limit_does_not_clear_earlier_model_failure():
    """An expected terminal response cannot hide an earlier infrastructure error."""
    model_calls = 0

    def model(request):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return httpx.Response(500, text="sensitive upstream error")
        return httpx.Response(400, json=_context_error())

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        assert (
            await client.post("/v1/chat/completions", json=_chat(token))
        ).status_code == 502
        assert (
            await client.post("/v1/chat/completions", json=_chat(token))
        ).status_code == 400
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(gym, model) as client:
        response = await client.post("/run", headers=_headers(), json=_row())
    assert response.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "transport", "server_error"])
async def test_unknown_gym_job_state_closes_admission(mode):
    """A lost Gym response requires operator reconciliation before more work."""

    async def gym(request):
        if mode == "transport":
            raise httpx.ConnectError("secret error body")
        if mode == "server_error":
            return httpx.Response(500, json={"error": "secret error body"})
        await asyncio.Event().wait()

    async with _bridge(
        gym, lambda _: httpx.Response(200), episode_timeout=0.02
    ) as client:
        result = await client.post("/run", headers=_headers(), json=_row())
        assert result.status_code == 503
        assert "secret error" not in result.text
        assert (await client.get("/health")).status_code == 503
        assert (
            await client.post("/run", headers=_headers(), json=_row())
        ).status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"stream": True}, {"n": 2}])
async def test_unsupported_completion_request_poisons_episode(change):
    """Streaming and multiple choices cannot bypass episode failure tracking."""

    def model(request):
        pytest.fail("Unsupported request reached AReaL")

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        result = await client.post("/v1/chat/completions", json=_chat(token, **change))
        assert result.status_code == 400
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(gym, model) as client:
        assert (
            await client.post("/run", headers=_headers(), json=_row())
        ).status_code == 502


@pytest.mark.asyncio
async def test_invalid_request_and_unknown_route_are_rejected():
    """Malformed inputs never launch jobs or forward requests to a model."""

    def unexpected(request):
        pytest.fail("Invalid request reached an upstream service")

    async with _bridge(unexpected, unexpected) as client:
        assert (await client.post("/run", json=_row())).status_code == 401
        assert (
            await client.post("/run", headers=_headers(), json=[])
        ).status_code == 400
        assert (
            await client.post("/run", headers=_headers(), content="{")
        ).status_code == 400
        assert (
            await client.post("/run", headers=_headers(), json={})
        ).status_code == 400
        assert (
            await client.post(
                "/run", headers=_headers(proxy="http://proxy/v1"), json=_row()
            )
        ).status_code == 400
        assert (
            await client.post("/v1/chat/completions", json={"model": "unknown"})
        ).status_code == 404
        assert (await client.post("/v1/chat/completions", json=[])).status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [{}, {"id": "x", "choices": []}, ["invalid"]])
async def test_invalid_model_completion_poisons_episode(result):
    """Incomplete responses must not become apparently valid failed samples."""

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        callback = await client.post("/v1/chat/completions", json=_chat(token))
        assert callback.status_code == 502
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(gym, lambda _: httpx.Response(200, json=result)) as client:
        assert (
            await client.post("/run", headers=_headers(), json=_row())
        ).status_code == 502


@pytest.mark.asyncio
async def test_openhands_null_message_fields_are_normalized_without_changing_tools():
    """Response-shaped messages become valid requests without editing tool schemas."""
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "run", "arguments": '{"command":"ls"}'},
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": ["string", "null"], "default": None}
                    },
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Fix the issue", "name": None},
        {
            "role": "assistant",
            "content": "Inspecting",
            "tool_calls": None,
            "function_call": None,
            "refusal": None,
            "annotations": None,
            "audio": None,
        },
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call-1", "content": "README.md"},
    ]
    original = deepcopy(messages)

    def model(request):
        body = json.loads(request.content)
        assert body["messages"] == [
            {"role": "user", "content": "Fix the issue"},
            {"role": "assistant", "content": "Inspecting"},
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": "call-1", "content": "README.md"},
        ]
        assert body["tools"] == tools
        return httpx.Response(200, json=_completion())

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        callback = await client.post(
            "/v1/chat/completions", json=_chat(token, messages=messages, tools=tools)
        )
        assert callback.status_code == 200
        return httpx.Response(200, json={"reward": 1})

    async with _bridge(gym, model) as client:
        response = await client.post("/run", headers=_headers(), json=_row())
    assert response.status_code == 200
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", [None, {}, [], "invalid", [None], ["invalid"]])
async def test_malformed_message_list_poisons_episode(messages):
    """Malformed histories cannot become apparently valid zero-reward samples."""

    def model(request):
        pytest.fail("Malformed messages reached AReaL")

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        callback = await client.post(
            "/v1/chat/completions", json=_chat(token, messages=messages)
        )
        assert callback.status_code == 400
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(gym, model) as client:
        response = await client.post("/run", headers=_headers(), json=_row())
    assert response.status_code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({}, 1024),
        ({"max_tokens": None, "max_completion_tokens": None}, 1024),
        ({"max_tokens": 42}, 42),
        ({"max_completion_tokens": 16}, 16),
        ({"max_tokens": 42, "max_completion_tokens": 16}, 16),
        ({"max_tokens": 16, "max_completion_tokens": 42}, 16),
        ({"max_tokens": 4096, "max_completion_tokens": 2048}, 1024),
        ({"max_tokens": 42, "max_completion_tokens": None}, 42),
    ],
)
async def test_completion_limits_enforce_episode_cap_and_preserve_stricter_limits(
    limits, expected
):
    """OpenHands null limits cannot silently remove the configured generation cap."""

    def model(request):
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == expected
        assert "max_tokens" not in body
        return httpx.Response(200, json=_completion())

    async def gym(request):
        body = json.loads(request.content)
        assert body["responses_create_params"]["max_output_tokens"] == 1024
        token = body["responses_create_params"]["model"]
        callback = await client.post(
            "/v1/chat/completions", json=_chat(token, **limits)
        )
        assert callback.status_code == 200
        return httpx.Response(200, json={"reward": 1})

    async with _bridge(gym, model) as client:
        response = await client.post("/run", headers=_headers(), json=_row())
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "1024"])
async def test_invalid_episode_token_limit_prevents_launch(value):
    """Every admitted attempt must have a finite positive integer token budget."""

    def unexpected(request):
        pytest.fail("Invalid token limit reached upstream")

    row = _row()
    row["responses_create_params"]["max_output_tokens"] = value
    async with _bridge(unexpected, unexpected) as client:
        response = await client.post("/run", headers=_headers(), json=row)
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024"])
async def test_invalid_completion_token_limit_poisons_episode(name, value):
    """Invalid callback limits cannot reach AReaL or masquerade as zero reward."""

    def model(request):
        pytest.fail("Invalid token limit reached AReaL")

    async def gym(request):
        token = json.loads(request.content)["responses_create_params"]["model"]
        callback = await client.post(
            "/v1/chat/completions", json=_chat(token, **{name: value})
        )
        assert callback.status_code == 400
        return httpx.Response(200, json={"reward": 0})

    async with _bridge(gym, model) as client:
        response = await client.post("/run", headers=_headers(), json=_row())
    assert response.status_code == 502
