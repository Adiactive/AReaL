# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import httpx
import pytest
from pydantic import TypeAdapter

from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData


@pytest.fixture
def proxy(monkeypatch):
    """Use real HTTP validation with a lightweight client at the model boundary."""
    calls = []

    async def create(messages, areal_cache, temperature, top_p, extra_body=None):
        calls.append(extra_body)
        return {"id": "test-completion", "choices": []}

    monkeypatch.setattr(
        srv,
        "_openai_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    monkeypatch.setattr(
        srv, "_session_cache", {"episode": SessionData(session_id="episode")}
    )
    monkeypatch.setattr(srv, "_api_key_to_session", {"episode-key": "episode"})
    monkeypatch.setattr(srv, "_session_to_api_key", {"episode": "episode-key"})
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    return calls


@pytest.mark.parametrize("nested", [None, {}, {"enable_thinking": True, "keep": 7}])
@pytest.mark.asyncio
async def test_http_template_options_override_nested_defaults(proxy, nested):
    """Top-level server extensions survive TypedDict validation and reach the client."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://proxy"
    ) as client:
        response = await client.post(
            "/chat/completions",
            headers={"Authorization": "Bearer episode-key"},
            json={
                "model": "policy",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
                "extra_body": {"chat_template_kwargs": nested},
            },
        )
    assert response.status_code == 200
    assert proxy == [
        {"chat_template_kwargs": (nested or {}) | {"enable_thinking": False}}
    ]


@pytest.mark.parametrize("nested", ["bad", [], 1, False])
@pytest.mark.asyncio
async def test_http_invalid_nested_template_options_return_validation_error(
    proxy, nested
):
    """Malformed extensions must fail as input errors before model execution."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://proxy"
    ) as client:
        response = await client.post(
            "/chat/completions",
            headers={"Authorization": "Bearer episode-key"},
            json={
                "model": "policy",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
                "extra_body": {"chat_template_kwargs": nested},
            },
        )
    assert response.status_code == 422
    assert proxy == []


@pytest.mark.parametrize("stream", [False, True])
def test_request_schema_retains_extensions_for_both_stream_modes(stream):
    """The extension supports both branches of the original OpenAI union."""
    request = TypeAdapter(srv._ChatCompletionRequest).validate_python(
        {
            "model": "policy",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    assert request["stream"] is stream
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("seed", [None, 0, 42])
@pytest.mark.asyncio
async def test_http_seed_warns_only_for_unsupported_explicit_values(
    proxy, monkeypatch, seed
):
    """A null seed is harmless; an ignored explicit seed must remain visible."""
    warnings = []
    monkeypatch.setattr(srv, "_warn_once", warnings.append)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://proxy"
    ) as client:
        response = await client.post(
            "/chat/completions",
            headers={"Authorization": "Bearer episode-key"},
            json={
                "model": "policy",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 1.0,
                "top_p": 1.0,
                "seed": seed,
            },
        )
    assert response.status_code == 200
    assert proxy == [None]
    assert warnings == (
        []
        if seed is None
        else [
            "dropped unsupported non-default arguments for areal client:\n"
            f"  seed: {seed}"
        ]
    )
