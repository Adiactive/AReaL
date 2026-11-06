# SPDX-License-Identifier: Apache-2.0

"""Context exhaustion is terminal and must not pollute captured trajectories."""

import threading

import httpx
import pytest

from tests.experimental.openai.vision_stubs import StubTokenizer

from areal.api import ModelRequest, ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.client import ArealOpenAI, ContextLengthExceededError
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData
from areal.utils.hf_utils import apply_chat_template

_MESSAGES = [{"role": "user", "content": "private prompt text"}]


class _RecordingEngine:
    def __init__(self, tokenizer: StubTokenizer) -> None:
        self.tokenizer = tokenizer
        self.requests: list[ModelRequest] = []

    async def agenerate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            input_tokens=list(request.input_ids),
            output_tokens=[101],
            output_logprobs=[-0.25],
            output_versions=[7],
            stop_reason="length",
            tokenizer=self.tokenizer,
        )


@pytest.fixture
def client_and_engine():
    """Use the production client, tokenizer path and cache without model downloads."""
    tokenizer = StubTokenizer()
    engine = _RecordingEngine(tokenizer)
    client = ArealOpenAI(
        engine=engine, tokenizer=tokenizer, chat_template_type="hf", api_key="test"
    )
    return client, engine


def _prompt_length(client: ArealOpenAI, messages: list[dict[str, str]]) -> int:
    return len(
        apply_chat_template(
            client.chat.completions.tokenizer,
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    )


@pytest.mark.parametrize("excess_tokens", [-1, 0, 1])
@pytest.mark.parametrize("limit_source", ["engine", "request", "minimum"])
@pytest.mark.asyncio
async def test_chat_context_boundary_preserves_generation_and_cache_semantics(
    client_and_engine, excess_tokens, limit_source
):
    """The tighter limit applies; exactly full prompts cannot generate a token."""
    client, engine = client_and_engine
    prompt_tokens = _prompt_length(client, _MESSAGES)
    max_tokens = prompt_tokens - excess_tokens
    request_options = {"max_completion_tokens": 8}
    if limit_source == "engine":
        client.chat.completions.engine_max_tokens = max_tokens
    elif limit_source == "request":
        request_options["max_total_tokens"] = max_tokens
    else:
        client.chat.completions.engine_max_tokens = max_tokens + 10
        request_options["max_total_tokens"] = max_tokens
    cache = InteractionCache()
    try:
        if excess_tokens >= 0:
            with pytest.raises(
                ContextLengthExceededError, match="context length"
            ) as exc:
                await client.chat.completions.create(
                    messages=_MESSAGES, areal_cache=cache, **request_options
                )
            assert exc.value.prompt_tokens == prompt_tokens
            assert exc.value.max_tokens == max_tokens
            assert _MESSAGES[0]["content"] not in str(exc.value)
            assert engine.requests == []
            assert cache == {}
        else:
            completion = await client.chat.completions.create(
                messages=_MESSAGES, areal_cache=cache, **request_options
            )
            assert len(engine.requests) == 1
            assert engine.requests[0].gconfig.max_new_tokens == 1
            assert engine.requests[0].gconfig.max_tokens == max_tokens
            assert set(cache) == {completion.id}
    finally:
        await client.close()


@pytest.mark.parametrize("excess_tokens", [-1, 0, 1])
@pytest.mark.asyncio
async def test_responses_context_boundary_preserves_generation_and_cache_semantics(
    client_and_engine, excess_tokens
):
    """Responses uses the same typed error and clears rejected interactions."""
    client, engine = client_and_engine
    prompt_tokens = _prompt_length(client, _MESSAGES)
    max_tokens = prompt_tokens - excess_tokens
    client.responses.engine_max_tokens = max_tokens
    cache = InteractionCache()
    try:
        if excess_tokens >= 0:
            with pytest.raises(
                ContextLengthExceededError, match="context length"
            ) as exc:
                await client.responses.create(
                    input=_MESSAGES, max_output_tokens=8, tools=[], areal_cache=cache
                )
            assert exc.value.prompt_tokens == prompt_tokens
            assert exc.value.max_tokens == max_tokens
            assert engine.requests == []
            assert cache == {}
        else:
            response = await client.responses.create(
                input=_MESSAGES, max_output_tokens=8, tools=[], areal_cache=cache
            )
            assert len(engine.requests) == 1
            assert engine.requests[0].gconfig.max_new_tokens == 1
            assert set(cache) == {response.id}
            assert response.usage.input_tokens_details.cached_tokens == 0
            assert response.usage.input_tokens_details.cache_write_tokens == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_failed_child_context_request_preserves_completed_interaction(
    client_and_engine,
):
    """Rejecting a later turn must leave earlier model records available to export."""
    client, engine = client_and_engine
    client.chat.completions.engine_max_tokens = 200
    cache = InteractionCache()
    try:
        completion = await client.chat.completions.create(
            messages=_MESSAGES, max_completion_tokens=1, areal_cache=cache
        )
        previous = cache[completion.id]
        child_messages = _MESSAGES + [
            completion.choices[0].message.model_dump(exclude_none=True),
            {"role": "user", "content": "long tool observation " * 20},
        ]
        with pytest.raises(ContextLengthExceededError):
            await client.chat.completions.create(
                messages=child_messages, max_completion_tokens=1, areal_cache=cache
            )
        assert len(engine.requests) == 1
        assert set(cache) == {completion.id}
        assert cache[completion.id] is previous
        assert previous.model_response.output_tokens == [101]
        assert previous.model_response.output_logprobs == [-0.25]
        assert previous.model_response.output_versions == [7]
        assert set(cache.export_interactions(style="individual")) == {completion.id}
    finally:
        await client.close()


@pytest.fixture
def proxy(monkeypatch, client_and_engine):
    """Use real HTTP validation, client tokenization and session storage."""
    client, engine = client_and_engine
    session = SessionData(session_id="episode")
    monkeypatch.setattr(srv, "_openai_client", client)
    monkeypatch.setattr(srv, "_session_cache", {"episode": session})
    monkeypatch.setattr(srv, "_api_key_to_session", {"episode-key": "episode"})
    monkeypatch.setattr(srv, "_session_to_api_key", {"episode": "episode-key"})
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    return client, engine, session


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("excess_tokens", [0, 1])
@pytest.mark.asyncio
async def test_chat_http_context_error_returns_safe_openai_400(
    proxy, stream, excess_tokens
):
    """Both streaming setup and regular calls expose a non-retryable input error."""
    client, engine, session = proxy
    prompt_tokens = _prompt_length(client, _MESSAGES)
    max_tokens = prompt_tokens - excess_tokens
    client.chat.completions.engine_max_tokens = max_tokens
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=srv.app), base_url="http://proxy"
        ) as http:
            response = await http.post(
                "/chat/completions",
                headers={"Authorization": "Bearer episode-key"},
                json={"model": "policy", "messages": _MESSAGES, "stream": stream},
            )
        assert response.status_code == 400
        assert response.json() == {
            "error": {
                "message": str(ContextLengthExceededError(prompt_tokens, max_tokens)),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded",
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
            }
        }
        assert _MESSAGES[0]["content"] not in response.text
        assert "episode-key" not in response.text
        assert engine.requests == []
        assert session.completions == {}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_non_context_value_error_remains_server_error(proxy):
    """An unrelated ValueError must not become a context termination signal."""
    client, engine, _ = proxy
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=srv.app), base_url="http://proxy"
        ) as http:
            response = await http.post(
                "/chat/completions",
                headers={"Authorization": "Bearer episode-key"},
                json={"model": "policy", "messages": []},
            )
        assert response.status_code == 500
        assert response.json() == {"detail": "messages cannot be empty"}
        assert engine.requests == []
    finally:
        await client.close()


@pytest.mark.parametrize(
    "path,body",
    [
        ("/responses", {"input": _MESSAGES}),
        ("/v1/messages", {"messages": _MESSAGES, "max_tokens": 1}),
        ("/v1/messages", {"messages": _MESSAGES, "max_tokens": 1, "stream": True}),
    ],
)
@pytest.mark.asyncio
async def test_other_api_wrappers_preserve_context_400(proxy, path, body):
    """Responses and both Anthropic wrappers must not hide the typed error."""
    client, engine, session = proxy
    max_tokens = _prompt_length(client, _MESSAGES)
    client.chat.completions.engine_max_tokens = max_tokens
    client.responses.engine_max_tokens = max_tokens
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=srv.app), base_url="http://proxy"
        ) as http:
            response = await http.post(
                path,
                headers={"Authorization": "Bearer episode-key"},
                json={"model": "policy", **body},
            )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert engine.requests == []
        assert session.completions == {}
    finally:
        await client.close()
