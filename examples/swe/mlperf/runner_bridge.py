"""Route self-hosted Gym episodes through AReaL's per-episode model proxy.

Run exactly one worker. Cancelling a request revokes model access, but cannot
stop Gym's Ray job. Started jobs retain their admission slot while draining;
an uncertain job outcome closes admission until the service is restarted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import math
from collections import Counter
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("MLPerfRunnerBridge")

_MEMBERSHIP_REFRESH_SECONDS = 5


@dataclass
class _Attempt:
    attempt_id: str
    received_at: float
    token: str
    proxy_url: str
    model: str
    max_output_tokens: int
    api_key: str = field(repr=False)
    key_digest: bytes = field(repr=False)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    model_requests: set[asyncio.Task] = field(default_factory=set, repr=False)
    first_callback_seen: bool = False
    started: bool = False
    active: bool = True
    invalid_callback: bool = False
    failed: bool = False


class _Metrics:
    def __init__(self) -> None:
        self.attempts = 0
        self.finished_without_callback = 0
        self.sandbox_failures = 0
        self.unknown_outcomes = 0
        self.seconds: Counter[int] = Counter()
        self.minimum = math.inf
        self.maximum = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "sandbox_failures": self.sandbox_failures,
            "finished_without_callback": self.finished_without_callback,
            "unknown_outcomes": self.unknown_outcomes,
            "first_llm_request_seconds": self.latency(),
        }

    def observe(self, seconds: float) -> None:
        self.minimum = min(self.minimum, seconds)
        self.maximum = max(self.maximum, seconds)
        self.seconds[math.ceil(seconds)] += 1

    def latency(self) -> dict[str, Any]:
        count = self.seconds.total()
        # Nearest-rank quantiles, rounded up to the one-second bucket edge.
        ranks = {
            name: math.ceil(count * q) for name, q in (("p50", 0.5), ("p99", 0.99))
        }
        quantiles = dict.fromkeys(ranks)
        cumulative = 0
        for upper, size in sorted(self.seconds.items()):
            cumulative += size
            for name, rank in ranks.items():
                if quantiles[name] is None and cumulative >= rank:
                    quantiles[name] = min(upper, self.maximum)
        return {
            "count": count,
            "min": self.minimum if count else None,
            "max": self.maximum if count else None,
            **quantiles,
            "cumulative_buckets": {
                str(limit): sum(
                    n for upper, n in self.seconds.items() if upper <= limit
                )
                for limit in (5, 15, 30, 60, 120, 300)
            },
        }


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status)


def _context_length_error(response: httpx.Response) -> JSONResponse | None:
    if response.status_code != 400:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != "context_length_exceeded":
        return None
    prompt_tokens, max_tokens = error.get("prompt_tokens"), error.get("max_tokens")
    if (
        type(prompt_tokens) is not int
        or type(max_tokens) is not int
        or max_tokens <= 0
        or prompt_tokens < max_tokens
    ):
        return None
    # Forward only the typed overflow, never upstream messages or prompt content.
    return JSONResponse(
        {
            "error": {
                "code": "context_length_exceeded",
                "type": "invalid_request_error",
                "param": "messages",
                "message": (
                    f"Prompt has {prompt_tokens} tokens, reaching or exceeding "
                    f"the context length limit of {max_tokens} tokens."
                ),
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
            }
        },
        status_code=400,
    )


def _root_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
            and not parsed.username
            and not parsed.password
        )
        parsed.port  # Validate a supplied port without including it in errors.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Expected an HTTP service root without /v1 or credentials")
    return value.rstrip("/")


def _bearer(request: Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token or any(c.isspace() for c in token):
        raise HTTPException(401, "An episode bearer key is required")
    return token


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(400, "Expected a JSON object") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    return body


def _live_ray_nodes(address: str) -> int:
    from ray.autoscaler.v2.sdk import get_cluster_status

    status = get_cluster_status(address, timeout=2)
    return len(status.active_nodes) + len(status.idle_nodes)


def create_app(
    gym_url: str,
    *,
    concurrency: int = 4,
    ray_head_address: str | None = None,
    episode_timeout: float = 3600,
    metrics_idle_reset_seconds: float = 900,
    gym_client: httpx.AsyncClient | None = None,
    model_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Create the bridge; injected clients remain owned by their caller."""
    gym_url = _root_url(gym_url)
    if concurrency < 1 or episode_timeout <= 0 or not math.isfinite(episode_timeout):
        raise ValueError("Concurrency and episode timeout must be positive")

    if not math.isfinite(metrics_idle_reset_seconds) or metrics_idle_reset_seconds <= 0:
        raise ValueError("Metrics idle reset seconds must be positive and finite")
    window = _Metrics()
    total = _Metrics()
    idle_since = None

    def window_snapshot() -> dict[str, Any]:
        return {"idle_reset_seconds": metrics_idle_reset_seconds, **window.snapshot()}

    def expire_idle_window() -> None:
        nonlocal window, idle_since
        if (
            idle_since is not None
            and monotonic() - idle_since >= metrics_idle_reset_seconds
        ):
            logger.info("metrics_window_reset %s", json.dumps(window_snapshot()))
            window = _Metrics()
            idle_since = None

    attempts: dict[str, _Attempt] = {}
    routes: dict[str, _Attempt] = {}
    jobs: set[asyncio.Task] = set()
    slots = asyncio.Semaphore(concurrency)
    admission_closed = False
    owned_gym_client = gym_client is None
    owned_model_client = model_client is None
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    gym_client = gym_client or httpx.AsyncClient(
        trust_env=False, follow_redirects=False, limits=limits
    )
    model_client = model_client or httpx.AsyncClient(
        trust_env=False, follow_redirects=False, limits=limits
    )

    def revoke(attempt: _Attempt) -> None:
        attempt.active = False
        routes.pop(attempt.token, None)
        attempt.api_key = ""
        for task in tuple(attempt.model_requests):
            task.cancel()

    def cancel(attempt: _Attempt) -> None:
        attempt.cancelled.set()
        revoke(attempt)

    running_runners = None

    async def refresh_membership(address: str) -> None:
        nonlocal running_runners
        while True:
            try:
                running_runners = await asyncio.to_thread(_live_ray_nodes, address)
            except Exception:
                running_runners = None
            await asyncio.sleep(_MEMBERSHIP_REFRESH_SECONDS)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        monitor = (
            asyncio.create_task(refresh_membership(ray_head_address))
            if ray_head_address
            else None
        )
        try:
            yield
        finally:
            if monitor is not None:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
        # Shutdown cannot prove that remote Ray jobs stopped. Restart only after
        # the operator has reconciled outstanding jobs on the sandbox host.
        for attempt in tuple(attempts.values()):
            cancel(attempt)
        for task in tuple(jobs):
            task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        if owned_gym_client:
            await gym_client.aclose()
        if owned_model_client:
            await model_client.aclose()

    app = FastAPI(lifespan=lifespan)

    async def acquire_slot(attempt: _Attempt) -> bool:
        acquire = asyncio.create_task(slots.acquire())
        cancelled = asyncio.create_task(attempt.cancelled.wait())
        try:
            await asyncio.wait(
                {acquire, cancelled}, return_when=asyncio.FIRST_COMPLETED
            )
            if attempt.cancelled.is_set():
                return False
            return True
        finally:
            cancelled.cancel()
            if not acquire.done():
                acquire.cancel()
            await asyncio.gather(acquire, cancelled, return_exceptions=True)
            if attempt.cancelled.is_set() and not acquire.cancelled():
                slots.release()

    async def execute(attempt: _Attempt, body: dict[str, Any]) -> JSONResponse:
        nonlocal admission_closed, idle_since
        acquired = False
        sandbox_failure = False
        unknown = False
        try:
            acquired = await acquire_slot(attempt)
            if not acquired:
                return _error(409, "Attempt cancelled")
            if admission_closed:
                unknown = True
                return _error(
                    503, "Admission closed: reconcile sandbox jobs and restart"
                )
            attempt.started = True
            try:
                response = await asyncio.wait_for(
                    gym_client.post(
                        f"{gym_url}/run", json=body, timeout=episode_timeout
                    ),
                    timeout=episode_timeout,
                )
            except (TimeoutError, httpx.HTTPError):
                sandbox_failure = True
                unknown = True
                admission_closed = True
                return _error(503, "Gym job state is unknown; admission closed")
            if response.status_code >= 500:
                sandbox_failure = True
                admission_closed = True
                return _error(503, "Gym job state is unknown; admission closed")
            if attempt.cancelled.is_set():
                return _error(409, "Attempt cancelled")
            if attempt.failed or attempt.invalid_callback:
                sandbox_failure = attempt.invalid_callback
                return _error(502, "Episode model request failed")
            if not response.is_success:
                unknown = True
                return _error(502, "Gym episode request failed")
            try:
                result = response.json()
            except ValueError:
                sandbox_failure = True
                return _error(502, "Gym returned an invalid episode result")
            if not isinstance(result, dict):
                sandbox_failure = True
                return _error(502, "Gym returned an invalid episode result")
            trajectory = result.get("response")
            resolved = result.get("resolved")
            reward = result.get("reward")
            sandbox_failure = (
                type(resolved) is not bool
                or type(reward) not in (int, float)
                or reward != resolved
                or not isinstance(trajectory, dict)
                or not isinstance(trajectory.get("output"), list)
                or not trajectory["output"]
                or trajectory.get("status") not in (None, "completed")
                or bool(trajectory.get("error"))
            )
            return JSONResponse(result)
        except asyncio.CancelledError:
            unknown = True
            if attempt.started:
                admission_closed = True
            raise
        except Exception:
            unknown = True
            admission_closed = True
            return _error(503, "Bridge job state is unknown; admission closed")
        finally:
            revoke(attempt)
            attempts.pop(attempt.attempt_id, None)
            for metrics in (window, total):
                if not attempt.cancelled.is_set() and not attempt.failed:
                    if sandbox_failure:
                        metrics.sandbox_failures += 1
                        if not attempt.first_callback_seen:
                            metrics.finished_without_callback += 1
                    if unknown:
                        metrics.unknown_outcomes += 1
            if not attempts:
                idle_since = monotonic()
            if acquired:
                slots.release()

    @app.get("/health")
    async def health() -> JSONResponse:
        expire_idle_window()
        return JSONResponse(
            {
                "status": "unavailable" if admission_closed else "ok",
                "active_attempts": sum(a.started for a in attempts.values()),
                "running_runners": running_runners if ray_head_address else 1,
                "queued_attempts": sum(not a.started for a in attempts.values()),
                "concurrency": concurrency,
                "total": total.snapshot(),
                "window": window_snapshot(),
            },
            status_code=503 if admission_closed else 200,
        )

    @app.post("/run")
    async def run(request: Request) -> JSONResponse:
        nonlocal idle_since
        received_at = monotonic()
        if admission_closed:
            return _error(503, "Admission closed: reconcile sandbox jobs and restart")
        key = _bearer(request)
        try:
            attempt_id = str(UUID(request.headers.get("x-areal-attempt-id", "")))
            proxy_url = _root_url(request.headers.get("x-areal-proxy-url", ""))
        except ValueError:
            return _error(400, "A valid attempt UUID and AReaL proxy root are required")
        if attempt_id in attempts:
            return _error(409, "Attempt already registered")
        body = await _body(request)
        params = body.get("responses_create_params")
        if not isinstance(params, dict) or not isinstance(params.get("model"), str):
            return _error(400, "responses_create_params.model is required")
        if not params["model"] or params.get("stream", False) is not False:
            return _error(400, "A model and non-streaming episode are required")
        if params.get("n", 1) != 1:
            return _error(400, "Only one completion per request is supported")
        max_output_tokens = params.get("max_output_tokens")
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            return _error(400, "max_output_tokens must be a positive integer")
        # Reading the body yields control; another caller may have registered
        # the same UUID while this request was still uploading.
        if attempt_id in attempts:
            return _error(409, "Attempt already registered")
        attempt = _Attempt(
            attempt_id=attempt_id,
            received_at=received_at,
            token=f"areal-{uuid4()}",
            proxy_url=proxy_url,
            model=params["model"],
            max_output_tokens=max_output_tokens,
            api_key=key,
            key_digest=hashlib.sha256(key.encode()).digest(),
        )
        forwarded = deepcopy(body)
        forwarded["responses_create_params"]["model"] = attempt.token
        expire_idle_window()
        idle_since = None
        total.attempts += 1
        window.attempts += 1
        attempts[attempt_id] = attempt
        routes[attempt.token] = attempt
        job = asyncio.create_task(execute(attempt, forwarded))
        jobs.add(job)
        job.add_done_callback(jobs.discard)

        stop_disconnect = asyncio.Event()

        async def wait_for_disconnect() -> None:
            while not stop_disconnect.is_set():
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.1)

        disconnect = asyncio.create_task(wait_for_disconnect())
        try:
            done, _ = await asyncio.wait(
                {job, disconnect}, return_when=asyncio.FIRST_COMPLETED
            )
            if job in done:
                return job.result()
            cancel(attempt)
            return _error(499, "Caller disconnected; started Gym job is draining")
        except asyncio.CancelledError:
            cancel(attempt)
            raise
        finally:
            # Starlette's disconnect poll can swallow task cancellation.
            stop_disconnect.set()
            disconnect.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect

    @app.delete("/attempts/{attempt_id}")
    async def cancel_attempt(attempt_id: str, request: Request) -> JSONResponse:
        key = _bearer(request)
        try:
            attempt_id = str(UUID(attempt_id))
        except ValueError:
            return _error(400, "A valid attempt UUID is required")
        attempt = attempts.get(attempt_id)
        if attempt is None:
            return _error(404, "Attempt not found")
        if not hmac.compare_digest(
            attempt.key_digest, hashlib.sha256(key.encode()).digest()
        ):
            return _error(403, "Episode key does not match")
        cancel(attempt)
        return JSONResponse(
            {
                "status": "draining" if attempt.started else "cancelled",
                "sandbox_job_stopped": False,
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        callback_received_at = monotonic()
        body = await _body(request)
        token = body.get("model")
        attempt = routes.get(token) if isinstance(token, str) else None
        if attempt is None or not attempt.active or not attempt.started:
            return _error(404, "Unknown or expired model route")
        if body.get("stream", False) is not False or body.get("n", 1) != 1:
            attempt.invalid_callback = True
            return _error(400, "Only non-streaming single completions are supported")
        messages = body.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(not isinstance(message, dict) for message in messages)
        ):
            attempt.invalid_callback = True
            return _error(400, "A non-empty list of message objects is required")
        token_limits = [attempt.max_output_tokens]
        for name in ("max_tokens", "max_completion_tokens"):
            value = body.get(name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                attempt.invalid_callback = True
                return _error(400, "Completion token limits must be positive integers")
            token_limits.append(value)
        forwarded = deepcopy(body)
        forwarded["model"] = attempt.model
        # The pinned Gym/OpenHands recipe can omit its configured output cap.
        # Always enforce the episode cap while respecting stricter tool requests.
        forwarded.pop("max_tokens", None)
        forwarded["max_completion_tokens"] = min(token_limits)
        # OpenHands replays response model_dump() objects, whose optional null
        # fields are invalid in OpenAI request TypedDicts. Null content is valid.
        forwarded["messages"] = [
            {
                key: value
                for key, value in message.items()
                if value is not None or key == "content"
            }
            for message in forwarded["messages"]
        ]
        if not attempt.first_callback_seen:
            for metrics in (window, total):
                metrics.observe(callback_received_at - attempt.received_at)
            attempt.first_callback_seen = True
        upstream = asyncio.create_task(
            model_client.post(
                f"{attempt.proxy_url}/chat/completions",
                headers={"Authorization": f"Bearer {attempt.api_key}"},
                json=forwarded,
                timeout=episode_timeout,
            )
        )
        attempt.model_requests.add(upstream)
        try:
            response = await upstream
            if not attempt.active:
                return _error(410, "Model route revoked")
            if not response.is_success:
                context_error = _context_length_error(response)
                if context_error is not None:
                    # Gym terminates and grades the existing patch. AReaL exports
                    # only recorded generations, not Gym's empty terminal response.
                    return context_error
                logger.warning(
                    "AReaL model request failed with HTTP %s", response.status_code
                )
                attempt.failed = True
                return _error(502, "AReaL model request failed")
            try:
                result = response.json()
            except ValueError:
                result = None
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("id"), str)
                or not result["id"]
                or not isinstance(result.get("choices"), list)
                or len(result["choices"]) != 1
            ):
                attempt.failed = True
                return _error(502, "AReaL returned an invalid completion")
            return JSONResponse(result)
        except asyncio.CancelledError:
            if not attempt.active:
                return _error(410, "Model route revoked")
            attempt.failed = True
            raise
        except (httpx.HTTPError, ValueError):
            attempt.failed = True
            return _error(502, "AReaL model request failed")
        finally:
            attempt.model_requests.discard(upstream)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-url", required=True)
    parser.add_argument("--ray-head-address")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--episode-timeout", type=float, default=3600)
    parser.add_argument("--metrics-idle-reset-seconds", type=float, default=900)
    args = parser.parse_args()
    app = create_app(
        args.gym_url,
        ray_head_address=args.ray_head_address,
        concurrency=args.concurrency,
        episode_timeout=args.episode_timeout,
        metrics_idle_reset_seconds=args.metrics_idle_reset_seconds,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
