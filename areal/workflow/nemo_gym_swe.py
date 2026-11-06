# SPDX-License-Identifier: Apache-2.0

"""Run remote NeMo Gym SWE episodes through AReaL's session proxy."""

import asyncio
import copy
import json
import math
import uuid
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

import httpx


class NeMoGymSWEError(RuntimeError):
    """An attempt failed without producing a trustworthy training reward."""


def _validate_url(value: str, name: str, *, root: bool = False) -> None:
    try:
        if not isinstance(value, str) or any(c.isspace() for c in value):
            raise ValueError
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (root and parsed.path not in {"", "/"})
        ):
            raise ValueError
        _ = parsed.port
    except ValueError:
        raise ValueError(f"{name} must be an HTTP(S) URL without credentials") from None


def _validate_number(value: Any, name: str, low: float, high: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} is outside its supported numeric range")


def _validate_generation(params: dict[str, Any]) -> None:
    if not isinstance(params.get("model"), str) or not params["model"].strip():
        raise ValueError("model must be a nonempty string")
    _validate_number(params["temperature"], "temperature", 0, 2)
    _validate_number(params["top_p"], "top_p", 0, 1)
    if params["top_p"] == 0:
        raise ValueError("top_p must be greater than zero")
    tokens = params["max_output_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer")


class NeMoGymSWEAgent:
    """Submit a native Gym row and return its verified binary reward.

    Use the inline AReaL proxy workflow so each episode supplies its own
    ``base_url`` and ``api_key``. The runner must implement the AReaL bridge
    headers and attempt-revocation endpoint. The model remains on AReaL's
    inference workers; this agent only submits work to the sandbox host.
    """

    def __init__(
        self,
        runner_url: str,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_output_tokens: int = 32768,
        request_timeout_seconds: float = 7200,
        cancellation_timeout_seconds: float = 10,
    ):
        _validate_url(runner_url, "runner_url")
        self.runner_url = runner_url.rstrip("/")
        self._generation_defaults = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
        }
        _validate_generation(self._generation_defaults)
        for name, value in (
            ("request_timeout_seconds", request_timeout_seconds),
            ("cancellation_timeout_seconds", cancellation_timeout_seconds),
        ):
            _validate_number(value, name, 0, math.inf)
            if value == 0:
                raise ValueError(f"{name} must be greater than zero")
        self.request_timeout_seconds = request_timeout_seconds
        self.cancellation_timeout_seconds = cancellation_timeout_seconds

    def _prepare_body(self, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Gym task must be an object")
        params = data.get("responses_create_params")
        if not isinstance(params, dict):
            raise ValueError("Gym task requires responses_create_params")
        metadata = params.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("responses_create_params.metadata must be an object")
        for name in ("instance_id", "dataset_name", "instance_dict"):
            value = metadata.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"metadata.{name} must be a nonempty string")
        try:
            instance = json.loads(metadata["instance_dict"])
        except (ValueError, TypeError):
            raise ValueError(
                "metadata.instance_dict must encode a JSON object"
            ) from None
        if not isinstance(instance, dict):
            raise ValueError("metadata.instance_dict must encode a JSON object")
        if not isinstance(params.get("input"), (str, list)):
            raise ValueError("responses_create_params.input must be a string or list")
        if params.get("stream", False) is not False:
            raise ValueError("Gym tasks require non-streaming responses")
        if type(params.get("n", 1)) is not int or params.get("n", 1) != 1:
            raise ValueError("Gym tasks require one completion per model request")
        body = copy.deepcopy(data)
        params = body["responses_create_params"]
        for name, value in self._generation_defaults.items():
            params.setdefault(name, value)
        _validate_generation(params)
        try:
            json.dumps(body, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("Gym task must contain valid JSON values") from None
        return body

    @staticmethod
    def _reward(result: Any) -> float:
        if not isinstance(result, dict):
            raise NeMoGymSWEError("Runner returned an invalid result")
        config = result.get("instance_config", {})
        if not isinstance(config, dict):
            raise NeMoGymSWEError("Runner returned invalid instance metadata")
        masks = [
            source["mask_sample"]
            for source in (config, result)
            if "mask_sample" in source
        ]
        # The reference trainer consumes Gym's reward without applying its
        # sample mask or agent-status metadata.
        if any(not isinstance(flag, bool) for flag in masks):
            raise NeMoGymSWEError("Runner did not return a valid sample mask")
        for name in ("agent_timed_out", "eval_timed_out"):
            flag = result.get(name)
            if flag is not None and not isinstance(flag, bool):
                raise NeMoGymSWEError("Runner returned an invalid timeout flag")
        error_kind = result.get("agent_error_kind")
        if error_kind is not None and not isinstance(error_kind, str):
            raise NeMoGymSWEError("Runner returned an invalid agent error kind")
        resolved, reward = result.get("resolved"), result.get("reward")
        if (
            not isinstance(resolved, bool)
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or reward != float(resolved)
        ):
            raise NeMoGymSWEError("Runner returned an invalid or inconsistent reward")
        response = result.get("response")
        output = response.get("output") if isinstance(response, dict) else None
        if not isinstance(output, list) or not output:
            raise NeMoGymSWEError("Runner returned an incomplete trajectory")
        if response.get("status") not in (None, "completed") or response.get("error"):
            raise NeMoGymSWEError("Runner returned an incomplete trajectory")
        return float(reward)

    async def _revoke_attempt(
        self, client: httpx.AsyncClient, attempt_id: str, headers: dict[str, str]
    ) -> None:
        """Revoke forwarding credentials; this does not kill Ray containers."""

        async def revoke() -> None:
            try:
                async with asyncio.timeout(self.cancellation_timeout_seconds):
                    await client.delete(
                        f"{self.runner_url}/attempts/{attempt_id}",
                        headers=headers,
                        timeout=self.cancellation_timeout_seconds,
                        follow_redirects=False,
                    )
            except Exception:
                pass

        task = asyncio.create_task(revoke())
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _run(
        self,
        client: httpx.AsyncClient,
        body: dict[str, Any],
        headers: dict[str, str],
        attempt_id: str,
    ) -> float:
        try:
            async with asyncio.timeout(self.request_timeout_seconds):
                response = await client.post(
                    f"{self.runner_url}/run",
                    json=body,
                    headers=headers,
                    timeout=self.request_timeout_seconds,
                    follow_redirects=False,
                )
                if not response.is_success:
                    try:
                        reason = HTTPStatus(response.status_code).phrase
                    except ValueError:
                        reason = "Unknown status"
                    raise NeMoGymSWEError(
                        f"Runner returned HTTP {response.status_code} ({reason})"
                    )
                try:
                    result = response.json()
                except (ValueError, UnicodeError):
                    raise NeMoGymSWEError("Runner returned invalid JSON") from None
                return self._reward(result)
        except asyncio.CancelledError:
            await self._revoke_attempt(client, attempt_id, headers)
            raise
        except (TimeoutError, httpx.TimeoutException):
            await self._revoke_attempt(client, attempt_id, headers)
            raise NeMoGymSWEError("Runner request timed out") from None
        except NeMoGymSWEError:
            await self._revoke_attempt(client, attempt_id, headers)
            raise
        except Exception:
            await self._revoke_attempt(client, attempt_id, headers)
            raise NeMoGymSWEError("Runner request failed") from None

    async def run(
        self,
        data: dict[str, Any],
        *,
        base_url: str,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> float:
        """Forward per-episode credentials in headers, never in the task body."""
        _validate_url(base_url, "base_url", root=True)
        if (
            not isinstance(api_key, str)
            or not api_key
            or any(ord(char) < 33 or ord(char) > 126 for char in api_key)
        ):
            raise ValueError("api_key must be a nonempty printable ASCII token")
        body = self._prepare_body(data)
        attempt_id = uuid.uuid4().hex
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-AReaL-Proxy-Url": base_url,
            "X-AReaL-Attempt-Id": attempt_id,
        }
        if http_client is not None:
            return await self._run(http_client, body, headers, attempt_id)
        async with httpx.AsyncClient(trust_env=False) as client:
            return await self._run(client, body, headers, attempt_id)
