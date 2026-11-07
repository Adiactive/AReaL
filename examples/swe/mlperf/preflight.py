# SPDX-License-Identifier: Apache-2.0
"""Validate MLPerf inputs and runner admission before allocating training workers."""

from dataclasses import asdict, dataclass
from typing import Any

import httpx

from examples.swe.mlperf.config import MLPerfSWEConfig

from areal.api.cli_args import GenerationHyperparameters
from areal.workflow.nemo_gym_swe import NeMoGymSWEAgent, _validate_url


@dataclass(frozen=True)
class RunnerHealth:
    active_attempts: int
    queued_attempts: int
    concurrency: int


def check_runner_health(
    runner_url: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> RunnerHealth:
    """Read bridge admission state once, without submitting an episode."""
    _validate_url(runner_url, "runner_url")
    with httpx.Client(
        timeout=10.0,
        trust_env=False,
        follow_redirects=False,
        transport=transport,
    ) as client:
        try:
            response = client.get(f"{runner_url.rstrip('/')}/health")
        except httpx.RequestError:
            raise ValueError("Runner health request failed") from None
    if response.status_code != 200:
        raise ValueError(f"Runner health returned HTTP {response.status_code}")
    try:
        body = response.json()
    except (ValueError, UnicodeError):
        raise ValueError("Runner health must return a JSON object") from None
    if not isinstance(body, dict) or body.get("status") != "ok":
        raise ValueError("Runner health status is not ok")
    for name in ("active_attempts", "queued_attempts", "concurrency"):
        value = body.get(name)
        if type(value) is not int or value < (1 if name == "concurrency" else 0):
            raise ValueError(f"Runner health has invalid {name}")
    if body["active_attempts"] > body["concurrency"]:
        raise ValueError("Runner health active_attempts exceeds concurrency")
    return RunnerHealth(
        active_attempts=body["active_attempts"],
        queued_attempts=body["queued_attempts"],
        concurrency=body["concurrency"],
    )


def prepare_workflow(
    config: MLPerfSWEConfig,
    generation: GenerationHyperparameters,
    tasks: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    """Validate the same task and sampling fields used when submitting attempts."""
    kwargs = asdict(config.nemo_gym) | {
        "temperature": 0.0 if generation.greedy else generation.temperature,
        "top_p": generation.top_p,
        "max_output_tokens": generation.max_new_tokens,
    }
    agent = NeMoGymSWEAgent(**kwargs)
    for row_number, task in enumerate(tasks, 1):
        try:
            agent._prepare_body(task)
        except ValueError as exc:
            raise ValueError(f"Invalid {split} task {row_number}: {exc}") from None
    return kwargs


def capacity_report(
    config: MLPerfSWEConfig, health: RunnerHealth
) -> tuple[list[str], list[str]]:
    """Describe configured attempt pressure; runner admission remains authoritative."""
    groups = config.rollout.max_concurrent_rollouts
    if groups is None:
        groups = config.rollout.consumer_batch_size
    if type(groups) is not int or groups < 1:
        raise ValueError("rollout concurrency must be a positive number of groups")
    generations = [("Training", config.gconfig)]
    if config.valid_dataset is not None:
        generations.append(("Validation", config.eval_gconfig))
    messages = [
        f"Runner admission capacity: {health.concurrency} concurrent attempts; "
        f"currently {health.active_attempts} active and {health.queued_attempts} queued."
    ]
    warnings = []
    demands = []
    for split, generation in generations:
        samples = generation.n_samples
        if type(samples) is not int or samples < 1:
            raise ValueError(f"{split} n_samples must be a positive integer")
        demand = groups * samples
        demands.append(demand)
        messages.append(
            f"{split} configured demand: {groups} groups x {samples} samples "
            f"= {demand} attempts before queue and staleness limits."
        )
        if demand > health.concurrency:
            warnings.append(
                f"{split} demand exceeds this runner's admission capacity; "
                "attempts may queue and queue time counts toward the runner request timeout."
            )
    if config.valid_dataset is not None and config.rollout.max_head_offpolicyness > 0:
        warnings.append(
            f"Training and validation may overlap ({sum(demands)} configured attempts). "
            "Paused training attempts can occupy shared runner slots needed by evaluation; "
            "Provision spare runner capacity for validation and budget timeouts "
            "for training attempts paused during the sweep."
        )
    messages.append(
        "These are configured demand estimates and this endpoint's admission limit, "
        "not a global sandbox quota; other workers or jobs may share the cluster."
    )
    return messages, warnings
