# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize complete four-attempt SWE validation sweeps."""

from typing import Any

import torch

from examples.swe.mlperf.config import MLPerfSWEConfig

_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
_FIELDS = (
    "rewards",
    "validation_task_index",
    "validation_sample_index",
    "validation_failed",
    "begin_of_trajectory",
)


def validate_validation_config(
    config: MLPerfSWEConfig, tasks: list[dict[str, Any]]
) -> None:
    """Require the reference sampling settings and an unfiltered fixed-weight sweep."""
    if config.valid_dataset is None:
        raise ValueError("MLPerf validation requires valid_dataset")
    if config.valid_dataset.shuffle or config.valid_dataset.drop_last:
        raise ValueError("Validation requires shuffle: false and drop_last: false")
    generation = config.eval_gconfig
    if (
        generation is None
        or isinstance(generation.n_samples, bool)
        or not isinstance(generation.n_samples, int)
        or generation.n_samples != 4
        or generation.greedy is not False
        or generation.temperature != 0.1
        or generation.top_p != 0.95
    ):
        raise ValueError(
            "MLPerf validation requires eval_gconfig n_samples: 4, greedy: false, "
            "temperature: 0.1, and top_p: 0.95"
        )
    for engine, target in ((config.actor, "rollout"), (config.rollout, "actor")):
        strategy = engine.scheduling_strategy
        if (
            strategy is not None
            and strategy.type == "colocation"
            and strategy.target == target
        ):
            raise ValueError(
                "MLPerf validation requires separate actor and rollout scheduling"
            )
    if config.dynamic_bs is not False:
        raise ValueError("MLPerf validation requires dynamic_bs: false")
    batch_size = config.train_dataset.batch_size
    consumer_batch_size = config.rollout.consumer_batch_size
    if (
        type(batch_size) is not int
        or batch_size < 1
        or type(consumer_batch_size) is not int
        or consumer_batch_size < 1
        or consumer_batch_size != batch_size
    ):
        raise ValueError(
            "MLPerf validation requires positive integer rollout.consumer_batch_size "
            "equal to train_dataset.batch_size"
        )
    task_ids = []
    for task in tasks:
        params = task.get("responses_create_params") if isinstance(task, dict) else None
        metadata = params.get("metadata") if isinstance(params, dict) else None
        task_ids.append(
            metadata.get("instance_id") if isinstance(metadata, dict) else None
        )
    _validate_task_ids(task_ids)


def _validate_task_ids(task_ids: list[str]) -> None:
    if not task_ids:
        raise ValueError("Validation tasks must not be empty")
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids):
        raise ValueError(
            "Validation tasks require nonempty metadata.instance_id strings"
        )
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Validation tasks require unique metadata.instance_id values")


def summarize_validation(
    groups: list[dict[str, torch.Tensor] | None], task_ids: list[str]
) -> dict[str, Any]:
    """Reject incomplete sweeps instead of measuring only the attempts that survived."""
    _validate_task_ids(task_ids)
    if len(groups) != len(task_ids):
        raise ValueError("Validation requires exactly one complete group per task")

    attempts = []
    seen_tasks = set()
    solved_tasks = 0
    all_solved_tasks = 0
    successful_attempts = 0
    failed_attempts = 0
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Validation group is missing or malformed")
        for name in _FIELDS:
            tensor = group.get(name)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 1
                or tensor.numel() != 4
                or tensor.device.type != "cpu"
            ):
                raise ValueError(
                    f"Validation {name} must be a localized four-entry tensor"
                )
        for name in ("validation_task_index", "validation_sample_index"):
            if group[name].dtype not in _INTEGER_DTYPES:
                raise ValueError(f"Validation {name} must contain integer indices")
        if group["validation_failed"].dtype != torch.bool:
            raise ValueError("Validation validation_failed must contain boolean flags")
        boundaries = group["begin_of_trajectory"]
        if boundaries.dtype not in _INTEGER_DTYPES | {torch.bool} or not torch.all(
            boundaries == 1
        ):
            raise ValueError("Validation requires one metric row per attempt")
        rewards = group["rewards"]
        if (
            rewards.dtype == torch.bool
            or rewards.is_complex()
            or not torch.all(
                torch.isfinite(rewards) & ((rewards == 0) | (rewards == 1))
            )
        ):
            raise ValueError("Validation rewards must be finite binary numbers")

        task_indices = group["validation_task_index"].tolist()
        task_index = task_indices[0]
        if len(set(task_indices)) != 1 or not 0 <= task_index < len(task_ids):
            raise ValueError("Validation group must identify exactly one known task")
        if task_index in seen_tasks:
            raise ValueError("Validation contains duplicate task groups")
        seen_tasks.add(task_index)
        sample_indices = group["validation_sample_index"].tolist()
        if sorted(sample_indices) != [0, 1, 2, 3]:
            raise ValueError(
                "Validation requires sample indices 0, 1, 2, 3 once per task"
            )

        reward_values = rewards.tolist()
        failure_values = group["validation_failed"].tolist()
        if any(
            failed and reward != 0
            for failed, reward in zip(failure_values, reward_values)
        ):
            raise ValueError("Failed validation attempts must have zero reward")
        successful_attempts += sum(reward_values)
        solved_tasks += int(any(reward_values))
        all_solved_tasks += int(all(reward_values))
        failed_attempts += sum(failure_values)
        for sample_index, reward, failed in zip(
            sample_indices, reward_values, failure_values
        ):
            attempts.append(
                {
                    "task_index": task_index,
                    "instance_id": task_ids[task_index],
                    "sample_index": sample_index,
                    "reward": float(reward),
                    "failed": bool(failed),
                }
            )

    num_tasks = len(task_ids)
    num_attempts = num_tasks * 4
    return {
        "pass_at_4": solved_tasks / num_tasks,
        "pass_all_4": all_solved_tasks / num_tasks,
        "pass_at_1": successful_attempts / num_attempts,
        "num_tasks": num_tasks,
        "num_attempts": num_attempts,
        "solved_tasks": solved_tasks,
        "failed_attempts": failed_attempts,
        "attempts": sorted(
            attempts, key=lambda row: (row["task_index"], row["sample_index"])
        ),
    }
