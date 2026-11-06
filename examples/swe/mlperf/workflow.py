# SPDX-License-Identifier: Apache-2.0

"""Run benchmark attempts with shared training and validation recovery."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import aiohttp
import httpx
import torch

from examples.swe.mlperf.config import AGENT_PATH

from areal.api import InferenceEngine, RolloutWorkflow
from areal.experimental.openai.proxy.workflow import (
    EmptyAgentSessionError,
    OpenAIProxyWorkflow,
)
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    interactions_to_trajectory,
)
from areal.infra import workflow_context
from areal.utils import logging, stats_tracker
from areal.workflow.nemo_gym_swe import NeMoGymSWEError

logger = logging.getLogger("RLTrainer")

VALIDATION_TASK_INDEX = "__mlperf_validation_task_index"


class MLPerfSWEWorkflow(RolloutWorkflow):
    """Keep failed attempt slots in training rewards and validation metrics."""

    def __init__(self, agent_workflow: OpenAIProxyWorkflow, *, fallback_token_id: int):
        if not isinstance(agent_workflow, OpenAIProxyWorkflow):
            raise TypeError("MLPerf requires an OpenAIProxyWorkflow child")
        if type(fallback_token_id) is not int or fallback_token_id < 0:
            raise ValueError("fallback_token_id must be a nonnegative integer")
        self.agent_workflow = agent_workflow
        self.fallback_token_id = fallback_token_id

    @classmethod
    def _from_workflow_resolver(
        cls,
        workflow_resolver: Callable[[Any, dict[str, Any]], RolloutWorkflow],
        agent_kwargs: dict[str, Any],
        fallback_token_id: int,
    ) -> MLPerfSWEWorkflow:
        return cls(
            workflow_resolver(AGENT_PATH, agent_kwargs),
            fallback_token_id=fallback_token_id,
        )

    @classmethod
    def _iter_workflow_specs(
        cls, workflow_kwargs: Mapping[str, Any]
    ) -> Iterable[tuple[str, dict[str, Any]]]:
        yield AGENT_PATH, dict(workflow_kwargs.get("agent_kwargs", {}))

    @staticmethod
    def _terminal_reward(interactions: Any) -> float | None:
        if not isinstance(interactions, dict) or not interactions:
            return None
        values = list(interactions.values())
        if not all(isinstance(item, InteractionWithTokenLogpReward) for item in values):
            return None
        references = [
            item.rollout_reward for item in values if item.rollout_reward is not None
        ]
        rewards = references or [values[-1].reward]
        if any(
            type(reward) not in (int, float)
            or not math.isfinite(reward)
            or reward not in (0, 1)
            for reward in rewards
        ):
            return None
        if any(reward != rewards[0] for reward in rewards):
            return None
        return float(rewards[0])

    def _failed_trajectory(self) -> dict[str, Any]:
        # Nonempty attention keeps packing/model forward valid; no token is a target.
        return {
            "input_ids": torch.full(
                (1, 2), self.fallback_token_id, dtype=torch.int64, device="cpu"
            ),
            "attention_mask": torch.ones((1, 2), dtype=torch.bool, device="cpu"),
            "loss_mask": torch.zeros((1, 2), dtype=torch.int64, device="cpu"),
            "logprobs": torch.zeros((1, 2), dtype=torch.float32, device="cpu"),
            "versions": torch.full((1, 2), -1, dtype=torch.int64, device="cpu"),
            "rewards": torch.zeros(1, dtype=torch.float32, device="cpu"),
            "is_truncated": torch.zeros(1, dtype=torch.bool, device="cpu"),
            "begin_of_trajectory": torch.ones(1, dtype=torch.int32, device="cpu"),
            "rollout_rewards": [0.0],
        }

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        context = workflow_context.get()
        task_index = data.get(VALIDATION_TASK_INDEX)
        if context.is_eval and (type(task_index) is not int or task_index < 0):
            raise ValueError("MLPerf validation requires a nonnegative task index")
        sample_index = context.sample_idx if context.sample_idx is not None else 0
        if type(sample_index) is not int or not 0 <= sample_index < context.group_size:
            raise ValueError("MLPerf received an invalid sample index")
        task = copy.deepcopy(data)
        task.pop(VALIDATION_TASK_INDEX, None)
        trajectory = None
        reward = None
        failure_reason = None
        try:
            interactions = await self.agent_workflow.arun_episode(engine, task)
        except (
            NeMoGymSWEError,
            EmptyAgentSessionError,
            aiohttp.ClientError,
            httpx.RequestError,
            TimeoutError,
        ) as exc:
            failure_reason = type(exc).__name__
        else:
            reward = self._terminal_reward(interactions)
            if reward is None:
                failure_reason = "invalid_reward"
            else:
                try:
                    # Validate before concatenation can pad missing token fields.
                    for interaction in interactions.values():
                        row = interaction.to_tensor_dict()
                        shape = row["input_ids"].shape
                        if (
                            len(shape) != 2
                            or shape[0] != 1
                            or any(
                                not isinstance(row.get(name), torch.Tensor)
                                or row[name].shape != shape
                                for name in (
                                    "attention_mask",
                                    "loss_mask",
                                    "logprobs",
                                    "versions",
                                )
                            )
                        ):
                            raise ValueError("Inconsistent proxy token fields")
                    trajectory = interactions_to_trajectory(interactions)
                    loss_mask = trajectory.get("loss_mask")
                    if (
                        not isinstance(loss_mask, torch.Tensor)
                        or loss_mask.ndim != 2
                        or not loss_mask.any()
                    ):
                        failure_reason = "no_generation_data"
                    else:
                        n_rows = trajectory["input_ids"].shape[0]
                        trajectory["rollout_rewards"] = [reward] * n_rows
                        boundaries = torch.zeros(
                            n_rows, dtype=torch.int32, device="cpu"
                        )
                        boundaries[0] = 1
                        trajectory["begin_of_trajectory"] = boundaries
                except Exception as exc:
                    failure_reason = f"postprocess_failed:{type(exc).__name__}"

        failed = failure_reason is not None
        if failed:
            reward = 0.0
            logger.warning(
                "MLPerf %s attempt failed: task=%s sample=%s reason=%s; "
                "retaining zero-reward slot.",
                "validation" if context.is_eval else "training",
                context.task_id,
                sample_index,
                failure_reason,
            )
        tracker = stats_tracker.get(workflow_context.stat_scope())
        tracker.scalar(
            mlperf_attempt_failed=float(failed), mlperf_attempt_reward=reward
        )
        tracker.denominator(
            mlperf_attempts=torch.ones(1, dtype=torch.bool, device="cpu"),
            mlperf_failed_attempts=torch.tensor(
                [failed], dtype=torch.bool, device="cpu"
            ),
        )
        if not context.is_eval:
            return self._failed_trajectory() if failed else trajectory
        return {
            "rewards": torch.tensor([reward], dtype=torch.float32, device="cpu"),
            "validation_task_index": torch.tensor(
                [task_index], dtype=torch.int64, device="cpu"
            ),
            "validation_sample_index": torch.tensor(
                [sample_index], dtype=torch.int64, device="cpu"
            ),
            "validation_failed": torch.tensor([failed], dtype=torch.bool, device="cpu"),
            "begin_of_trajectory": torch.ones(1, dtype=torch.int32, device="cpu"),
        }

    async def _afinalize_processor_cache_group(
        self, context: workflow_context.WorkflowContext
    ) -> None:
        await self.agent_workflow._afinalize_processor_cache_group(context)
