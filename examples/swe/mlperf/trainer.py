# SPDX-License-Identifier: Apache-2.0
"""Inline MLPerf validation using AReaL's proxy and evaluation workers."""

import copy
import json
import time
from pathlib import Path
from typing import Any

from examples.swe.mlperf.config import MLPerfSWEConfig
from examples.swe.mlperf.validation import (
    summarize_validation,
    validate_validation_config,
)
from examples.swe.mlperf.workflow import VALIDATION_TASK_INDEX

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.rpc.rtensor import RTensor
from areal.trainer.rl_trainer import PPOTrainer, is_single_controller
from areal.utils import logging
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("RLTrainer")


class MLPerfSWETrainer(PPOTrainer):
    """Evaluate complete four-attempt groups after selected optimizer updates."""

    def __init__(self, config: MLPerfSWEConfig, *args: Any, **kwargs: Any):
        self._validation_stats: dict[str, float] = {}
        self._last_validation: dict[str, Any] | None = None
        self._completed_steps = 0
        super().__init__(config, *args, **kwargs)

    def _init_rollout(
        self,
        rollout_config: InferenceEngineConfig,
        is_eval: bool = False,
        lora_path: str | None = None,
    ):
        if is_eval:
            rollout_config = copy.deepcopy(rollout_config)
            # Validation exports metric rows; agent traces remain with the runner.
            rollout_config.dump_to_file = False
            rollout_config.check_trajectory_format = False
            rollout_config.agent.engine_max_tokens = self.config.eval_gconfig.max_tokens
        return super()._init_rollout(rollout_config, is_eval, lora_path)

    def _write_validation_artifact(self, name: str, value: dict[str, Any]) -> None:
        directory = (
            Path(StatsLogger.get_log_path(self.config.stats_logger)) / "validation"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def _write_validation_status(self, status: str) -> None:
        result = {
            key: value
            for key, value in (self._last_validation or {}).items()
            if key != "attempts"
        }
        result.update(
            status=status,
            completed_steps=self._completed_steps,
            target=self.config.validation.target,
            target_reached=status == "target_reached",
        )
        self._write_validation_artifact("status.json", result)
        logger.info("MLPerf validation status: %s", status)

    def train(self, *args: Any, **kwargs: Any) -> None:
        try:
            super().train(*args, **kwargs)
        except BaseException:
            try:
                self._write_validation_status("aborted")
            except OSError:
                logger.exception("Failed to write validation status")
            raise
        if self._last_validation is None:
            status = "not_evaluated"
        elif self._last_validation["target_reached"]:
            status = "target_reached"
        elif self.config.validation.target is None:
            status = "completed"
        else:
            status = "target_not_reached"
        self._write_validation_status(status)

    def _evaluate(
        self,
        eval_workflow,
        eval_workflow_kwargs,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ) -> None:
        self._completed_steps = global_step + 1
        self._validation_stats = {}
        if not self.config.validation.should_evaluate(self._completed_steps):
            return
        if not is_single_controller():
            raise ValueError("MLPerf validation requires single-controller mode")
        if (
            self.valid_dataloader is None
            or self.eval_rollout is None
            or eval_workflow is None
        ):
            raise ValueError(
                "MLPerf validation requires a dataset and evaluation workflow"
            )

        tasks = [task for batch in self.valid_dataloader for task in batch]
        validate_validation_config(self.config, tasks)
        task_ids = [
            task["responses_create_params"]["metadata"]["instance_id"] for task in tasks
        ]
        started = time.monotonic()
        groups = []
        try:
            for task_index, task in enumerate(tasks):
                task = copy.deepcopy(task)
                task[VALIDATION_TASK_INDEX] = task_index
                params = task["responses_create_params"]
                generation = self.config.eval_gconfig
                params["temperature"] = generation.temperature
                params["top_p"] = generation.top_p
                params["max_output_tokens"] = min(
                    params.get("max_output_tokens") or generation.max_new_tokens,
                    generation.max_new_tokens,
                )
                self.eval_rollout.submit(
                    task,
                    eval_workflow,
                    eval_workflow_kwargs,
                    group_size=4,
                    is_eval=True,
                )
            groups = self.eval_rollout.wait(len(tasks), timeout=None)
            summary = summarize_validation(RTensor.localize(groups), task_ids)
        finally:
            if groups:
                self.actor.clear_batches(*groups)

        target = self.config.validation.target
        reached = target is not None and summary["pass_at_4"] >= target
        summary.update(
            completed_step=self._completed_steps,
            model_version=self._completed_steps,
            duration_seconds=time.monotonic() - started,
            target=target,
            target_reached=reached,
        )
        self._write_validation_artifact(
            f"step_{self._completed_steps:06d}.json", summary
        )
        self._last_validation = summary
        self._validation_stats = {
            f"mlperf_validation/{key}": float(value)
            for key, value in summary.items()
            if key != "attempts" and isinstance(value, (int, float))
        }
        logger.info(
            "Validation after update %s: pass@4=%.6f (%s/%s tasks), "
            "pass@1=%.6f, failed attempts=%s/%s, target reached=%s",
            self._completed_steps,
            summary["pass_at_4"],
            summary["solved_tasks"],
            summary["num_tasks"],
            summary["pass_at_1"],
            summary["failed_attempts"],
            summary["num_attempts"],
            reached,
        )

    def _export_and_commit_stats(
        self,
        epoch: int,
        epoch_step: int,
        global_step: int,
        actor_stats: dict[str, float] | None = None,
    ):
        stats = (
            dict(actor_stats) if actor_stats is not None else self.actor.export_stats()
        )
        stats.update(self._validation_stats)
        super()._export_and_commit_stats(epoch, epoch_step, global_step, stats)
