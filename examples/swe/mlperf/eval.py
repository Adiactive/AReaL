# SPDX-License-Identifier: Apache-2.0
"""Evaluate an exported checkpoint through AReaL's proxy, without training engines."""

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import swanlab
from omegaconf import OmegaConf

from examples.swe.mlperf.config import MLPerfSWEEvalConfig, validate_proxy_config
from examples.swe.mlperf.preflight import check_runner_health, prepare_workflow
from examples.swe.mlperf.train import load_tasks
from examples.swe.mlperf.validation import (
    run_validation_sweep,
    validate_validation_config,
)

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    load_expr_config,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.infra.rpc.rtensor import RTensor
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("RLTrainer")


def validate_eval_config(
    config: MLPerfSWEEvalConfig, tasks: list[dict[str, Any]]
) -> None:
    """Reject settings that require a training actor or uninitialized weights."""
    validate_proxy_config(config, evaluation_only=True)
    validate_validation_config(config, tasks, evaluation_only=True)
    allocation = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    if allocation.backend != "vllm":
        raise ValueError("Standalone MLPerf evaluation currently requires vllm")
    if (
        config.rollout.initial_weight_source != "checkpoint"
        or config.vllm.load_format == "dummy"
    ):
        raise ValueError(
            "Standalone evaluation requires checkpoint weights, not actor or dummy weights"
        )
    if config.rollout.scheduling_strategy.type != "separation":
        raise ValueError(
            "Standalone evaluation requires separate inference worker allocation"
        )
    if config.enable_offload:
        raise ValueError("Standalone evaluation does not support training offload")
    if config.rollout.use_lora:
        raise ValueError("Standalone evaluation requires a full or merged checkpoint")
    if not config.vllm.model:
        raise ValueError("vllm.model must identify the exported checkpoint")
    if config.scheduler.type not in ("local", "ray", "slurm"):
        raise ValueError(f"Unsupported scheduler: {config.scheduler.type}")


def create_controller(config: MLPerfSWEEvalConfig):
    """Allocate only inference resources; no trainer or actor is constructed."""
    from areal.engine import RemotevLLMEngine
    from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler

    scheduler_cls = {
        "local": LocalScheduler,
        "ray": RayScheduler,
        "slurm": SlurmScheduler,
    }[config.scheduler.type]
    rollout = copy.deepcopy(config.rollout)
    # A fixed checkpoint has no optimizer updates to advance admission versions.
    rollout.max_head_offpolicyness = int(1e12)
    rollout.dump_to_file = False
    rollout.check_trajectory_format = False
    rollout.agent.engine_max_tokens = config.eval_gconfig.max_tokens
    allocation = ModelAllocation.from_str(rollout.backend, name="rollout")
    server_args = vLLMConfig.build_args(
        vllm_config=config.vllm,
        tp_size=allocation.parallel.tp_size,
        pp_size=allocation.parallel.pp_size,
    )
    scheduler = scheduler_cls(exp_config=config)
    try:
        controller = RemotevLLMEngine.as_controller(rollout, scheduler)
    except BaseException:
        scheduler.delete_workers()
        raise
    return controller, scheduler, server_args


def clear_metric_batches(*groups: dict[str, Any]) -> None:
    """Release rollout metric storage directly, without a TrainController."""

    async def clear():
        await asyncio.gather(
            *(
                RTensor.clear_node(address, shard_ids)
                for address, shard_ids in RTensor.collect_shards(groups).items()
            )
        )

    run_async_task(clear)


def write_artifact(directory: Path, name: str, value: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _start_swanlab(config: MLPerfSWEEvalConfig, num_tasks: int):
    tracking = config.stats_logger.swanlab
    if tracking.mode == "disabled":
        return None
    try:
        if tracking.mode == "cloud" and tracking.api_key:
            swanlab.login(tracking.api_key)
        # Keep proxy credentials and task contents out of tracking metadata.
        metadata = dict(tracking.config or {})
        metadata.update(
            checkpoint=config.vllm.model,
            num_tasks=num_tasks,
            seed=config.seed,
            n_samples=config.eval_gconfig.n_samples,
            temperature=config.eval_gconfig.temperature,
            top_p=config.eval_gconfig.top_p,
            max_tokens=config.eval_gconfig.max_tokens,
            max_new_tokens=config.eval_gconfig.max_new_tokens,
            validation_target=config.validation.target,
        )
        return swanlab.init(
            project=tracking.project or config.experiment_name,
            experiment_name=tracking.name or f"{config.trial_name}_eval",
            config=metadata,
            logdir=tracking.logdir or StatsLogger.get_log_path(config.stats_logger),
            mode=tracking.mode,
            settings=swanlab.Settings(
                log_proxy_type="none", hardware_monitor=False, metadata_collect=False
            ),
        )
    except Exception:
        logger.exception("Failed to initialize SwanLab evaluation tracking")
        return None


def _log_swanlab(tracker: Any, summary: dict[str, Any]) -> None:
    if tracker is None:
        return
    try:
        tracker.log(
            {
                f"mlperf_validation/{key}": float(value)
                for key, value in summary.items()
                if isinstance(value, (int, float))
            },
            step=0,
        )
    except Exception:
        logger.exception("Failed to report evaluation metrics to SwanLab")


def evaluate_checkpoint(
    config: MLPerfSWEEvalConfig,
    tasks: list[dict[str, Any]],
    agent_kwargs: dict[str, Any],
) -> dict[str, Any]:
    from areal.utils import seeding
    from areal.utils.hf_utils import load_hf_tokenizer

    directory = Path(StatsLogger.get_log_path(config.stats_logger)) / "validation"
    started = time.monotonic()
    tracker = None
    completed = False
    try:
        tracker = _start_swanlab(config, len(tasks))
        seeding.set_random_seed(config.seed, key="eval")
        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        fallback = tokenizer.pad_token_id
        if fallback is None:
            fallback = tokenizer.eos_token_id
        controller, scheduler, server_args = create_controller(config)
        try:
            controller.initialize(role="eval-rollout", server_args=server_args)
            controller.start_proxy()
            summary = run_validation_sweep(
                controller,
                tasks,
                config.eval_gconfig,
                "examples.swe.mlperf.workflow.MLPerfSWEWorkflow",
                {
                    "agent_kwargs": agent_kwargs,
                    "fallback_token_id": fallback if fallback is not None else 0,
                },
                clear_metric_batches,
            )
        finally:
            primary_error = sys.exception()
            try:
                try:
                    controller.destroy()
                finally:
                    scheduler.delete_workers()
            except Exception:
                if primary_error is None:
                    raise
                logger.exception("Failed to clean up evaluation workers")
        target = config.validation.target
        reached = target is not None and summary["pass_at_4"] >= target
        summary.update(
            checkpoint=config.vllm.model,
            duration_seconds=time.monotonic() - started,
            target=target,
            target_reached=reached,
        )
        write_artifact(directory, "checkpoint.json", summary)
        write_artifact(
            directory,
            "status.json",
            {key: value for key, value in summary.items() if key != "attempts"}
            | {
                "status": "completed"
                if target is None
                else "target_reached"
                if reached
                else "target_not_reached"
            },
        )
        logger.info(
            "Checkpoint evaluation: pass@4=%.6f, pass@1=%.6f, failed attempts=%s/%s",
            summary["pass_at_4"],
            summary["pass_at_1"],
            summary["failed_attempts"],
            summary["num_attempts"],
        )
        completed = True
        _log_swanlab(tracker, summary | {"completed": True, "aborted": False})
        return summary
    except BaseException:
        _log_swanlab(
            tracker,
            {
                "completed": False,
                "aborted": True,
                "duration_seconds": time.monotonic() - started,
            },
        )
        try:
            write_artifact(
                directory,
                "status.json",
                {"status": "aborted", "checkpoint": config.vllm.model},
            )
        except OSError:
            logger.exception("Failed to write evaluation status")
        raise
    finally:
        if tracker is not None:
            try:
                tracker.finish(interrupt=not completed)
            except Exception:
                logger.exception("Failed to close SwanLab evaluation tracking")


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-only", action="store_true")
    options, config_args = parser.parse_known_args(args)
    if os.getenv("AREAL_SPMD_MODE") == "1":
        raise ValueError("Standalone MLPerf evaluation requires single-controller mode")
    if options.check_only:
        raw, _ = parse_cli_args(config_args)
        config = OmegaConf.to_object(to_structured_cfg(raw, MLPerfSWEEvalConfig))
    else:
        config, _ = load_expr_config(config_args, MLPerfSWEEvalConfig)
    if config.valid_dataset is None:
        raise ValueError("MLPerf evaluation requires valid_dataset")
    tasks = load_tasks(config.valid_dataset.path)
    validate_eval_config(config, tasks)
    kwargs = prepare_workflow(config, config.eval_gconfig, tasks, split="validation")
    health = check_runner_health(config.nemo_gym.runner_url)
    logger.info(
        "Validated %s tasks (%s attempts); runner capacity=%s, active=%s, queued=%s",
        len(tasks),
        4 * len(tasks),
        health.concurrency,
        health.active_attempts,
        health.queued_attempts,
    )
    if options.check_only:
        logger.info(
            "Preflight passed; no inference workers or sandbox attempts were started."
        )
        return
    logging.setup_file_logging(
        str(Path(StatsLogger.get_log_path(config.stats_logger)) / "eval.log")
    )
    evaluate_checkpoint(config, tasks, kwargs)


if __name__ == "__main__":
    main(sys.argv[1:])
