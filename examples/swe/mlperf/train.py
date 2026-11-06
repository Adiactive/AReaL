# SPDX-License-Identifier: Apache-2.0
"""Train through the private NeMo Gym bridge using AReaL's own proxy."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from examples.swe.mlperf.config import (
    MLPerfSWEConfig,
    validate_proxy_config,
)
from examples.swe.mlperf.preflight import (
    capacity_report,
    check_runner_health,
    prepare_workflow,
)
from examples.swe.mlperf.validation import validate_validation_config

from areal.api.cli_args import load_expr_config, parse_cli_args, to_structured_cfg
from areal.utils import logging


def load_tasks(path: str) -> list[dict[str, Any]]:
    """Keep native Gym JSONL records intact, including grader metadata."""
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError(f"Invalid task JSON at line {line_number}") from None
            if not isinstance(row, dict) or not isinstance(
                row.get("responses_create_params"), dict
            ):
                raise ValueError(f"Expected a native Gym task at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("Task JSONL is empty")
    return rows


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-only", action="store_true")
    options, training_args = parser.parse_known_args(args)
    if os.getenv("AREAL_SPMD_MODE") == "1":
        raise ValueError("The AReaL proxy requires single-controller mode")
    if options.check_only:
        raw_config, _ = parse_cli_args(training_args)
        config = OmegaConf.to_object(to_structured_cfg(raw_config, MLPerfSWEConfig))
    else:
        config, _ = load_expr_config(training_args, MLPerfSWEConfig)
    validate_proxy_config(config)
    train_dataset = load_tasks(config.train_dataset.path)
    valid_dataset = (
        load_tasks(config.valid_dataset.path) if config.valid_dataset else None
    )
    kwargs = prepare_workflow(config, config.gconfig, train_dataset, split="training")
    eval_kwargs = (
        prepare_workflow(config, config.eval_gconfig, valid_dataset, split="validation")
        if valid_dataset is not None
        else None
    )
    if valid_dataset is not None:
        validate_validation_config(config, valid_dataset)
    health = check_runner_health(config.nemo_gym.runner_url)
    messages, warnings = capacity_report(config, health)
    logger = logging.getLogger("RLTrainer")
    logger.info(
        "Validated %s training tasks and %s validation tasks.",
        len(train_dataset),
        len(valid_dataset) if valid_dataset is not None else 0,
    )
    for message in messages:
        logger.info(message)
    for warning in warnings:
        logger.warning(warning)
    if options.check_only:
        logger.info(
            "Preflight passed without starting training workers or submitting an attempt. "
            "Runner health does not verify model callbacks or sandbox execution."
        )
        return

    from areal import PPOTrainer

    trainer_cls = PPOTrainer
    if valid_dataset is not None:
        from examples.swe.mlperf.trainer import MLPerfSWETrainer

        trainer_cls = MLPerfSWETrainer

    with trainer_cls(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        fallback_token_id = trainer.tokenizer.pad_token_id
        if fallback_token_id is None:
            fallback_token_id = trainer.tokenizer.eos_token_id
        if fallback_token_id is None:
            fallback_token_id = 0
        workflow_path = "examples.swe.mlperf.workflow.MLPerfSWEWorkflow"
        trainer.train(
            workflow=workflow_path,
            workflow_kwargs={
                "agent_kwargs": kwargs,
                "fallback_token_id": fallback_token_id,
            },
            eval_workflow=workflow_path if valid_dataset is not None else None,
            eval_workflow_kwargs=(
                {
                    "agent_kwargs": eval_kwargs,
                    "fallback_token_id": fallback_token_id,
                }
                if valid_dataset is not None
                else None
            ),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
