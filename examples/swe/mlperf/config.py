# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field

from areal.api.cli_args import (
    BaseExperimentConfig,
    EvaluatorConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    PPOConfig,
    RecoverConfig,
    SaverConfig,
)

AGENT_PATH = "areal.workflow.nemo_gym_swe.NeMoGymSWEAgent"


@dataclass
class NeMoGymConfig:
    """Connection to the private sandbox bridge, not the model server."""

    runner_url: str = ""
    model: str = ""
    request_timeout_seconds: float = 7200.0
    cancellation_timeout_seconds: float = 10.0


@dataclass
class MLPerfValidationConfig:
    """Evaluate after completed optimizer steps and report the pass@4 target."""

    start_step: int = 20
    interval: int = 1
    target: float | None = 0.7

    def __post_init__(self) -> None:
        for name in ("start_step", "interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"validation.{name} must be a positive integer")
        if self.target is not None and (
            isinstance(self.target, bool)
            or not isinstance(self.target, (int, float))
            or not math.isfinite(self.target)
            or not 0 <= self.target <= 1
        ):
            raise ValueError("validation.target must be finite in [0, 1] or null")

    def should_evaluate(self, completed_steps: int) -> bool:
        return (
            completed_steps >= self.start_step and completed_steps % self.interval == 0
        )


@dataclass
class MLPerfSWEConfig(PPOConfig):
    nemo_gym: NeMoGymConfig = field(default_factory=NeMoGymConfig)
    validation: MLPerfValidationConfig = field(default_factory=MLPerfValidationConfig)


@dataclass
class MLPerfSWEEvalConfig(BaseExperimentConfig):
    """Evaluate an exported checkpoint without constructing training engines."""

    saver: SaverConfig | None = None
    evaluator: EvaluatorConfig | None = None
    recover: RecoverConfig | None = None

    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    eval_gconfig: GenerationHyperparameters = field(
        default_factory=lambda: GenerationHyperparameters(
            n_samples=4, temperature=0.1, top_p=0.95
        )
    )
    nemo_gym: NeMoGymConfig = field(default_factory=NeMoGymConfig)
    validation: MLPerfValidationConfig = field(default_factory=MLPerfValidationConfig)


def validate_proxy_config(
    config: MLPerfSWEConfig | MLPerfSWEEvalConfig, *, evaluation_only: bool = False
) -> None:
    """Require the proxy path and supported multi-turn export semantics."""
    if config.rollout._version != "v1" or (
        not evaluation_only and config.actor._version != "v1"
    ):
        raise ValueError("MLPerf SWE currently requires actor/rollout _version: v1")
    agent = config.rollout.agent
    if agent is None or agent.agent_cls_path != AGENT_PATH:
        raise ValueError(f"rollout.agent.agent_cls_path must be {AGENT_PATH}")
    if agent.mode != "inline" or agent.export_style != "individual":
        raise ValueError("MLPerf SWE requires inline mode and individual export")
    if agent.chat_template_type != "hf":
        raise ValueError("MLPerf SWE requires the hf chat template")
    timeouts = {
        "nemo_gym.request_timeout_seconds": config.nemo_gym.request_timeout_seconds,
        "nemo_gym.cancellation_timeout_seconds": config.nemo_gym.cancellation_timeout_seconds,
        "rollout.request_timeout": config.rollout.request_timeout,
        "rollout.agent.session_timeout_seconds": agent.session_timeout_seconds,
    }
    for name, value in timeouts.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite timeout")
    episode_timeout = (
        config.nemo_gym.request_timeout_seconds
        + config.nemo_gym.cancellation_timeout_seconds
    )
    if config.rollout.request_timeout <= episode_timeout:
        raise ValueError(
            "rollout.request_timeout must exceed the runner request timeout "
            "plus cancellation timeout"
        )
    if agent.session_timeout_seconds <= episode_timeout:
        raise ValueError(
            "Proxy session timeout must exceed the runner request timeout "
            "plus cancellation timeout"
        )
    if evaluation_only:
        return
    if config.train_dataset.shuffle:
        raise ValueError("MLPerf SWE requires train_dataset.shuffle: false")
    adv_norm = config.actor.adv_norm
    if adv_norm is not None and "group" in (
        adv_norm.mean_level,
        adv_norm.std_level,
    ):
        raise ValueError(
            "Individual multi-turn export does not support group advantage "
            "normalization; use batch-level adv_norm or null"
        )
