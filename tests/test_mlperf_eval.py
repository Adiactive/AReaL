# SPDX-License-Identifier: Apache-2.0

import builtins
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from omegaconf import OmegaConf

from examples.swe.mlperf import eval as evaluation
from examples.swe.mlperf.config import MLPerfSWEEvalConfig
from examples.swe.mlperf.preflight import RunnerHealth
from examples.swe.mlperf.workflow import VALIDATION_TASK_INDEX

from areal.api.cli_args import to_structured_cfg

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples/swe/mlperf/qwen3_8_27b_eval_npu.yaml"
)


def _task(name="task-a"):
    return {
        "agent_ref": {"name": "reference_agent"},
        "responses_create_params": {
            "input": "Fix the repository",
            "max_output_tokens": 1024,
            "metadata": {
                "instance_id": name,
                "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                "instance_dict": '{"tests": ["test_patch"]}',
            },
        },
    }


def _group(index, rewards, failed):
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "validation_task_index": torch.full((4,), index, dtype=torch.int64),
        "validation_sample_index": torch.arange(4, dtype=torch.int64),
        "validation_failed": torch.tensor(failed, dtype=torch.bool),
        "begin_of_trajectory": torch.ones(4, dtype=torch.int32),
    }


@pytest.fixture(autouse=True)
def swanlab_sdk(monkeypatch):
    sdk = Mock()
    monkeypatch.setattr(evaluation, "swanlab", sdk)
    return sdk


@pytest.fixture
def eval_config(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(_task()) + "\n")
    for name, value in {
        "AREAL_SWE_MODEL_PATH": str(tmp_path / "checkpoint"),
        "AREAL_SWE_OUTPUT_DIR": str(tmp_path / "output"),
        "AREAL_SWE_VALID_JSONL": str(tasks),
        "AREAL_SWE_RUNNER_URL": "http://runner:8080",
        "AREAL_PROXY_ADMIN_KEY": "test-only-key",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AREAL_SWE_TRAIN_JSONL", raising=False)
    monkeypatch.delenv("AREAL_SPMD_MODE", raising=False)
    return OmegaConf.to_object(
        to_structured_cfg(OmegaConf.load(EXAMPLE), MLPerfSWEEvalConfig)
    )


def test_example_preflight_does_not_import_trainer_or_launch_workers(
    eval_config, monkeypatch
):
    """The supplied YAML needs no actor or training data, even during preflight."""
    assert not hasattr(eval_config, "actor")
    assert eval_config.saver is None
    assert eval_config.evaluator is None
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("areal.trainer", "megatron", "mindspeed"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    health = Mock(return_value=RunnerHealth(0, 0, 4))
    launch = Mock(side_effect=AssertionError("Preflight must not launch workers"))
    monkeypatch.setattr(evaluation, "check_runner_health", health)
    monkeypatch.setattr(evaluation, "evaluate_checkpoint", launch)

    evaluation.main(["--config", str(EXAMPLE), "--check-only"])

    health.assert_called_once_with("http://runner:8080")
    launch.assert_not_called()


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("rollout", "initial_weight_source", "actor"),
        ("vllm", "load_format", "dummy"),
        ("rollout", "use_lora", True),
        ("rollout", "backend", "sglang:d1t4"),
        ("eval_gconfig", "n_samples", 1),
        ("eval_gconfig", "temperature", 0.8),
        ("valid_dataset", "shuffle", True),
    ],
)
def test_eval_invalid_settings_rejected(eval_config, section, field, value):
    """Standalone evaluation enforces real weights and the validation protocol."""
    setattr(getattr(eval_config, section), field, value)
    with pytest.raises(ValueError):
        evaluation.validate_eval_config(eval_config, [_task()])


def test_create_controller_only_builds_inference(eval_config, monkeypatch):
    """The controller receives checkpoint server arguments and metric-only settings."""
    import areal.engine
    import areal.infra

    scheduler = Mock()
    scheduler_cls = Mock(return_value=scheduler)
    engine = Mock()
    monkeypatch.setattr(areal.infra, "LocalScheduler", scheduler_cls)
    monkeypatch.setattr(areal.engine, "RemotevLLMEngine", engine)
    original = copy.deepcopy(eval_config.rollout)

    controller, actual_scheduler, server_args = evaluation.create_controller(
        eval_config
    )

    assert controller is engine.as_controller.return_value
    assert actual_scheduler is scheduler
    rollout = engine.as_controller.call_args.args[0]
    assert rollout.max_head_offpolicyness == int(1e12)
    assert not rollout.check_trajectory_format
    assert not rollout.dump_to_file
    assert rollout.agent.engine_max_tokens == eval_config.eval_gconfig.max_tokens
    assert server_args["model"] == eval_config.vllm.model
    assert server_args["load_format"] != "dummy"
    assert server_args["tensor_parallel_size"] == 4
    assert eval_config.rollout == original


@pytest.fixture
def runtime(eval_config, tmp_path, monkeypatch):
    from areal.utils import hf_utils, seeding

    controller, scheduler = Mock(), Mock()
    groups = [
        _group(1, [0, 0, 0, 0], [False, False, False, False]),
        _group(0, [1, 0, 0, 1], [False, False, True, False]),
    ]
    controller.wait.return_value = groups
    monkeypatch.setattr(
        hf_utils,
        "load_hf_tokenizer",
        lambda _: SimpleNamespace(pad_token_id=None, eos_token_id=7),
    )
    monkeypatch.setattr(seeding, "set_random_seed", Mock())
    monkeypatch.setattr(
        evaluation,
        "create_controller",
        lambda _: (controller, scheduler, {"model": "checkpoint"}),
    )
    monkeypatch.setattr(evaluation.StatsLogger, "get_log_path", lambda _: str(tmp_path))
    return controller, scheduler, groups


def test_eval_sweep_preserves_failed_slots_and_writes_checkpoint_metrics(
    eval_config, runtime, tmp_path
):
    """A full sweep retains failures and task identities despite out-of-order results."""
    controller, scheduler, _ = runtime
    tasks = [_task(), _task("task-b")]
    original = copy.deepcopy(tasks)
    eval_config.validation.start_step = 1000
    eval_config.validation.interval = 1000

    summary = evaluation.evaluate_checkpoint(
        eval_config, tasks, {"model": "checkpoint"}
    )

    assert summary["pass_at_4"] == 0.5
    assert summary["pass_at_1"] == 0.25
    assert summary["failed_attempts"] == 1
    assert summary["num_attempts"] == 8
    assert summary["checkpoint"] == eval_config.vllm.model
    assert tasks == original
    controller.initialize.assert_called_once_with(
        role="eval-rollout", server_args={"model": "checkpoint"}
    )
    controller.start_proxy.assert_called_once()
    controller.wait.assert_called_once_with(2, timeout=None)
    for index, call in enumerate(controller.submit.call_args_list):
        data, workflow, kwargs = call.args
        assert data[VALIDATION_TASK_INDEX] == index
        assert data["responses_create_params"]["temperature"] == 0.1
        assert data["responses_create_params"]["top_p"] == 0.95
        assert data["responses_create_params"]["max_output_tokens"] == 1024
        assert workflow == "examples.swe.mlperf.workflow.MLPerfSWEWorkflow"
        assert kwargs["fallback_token_id"] == 7
        assert call.kwargs == {"group_size": 4, "is_eval": True}
    controller.destroy.assert_called_once()
    scheduler.delete_workers.assert_called_once()
    saved = json.loads((tmp_path / "validation/checkpoint.json").read_text())
    assert saved == summary
    assert (
        json.loads((tmp_path / "validation/status.json").read_text())["status"]
        == "target_not_reached"
    )


@pytest.mark.parametrize("stage", ["initialize", "start_proxy", "wait", "incomplete"])
def test_eval_failure_shuts_down_and_marks_aborted(
    eval_config, runtime, tmp_path, stage
):
    """Partial startup and incomplete sweeps do not leave workers running or emit scores."""
    controller, scheduler, _ = runtime
    if stage == "incomplete":
        controller.wait.return_value = [None]
    else:
        getattr(controller, stage).side_effect = RuntimeError("evaluation failed")
    with pytest.raises((RuntimeError, ValueError)):
        evaluation.evaluate_checkpoint(eval_config, [_task()], {})
    controller.destroy.assert_called_once()
    scheduler.delete_workers.assert_called_once()
    assert not (tmp_path / "validation/checkpoint.json").exists()
    assert (
        json.loads((tmp_path / "validation/status.json").read_text())["status"]
        == "aborted"
    )


def test_eval_cancellation_survives_shutdown_error(eval_config, runtime, tmp_path):
    """Cleanup errors must not replace the original cancellation."""
    controller, scheduler, _ = runtime
    controller.wait.side_effect = KeyboardInterrupt()
    controller.destroy.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate_checkpoint(eval_config, [_task()], {})
    scheduler.delete_workers.assert_called_once()


def test_metric_cleanup_deletes_remote_shards_without_actor(monkeypatch):
    """Metric storage is released through RTensor's storage API."""
    from areal.infra.rpc import rtensor

    backend = Mock()

    backend.delete = AsyncMock(return_value={})
    monkeypatch.setattr(rtensor, "get_backend", lambda: backend)
    tensor = rtensor.RTensor(
        data=torch.tensor([1.0]),
        shard=rtensor.TensorShardInfo(shard_id="metric-shard", node_addr="storage:1"),
    )
    evaluation.clear_metric_batches({"rewards": tensor})
    backend.delete.assert_awaited_once_with("storage:1", ["metric-shard"])


def test_eval_main_runs_example_without_training_imports(
    eval_config, runtime, monkeypatch, tmp_path
):
    """The real CLI loads the example and completes an inference-only sweep."""
    controller, scheduler, groups = runtime
    controller.wait.return_value = [groups[1]]
    monkeypatch.setattr(
        evaluation, "check_runner_health", lambda _: RunnerHealth(0, 0, 4)
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("areal.trainer", "megatron", "mindspeed"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    evaluation.main(["--config", str(EXAMPLE)])

    controller.start_proxy.assert_called_once()
    scheduler.delete_workers.assert_called_once()
    saved = json.loads((tmp_path / "validation/checkpoint.json").read_text())
    assert saved["num_attempts"] == 4
    assert saved["target_reached"] is True
    assert (tmp_path / "config.yaml").is_file()


@pytest.mark.parametrize("mode", ["offline", "local", "cloud"])
def test_eval_swanlab_reports_final_metrics_and_checkpoint_metadata(
    eval_config, runtime, tmp_path, swanlab_sdk, mode
):
    """Tracking uses the same sweep metrics while excluding credentials and rows."""
    tracking = eval_config.stats_logger.swanlab
    tracking.mode = mode
    tracking.api_key = "tracking-secret"
    summary = evaluation.evaluate_checkpoint(
        eval_config, [_task(), _task("task-b")], {}
    )

    if mode == "cloud":
        swanlab_sdk.login.assert_called_once_with("tracking-secret")
    else:
        swanlab_sdk.login.assert_not_called()
    settings = swanlab_sdk.init.call_args.kwargs
    assert settings["mode"] == mode
    assert settings["project"] == eval_config.experiment_name
    assert settings["experiment_name"] == f"{eval_config.trial_name}_eval"
    assert settings["logdir"] == str(tmp_path)
    assert settings["config"]["checkpoint"] == eval_config.vllm.model
    assert settings["config"]["n_samples"] == 4
    assert "test-only-key" not in json.dumps(settings["config"])
    assert "tracking-secret" not in json.dumps(settings["config"])
    assert "stats_logger" not in settings["config"]
    swanlab_sdk.Settings.assert_called_once_with(
        log_proxy_type="none", hardware_monitor=False, metadata_collect=False
    )
    tracker = swanlab_sdk.init.return_value
    metrics = tracker.log.call_args.args[0]
    assert metrics["mlperf_validation/pass_at_4"] == summary["pass_at_4"]
    assert metrics["mlperf_validation/pass_at_1"] == summary["pass_at_1"]
    assert metrics["mlperf_validation/failed_attempts"] == 1
    assert metrics["mlperf_validation/num_attempts"] == 8
    assert metrics["mlperf_validation/completed"] == 1
    assert metrics["mlperf_validation/aborted"] == 0
    assert metrics["mlperf_validation/duration_seconds"] >= 0
    assert "mlperf_validation/attempts" not in metrics
    assert "mlperf_validation/checkpoint" not in metrics
    assert tracker.log.call_args.kwargs == {"step": 0}
    tracker.finish.assert_called_once_with(interrupt=False)


def test_eval_swanlab_honors_tracking_overrides(eval_config, swanlab_sdk, tmp_path):
    """Existing SwanLab configuration supplies names, directory, and custom metadata."""
    tracking = eval_config.stats_logger.swanlab
    tracking.mode = "cloud"
    tracking.api_key = None
    tracking.project = "checkpoint-comparison"
    tracking.name = "checkpoint-42"
    tracking.logdir = str(tmp_path / "swanlab")
    tracking.config = {"checkpoint": "stale", "label": "candidate"}
    evaluation._start_swanlab(eval_config, 251)
    swanlab_sdk.login.assert_not_called()
    options = swanlab_sdk.init.call_args.kwargs
    assert options["project"] == tracking.project
    assert options["experiment_name"] == tracking.name
    assert options["logdir"] == tracking.logdir
    assert options["config"]["label"] == "candidate"
    assert options["config"]["checkpoint"] == eval_config.vllm.model
    assert options["config"]["num_tasks"] == 251
    assert tracking.config["checkpoint"] == "stale"


def test_eval_swanlab_disabled_makes_no_tracking_calls(
    eval_config, runtime, swanlab_sdk
):
    """Default evaluation never initializes a tracking run or logs in."""
    evaluation.evaluate_checkpoint(eval_config, [_task(), _task("task-b")], {})
    assert swanlab_sdk.mock_calls == []


def test_eval_preflight_with_swanlab_enabled_does_not_start_tracking(
    eval_config, monkeypatch, swanlab_sdk
):
    """Checking configuration must not initialize or authenticate a tracking run."""
    monkeypatch.setattr(
        evaluation, "check_runner_health", lambda _: RunnerHealth(0, 0, 4)
    )
    evaluation.main(
        ["--config", str(EXAMPLE), "--check-only", "++stats_logger.swanlab.mode=cloud"]
    )
    assert swanlab_sdk.mock_calls == []


@pytest.mark.parametrize("error", [RuntimeError("runner failed"), KeyboardInterrupt()])
def test_eval_swanlab_failure_closes_run_without_partial_scores(
    eval_config, runtime, tmp_path, swanlab_sdk, error
):
    """Failed sweeps retain their original exception even when tracker cleanup fails."""
    eval_config.stats_logger.swanlab.mode = "offline"
    controller, scheduler, _ = runtime
    controller.wait.side_effect = error
    tracker = swanlab_sdk.init.return_value
    tracker.finish.side_effect = RuntimeError("tracking shutdown failed")
    with pytest.raises(type(error)) as exc:
        evaluation.evaluate_checkpoint(eval_config, [_task()], {})
    assert exc.value is error
    controller.destroy.assert_called_once()
    scheduler.delete_workers.assert_called_once()
    metrics = tracker.log.call_args.args[0]
    assert metrics["mlperf_validation/aborted"] == 1
    assert metrics["mlperf_validation/completed"] == 0
    assert not any("pass_at" in name for name in metrics)
    tracker.finish.assert_called_once_with(interrupt=True)
    assert (
        json.loads((tmp_path / "validation/status.json").read_text())["status"]
        == "aborted"
    )


@pytest.mark.parametrize("stage", ["login", "init", "log", "finish"])
def test_eval_tracking_error_does_not_invalidate_local_results(
    eval_config, runtime, tmp_path, swanlab_sdk, stage
):
    """Tracker availability cannot turn a completed evaluation into an aborted sweep."""
    eval_config.stats_logger.swanlab.mode = "cloud"
    eval_config.stats_logger.swanlab.api_key = "tracking-secret"
    target = (
        swanlab_sdk if stage in ("login", "init") else swanlab_sdk.init.return_value
    )
    getattr(target, stage).side_effect = RuntimeError("tracking unavailable")
    summary = evaluation.evaluate_checkpoint(
        eval_config, [_task(), _task("task-b")], {}
    )
    assert summary["pass_at_4"] == 0.5
    assert json.loads((tmp_path / "validation/checkpoint.json").read_text()) == summary
    assert (
        json.loads((tmp_path / "validation/status.json").read_text())["status"]
        != "aborted"
    )


def test_eval_swanlab_real_offline_sdk_writes_locally(
    eval_config, monkeypatch, tmp_path
):
    """The installed SDK can record and close an offline run without authentication."""
    import swanlab

    monkeypatch.setattr(evaluation, "swanlab", swanlab)
    login = Mock(side_effect=AssertionError("Offline evaluation must not log in"))
    monkeypatch.setattr(swanlab, "login", login)
    tracking = eval_config.stats_logger.swanlab
    tracking.mode = "offline"
    tracking.project = "mlperf-eval-test"
    tracking.logdir = str(tmp_path / "offline-tracking")
    tracker = evaluation._start_swanlab(eval_config, 1)
    assert tracker is not None
    try:
        evaluation._log_swanlab(tracker, {"pass_at_4": 1.0, "num_attempts": 4})
    finally:
        tracker.finish()
    login.assert_not_called()
    assert any(Path(tracking.logdir).iterdir())
