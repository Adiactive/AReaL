# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples.swe.mlperf import trainer as trainer_module
from examples.swe.mlperf.config import MLPerfValidationConfig
from examples.swe.mlperf.trainer import MLPerfSWETrainer
from examples.swe.mlperf.workflow import VALIDATION_TASK_INDEX

from areal.trainer import rl_trainer
from areal.trainer.rl_trainer import PPOTrainer


def _task(instance_id, **params):
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": f"Fix {instance_id}"}],
            "metadata": {"instance_id": instance_id, "nested": {"preserve": True}},
            "temperature": 0.7,
            "top_p": 0.8,
            **params,
        },
        "benchmark_metadata": {"split": "validation"},
    }


def _group(task_index, rewards):
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "validation_task_index": torch.full((4,), task_index, dtype=torch.int64),
        "validation_sample_index": torch.arange(4, dtype=torch.int64),
        "validation_failed": torch.zeros(4, dtype=torch.bool),
        "begin_of_trajectory": torch.ones(4, dtype=torch.int32),
    }


@pytest.fixture
def trainer_factory(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer_module, "is_single_controller", lambda: True)
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    monkeypatch.setattr(
        trainer_module.StatsLogger, "get_log_path", lambda config: str(tmp_path)
    )

    def create(tasks=None, groups=None):
        tasks = tasks if tasks is not None else [_task("task-a")]
        trainer = object.__new__(MLPerfSWETrainer)
        trainer.config = SimpleNamespace(
            validation=MLPerfValidationConfig(start_step=1),
            valid_dataset=SimpleNamespace(shuffle=False, drop_last=False),
            eval_gconfig=SimpleNamespace(
                n_samples=4,
                greedy=False,
                temperature=0.1,
                top_p=0.95,
                max_new_tokens=4096,
                max_tokens=8192,
            ),
            actor=SimpleNamespace(scheduling_strategy=None),
            rollout=SimpleNamespace(
                max_head_offpolicyness=0,
                consumer_batch_size=1,
                scheduling_strategy=None,
            ),
            train_dataset=SimpleNamespace(batch_size=1),
            dynamic_bs=False,
            stats_logger=object(),
        )
        trainer._validation_stats = {}
        trainer._last_validation = None
        trainer._completed_steps = 0
        trainer.valid_dataloader = [tasks[:1], tasks[1:]]
        trainer.actor = Mock()
        trainer.actor.export_stats.return_value = {"actor/reward": 0.25}
        trainer.rollout = Mock()
        trainer.rollout.export_stats.return_value = {"rollout/count": 4}
        trainer.eval_rollout = Mock()
        trainer.eval_rollout.export_stats.return_value = {"evaluation/count": 4}
        trainer.stats_logger = Mock()
        trainer.eval_rollout.wait.return_value = (
            groups if groups is not None else [_group(0, [1, 0, 0, 0])]
        )
        return trainer

    return create


def _evaluate(trainer, global_step=0):
    return trainer._evaluate(
        "validation-workflow",
        {"agent_kwargs": {"temperature": 0.1}},
        0,
        global_step,
        global_step,
    )


def test_validation_unscheduled_step_skips_submission_and_clears_metrics(
    trainer_factory, tmp_path
):
    """A skipped sweep cannot emit stale metrics from an earlier result."""
    trainer = trainer_factory()
    trainer.config.validation = MLPerfValidationConfig(start_step=20)
    trainer._validation_stats = {"mlperf_validation/pass_at_4": 1.0}

    assert _evaluate(trainer, global_step=18) is None

    assert trainer._completed_steps == 19
    assert trainer._validation_stats == {}
    trainer.eval_rollout.submit.assert_not_called()
    trainer.eval_rollout.wait.assert_not_called()
    trainer.actor.clear_batches.assert_not_called()
    assert not (tmp_path / "validation").exists()


def test_validation_waits_for_full_unordered_sweep_before_comparing_target(
    trainer_factory, tmp_path
):
    """Later failures stay in the denominator even when early groups all succeed."""
    tasks = [_task("task-c"), _task("task-a"), _task("task-b")]
    groups = [_group(1, [1] * 4), _group(0, [1, 0, 0, 0]), _group(2, [0] * 4)]
    trainer = trainer_factory(tasks, groups)

    def wait(count, timeout):
        assert count == 3
        assert timeout is None
        assert trainer.eval_rollout.submit.call_count == 3
        assert trainer._last_validation is None
        return groups

    trainer.eval_rollout.wait.side_effect = wait

    assert _evaluate(trainer) is None

    assert trainer._last_validation["num_tasks"] == 3
    assert trainer._last_validation["num_attempts"] == 12
    assert trainer._last_validation["pass_at_4"] == pytest.approx(2 / 3)
    assert trainer._last_validation["target_reached"] is False
    assert trainer._last_validation["model_version"] == 1
    assert trainer.actor.clear_batches.call_count == 1
    assert all(
        actual is expected
        for actual, expected in zip(
            trainer.actor.clear_batches.call_args.args, groups, strict=True
        )
    )
    artifact = json.loads((tmp_path / "validation/step_000001.json").read_text())
    assert artifact["completed_step"] == 1
    assert len(artifact["attempts"]) == 12
    assert [attempt["instance_id"] for attempt in artifact["attempts"]] == [
        task for task in ("task-c", "task-a", "task-b") for _ in range(4)
    ]
    assert list((tmp_path / "validation").glob("*.tmp")) == []


def test_validation_sampling_overrides_rows_without_mutating_tasks(trainer_factory):
    """Validation uses its own sampling while retaining smaller row token budgets."""
    tasks = [
        _task("task-a", max_output_tokens=8192),
        _task("task-b", max_output_tokens=1024),
        _task("task-c"),
    ]
    snapshot = copy.deepcopy(tasks)
    trainer = trainer_factory(tasks, [_group(index, [0] * 4) for index in range(3)])

    _evaluate(trainer)

    assert tasks == snapshot
    for index, (submission, cap) in enumerate(
        zip(trainer.eval_rollout.submit.call_args_list, (4096, 1024, 4096), strict=True)
    ):
        task, workflow, kwargs = submission.args
        assert task is not tasks[index]
        assert task[VALIDATION_TASK_INDEX] == index
        assert workflow == "validation-workflow"
        assert kwargs == {"agent_kwargs": {"temperature": 0.1}}
        assert submission.kwargs == {"group_size": 4, "is_eval": True}
        params = task["responses_create_params"]
        assert params["temperature"] == 0.1
        assert params["top_p"] == 0.95
        assert params["max_output_tokens"] == cap
        assert (
            params["metadata"] == snapshot[index]["responses_create_params"]["metadata"]
        )
        assert (
            params["metadata"]
            is not tasks[index]["responses_create_params"]["metadata"]
        )


@pytest.mark.parametrize("groups", [[None], [{"rewards": torch.tensor([1.0])}]])
def test_validation_malformed_group_is_cleaned_without_success_artifact(
    trainer_factory, tmp_path, groups
):
    """Invalid result structures abort the sweep and release received batch storage."""
    trainer = trainer_factory(groups=groups)

    with pytest.raises(ValueError):
        _evaluate(trainer)

    assert trainer.actor.clear_batches.call_count == 1
    assert trainer.actor.clear_batches.call_args.args[0] is groups[0]
    assert trainer._last_validation is None
    assert trainer._validation_stats == {}
    assert not (tmp_path / "validation").exists()


def test_validation_stats_merge_once_without_mutating_supplied_actor_stats(
    trainer_factory,
):
    """Each evaluation step commits its metrics once and later skipped steps omit them."""
    trainer = trainer_factory()
    _evaluate(trainer)
    actor_stats = {"actor/reward": 0.4}

    trainer._export_and_commit_stats(0, 0, 0, actor_stats)

    trainer.stats_logger.commit.assert_called_once()
    trainer.actor.export_stats.assert_not_called()
    assert actor_stats == {"actor/reward": 0.4}
    committed = trainer.stats_logger.commit.call_args.args[3]
    assert committed["actor/reward"] == 0.4
    assert committed["mlperf_validation/pass_at_4"] == 1.0
    assert committed["mlperf_validation/num_attempts"] == 4
    assert "mlperf_validation/attempts" not in committed
    assert committed["rollout/count"] == committed["evaluation/count"] == 4

    trainer.config.validation = MLPerfValidationConfig(start_step=20)
    _evaluate(trainer, global_step=1)
    trainer._export_and_commit_stats(0, 1, 1)

    assert trainer.stats_logger.commit.call_count == 2
    trainer.actor.export_stats.assert_called_once()
    next_stats = trainer.stats_logger.commit.call_args.args[3]
    assert not any(key.startswith("mlperf_validation/") for key in next_stats)


@pytest.mark.parametrize(
    "rewards,target,start_step,status",
    [
        ([1, 0, 0, 0], 0.7, 1, "target_reached"),
        ([0, 0, 0, 0], 0.7, 1, "target_not_reached"),
        ([1, 0, 0, 0], 0.7, 20, "not_evaluated"),
        ([1, 0, 0, 0], None, 1, "completed"),
    ],
)
def test_validation_train_writes_terminal_status(
    trainer_factory, monkeypatch, tmp_path, rewards, target, start_step, status
):
    """The final artifact distinguishes convergence, incomplete progress, and no sweep."""
    trainer = trainer_factory(groups=[_group(0, rewards)])
    trainer.config.validation = MLPerfValidationConfig(
        start_step=start_step, target=target
    )
    monkeypatch.setattr(PPOTrainer, "train", lambda self: _evaluate(self))

    trainer.train()

    artifact = json.loads((tmp_path / "validation/status.json").read_text())
    assert artifact["status"] == status
    assert artifact["target"] == target
    assert artifact["target_reached"] is (status == "target_reached")
    assert artifact["completed_steps"] == 1
    assert "attempts" not in artifact
    assert (tmp_path / "validation/step_000001.json").exists() is (
        status != "not_evaluated"
    )


def test_validation_target_reached_keeps_evaluating_and_reports_latest_sweep(
    trainer_factory, monkeypatch, tmp_path
):
    """Reaching the target leaves later sweeps active and preserves each result."""
    trainer = trainer_factory()
    trainer.eval_rollout.wait.side_effect = [
        [_group(0, [1, 0, 0, 0])],
        [_group(0, [0, 0, 0, 0])],
    ]

    def train(self):
        for global_step in range(2):
            assert _evaluate(self, global_step=global_step) is None

    monkeypatch.setattr(PPOTrainer, "train", train)

    trainer.train()

    assert trainer.eval_rollout.submit.call_count == 2
    assert trainer.eval_rollout.wait.call_count == 2
    first = json.loads((tmp_path / "validation/step_000001.json").read_text())
    latest = json.loads((tmp_path / "validation/step_000002.json").read_text())
    status = json.loads((tmp_path / "validation/status.json").read_text())
    assert first["target_reached"] is True
    assert first["pass_at_4"] == 1.0
    assert latest["target_reached"] is False
    assert latest["pass_at_4"] == 0.0
    assert status["status"] == "target_not_reached"
    assert status["completed_steps"] == 2
    assert status["target_reached"] is False
    assert status["pass_at_4"] == latest["pass_at_4"]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_validation_train_failure_writes_aborted_and_preserves_error(
    trainer_factory, monkeypatch, tmp_path, error_type
):
    """Failures and interruption never produce a success status."""
    trainer = trainer_factory()
    error = error_type("stopped during training")

    def train(self):
        _evaluate(self)
        raise error

    monkeypatch.setattr(PPOTrainer, "train", train)

    with pytest.raises(error_type) as exc_info:
        trainer.train()

    assert exc_info.value is error
    artifact = json.loads((tmp_path / "validation/status.json").read_text())
    assert artifact["status"] == "aborted"
    assert artifact["target_reached"] is False
    assert artifact["pass_at_4"] == 1.0


@pytest.mark.parametrize("is_eval", [False, True])
def test_validation_rollout_configuration_changes_only_evaluation_copy(
    monkeypatch, is_eval
):
    """Metric-only evaluation must not alter training trajectory export checks."""
    trainer = object.__new__(MLPerfSWETrainer)
    trainer.config = SimpleNamespace(eval_gconfig=SimpleNamespace(max_tokens=16384))
    rollout_config = SimpleNamespace(
        agent=SimpleNamespace(engine_max_tokens=32768),
        dump_to_file=True,
        check_trajectory_format=True,
        nested={"preserve": [1]},
    )
    original = copy.deepcopy(rollout_config)
    initialized = Mock(return_value=object())
    monkeypatch.setattr(PPOTrainer, "_init_rollout", initialized)

    result = trainer._init_rollout(rollout_config, is_eval=is_eval, lora_path="adapter")

    assert result is initialized.return_value
    actual, eval_flag, lora_path = initialized.call_args.args
    assert eval_flag is is_eval
    assert lora_path == "adapter"
    assert actual.dump_to_file is (not is_eval)
    assert actual.check_trajectory_format is (not is_eval)
    assert actual.agent.engine_max_tokens == (16384 if is_eval else 32768)
    assert rollout_config.agent.engine_max_tokens == 32768
    assert rollout_config == original
    if is_eval:
        assert actual is not rollout_config
        assert actual.agent is not rollout_config.agent
        assert actual.nested is not rollout_config.nested
    else:
        assert actual is rollout_config
