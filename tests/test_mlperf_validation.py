# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from examples.swe.mlperf.config import MLPerfValidationConfig
from examples.swe.mlperf.validation import (
    summarize_validation,
    validate_validation_config,
)


def _group(task_index, rewards, failed=None, sample_indices=None):
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "validation_task_index": torch.full((4,), task_index, dtype=torch.int64),
        "validation_sample_index": torch.tensor(
            sample_indices if sample_indices is not None else [0, 1, 2, 3],
            dtype=torch.int64,
        ),
        "validation_failed": torch.tensor(
            failed if failed is not None else [False] * 4, dtype=torch.bool
        ),
        "begin_of_trajectory": torch.ones(4, dtype=torch.int32),
    }


def _config():
    return SimpleNamespace(
        valid_dataset=SimpleNamespace(shuffle=False, drop_last=False),
        eval_gconfig=SimpleNamespace(
            n_samples=4, greedy=False, temperature=0.1, top_p=0.95
        ),
        actor=SimpleNamespace(scheduling_strategy=None),
        rollout=SimpleNamespace(
            max_head_offpolicyness=0, consumer_batch_size=1, scheduling_strategy=None
        ),
        train_dataset=SimpleNamespace(batch_size=1),
        dynamic_bs=False,
    )


def _tasks(*instance_ids):
    return [
        {"responses_create_params": {"metadata": {"instance_id": instance_id}}}
        for instance_id in instance_ids
    ]


def test_validation_mixed_success_keeps_failed_slots_and_sorts_attempts():
    """Four independent attempts contribute to fixed task and attempt denominators."""
    groups = [
        _group(2, [0, 0, 0, 0], failed=[True] * 4),
        _group(0, [0, 1, 0, 0], sample_indices=[3, 2, 1, 0]),
        _group(1, [1, 1, 1, 1]),
    ]

    result = summarize_validation(groups, ["a", "b", "c"])

    assert result["pass_at_4"] == pytest.approx(2 / 3)
    assert result["pass_all_4"] == pytest.approx(1 / 3)
    assert result["pass_at_1"] == pytest.approx(5 / 12)
    assert result["num_tasks"] == 3
    assert result["num_attempts"] == 12
    assert result["solved_tasks"] == 2
    assert result["failed_attempts"] == 4
    assert [(a["task_index"], a["sample_index"]) for a in result["attempts"]] == [
        (task, sample) for task in range(3) for sample in range(4)
    ]
    assert result["attempts"][2] == {
        "task_index": 0,
        "instance_id": "a",
        "sample_index": 2,
        "reward": 1.0,
        "failed": False,
    }
    json.dumps(result, allow_nan=False)


def test_validation_all_attempts_failed_reports_zero_with_full_denominator():
    """Failed slots remain measured attempts instead of disappearing from the sweep."""
    result = summarize_validation([_group(0, [0] * 4, failed=[True] * 4)], ["a"])

    assert result["pass_at_4"] == 0
    assert result["pass_all_4"] == 0
    assert result["pass_at_1"] == 0
    assert result["num_attempts"] == result["failed_attempts"] == 4


@pytest.mark.parametrize("solved_tasks,passes_target", [(175, False), (176, True)])
def test_validation_full_split_threshold_counts_solved_tasks(
    solved_tasks, passes_target
):
    """The 0.7 target on 251 tasks requires at least 176 solved tasks."""
    groups = [
        _group(index, [int(index < solved_tasks), 0, 0, 0]) for index in range(251)
    ]

    result = summarize_validation(groups, [f"task-{index}" for index in range(251)])

    assert result["num_tasks"] == 251
    assert result["num_attempts"] == 1004
    assert result["solved_tasks"] == solved_tasks
    assert (result["pass_at_4"] >= MLPerfValidationConfig().target) is passes_target


@pytest.mark.parametrize("task_ids", [[], [""], [" "], [None], [4], ["a", "a"]])
def test_validation_invalid_task_ids_abort(task_ids):
    """Missing or ambiguous task identities must never produce a partial score."""
    with pytest.raises(ValueError):
        summarize_validation([], task_ids)


@pytest.mark.parametrize(
    "groups",
    [[], [None], [{}, {}], [None, None]],
)
def test_validation_missing_groups_abort(groups):
    """A complete sweep requires exactly one well-formed group for each task."""
    with pytest.raises(ValueError):
        summarize_validation(groups, ["a"])


def test_validation_duplicate_task_groups_abort():
    """Duplicating one task cannot replace an omitted task."""
    with pytest.raises(ValueError, match="duplicate task"):
        summarize_validation([_group(0, [1] * 4), _group(0, [1] * 4)], ["a", "b"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("rewards", None),
        ("rewards", [1, 0, 0, 0]),
        ("rewards", torch.tensor([[1, 0, 0, 0]])),
        ("rewards", torch.tensor([1.0, 0.0, 0.0])),
        ("rewards", torch.tensor([float("nan"), 0, 0, 0])),
        ("rewards", torch.tensor([float("inf"), 0, 0, 0])),
        ("rewards", torch.tensor([-1, 0, 0, 0])),
        ("rewards", torch.tensor([0.5, 0, 0, 0])),
        ("rewards", torch.tensor([True, False, False, False])),
        ("rewards", torch.tensor([1j, 0, 0, 0])),
        ("validation_task_index", torch.tensor([0, 0, 0, 1])),
        ("validation_task_index", torch.tensor([-1] * 4)),
        ("validation_task_index", torch.tensor([1] * 4)),
        ("validation_task_index", torch.zeros(4, dtype=torch.float32)),
        ("validation_sample_index", torch.tensor([0, 0, 2, 3])),
        ("validation_sample_index", torch.tensor([0, 1, 2, 4])),
        ("validation_sample_index", torch.tensor([-1, 0, 1, 2])),
        ("validation_sample_index", torch.arange(4, dtype=torch.float32)),
        ("validation_failed", torch.tensor([0, 0, 0, 0])),
        ("validation_failed", torch.tensor([True, False, False, False])),
        ("begin_of_trajectory", torch.tensor([1, 0, 1, 1])),
        ("begin_of_trajectory", torch.tensor([1, 2, 1, 1])),
        ("begin_of_trajectory", torch.ones(4, dtype=torch.float32)),
    ],
)
def test_validation_malformed_attempt_data_aborts(field, value):
    """Invalid rewards, identity, or boundaries must fail before reporting metrics."""
    group = _group(0, [1, 0, 0, 0])
    group[field] = value

    with pytest.raises(ValueError):
        summarize_validation([group], ["a"])


def test_validation_default_schedule_uses_completed_optimizer_steps():
    """Default validation starts after update20 and continues after each update."""
    schedule = MLPerfValidationConfig()

    assert not schedule.should_evaluate(0)
    assert not schedule.should_evaluate(19)
    assert schedule.should_evaluate(20)
    assert schedule.should_evaluate(21)
    assert schedule.target == 0.7


def test_validation_schedule_interval_is_global_step_modulo():
    """The reference period is measured from step0, not offset from its start step."""
    schedule = MLPerfValidationConfig(start_step=3, interval=2, target=None)

    assert [step for step in range(7) if schedule.should_evaluate(step)] == [4, 6]
    assert schedule.target is None


@pytest.mark.parametrize("name", ["start_step", "interval"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "1", None])
def test_validation_schedule_invalid_step_fields_rejected(name, value):
    """Schedule bounds require positive integers rather than coercion."""
    with pytest.raises(ValueError, match=name):
        MLPerfValidationConfig(**{name: value})


@pytest.mark.parametrize(
    "target", [-0.01, 1.01, True, False, "0.7", float("nan"), float("inf")]
)
def test_validation_invalid_target_rejected(target):
    """A stopping threshold must be a finite probability or explicitly disabled."""
    with pytest.raises(ValueError, match="target"):
        MLPerfValidationConfig(target=target)


@pytest.mark.parametrize("target", [None, 0, 1, 0.7])
def test_validation_valid_target_accepted(target):
    """Both probability boundaries and disabled stopping are supported."""
    assert MLPerfValidationConfig(target=target).target == target


def test_validation_config_accepts_complete_small_sweep():
    """Small smoke splits use the same complete-sweep semantics as the full split."""
    validate_validation_config(_config(), _tasks("a", "b"))


@pytest.mark.parametrize(
    "section,name,value",
    [
        ("valid_dataset", "shuffle", True),
        ("valid_dataset", "drop_last", True),
        ("eval_gconfig", "n_samples", 1),
        ("eval_gconfig", "n_samples", 4.0),
        ("eval_gconfig", "n_samples", True),
        ("eval_gconfig", "greedy", True),
        ("eval_gconfig", "temperature", 1.0),
        ("eval_gconfig", "top_p", 1.0),
        ("rollout", "max_head_offpolicyness", 1),
    ],
)
def test_validation_config_incompatible_settings_rejected(section, name, value):
    """Validation cannot silently use training sampling or omit held-out tasks."""
    config = _config()
    setattr(getattr(config, section), name, value)

    with pytest.raises(ValueError):
        validate_validation_config(config, _tasks("a"))


@pytest.mark.parametrize("section", ["valid_dataset", "eval_gconfig"])
def test_validation_config_missing_section_rejected(section):
    """The protocol requires explicit validation data and generation settings."""
    config = _config()
    setattr(config, section, None)

    with pytest.raises(ValueError):
        validate_validation_config(config, _tasks("a"))


@pytest.mark.parametrize(
    "tasks",
    [
        [],
        _tasks("a", "a"),
        _tasks(""),
        _tasks(None),
        [{}],
        [None],
        [{"responses_create_params": {}}],
    ],
)
def test_validation_config_invalid_dataset_rejected(tasks):
    """Every supplied validation row must have a unique stable instance identity."""
    with pytest.raises(ValueError):
        validate_validation_config(_config(), tasks)


@pytest.mark.parametrize("value", [True, None, 0])
def test_validation_rejects_dynamic_batching(value):
    """Fixed batch consumption must drain admitted training jobs before validation."""
    config = _config()
    config.dynamic_bs = value

    with pytest.raises(ValueError, match="dynamic_bs"):
        validate_validation_config(config, _tasks("a"))


@pytest.mark.parametrize(
    "consumer_batch_size,batch_size",
    [
        (2, 1),
        (1, 2),
        (0, 0),
        (-1, -1),
        (True, True),
        (1.0, 1),
        (1, 1.0),
        (None, 1),
        (1, None),
        (1, True),
        (True, 1),
    ],
)
def test_validation_rejects_incompatible_admission_batch_sizes(
    consumer_batch_size, batch_size
):
    """The admission window must equal one positive fixed training batch."""
    config = _config()
    config.rollout.consumer_batch_size = consumer_batch_size
    config.train_dataset.batch_size = batch_size

    with pytest.raises(ValueError, match="consumer_batch_size"):
        validate_validation_config(config, _tasks("a"))


def test_validation_accepts_matching_larger_training_batch():
    """The no-prefetch invariant supports any matching positive fixed batch size."""
    config = _config()
    config.rollout.consumer_batch_size = 16
    config.train_dataset.batch_size = 16

    validate_validation_config(config, _tasks("a"))


@pytest.mark.parametrize("role,target", [("actor", "rollout"), ("rollout", "actor")])
def test_validation_rejects_actor_rollout_colocation(role, target):
    """Shared actor/rollout workers can resume training before a validation sweep."""
    config = _config()
    getattr(config, role).scheduling_strategy = SimpleNamespace(
        type="colocation", target=target
    )

    with pytest.raises(ValueError, match="separate actor and rollout"):
        validate_validation_config(config, _tasks("a"))


@pytest.mark.parametrize("role,target", [("actor", "rollout"), ("rollout", "actor")])
def test_validation_accepts_explicit_actor_rollout_separation(role, target):
    """A dormant target name without colocation does not share actor/rollout workers."""
    config = _config()
    getattr(config, role).scheduling_strategy = SimpleNamespace(
        type="separation", target=target
    )

    validate_validation_config(config, _tasks("a"))


def test_validation_accepts_colocation_with_unrelated_role():
    """The supported-mode guard matches actor/rollout colocation only."""
    config = _config()
    config.actor.scheduling_strategy = SimpleNamespace(
        type="colocation", target="critic"
    )

    validate_validation_config(config, _tasks("a"))
