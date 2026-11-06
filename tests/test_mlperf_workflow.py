# SPDX-License-Identifier: Apache-2.0

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import httpx
import pytest
import torch

from examples.swe.mlperf import workflow as mlperf_module
from examples.swe.mlperf.config import AGENT_PATH
from examples.swe.mlperf.workflow import VALIDATION_TASK_INDEX, MLPerfSWEWorkflow

from areal.api import ModelResponse
from areal.api.cli_args import AgentConfig, NormConfig, PPOActorConfig
from areal.experimental.openai.proxy import workflow as proxy_module
from areal.experimental.openai.proxy.workflow import (
    EmptyAgentSessionError,
    OpenAIProxyWorkflow,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow, RemoteInfEngine
from areal.infra.workflow_context import WorkflowContext
from areal.infra.workflow_executor import check_trajectory_format
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils.data import pack_tensor_dict
from areal.workflow.nemo_gym_swe import NeMoGymSWEAgent, NeMoGymSWEError


@pytest.fixture(autouse=True)
def eval_context():
    original = workflow_context.get()
    workflow_context.set(WorkflowContext(is_eval=True, task_id=17))
    yield
    workflow_context.set(original)


@pytest.fixture(autouse=True)
def attempt_metrics(monkeypatch):
    """Keep workflow counters isolated from unrelated tests and actor metrics."""
    tracker = Mock()
    monkeypatch.setattr(mlperf_module, "stats_tracker", Mock())
    mlperf_module.stats_tracker.get.return_value = tracker
    return tracker


@pytest.fixture
def make_proxy(monkeypatch):
    """Use the real proxy workflow with only its HTTP boundary replaced."""
    monkeypatch.setattr(workflow_context, "get_aiohttp_session", AsyncMock())
    monkeypatch.setattr(proxy_module, "_log_interaction_reward_metrics", Mock())
    finalizer = AsyncMock()
    monkeypatch.setattr(proxy_module, "post_json", finalizer)

    def build(exports, run_agent=None):
        sessions = []

        class ProxyClient:
            def __init__(self, **kwargs):
                self.sample_index = workflow_context.get().sample_idx or 0
                self.session_api_key = f"key-{self.sample_index}"
                self.closed = False
                self.exported = False
                sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                self.closed = True

            async def set_last_reward(self, reward):
                self.assigned_reward = reward

            async def export_interactions(self, **kwargs):
                self.exported = True
                return deepcopy(exports[self.sample_index])

        monkeypatch.setattr(proxy_module, "OpenAIProxyClient", ProxyClient)
        agent = NeMoGymSWEAgent(runner_url="http://runner", model="test-model")
        proxy = OpenAIProxyWorkflow(
            mode="inline", agent=agent, proxy_addr="http://proxy"
        )
        proxy._grant_capacity = AsyncMock()
        proxy._run_agent = run_agent or AsyncMock(return_value=1.0)
        return proxy, sessions, finalizer

    return build


def _interactions(*rewards, rollout_reward=None):
    result = {
        f"turn-{index}": InteractionWithTokenLogpReward(
            reward=reward,
            model_response=ModelResponse(
                input_tokens=[1, 2],
                output_tokens=[3],
                output_logprobs=[-0.1],
                output_versions=[0],
            ),
        )
        for index, reward in enumerate(rewards)
    }
    if result:
        next(reversed(result.values())).rollout_reward = rollout_reward
    return result


def _assert_failed_attempt(result, *, is_eval):
    assert result["rewards"].tolist() == [0.0]
    assert result["begin_of_trajectory"].tolist() == [1]
    if is_eval:
        assert result["validation_failed"].tolist() == [True]
        return
    assert check_trajectory_format(result, batch_size=1)
    assert result["input_ids"].tolist() == [[7, 7]]
    assert result["attention_mask"].tolist() == [[True, True]]
    assert result["loss_mask"].count_nonzero() == 0
    assert result["logprobs"].count_nonzero() == 0
    assert result["versions"].tolist() == [[-1, -1]]
    assert result["is_truncated"].tolist() == [False]
    assert result["rollout_rewards"] == [0.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eval", [False, True])
async def test_group_preserves_failed_slot_siblings_and_multiturn_reward(
    make_proxy, is_eval
):
    """A failed attempt remains one slot while siblings retain all usable turns."""
    workflow_context.set(WorkflowContext(is_eval=is_eval, task_id=17))
    finished = []
    failed = asyncio.Event()
    exports = [
        _interactions(0.25, 0.5, rollout_reward=1.0),
        _interactions(1.0),
        _interactions(0.0),
        _interactions(1.0),
    ]

    async def run_agent(api_key, data):
        index = workflow_context.get().sample_idx
        assert VALIDATION_TASK_INDEX not in data
        data["responses_create_params"]["input"].append("local mutation")
        if index == 1:
            failed.set()
            raise NeMoGymSWEError("runner disconnected after a partial generation")
        await failed.wait()
        finished.append(index)
        return 0.0 if index == 2 else 1.0

    proxy, sessions, finalizer = make_proxy(exports, run_agent)
    workflow = GroupedRolloutWorkflow(
        MLPerfSWEWorkflow(proxy, fallback_token_id=7), 4, Mock()
    )
    task = {
        VALIDATION_TASK_INDEX: 8,
        "responses_create_params": {"input": ["original"]},
    }
    original = deepcopy(task)

    result = await asyncio.wait_for(workflow.arun_episode(None, task), timeout=2)

    assert task == original
    assert sorted(finished) == [0, 2, 3]
    assert len(sessions) == 4 and all(session.closed for session in sessions)
    assert not next(s for s in sessions if s.sample_index == 1).exported
    finalizer.assert_awaited_once()
    scope = "eval" if is_eval else "train"
    assert finalizer.call_args.kwargs["payload"].group_id == f"{scope}:17"
    if is_eval:
        expected = {
            "rewards": torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            "validation_task_index": torch.tensor([8, 8, 8, 8], dtype=torch.int64),
            "validation_sample_index": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            "validation_failed": torch.tensor([False, True, False, False]),
            "begin_of_trajectory": torch.ones(4, dtype=torch.int32),
        }
        assert result.keys() == expected.keys()
        for name, value in expected.items():
            assert result[name].device.type == "cpu"
            torch.testing.assert_close(result[name], value, rtol=0, atol=0)
        return

    assert check_trajectory_format(result, batch_size=5)
    assert result["rewards"].tolist() == [0.25, 0.5, 0.0, 0.0, 1.0]
    assert result["rollout_rewards"] == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert result["begin_of_trajectory"].tolist() == [1, 0, 1, 1, 1]
    assert result["loss_mask"].sum(-1).tolist() == [1, 1, 0, 1, 1]
    assert result["attention_mask"].sum(-1).tolist() == [3, 3, 2, 3, 3]
    assert result["input_ids"][2, :2].tolist() == [7, 7]
    assert not any(name.startswith("validation_") for name in result)

    # The failed zero participates once, even though its own tokens carry no loss.
    norm = NormConfig(
        mean_level="group", std_level="group", group_size=4, std_unbiased=False
    )
    actor = PPOActor(
        PPOActorConfig(
            path="dummy", reward_norm=norm, recompute_logprob=False, kl_ctl=0.0
        ),
        engine=None,
    )
    advantages = actor.compute_advantages([result])[0]
    reference_rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    expected = (result["rewards"] - reference_rewards.mean()) / (
        reference_rewards.std(unbiased=False) + norm.eps
    )
    active = advantages["loss_mask"].bool()
    torch.testing.assert_close(
        advantages["advantages"][active],
        expected[:, None].expand_as(advantages["advantages"])[active],
        rtol=1e-6,
        atol=1e-6,
    )
    assert advantages["advantages"][2].count_nonzero() == 0
    assert advantages["loss_mask"][2].count_nonzero() == 0
    packed = pack_tensor_dict(advantages)
    assert packed["cu_seqlens"].tolist() == [0, 3, 6, 8, 11, 14]
    assert packed["loss_mask"][6:8].count_nonzero() == 0
    assert packed["advantages"][6:8].count_nonzero() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        NeMoGymSWEError("runner error"),
        EmptyAgentSessionError("empty session"),
        aiohttp.ClientConnectionError("proxy disconnected"),
        httpx.ConnectError("connection failed"),
        TimeoutError("request expired"),
    ],
)
async def test_expected_failure_before_first_token_retains_zero_slot(
    make_proxy, failure, is_eval
):
    """Expected runner/proxy failures retain a slot in both training and evaluation."""
    workflow_context.set(WorkflowContext(is_eval=is_eval, task_id=17))
    proxy, sessions, _ = make_proxy([{}], AsyncMock(side_effect=failure))
    result = await MLPerfSWEWorkflow(proxy, fallback_token_id=7).arun_episode(
        None, {VALIDATION_TASK_INDEX: 0}
    )
    _assert_failed_attempt(result, is_eval=is_eval)
    assert all(session.closed for session in sessions)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize(
    "export",
    [
        {},
        None,
        _interactions(None),
        _interactions(True),
        _interactions(0.5),
        _interactions(float("nan")),
        _interactions(float("inf")),
        _interactions(1.0, rollout_reward=-1.0),
        {
            "first": next(iter(_interactions(1.0, rollout_reward=0.0).values())),
            "last": next(iter(_interactions(1.0, rollout_reward=1.0).values())),
        },
        {"missing-tokens": InteractionWithTokenLogpReward(reward=1.0)},
        *[
            pytest.param(
                {
                    "incomplete-generation": InteractionWithTokenLogpReward(
                        reward=1.0,
                        model_response=ModelResponse(
                            input_tokens=[1, 2],
                            output_tokens=[3, 4],
                            output_logprobs=logprobs,
                            output_versions=versions,
                        ),
                    )
                },
                id=case,
            )
            for case, logprobs, versions in [
                ("missing-logprobs", [], [0, 0]),
                ("short-logprobs", [-0.1], [0, 0]),
                ("missing-versions", [-0.1, -0.2], []),
                ("short-versions", [-0.1, -0.2], [0]),
            ]
        ],
        {
            "empty-generation": InteractionWithTokenLogpReward(
                reward=1.0,
                model_response=ModelResponse(
                    input_tokens=[1, 2],
                    output_tokens=[],
                    output_logprobs=[],
                    output_versions=[],
                ),
            )
        },
    ],
)
async def test_empty_or_unusable_export_retains_failed_slot(
    make_proxy, export, is_eval
):
    """Incomplete exports and invalid raw rewards cannot train or score success."""
    workflow_context.set(WorkflowContext(is_eval=is_eval, task_id=17))
    proxy, _, _ = make_proxy([export])
    result = await MLPerfSWEWorkflow(proxy, fallback_token_id=7).arun_episode(
        None, {VALIDATION_TASK_INDEX: 0}
    )
    _assert_failed_attempt(result, is_eval=is_eval)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eval", [False, True])
async def test_result_conversion_failure_retains_zero_slot(
    make_proxy, monkeypatch, is_eval
):
    """A parsed result that cannot produce trajectory tensors becomes a fallback."""
    workflow_context.set(WorkflowContext(is_eval=is_eval, task_id=17))
    proxy, _, _ = make_proxy([_interactions(1.0)])
    monkeypatch.setattr(
        InteractionWithTokenLogpReward,
        "to_tensor_dict",
        Mock(side_effect=ValueError("inconsistent token fields")),
    )

    result = await MLPerfSWEWorkflow(proxy, fallback_token_id=7).arun_episode(
        None, {VALIDATION_TASK_INDEX: 0}
    )

    _assert_failed_attempt(result, is_eval=is_eval)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eval", [False, True])
async def test_attempt_metrics_include_failures_and_log_only_safe_reason(
    make_proxy, monkeypatch, attempt_metrics, is_eval
):
    """Failed placeholders remain observable without putting exception text in new logs."""
    workflow_context.set(WorkflowContext(is_eval=is_eval, task_id=17))
    secret = "sensitive-session-token"
    proxy, _, _ = make_proxy(
        [_interactions(1.0)],
        AsyncMock(side_effect=[NeMoGymSWEError(secret), 1.0]),
    )
    logger = Mock()
    monkeypatch.setattr(mlperf_module, "logger", logger)
    workflow = MLPerfSWEWorkflow(proxy, fallback_token_id=7)

    await workflow.arun_episode(None, {VALIDATION_TASK_INDEX: 0})
    await workflow.arun_episode(None, {VALIDATION_TASK_INDEX: 0})

    assert [call.kwargs for call in attempt_metrics.scalar.call_args_list] == [
        {"mlperf_attempt_failed": 1.0, "mlperf_attempt_reward": 0.0},
        {"mlperf_attempt_failed": 0.0, "mlperf_attempt_reward": 1.0},
    ]
    counters = [call.kwargs for call in attempt_metrics.denominator.call_args_list]
    assert sum(item["mlperf_attempts"].sum() for item in counters) == 2
    assert sum(item["mlperf_failed_attempts"].sum() for item in counters) == 1
    for call in mlperf_module.stats_tracker.get.call_args_list:
        assert call.args == (workflow_context.stat_scope(),)
    logger.warning.assert_called_once()
    args = logger.warning.call_args.args
    message = args[0] % args[1:]
    assert "NeMoGymSWEError" in message
    assert secret not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize("reward", [0.0, 1.0])
async def test_missing_reference_uses_terminal_reward(make_proxy, reward, is_eval):
    """Earlier discounted turn scores cannot replace the terminal group reference."""
    workflow_context.set(WorkflowContext(is_eval=is_eval, task_id=17))
    proxy, _, _ = make_proxy([_interactions(0.25, reward)])
    result = await MLPerfSWEWorkflow(proxy, fallback_token_id=7).arun_episode(
        None, {VALIDATION_TASK_INDEX: 0}
    )
    if is_eval:
        assert result["rewards"].tolist() == [reward]
        assert result["validation_failed"].tolist() == [False]
    else:
        assert result["rewards"].tolist() == [0.25, reward]
        assert result["rollout_rewards"] == [reward, reward]
        assert result["begin_of_trajectory"].tolist() == [1, 0]
        assert result["loss_mask"].sum(-1).tolist() == [1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize(
    "failure", [ValueError("programming bug"), asyncio.CancelledError()]
)
async def test_unexpected_error_or_cancel_propagates_and_finalizes(
    make_proxy, failure, is_eval
):
    """Cancellation and unexpected defects must not silently become failed slots."""
    workflow_context.set(WorkflowContext(is_eval=is_eval, task_id=17))
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def run_agent(api_key, data):
        if workflow_context.get().sample_idx == 0:
            await sibling_started.wait()
            raise failure
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    proxy, sessions, finalizer = make_proxy([{}, {}], run_agent)
    grouped = GroupedRolloutWorkflow(
        MLPerfSWEWorkflow(proxy, fallback_token_id=7), 2, Mock()
    )
    with pytest.raises(type(failure)) as info:
        await asyncio.wait_for(
            grouped.arun_episode(None, {VALIDATION_TASK_INDEX: 0}), timeout=2
        )
    assert info.value is failure
    assert sibling_cancelled.is_set()
    assert all(session.closed for session in sessions)
    finalizer.assert_awaited_once()


@pytest.mark.asyncio
async def test_training_all_failed_group_keeps_valid_placeholder_rows(make_proxy):
    """A fully failed rollout group remains representable before the optimizer."""
    workflow_context.set(WorkflowContext(is_eval=False, task_id=17))
    proxy, sessions, _ = make_proxy(
        [{}, {}, {}, {}], AsyncMock(side_effect=NeMoGymSWEError("runner unavailable"))
    )
    grouped = GroupedRolloutWorkflow(
        MLPerfSWEWorkflow(proxy, fallback_token_id=7), 4, Mock()
    )

    result = await grouped.arun_episode(None, {})

    assert check_trajectory_format(result, batch_size=4)
    assert result["input_ids"].shape == (4, 2)
    assert result["rewards"].tolist() == [0.0] * 4
    assert result["rollout_rewards"] == [0.0] * 4
    assert result["begin_of_trajectory"].tolist() == [1] * 4
    assert result["loss_mask"].count_nonzero() == 0
    assert result["attention_mask"].all()
    assert len(sessions) == 4 and all(session.closed for session in sessions)


@pytest.mark.asyncio
async def test_validation_invalid_task_or_sample_identity_fails_before_runner(
    make_proxy,
):
    """Validation results must retain unambiguous benchmark row and attempt identity."""
    proxy, _, _ = make_proxy([_interactions(1.0)])
    workflow = MLPerfSWEWorkflow(proxy, fallback_token_id=7)
    for task_index in (None, -1, True, "0"):
        with pytest.raises(ValueError, match="nonnegative task index"):
            await workflow.arun_episode(None, {VALIDATION_TASK_INDEX: task_index})
    for sample_index in (-1, True, "0", 4):
        workflow_context.set(
            WorkflowContext(is_eval=True, sample_idx=sample_index, group_size=4)
        )
        with pytest.raises(ValueError, match="invalid sample index"):
            await workflow.arun_episode(None, {VALIDATION_TASK_INDEX: 0})
    proxy._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_proxy_empty_export_exposes_typed_runtime_error(make_proxy):
    """The shared wrapper recognizes empty exports without catching all RuntimeError."""
    proxy, _, _ = make_proxy([{}])
    with pytest.raises(
        EmptyAgentSessionError, match="no complete interactions"
    ) as info:
        await proxy.arun_episode(None, {})
    assert isinstance(info.value, RuntimeError)


@pytest.mark.parametrize("token_id", [None, -1, True, 1.5])
def test_workflow_invalid_fallback_token_rejected(make_proxy, token_id):
    """Placeholders must use a configured nonnegative integer token ID."""
    proxy, _, _ = make_proxy([{}])
    with pytest.raises(ValueError, match="fallback_token_id"):
        MLPerfSWEWorkflow(proxy, fallback_token_id=token_id)


def test_workflow_non_proxy_child_rejected():
    """The shared adapter requires the proxy session and export contract."""
    with pytest.raises(TypeError, match="OpenAIProxyWorkflow"):
        MLPerfSWEWorkflow(object(), fallback_token_id=7)


def test_workflow_factory_resolves_proxy_child_and_is_detected():
    """The normal inference resolver supplies proxy settings and detects the agent."""
    engine = object.__new__(RemoteInfEngine)
    engine.logger = Mock()
    engine.config = SimpleNamespace(
        agent=AgentConfig(mode="inline", agent_cls_path=AGENT_PATH)
    )
    engine._proxy_gateway_addr = None
    kwargs = {
        "agent_kwargs": {"runner_url": "http://runner", "model": "test-model"},
        "fallback_token_id": 7,
    }
    path = "examples.swe.mlperf.workflow.MLPerfSWEWorkflow"

    workflow = engine._resolve_workflow(
        path, kwargs, group_size=4, proxy_addr="http://proxy"
    )

    assert isinstance(workflow, GroupedRolloutWorkflow)
    assert isinstance(workflow.workflow, MLPerfSWEWorkflow)
    child = workflow.workflow.agent_workflow
    assert isinstance(child, OpenAIProxyWorkflow)
    assert isinstance(child.agent, NeMoGymSWEAgent)
    assert child.proxy_addr == "http://proxy"
    assert list(MLPerfSWEWorkflow._iter_workflow_specs(kwargs)) == [
        (AGENT_PATH, kwargs["agent_kwargs"])
    ]
    trainer = object.__new__(PPOTrainer)
    assert trainer._requires_proxy_workflow(path, kwargs)
