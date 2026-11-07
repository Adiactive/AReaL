# SPDX-License-Identifier: Apache-2.0

import builtins
import json
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import OmegaConf

from examples.swe.mlperf import train
from examples.swe.mlperf.config import AGENT_PATH, NeMoGymConfig
from examples.swe.mlperf.preflight import (
    RunnerHealth,
    capacity_report,
    check_runner_health,
    prepare_workflow,
)


def _task():
    return {
        "agent_ref": {"name": "reference_agent"},
        "responses_create_params": {
            "input": "Fix the repository",
            "metadata": {
                "instance_id": "example-task",
                "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                "instance_dict": '{"tests": ["test_patch"], "hidden": true}',
            },
        },
    }


def _config():
    generation = SimpleNamespace(
        greedy=False, temperature=0.8, top_p=0.95, max_new_tokens=4096, n_samples=2
    )
    return SimpleNamespace(
        nemo_gym=NeMoGymConfig(runner_url="http://runner:8080", model="policy"),
        rollout=SimpleNamespace(
            max_concurrent_rollouts=3,
            consumer_batch_size=2,
            max_head_offpolicyness=0,
        ),
        gconfig=generation,
        eval_gconfig=SimpleNamespace(**vars(generation) | {"n_samples": 1}),
        valid_dataset=None,
    )


def _healthy():
    return {
        "status": "ok",
        "active_attempts": 1,
        "queued_attempts": 2,
        "concurrency": 4,
    }


def test_runner_health_reads_only_admission_without_credentials(monkeypatch):
    """Readiness is one unauthenticated GET and ignores host proxy settings."""
    requests = []
    monkeypatch.setenv("HTTP_PROXY", "http://unusable-proxy.invalid:8080")

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=_healthy())

    health = check_runner_health(
        "http://runner:8080/", transport=httpx.MockTransport(respond)
    )
    assert health == RunnerHealth(active_attempts=1, queued_attempts=2, concurrency=4)
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == "http://runner:8080/health"
    assert "authorization" not in requests[0].headers


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "unavailable"),
        ("active_attempts", -1),
        ("active_attempts", True),
        ("active_attempts", 5),
        ("queued_attempts", -1),
        ("queued_attempts", "0"),
        ("queued_attempts", None),
        ("concurrency", 0),
        ("concurrency", 1.5),
        ("concurrency", True),
    ],
)
def test_runner_health_rejects_unhealthy_or_invalid_admission(field, value):
    """An OK HTTP response alone is insufficient to admit training."""
    body = _healthy() | {field: value}
    with pytest.raises(ValueError, match="Runner health"):
        check_runner_health(
            "http://runner:8080",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body)
            ),
        )


@pytest.mark.parametrize("status", [301, 404, 503])
def test_runner_health_rejects_http_failure_without_retry_or_redirect(status):
    """A failed health response is returned without redirecting or replaying it."""
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            status, json=_healthy(), headers={"location": "http://other/health"}
        )

    with pytest.raises(ValueError, match=f"HTTP {status}"):
        check_runner_health(
            "http://runner:8080", transport=httpx.MockTransport(respond)
        )
    assert len(requests) == 1


@pytest.mark.parametrize("body", [b"not JSON", b"[]", b"{}"])
def test_runner_health_rejects_noncontract_response(body):
    """A generic server health endpoint is not a valid AReaL bridge response."""
    with pytest.raises(ValueError, match="Runner health"):
        check_runner_health(
            "http://runner:8080",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=body)
            ),
        )


def test_prepare_workflow_validates_native_task_without_rewriting_metadata():
    """Preflight applies agent validation while preserving the published task."""
    task = _task()
    original = deepcopy(task)
    kwargs = prepare_workflow(_config(), _config().gconfig, [task], split="training")
    assert kwargs["max_output_tokens"] == 4096
    assert kwargs["temperature"] == 0.8
    assert task == original


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_metadata", "metadata"),
        ("missing_instance_id", "instance_id"),
        ("invalid_instance_json", "instance_dict"),
        ("invalid_generation", "top_p"),
        ("streaming", "non-streaming"),
        ("null_stream", "non-streaming"),
        ("multiple_completions", "one completion"),
        ("boolean_completions", "one completion"),
    ],
)
def test_prepare_workflow_rejects_bad_task_before_submission(mutation, match):
    """Malformed metadata and per-task sampling overrides fail locally."""
    task = _task()
    params = task["responses_create_params"]
    if mutation == "missing_metadata":
        del params["metadata"]
    elif mutation == "missing_instance_id":
        del params["metadata"]["instance_id"]
    elif mutation == "invalid_instance_json":
        params["metadata"]["instance_dict"] = "[]"
    elif mutation == "invalid_generation":
        params["top_p"] = 0
    elif mutation == "streaming":
        params["stream"] = True
    elif mutation == "null_stream":
        params["stream"] = None
    elif mutation == "multiple_completions":
        params["n"] = 2
    elif mutation == "boolean_completions":
        params["n"] = True
    with pytest.raises(ValueError, match=f"Invalid training task 1: .*{match}"):
        prepare_workflow(_config(), _config().gconfig, [task], split="training")


def test_capacity_report_distinguishes_pressure_from_global_quota():
    """Per-episode sample fan-out can queue without becoming a global hard cap."""
    config = _config()
    messages, warnings = capacity_report(config, RunnerHealth(0, 0, 4))
    assert any("3 groups x 2 samples = 6 attempts" in message for message in messages)
    assert any("not a global sandbox quota" in message for message in messages)
    assert any("queue time" in warning for warning in warnings)


def test_capacity_report_warns_about_shared_runner_eval_overlap():
    """Prefetched training attempts may block evaluation on the same runner."""
    config = _config()
    config.valid_dataset = object()
    config.rollout.max_head_offpolicyness = 1
    messages, warnings = capacity_report(config, RunnerHealth(0, 0, 12))
    assert any(
        "Validation configured demand: 3 groups x 1 samples" in x for x in messages
    )
    assert any("9 configured attempts" in warning for warning in warnings)
    assert any("spare runner capacity" in warning for warning in warnings)
    assert any("budget timeouts" in warning for warning in warnings)


def test_capacity_report_uses_consumer_batch_when_group_limit_unset():
    """The preflight mirrors the workflow executor's default group count."""
    config = _config()
    config.rollout.max_concurrent_rollouts = None
    messages, warnings = capacity_report(config, RunnerHealth(0, 0, 4))
    assert any("2 groups x 2 samples = 4 attempts" in message for message in messages)
    assert not warnings


def _write_config(tmp_path, task):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(task) + "\n")
    config = {
        "experiment_name": "mlperf-preflight-test",
        "trial_name": "trial0",
        "actor": {
            "backend": "fsdp:d1p1t1",
            "_version": "v1",
            "experiment_name": "${experiment_name}",
            "trial_name": "${trial_name}",
        },
        "rollout": {
            "backend": "sglang:d1p1t1",
            "_version": "v1",
            "request_timeout": 7500,
            "max_concurrent_rollouts": 1,
            "agent": {
                "agent_cls_path": AGENT_PATH,
                "admin_api_key": "test-key",
                "session_timeout_seconds": 10800,
            },
        },
        "train_dataset": {"path": str(tasks), "shuffle": False},
        "valid_dataset": None,
        "gconfig": {"n_samples": 2},
        "nemo_gym": {"runner_url": "http://runner:8080", "model": "policy"},
    }
    for section in ("saver", "recover", "evaluator", "stats_logger", "perf_tracer"):
        config[section] = {
            "experiment_name": "${experiment_name}",
            "trial_name": "${trial_name}",
            "fileroot": str(tmp_path / "logs"),
        }
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    return path


@pytest.mark.parametrize("outcome", ["healthy", "refused", "invalid_task"])
def test_check_only_never_imports_trainer_or_submits_attempt(
    tmp_path, monkeypatch, outcome
):
    """The actual entrypoint validates typed config and exits before trainer import."""
    task = _task()
    if outcome == "invalid_task":
        del task["responses_create_params"]["metadata"]
    path = _write_config(tmp_path, task)
    requests = []

    def respond(request):
        requests.append(request)
        if outcome == "refused":
            raise httpx.ConnectError("Connection refused", request=request)
        return httpx.Response(200, json=_healthy())

    monkeypatch.setattr(
        train,
        "check_runner_health",
        partial(check_runner_health, transport=httpx.MockTransport(respond)),
    )
    monkeypatch.delenv("AREAL_SPMD_MODE", raising=False)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "areal" and "PPOTrainer" in fromlist:
            raise AssertionError("Preflight imported PPOTrainer")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = ["--config", str(path), "--check-only"]
    if outcome == "healthy":
        train.main(args)
    else:
        match = "Runner health request failed" if outcome == "refused" else "metadata"
        with pytest.raises(ValueError, match=match):
            train.main(args)
    assert len(requests) == (0 if outcome == "invalid_task" else 1)
    assert all(
        request.method == "GET" and request.url.path == "/health"
        for request in requests
    )
    assert set(tmp_path.iterdir()) == {path, tmp_path / "tasks.jsonl"}


def _enable_validation(path):
    config = OmegaConf.load(path)
    config.valid_dataset = {"path": config.train_dataset.path}
    config.rollout.max_head_offpolicyness = 0
    config.eval_gconfig = {"n_samples": 4, "temperature": 0.1, "top_p": 0.95}
    OmegaConf.save(config, path)


@pytest.mark.parametrize("max_head_offpolicyness", [0, 1, 2])
@pytest.mark.parametrize("valid_sampling", [True, False])
def test_validation_preflight_checks_protocol_without_importing_trainer(
    tmp_path, monkeypatch, valid_sampling, max_head_offpolicyness
):
    path = _write_config(tmp_path, _task())
    _enable_validation(path)
    config = OmegaConf.load(path)
    config.rollout.max_head_offpolicyness = max_head_offpolicyness
    if not valid_sampling:
        config.eval_gconfig.n_samples = 1
    OmegaConf.save(config, path)
    health_calls = []
    monkeypatch.setattr(
        train,
        "check_runner_health",
        lambda url: health_calls.append(url) or RunnerHealth(0, 0, 4),
    )
    monkeypatch.delenv("AREAL_SPMD_MODE", raising=False)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "examples.swe.mlperf.trainer" or (
            name == "areal" and "PPOTrainer" in fromlist
        ):
            raise AssertionError("Preflight imported a trainer")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    if valid_sampling:
        train.main(["--config", str(path), "--check-only"])
        assert health_calls == ["http://runner:8080"]
    else:
        with pytest.raises(ValueError, match="n_samples: 4"):
            train.main(["--config", str(path), "--check-only"])
        assert not health_calls


@pytest.mark.parametrize("validation_enabled", [True, False])
@pytest.mark.parametrize(
    "pad_token_id,eos_token_id,fallback_token_id",
    [(42, 17, 42), (0, 17, 0), (None, 17, 17), (None, None, 0)],
)
def test_entrypoint_selects_shared_workflow_and_tokenizer_fallback(
    tmp_path,
    monkeypatch,
    validation_enabled,
    pad_token_id,
    eos_token_id,
    fallback_token_id,
):
    """Both splits use the shared wrapper with an existing tokenizer's safe ID."""
    path = _write_config(tmp_path, _task())
    if validation_enabled:
        _enable_validation(path)
    calls = {}

    class FakeTrainer:
        def __init__(self, config, **kwargs):
            calls["datasets"] = kwargs
            self.tokenizer = SimpleNamespace(
                pad_token_id=pad_token_id, eos_token_id=eos_token_id
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def train(self, **kwargs):
            calls["train"] = kwargs

    real_import = builtins.__import__

    def trainer_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "areal" and "PPOTrainer" in fromlist:
            calls["trainer"] = "standard"
            return SimpleNamespace(PPOTrainer=FakeTrainer)
        if name == "examples.swe.mlperf.trainer":
            calls["trainer"] = "mlperf"
            return SimpleNamespace(MLPerfSWETrainer=FakeTrainer)
        return real_import(name, globals, locals, fromlist, level)

    def load_config(args, config_cls):
        raw, _ = train.parse_cli_args(args)
        return OmegaConf.to_object(train.to_structured_cfg(raw, config_cls)), None

    monkeypatch.setattr(builtins, "__import__", trainer_import)
    monkeypatch.setattr(train, "load_expr_config", load_config)
    monkeypatch.setattr(train, "check_runner_health", lambda url: RunnerHealth(0, 0, 4))
    monkeypatch.delenv("AREAL_SPMD_MODE", raising=False)
    train.main(["--config", str(path)])
    workflow_path = "examples.swe.mlperf.workflow.MLPerfSWEWorkflow"
    assert calls["train"]["workflow"] == workflow_path
    training_kwargs = calls["train"]["workflow_kwargs"]
    assert training_kwargs["fallback_token_id"] == fallback_token_id
    assert training_kwargs["agent_kwargs"]["runner_url"] == "http://runner:8080"
    if validation_enabled:
        assert calls["trainer"] == "mlperf"
        assert calls["datasets"]["valid_dataset"] == [_task()]
        assert calls["train"]["eval_workflow"] == workflow_path
        eval_kwargs = calls["train"]["eval_workflow_kwargs"]
        assert eval_kwargs["fallback_token_id"] == fallback_token_id
        kwargs = eval_kwargs["agent_kwargs"]
        assert kwargs["temperature"] == 0.1
        assert kwargs["top_p"] == 0.95
    else:
        assert calls["trainer"] == "standard"
        assert calls["datasets"]["valid_dataset"] is None
        assert calls["train"]["eval_workflow"] is None
        assert calls["train"]["eval_workflow_kwargs"] is None
