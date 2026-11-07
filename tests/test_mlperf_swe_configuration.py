# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples.swe.mlperf import serve
from examples.swe.mlperf.config import AGENT_PATH, validate_proxy_config
from examples.swe.mlperf.configure_runner import configure
from examples.swe.mlperf.serve import service_settings
from examples.swe.mlperf.train import load_tasks


def _runner_config():
    return {
        "head_server": {"host": "127.0.0.1", "port": 8081},
        "policy_base_url": "http://old-model:9000/v1",
        "policy_api_key": "old-secret",
        "policy_model_name": "old-model",
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "host": "127.0.0.1",
                    "port": 8082,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            }
        },
        "swe_agents": {
            "responses_api_agents": {
                "swe_agents": {"host": "0.0.0.0", "port": 8080, "concurrency": 4}
            }
        },
    }


def test_runner_configuration_replaces_fixed_credentials_and_keeps_sampling():
    """Only the bridge is published; model requests cannot use the old credential."""
    original = _runner_config()
    snapshot = deepcopy(original)
    result = configure(original, 8080, 8083)
    assert original == snapshot
    assert "old-secret" not in json.dumps(result)
    model = result["policy_model"]["responses_api_models"]["vllm_model"]
    assert model["entrypoint"] == "areal_app.py"
    assert model["return_token_id_information"] is False
    assert model["uses_reasoning_parser"] is False
    assert model["chat_template_kwargs"] == {"enable_thinking": False}
    assert service_settings(result) == (
        "http://127.0.0.1:8083",
        8080,
        4,
        ["http://127.0.0.1:8083", "http://127.0.0.1:8082", "http://127.0.0.1:8081"],
    )


@pytest.mark.parametrize(
    "bridge,gym", [(8080, 8080), (8081, 8083), (0, 8083), (8080, 65536)]
)
def test_runner_configuration_rejects_conflicting_or_invalid_ports(bridge, gym):
    """Bad local topology fails before launching any services."""
    with pytest.raises(ValueError):
        configure(_runner_config(), bridge, gym)


def test_runner_configuration_distributed_callbacks_keep_internal_routes_local():
    """Workers use the head IP while Gym and episode forwarding remain local."""
    original = _runner_config()
    original["ray_head_node_address"] = "192.0.2.10:6379"
    snapshot = deepcopy(original)
    result = configure(original, 8080, 8083, advertised_host="192.0.2.10")
    assert original == snapshot
    assert result["ray_head_node_address"] == "192.0.2.10:6379"
    assert result["policy_base_url"] == "http://127.0.0.1:8080/v1"
    assert service_settings(result) == (
        "http://127.0.0.1:8083",
        8080,
        4,
        ["http://127.0.0.1:8083", "http://192.0.2.10:8082", "http://192.0.2.10:8081"],
    )


@pytest.mark.parametrize(
    "runner_settings,cli_timeout,expected",
    [
        (None, None, 3600),
        ({}, None, 3600),
        ({"episode_timeout": 10800}, None, 10800),
        ({"episode_timeout": 0.5}, None, 0.5),
        (None, "7200", 7200),
        ({"episode_timeout": 10800}, "14400", 14400),
    ],
)
def test_serve_episode_timeout_precedence_reaches_bridge(
    tmp_path, monkeypatch, runner_settings, cli_timeout, expected
):
    """The actual bridge command receives the CLI, config, or legacy timeout."""
    config = configure(_runner_config(), 8080, 8083)
    if runner_settings is not None:
        config["swe_runner"] = runner_settings
    path = tmp_path / "site.yaml"
    path.write_text(json.dumps(config))
    argv = ["serve.py", "--config", str(path)]
    if cli_timeout is not None:
        argv += ["--episode-timeout", cli_timeout]
    monkeypatch.setattr(sys, "argv", argv)
    gym, bridge = MagicMock(), MagicMock()
    gym.poll.side_effect = [None, 1]
    launch = MagicMock(side_effect=[gym, bridge])
    monkeypatch.setattr(serve.subprocess, "Popen", launch)
    monkeypatch.setattr(serve.signal, "signal", MagicMock())
    monkeypatch.setattr(serve.os, "killpg", MagicMock())
    client = MagicMock()
    client.__enter__.return_value.get.return_value.is_success = True
    monkeypatch.setattr(serve.httpx, "Client", MagicMock(return_value=client))

    with pytest.raises(RuntimeError, match="A runner service exited"):
        serve.main()

    assert launch.call_count == 2
    command = launch.call_args_list[1].args[0]
    assert Path(command[1]).name == "runner_bridge.py"
    assert float(command[command.index("--episode-timeout") + 1]) == expected


@pytest.mark.parametrize("source", ["config", "cli"])
@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_serve_invalid_episode_timeout_rejected_before_launch(
    tmp_path, monkeypatch, source, timeout
):
    """Invalid limits fail before Gym or the bridge can start."""
    config = configure(_runner_config(), 8080, 8083)
    config["swe_runner"] = {"episode_timeout": timeout if source == "config" else 10800}
    path = tmp_path / "site.yaml"
    path.write_text(json.dumps(config))
    argv = ["serve.py", "--config", str(path)]
    if source == "cli":
        argv += [f"--episode-timeout={timeout}"]
    monkeypatch.setattr(sys, "argv", argv)
    launch = MagicMock()
    monkeypatch.setattr(serve.subprocess, "Popen", launch)

    with pytest.raises(ValueError, match="Episode timeout must be a positive finite"):
        serve.main()

    launch.assert_not_called()


@pytest.mark.parametrize("timeout", [True, False, None, "10800", [], {}])
def test_serve_nonnumeric_config_timeout_rejected_before_launch(
    tmp_path, monkeypatch, timeout
):
    """Configuration values must be numbers, not coercible strings or booleans."""
    config = configure(_runner_config(), 8080, 8083)
    config["swe_runner"] = {"episode_timeout": timeout}
    path = tmp_path / "site.yaml"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["serve.py", "--config", str(path)])
    launch = MagicMock()
    monkeypatch.setattr(serve.subprocess, "Popen", launch)

    with pytest.raises(ValueError, match="Episode timeout must be a positive finite"):
        serve.main()

    launch.assert_not_called()


def test_runner_configuration_explicit_ray_cluster_overrides_input():
    """An explicit GCS address is propagated without silently starting local Ray."""
    original = _runner_config()
    original["ray_head_node_address"] = "127.0.0.1:6379"
    result = configure(
        original,
        8080,
        8083,
        advertised_host="192.0.2.10",
        ray_head_address="192.0.2.10:6380",
    )
    assert result["ray_head_node_address"] == "192.0.2.10:6380"
    assert original["ray_head_node_address"] == "127.0.0.1:6379"


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "255.255.255.255",
        "224.0.0.1",
        "::1",
        "localhost",
        "http://192.0.2.10",
        "user@192.0.2.10",
        "192.0.2.10:8081",
        "192.0.2.999",
        "",
        None,
    ],
)
def test_runner_configuration_rejects_unusable_callback_hosts(host):
    """A wildcard or URL cannot be advertised to remote sandbox clients."""
    with pytest.raises(ValueError, match="IPv4"):
        configure(_runner_config(), 8080, 8083, advertised_host=host)


@pytest.mark.parametrize(
    "address",
    [
        "auto",
        "ray://192.0.2.10:10001",
        "user@192.0.2.10:6379",
        "0.0.0.0:6379",
        "192.0.2.10:0",
        "192.0.2.10:65536",
        "192.0.2.10:bad",
        "192.0.2.10",
    ],
)
def test_runner_configuration_rejects_invalid_ray_gcs_addresses(address):
    """The selected cluster must have an explicit usable GCS address."""
    with pytest.raises(ValueError):
        configure(_runner_config(), 8080, 8083, ray_head_address=address)


@pytest.mark.parametrize("server", ["head", "model"])
@pytest.mark.parametrize("port", [0, 65536, True, "8082", 8080])
def test_runner_configuration_rejects_invalid_callback_ports(server, port):
    """All published and internal ports are validated before process launch."""
    config = _runner_config()
    target = (
        config["head_server"]
        if server == "head"
        else config["policy_model"]["responses_api_models"]["vllm_model"]
    )
    target["port"] = port
    with pytest.raises(ValueError):
        configure(config, 8080, 8083)


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_gym_public",
        "mismatched_callback",
        "wildcard_callbacks",
        "bridge_credentials",
        "bridge_query",
        "bridge_fragment",
        "bridge_remote",
        "port_collision",
        "zero_concurrency",
        "invalid_ray",
    ],
)
def test_service_settings_rejects_manually_broken_topology(mutation):
    """Editing generated JSON must not bypass callback and internal-route checks."""
    config = configure(_runner_config(), 8080, 8083, advertised_host="192.0.2.10")
    swe = config["swe_agents"]["responses_api_agents"]["swe_agents"]
    model = config["policy_model"]["responses_api_models"]["vllm_model"]
    if mutation == "raw_gym_public":
        swe["host"] = "0.0.0.0"
    elif mutation == "mismatched_callback":
        model["host"] = "127.0.0.1"
    elif mutation == "wildcard_callbacks":
        model["host"] = config["head_server"]["host"] = "0.0.0.0"
    elif mutation == "bridge_credentials":
        config["policy_base_url"] = "http://user:secret@127.0.0.1:8080/v1"
    elif mutation == "bridge_query":
        config["policy_base_url"] += "?x=1"
    elif mutation == "bridge_fragment":
        config["policy_base_url"] += "#fragment"
    elif mutation == "bridge_remote":
        config["policy_base_url"] = "http://192.0.2.10:8080/v1"
    elif mutation == "port_collision":
        model["port"] = 8080
    elif mutation == "zero_concurrency":
        swe["concurrency"] = 0
    elif mutation == "invalid_ray":
        config["ray_head_node_address"] = "0.0.0.0:6379"
    with pytest.raises(ValueError):
        service_settings(config)


def test_runner_config_cli_creates_routable_private_config_without_overwriting(
    tmp_path,
):
    """The distributed CLI writes a private config and preserves an existing file."""
    source = tmp_path / "fixed.json"
    target = tmp_path / "areal.json"
    source.write_text(json.dumps(_runner_config()))
    script = (
        Path(__file__).resolve().parents[1] / "examples/swe/mlperf/configure_runner.py"
    )
    command = [
        sys.executable,
        str(script),
        "--input",
        str(source),
        "--output",
        str(target),
        "--advertised-host",
        "192.0.2.10",
        "--ray-head-address",
        "192.0.2.10:6379",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    encoded = target.read_bytes()
    config = json.loads(encoded)
    assert config["ray_head_node_address"] == "192.0.2.10:6379"
    assert "http://192.0.2.10:8082" in service_settings(config)[3]
    assert target.stat().st_mode & 0o777 == 0o600
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0
    assert target.read_bytes() == encoded


def test_native_jsonl_preserves_heterogeneous_metadata(tmp_path):
    """Reading records must not coerce fields or drop hidden grader metadata."""
    rows = [
        {
            "responses_create_params": {
                "input": "task",
                "metadata": {"instance_dict": '{"test": [1, 2]}'},
            }
        },
        {
            "responses_create_params": {
                "input": [],
                "temperature": 0,
                "extra": {"seed": 7},
            }
        },
    ]
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n\n".join(map(json.dumps, rows)))
    assert load_tasks(str(path)) == rows


@pytest.mark.parametrize("text", ["\n", "[]\n", "{}\n", "not json\n"])
def test_native_jsonl_rejects_invalid_records(tmp_path, text):
    """Input errors fail locally before allocating training workers."""
    path = tmp_path / "tasks.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_tasks(str(path))


def _training_config():
    return SimpleNamespace(
        actor=SimpleNamespace(_version="v1", adv_norm=None),
        rollout=SimpleNamespace(
            _version="v1",
            request_timeout=7500,
            agent=SimpleNamespace(
                agent_cls_path=AGENT_PATH,
                mode="inline",
                export_style="individual",
                chat_template_type="hf",
                session_timeout_seconds=10800,
            ),
        ),
        nemo_gym=SimpleNamespace(
            request_timeout_seconds=7200, cancellation_timeout_seconds=10
        ),
        train_dataset=SimpleNamespace(shuffle=False),
    )


def test_proxy_config_allows_episode_reward_groups_but_rejects_advantage_groups():
    """Variable turn counts require metadata-aware normalization."""
    config = _training_config()
    config.actor.reward_norm = SimpleNamespace(mean_level="group", std_level="group")
    validate_proxy_config(config)
    config.actor.adv_norm = SimpleNamespace(mean_level="group", std_level="batch")
    with pytest.raises(ValueError, match="advantage"):
        validate_proxy_config(config)


def test_proxy_config_rejects_session_timeout_shorter_than_queue_wait():
    """A queued episode must not outlive its inactive proxy session."""
    config = _training_config()
    config.rollout.agent.session_timeout_seconds = 3600
    with pytest.raises(ValueError, match="timeout"):
        validate_proxy_config(config)


@pytest.mark.parametrize("outer_timeout", [3600, 7200, 7210])
def test_proxy_config_rejects_rpc_timeout_before_episode_cleanup(outer_timeout):
    """The outer RPC must allow the complete attempt and credential revocation."""
    config = _training_config()
    config.rollout.request_timeout = outer_timeout
    with pytest.raises(ValueError, match="rollout.request_timeout"):
        validate_proxy_config(config)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_proxy_config_rejects_invalid_timeout(timeout):
    """Invalid timeout values cannot bypass the nesting checks."""
    config = _training_config()
    config.nemo_gym.cancellation_timeout_seconds = timeout
    with pytest.raises(ValueError, match="positive finite timeout"):
        validate_proxy_config(config)


def test_proxy_config_rejects_shuffled_benchmark_tasks():
    """Training must preserve the published MLPerf task order."""
    config = _training_config()
    config.train_dataset.shuffle = True
    with pytest.raises(ValueError, match="shuffle"):
        validate_proxy_config(config)
