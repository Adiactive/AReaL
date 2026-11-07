# SPDX-License-Identifier: Apache-2.0

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER_DIR = Path(__file__).resolve().parents[1] / "examples/swe/mlperf/runner_cluster"


@pytest.fixture
def generate_config(tmp_path, monkeypatch):
    """Run the real wrapper and generator, replacing only Docker invocation."""
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.delenv("SWE_EPISODE_TIMEOUT", raising=False)
    output = tmp_path / "site.yaml"

    def generate(extra_env="", extra_args=()):
        env_file = tmp_path / "configure.env"
        env_file.write_text(
            f"SWE_DEPLOY_DIR={shlex.quote(str(tmp_path / 'deployment'))}\n"
            "SWE_SERVICE=areal\n"
            "SWE_CONFIG_FILE=site.yaml\n"
            "SWE_HEAD_IP=192.0.2.10\n"
            "SWE_HEAD_ADDRESS=192.0.2.10:16379\n"
            "SWE_SIF_DIR=/srv/sif\n"
            "SWE_STATE_DIR=/srv/state\n"
            "SWE_RESULTS_DIR=/srv/results\n" + extra_env
        )
        wrapper = subprocess.run(
            [
                "bash",
                str(RUNNER_DIR / "configure.sh"),
                "--env-file",
                str(env_file),
                *extra_args,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        docker_args = json.loads(wrapper.stdout)
        args = docker_args[docker_args.index("/opt/swe-runner/configure.py") + 1 :]
        args[args.index("--output") + 1] = str(output)
        result = subprocess.run(
            [sys.executable, str(RUNNER_DIR / "configure.py"), *args],
            capture_output=True,
            text=True,
        )
        return result, output

    return generate


@pytest.mark.parametrize(
    "extra_env,extra_args,expected",
    [
        ("", (), 3600),
        ("SWE_EPISODE_TIMEOUT=10800\n", (), 10800),
        ("SWE_EPISODE_TIMEOUT=10800\n", ("--episode-timeout", "14400"), 14400),
        ("SWE_EPISODE_TIMEOUT=0.5\n", (), 0.5),
    ],
)
def test_configure_wrapper_timeout_written_to_site_yaml(
    generate_config, extra_env, extra_args, expected
):
    """Defaults, env values and CLI overrides reach the generated deployment."""
    result, output = generate_config(extra_env, extra_args)
    assert result.returncode == 0, result.stderr
    config = json.loads(output.read_text())
    assert config["swe_runner"]["episode_timeout"] == expected
    assert config["swe_runner"]["sif_dir"] == "/srv/sif"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "true", "invalid"])
def test_configure_invalid_timeout_rejected_without_output(generate_config, value):
    """The generator rejects unusable deadlines before writing a config."""
    result, output = generate_config(extra_args=(f"--episode-timeout={value}",))
    assert result.returncode != 0
    assert "--episode-timeout" in result.stderr
    assert not output.exists()


def test_configure_standalone_service_omits_bridge_timeout(generate_config):
    """The AReaL-only environment setting does not affect fixed-model service."""
    result, output = generate_config(
        "SWE_SERVICE=standalone\n"
        "SWE_MODEL_BASE_URL=http://192.0.2.20:8000/v1\n"
        "SWE_MODEL_NAME=test-model\n"
        "SWE_EPISODE_TIMEOUT=invalid\n"
    )
    assert result.returncode == 0, result.stderr
    assert "episode_timeout" not in json.loads(output.read_text())["swe_runner"]


@pytest.mark.parametrize("extra_args", [(), ("--episode-timeout", "10800")])
def test_configure_service_only_preserves_legacy_metadata(tmp_path, extra_args):
    """Adding launch settings requires paths; legacy service configs stay intact."""
    output = tmp_path / "site.yaml"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_DIR / "configure.py"),
            "--service",
            "areal",
            "--output",
            str(output),
            "--advertised-host",
            "192.0.2.10",
            "--ray-head-address",
            "192.0.2.10:16379",
            *extra_args,
        ],
        capture_output=True,
        text=True,
    )
    if extra_args:
        assert result.returncode != 0
        assert "launch settings require" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "swe_runner" not in json.loads(output.read_text())
