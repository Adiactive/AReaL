# SPDX-License-Identifier: Apache-2.0

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER_DIR = Path(__file__).resolve().parents[1] / "examples/swe/mlperf/runner_cluster"
PATCH_PATH = RUNNER_DIR / "patch-sandbox-isolation.py"
patch_source = runpy.run_path(str(PATCH_PATH))["patch_source"]

GYM_SOURCE = """class Gym:
    def _build_apptainer_command(self, params, command):
        mount_args = [
            f"--mount type=bind,src={params.setup},dst=/openhands_setup/OpenHands,ro",
            f"--mount type=bind,src={params.setup},dst={params.setup},ro",
            f"--mount type=bind,src={params.results},dst=/trajectories_mount",
        ]
        if command.mode == "eval":
            mount_args.append(f"--mount type=bind,src={params.evaluator},dst=/r2egym_setup")
        mount_str = " ".join(mount_args)
        env_args = ""
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --cleanenv --pid --no-mount home,tmp,bind-paths "
            f"{env_args}"
            f"{mount_str} "
            f" {params.container} bash /container_scripts/{command.mode}_script.sh"
        )
        return apptainer_cmd

OTHER_COMMAND = "apptainer exec --writable-tmpfs --cleanenv --pid --no-mount home,tmp,bind-paths "
"""


@pytest.mark.parametrize("mode", ["agent", "eval"])
def test_patch_command_both_modes_preserves_explicit_mounts(mode):
    """Generated commands change only the automatic CWD bind policy."""
    params = SimpleNamespace(
        setup="/runtime/OpenHands",
        results="/results/episode",
        evaluator="/runtime/r2e",
        container="/images/task.sif",
    )
    original_namespace = {}
    patched_namespace = {}
    exec(compile(GYM_SOURCE, "original.py", "exec"), original_namespace)
    exec(compile(patch_source(GYM_SOURCE), "patched.py", "exec"), patched_namespace)
    command = SimpleNamespace(mode=mode)
    original = original_namespace["Gym"]()._build_apptainer_command(params, command)
    patched = patched_namespace["Gym"]()._build_apptainer_command(params, command)

    assert patched == original.replace(
        "--no-mount home,tmp,bind-paths ", "--no-mount home,tmp,bind-paths,cwd "
    )
    assert patched_namespace["OTHER_COMMAND"] == original_namespace["OTHER_COMMAND"]


def test_patch_already_isolated_source_is_unchanged():
    """Reapplying the build patch does not duplicate flags."""
    patched = patch_source(GYM_SOURCE)
    assert patch_source(patched) == patched


@pytest.mark.parametrize(
    "source",
    [
        GYM_SOURCE.replace("_build_apptainer_command", "renamed_builder"),
        GYM_SOURCE + GYM_SOURCE.replace("class Gym:", "class AnotherGym:"),
        GYM_SOURCE.replace("--pid ", ""),
        GYM_SOURCE.replace("home,tmp,bind-paths ", "home,tmp,bind-paths,sys "),
        GYM_SOURCE.replace(
            "home,tmp,bind-paths ",
            "home,tmp,bind-paths --no-mount home,tmp,bind-paths,cwd ",
        ),
        GYM_SOURCE.replace(
            "home,tmp,bind-paths ",
            "home,tmp,bind-paths,cwd --no-mount home,tmp,bind-paths,cwd ",
        ),
    ],
)
def test_patch_upstream_drift_is_rejected(source):
    """Missing, changed, duplicate and mixed command forms fail closed."""
    with pytest.raises(ValueError):
        patch_source(source)


def test_patch_cli_drift_leaves_source_intact(tmp_path):
    """Build failure must not leave a partially patched upstream file."""
    source = tmp_path / "app.py"
    original = GYM_SOURCE.replace("--pid ", "")
    source.write_text(original)
    result = subprocess.run(
        [sys.executable, str(PATCH_PATH), str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refusing to patch" in result.stderr
    assert source.read_text() == original


def test_build_context_contains_patch_before_runtime_preparation(tmp_path, monkeypatch):
    """The portable build context wires the patch before building Gym's wheel."""
    monkeypatch.delenv("DOCKER_DEFAULT_PLATFORM", raising=False)
    context = tmp_path / "context"
    subprocess.run(
        ["bash", str(RUNNER_DIR / "build.sh"), "--prepare-context", str(context)],
        check=True,
        capture_output=True,
        text=True,
    )
    helper = context / "runner" / PATCH_PATH.name
    assert helper.read_bytes() == PATCH_PATH.read_bytes()
    dockerfile = (context / "Dockerfile").read_text()
    assert dockerfile.index(f"runner/{PATCH_PATH.name}") < dockerfile.index(
        "RUN SWE_RUNNER_BUILD=1"
    )
    preparation = (context / "runner/prepare-runtime.sh").read_text()
    assert (
        preparation.index('apply_patches "${gym_dir}"')
        < preparation.index(f'python3 "${{script_dir}}/{PATCH_PATH.name}"')
        < preparation.index("uv build ")
    )
    source = tmp_path / "app.py"
    source.write_text(GYM_SOURCE)
    subprocess.run([sys.executable, str(helper), str(source)], check=True)
    assert source.read_text() == patch_source(GYM_SOURCE)


def test_preflight_uses_worker_cwd_and_both_dependency_aliases(tmp_path):
    """Exercise the real shell array, including writable episode exceptions."""
    source = (RUNNER_DIR / "preflight.sh").read_text()
    setup = (
        "apptainer_args=("
        + source.split("apptainer_args=(", 1)[1].split("apptainer --version", 1)[0]
    )
    env = dict(
        os.environ,
        runner_root=str(tmp_path),
        openhands_root=str(tmp_path / "shared setup"),
        probe_dir=str(tmp_path / "probe"),
        r2e_root=str(tmp_path / "r2e"),
    )
    result = subprocess.run(
        ["bash", "-euc", setup + '\nprintf "%s\\0" "$PWD" "${apptainer_args[@]}"'],
        env=env,
        check=True,
        capture_output=True,
    )
    cwd, *args = result.stdout.decode().rstrip("\0").split("\0")
    assert cwd == str(tmp_path)
    assert args[args.index("--no-mount") + 1] == "home,tmp,bind-paths,cwd"
    for root in ("/openhands_setup", env["openhands_root"]):
        for directory in ("OpenHands", "miniforge3"):
            assert (
                f"type=bind,src={env['openhands_root']}/{directory},dst={root}/{directory},ro"
                in args
            )
        for directory in (".eval_sessions", "logs", "evaluation/oh"):
            assert (
                f"type=bind,src={env['openhands_root']}/OpenHands/{directory},dst={root}/OpenHands/{directory}"
                in args
            )


@pytest.mark.parametrize(
    "bad_path",
    [
        None,
        "/runtime/setup/OpenHands",
        "/runtime/setup/OpenHands/.venv",
        "/openhands_setup/OpenHands",
        "/openhands_setup/OpenHands/.venv",
        "/runtime/setup/miniforge3",
        "/openhands_setup/miniforge3",
        "/runtime/setup/OpenHands/logs",
    ],
)
def test_preflight_dependency_probe_checks_effective_mount_flags(monkeypatch, bad_path):
    """Reject writable aliases even when mount arguments requested read-only."""
    source = (RUNNER_DIR / "preflight.sh").read_text()
    probe = source.split("<<'DEPENDENCY_PROBE'\n", 1)[1].split("\nDEPENDENCY_PROBE", 1)[
        0
    ]
    checked = []

    def statvfs(path):
        path = str(path)
        checked.append(path)
        readonly = not path.endswith(("/.eval_sessions", "/logs", "/evaluation/oh"))
        if path == bad_path:
            readonly = not readonly
        return SimpleNamespace(f_flag=os.ST_RDONLY if readonly else 0)

    monkeypatch.setattr(os, "statvfs", statvfs)
    monkeypatch.setattr(sys, "argv", ["-", "/runtime/setup"])
    if bad_path:
        with pytest.raises(SystemExit, match=bad_path):
            exec(compile(probe, "preflight-probe.py", "exec"), {})
    else:
        exec(compile(probe, "preflight-probe.py", "exec"), {})
        assert len(checked) == 12
