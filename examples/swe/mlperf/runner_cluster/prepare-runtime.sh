#!/usr/bin/env bash
# Build-time preparation only. Run in the Dockerfile, never on the training host.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=runtime-pins.env
source "${script_dir}/runtime-pins.env"

[[ $(id -u) == 0 ]] || {
    echo 'Preparation requires the Ubuntu build container as root.' >&2
    exit 1
}

machine=$(uname -m)
case ${machine} in
    x86_64)
        package_arch=amd64
        miniforge_sha256=${MINIFORGE_INSTALLER_SHA256_AMD64}
        jq_sha256=${JQ_BINARY_SHA256_AMD64}
        ;;
    aarch64)
        package_arch=arm64
        miniforge_sha256=${MINIFORGE_INSTALLER_SHA256_ARM64}
        jq_sha256=${JQ_BINARY_SHA256_ARM64}
        ;;
    *) echo "Unsupported build architecture: ${machine}" >&2; exit 1 ;;
esac
[[ $(dpkg --print-architecture) == "${package_arch}" ]] || {
    echo 'Build container packages must match the native CPU architecture.' >&2
    exit 1
}
[[ ${SWE_RUNNER_BUILD:-} == 1 ]] || {
    echo 'Preparation is restricted to the Dockerfile build step (SWE_RUNNER_BUILD=1).' >&2
    exit 1
}

gym_dir=/opt/nemo-gym
reference_dir=/opt/mlperf-reference
swe_dir=${gym_dir}/responses_api_agents/swe_agents
github_base=${GITHUB_BASE_URL:-https://github.com}
github_download_base=${GITHUB_DOWNLOAD_BASE_URL:-https://github.com}
provenance_dir=/opt/swe-runner/provenance
mkdir -p "${provenance_dir}"

# Also covers Git dependencies installed by the upstream Poetry/pip scripts.
if [[ ${github_base%/} != https://github.com ]]; then
    git config --global "url.${github_base%/}/.insteadOf" https://github.com/
fi
printf 'channel_alias: %s\n' "${CONDA_CHANNEL_ALIAS:-https://conda.anaconda.org}" > /root/.condarc

checkout_pinned() {
    local repository=$1 revision=$2 destination=$3
    local attempt delay=5 max_attempts=4 actual_revision
    [[ ${revision} =~ ^[0-9a-f]{40}$ ]] || {
        echo "Expected an immutable commit for ${repository}" >&2
        exit 1
    }
    [[ ! -e ${destination} ]] || {
        echo "Refusing to overwrite existing source tree ${destination}" >&2
        exit 1
    }
    git init -q "${destination}"
    git -C "${destination}" remote add origin "${github_base%/}/${repository}.git"
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        echo "Fetching ${repository} at ${revision} (attempt ${attempt}/${max_attempts})" >&2
        if git -C "${destination}" fetch --depth 1 origin "${revision}"; then
            break
        fi
        if ((attempt == max_attempts)); then
            echo "Failed to fetch ${repository} at ${revision} after ${max_attempts} attempts" >&2
            return 1
        fi
        echo "Retrying ${repository} at ${revision} in ${delay} seconds" >&2
        sleep "${delay}"
        delay=$((delay * 2))
    done
    git -C "${destination}" checkout -q --detach FETCH_HEAD
    actual_revision=$(git -C "${destination}" rev-parse HEAD)
    [[ ${actual_revision} == "${revision}" ]] || {
        echo "Pinned commit mismatch for ${repository}: expected ${revision}, got ${actual_revision}" >&2
        return 1
    }
}

apply_patches() {
    local destination=$1 patch_dir=$2 patch_file
    local -a patches=("${patch_dir}"/*.patch)
    [[ -f ${patches[0]} ]] || {
        echo "No reference patches found in ${patch_dir}" >&2
        exit 1
    }
    for patch_file in "${patches[@]}"; do
        if git -C "${destination}" apply --check "${patch_file}"; then
            git -C "${destination}" apply "${patch_file}"
        elif git -C "${destination}" apply --reverse --check "${patch_file}"; then
            echo "Already applied: ${patch_file##*/}"
        else
            echo "Reference patch does not match pinned source: ${patch_file}" >&2
            exit 1
        fi
        sha256sum "${patch_file}" >> "${provenance_dir}/patches.sha256"
    done
}

download_checked() {
    local url=$1 digest=$2 destination=$3
    curl --fail --location --retry 3 "${url}" --output "${destination}"
    printf '%s  %s\n' "${digest}" "${destination}" | sha256sum --check --strict
}

checkout_pinned NVIDIA-NeMo/RL "${NEMO_RL_REVISION}" "${reference_dir}"
[[ $(git -C "${reference_dir}" rev-parse HEAD:3rdparty/Gym-workspace/Gym) == "${NEMO_GYM_REVISION}" ]]
checkout_pinned NVIDIA-NeMo/Gym "${NEMO_GYM_REVISION}" "${gym_dir}"
apply_patches "${gym_dir}" "${reference_dir}/docker/mlperf/patches/gym"

# Apply before building the Gym wheel so packaged and source modules agree.
python3 "${script_dir}/patch-sandbox-isolation.py" "${swe_dir}/app.py"
offline_wheel_dir=${script_dir}/offline-wheels
bash "${script_dir}/prepare-offline-wheels.sh" "${offline_wheel_dir}"
cp "${offline_wheel_dir}/SHA256SUMS" "${provenance_dir}/offline-wheels.sha256"
python3 "${script_dir}/patch-offline-eval.py" "${swe_dir}/app.py" \
    --wheel-dir "${offline_wheel_dir}"

# Build before adding any harness clones/venvs below Gym's package directories;
# otherwise setuptools namespace discovery can package those nested trees too.
wheel_dir=/opt/swe-runner/wheels
mkdir -p "${wheel_dir}"
uv build --python "${PYTHON_VERSION}" --wheel --out-dir "${wheel_dir}" "${gym_dir}"
gym_wheels=("${wheel_dir}"/*.whl)
[[ ${#gym_wheels[@]} == 1 && -f ${gym_wheels[0]} ]]
gym_wheel=${gym_wheels[0]}
sha256sum "${gym_wheel}" > "${provenance_dir}/gym-wheel.sha256"

cd "${gym_dir}"
uv venv --seed --python "${PYTHON_VERSION}" .venv
uv sync --locked --python "${PYTHON_VERSION}"
root_python=${gym_dir}/.venv/bin/python
"${root_python}" - <<'PY' > /opt/swe-runner/provenance/server-constraints.txt
from importlib.metadata import version

print(f"ray[default]=={version('ray')}")
print(f"openai=={version('openai')}")
PY

# These are the same separate server environments that `gym env start` creates.
# Prebuilding them makes skip_venv_if_present useful at runtime.
uv venv --seed --python "${root_python}" "${swe_dir}/.venv"
(
    cd "${swe_dir}"
    uv pip install --python .venv/bin/python --constraint "${provenance_dir}/server-constraints.txt" \
        --requirement requirements.txt
)
model_dir=${gym_dir}/responses_api_models/vllm_model
uv venv --seed --python "${root_python}" "${model_dir}/.venv"
(
    cd "${model_dir}"
    uv pip install --python .venv/bin/python --constraint "${provenance_dir}/server-constraints.txt" --editable .
)

# Preinstall a pinned Miniforge and jq so the upstream script does not download
# the moving Miniforge 'latest' release. Its package solver still runs at build
# time; its resolved environment is recorded below.
openhands_setup=${swe_dir}/swe_openhands_setup
mkdir -p "${openhands_setup}"
download_checked \
    "${github_download_base%/}/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${machine}.sh" \
    "${miniforge_sha256}" /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p "${openhands_setup}/miniforge3"
rm /tmp/miniforge.sh
download_checked \
    "${github_download_base%/}/jqlang/jq/releases/download/jq-${JQ_VERSION}/jq-linux-${package_arch}" \
    "${jq_sha256}" "${openhands_setup}/miniforge3/bin/jq"
chmod 0755 "${openhands_setup}/miniforge3/bin/jq"
checkout_pinned sdevare-nv/nv-OpenHands "${OPENHANDS_REVISION}" "${openhands_setup}/OpenHands"
# Include the underlying exception when Poetry reports a generic connection
# failure. One verbosity level avoids HTTP debug logging and credential output.
python3 - "${openhands_setup}/OpenHands/Makefile" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
if "poetry install --" not in text:
    raise SystemExit("Pinned OpenHands Poetry install commands were not found")
path.write_text(text.replace("poetry install --", "poetry install -v --"))
PY
python3 "${script_dir}/patch-poetry-markers.py" \
    --setup-script "${swe_dir}/setup_scripts/openhands.sh"
SETUP_DIR="${openhands_setup}" \
    MINIFORGE_DIR="${openhands_setup}/miniforge3" \
    OPENHANDS_DIR="${openhands_setup}/OpenHands" \
    AGENT_FRAMEWORK_REPO="${github_base%/}/sdevare-nv/nv-OpenHands.git" \
    AGENT_FRAMEWORK_COMMIT="${OPENHANDS_REVISION}" \
    SWE_POETRY_MARKER_FIX="${script_dir}/patch-poetry-markers.py" \
    SWE_BUILD_PROVENANCE_DIR="${provenance_dir}" \
    bash "${swe_dir}/setup_scripts/openhands.sh"

# OpenHands' lock has its own Gym dependency. Install a wheel from this exact
# prepared checkout: an editable install would point outside the OpenHands tree
# mounted inside the SIF. Retain OpenHands' OpenAI compatibility requirement.
oh_python=${openhands_setup}/OpenHands/.venv/bin/python
uv pip install --python "${oh_python}" "${gym_wheel}" 'openai==1.99.9'

prepare_eval_tools() {
    local setup_dir=$1
    mkdir -p "${setup_dir}/uv" "${setup_dir}/python"
    # Each evaluator's portable Python must live beneath its bind-mounted tree.
    # A symlink to /opt/python would break inside task SIFs.
    install -m 0755 "$(command -v uv)" "${setup_dir}/uv/uv"
    UV_PYTHON_INSTALL_DIR="${setup_dir}/python" uv python install "${PYTHON_VERSION}"
}

setup_dir=${swe_dir}/swe_swebench_setup
prepare_eval_tools "${setup_dir}"
checkout_pinned HeyyyyyyG/SWE-bench "${SWEBENCH_REVISION}" "${setup_dir}/SWE-bench"
SETUP_DIR="${setup_dir}" UV_DIR="${setup_dir}/uv" PYTHON_DIR="${setup_dir}/python" \
    SWEBENCH_DIR="${setup_dir}/SWE-bench" \
    SWEBENCH_REPO="${github_base%/}/HeyyyyyyG/SWE-bench.git" SWEBENCH_COMMIT="${SWEBENCH_REVISION}" \
    bash "${swe_dir}/setup_scripts/swebench.sh"

setup_dir=${swe_dir}/swe_swebench_multilingual_setup
prepare_eval_tools "${setup_dir}"
checkout_pinned Kipok/SWE-bench "${SWEBENCH_MULTILINGUAL_REVISION}" "${setup_dir}/SWE-bench_Multilingual"
SETUP_DIR="${setup_dir}" UV_DIR="${setup_dir}/uv" PYTHON_DIR="${setup_dir}/python" \
    SWEBENCH_DIR="${setup_dir}/SWE-bench_Multilingual" \
    SWEBENCH_REPO="${github_base%/}/Kipok/SWE-bench.git" SWEBENCH_COMMIT="${SWEBENCH_MULTILINGUAL_REVISION}" \
    bash "${swe_dir}/setup_scripts/swebench_multilingual.sh"

setup_dir=${swe_dir}/swe_r2e_gym_setup
prepare_eval_tools "${setup_dir}"
checkout_pinned sdevare-nv/nv-R2E-Gym "${R2E_GYM_REVISION}" "${setup_dir}/R2E-Gym"
# The reference patch runs `python -` before creating the evaluator venv.
# Supply the existing Gym interpreter for that setup process only.
SETUP_DIR="${setup_dir}" UV_DIR="${setup_dir}/uv" PYTHON_DIR="${setup_dir}/python" \
    PATH="${root_python%/*}:${PATH}" \
    R2E_GYM_DIR="${setup_dir}/R2E-Gym" \
    EVAL_HARNESS_REPO="${github_base%/}/sdevare-nv/nv-R2E-Gym.git" EVAL_HARNESS_COMMIT="${R2E_GYM_REVISION}" \
    bash "${swe_dir}/setup_scripts/r2e_gym.sh"
apply_patches "${setup_dir}/R2E-Gym" "${reference_dir}/docker/mlperf/patches/r2e"

setup_dir=${swe_dir}/swe_rebench_setup
mkdir -p "${setup_dir}"
checkout_pinned SWE-rebench/SWE-rebench-V2 "${SWE_REBENCH_REVISION}" "${setup_dir}/SWE-rebench-V2"
SETUP_DIR="${setup_dir}" REBENCH_DIR="${setup_dir}/SWE-rebench-V2" \
    bash "${swe_dir}/setup_scripts/swe_rebench.sh"

# Check imports in the actual environments, and record their resolved packages.
# No model requests, Ray cluster, or SIF execution occurs during image building.
"${swe_dir}/.venv/bin/python" -c \
    'from responses_api_agents.swe_agents.app import SWEBenchWrapper; assert "base_results_dir" in SWEBenchWrapper.model_fields["config"].annotation.model_fields'
"${model_dir}/.venv/bin/python" -c 'from responses_api_models.vllm_model.app import VLLMModel'
"${oh_python}" -c 'from openhands.agenthub.nemo_gym_client import NemoGymClient'
"${swe_dir}/swe_r2e_gym_setup/R2E-Gym/venv/bin/python" -c \
    'import r2egym; from r2egym.agenthub.runtime.local import LocalRuntime'
rebench_log_parsers=${swe_dir}/swe_rebench_setup/SWE-rebench-V2/lib/agent/log_parsers.py
[[ -f ${rebench_log_parsers} ]] || {
    echo "Missing pinned SWE-rebench log parser: ${rebench_log_parsers}" >&2
    exit 1
}

export RUNNER_GYM_DIR=${gym_dir} RUNNER_REFERENCE_DIR=${reference_dir}
export RUNNER_PROVENANCE_DIR=${provenance_dir}
"${root_python}" - <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import subprocess

gym = pathlib.Path(os.environ["RUNNER_GYM_DIR"])
swe = gym / "responses_api_agents/swe_agents"
provenance = pathlib.Path(os.environ["RUNNER_PROVENANCE_DIR"])
trees = {
    "nemo_rl": pathlib.Path(os.environ["RUNNER_REFERENCE_DIR"]),
    "nemo_gym": gym,
    "openhands": swe / "swe_openhands_setup/OpenHands",
    "r2e_gym": swe / "swe_r2e_gym_setup/R2E-Gym",
    "swebench": swe / "swe_swebench_setup/SWE-bench",
    "swebench_multilingual": swe / "swe_swebench_multilingual_setup/SWE-bench_Multilingual",
    "swe_rebench": swe / "swe_rebench_setup/SWE-rebench-V2",
}
versions = {
    name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    for name, path in trees.items()
}
environments = {
    "gym": gym / ".venv/bin/python",
    "swe_server": swe / ".venv/bin/python",
    "model_server": gym / "responses_api_models/vllm_model/.venv/bin/python",
    "openhands": trees["openhands"] / ".venv/bin/python",
    "r2e_gym": trees["r2e_gym"] / "venv/bin/python",
    "swebench": trees["swebench"] / "venv/bin/python",
    "swebench_multilingual": trees["swebench_multilingual"] / "venv/bin/python",
}
for name, python in environments.items():
    # Gym deliberately excludes optional mlflow dependencies. A raw `pip check`
    # would reject that reference layout; validate the actual imports above.
    packages = subprocess.check_output(["uv", "pip", "freeze", "--python", str(python)], text=True)
    (provenance / f"{name}.requirements.txt").write_text(packages)
    (provenance / f"{name}.python.txt").write_text(
        subprocess.check_output([str(python), "--version"], text=True)
    )
for name, path in trees.items():
    diff = subprocess.check_output(["git", "-C", str(path), "diff", "--binary"])
    (provenance / f"{name}.patch").write_bytes(diff)
    versions[f"{name}_diff_sha256"] = hashlib.sha256(diff).hexdigest()
versions["apptainer"] = subprocess.check_output(["apptainer", "--version"], text=True).strip()
versions["uv"] = subprocess.check_output(["uv", "--version"], text=True).strip()
versions["machine"] = platform.machine()
versions["package_architecture"] = subprocess.check_output(
    ["dpkg", "--print-architecture"], text=True
).strip()
versions["build_insecure_tls"] = os.environ.get("SWE_BUILD_INSECURE_TLS", "0") == "1"
pathlib.Path("/opt/swe-runner/runtime-manifest.json").write_text(json.dumps(versions, indent=2) + "\n")
(provenance / "dpkg.txt").write_text(
    subprocess.check_output(["dpkg-query", "-W", "-f=${Package}\t${Version}\n"], text=True)
)
PY
"${openhands_setup}/miniforge3/bin/conda" list --explicit > "${provenance_dir}/miniforge-explicit.txt"

# No build caches are required once all environments have been populated.
uv cache clean
"${openhands_setup}/miniforge3/bin/conda" clean --all --yes
rm -rf /root/.cache/pip /root/.cache/pypoetry/cache
echo 'CPU SWE runtime prepared; validate nested Apptainer and a real episode on the target host.'
