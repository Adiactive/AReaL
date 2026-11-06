#!/usr/bin/env bash
# Run inside the prepared runner image, preferably with Docker --network none.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash preflight.sh /absolute/path/to/task.sif

Run through start.sh --preflight FILE to supply runtime mounts and disable network.
Checks the baked Python/harness environments, read-only SIF mount, writable scratch,
native SIF architecture, and execution with the runner's namespace/overlay options.
Two disposable executions check writable state and isolation between attempts.
An explicit disposable /etc/hosts bind prevents changing the outer container's file.

This does not contact a model or any external service, download dependencies, run
an agent episode, or establish benchmark grading correctness. No SIF is modified.
USAGE
}

die() { echo "$*" >&2; exit 1; }
if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
[[ $# -eq 1 ]] || { usage >&2; exit 1; }
[[ $1 == /* && -f $1 && -r $1 ]] || die 'Provide an absolute path to a readable local SIF file.'
sif_path=$(realpath -e -- "$1")
[[ $sif_path != *,* ]] || die 'SIF paths must not contain commas.'
case $(uname -m) in
    x86_64) host_arch=amd64 ;;
    aarch64|arm64) host_arch=arm64 ;;
    *) die 'The runner host must be x86_64 or ARM64.' ;;
esac
for binary in apptainer timeout findmnt; do
    command -v "$binary" >/dev/null || die "Missing runtime command: $binary"
done
# Apptainer 1.5.5 uses sif v2.24.0, whose header prints Primary Architecture.
# Read the image header before executing anything inside the task environment.
sif_header=$(apptainer sif header "$sif_path") || die 'Unable to read the SIF header.'
sif_arch=$(awk '/^Primary Architecture:/ {print $3}' <<< "$sif_header")
[[ -n $sif_arch ]] || die 'The SIF has no recognized primary architecture; select an image with native architecture metadata.'
[[ $sif_arch == "$host_arch" ]] || die "SIF architecture is $sif_arch but this runner is $host_arch. Supply a matching task SIF; CPU emulation is unsupported."
printf 'Native architecture verified: runner=%s SIF=%s\n' "$host_arch" "$sif_arch"
[[ -c /dev/fuse ]] || die '/dev/fuse is not exposed to the runner container.'
mount_options=$(findmnt --noheadings --output VFS-OPTIONS --target "$sif_path")
case ",$mount_options," in
    *,ro,*) ;;
    *) die 'The SIF must be mounted read-only. Use start.sh with its read-only SIF bind.' ;;
esac

runner_root=/opt/nemo-gym
harness_root=$runner_root/responses_api_agents/swe_agents
openhands_root=$harness_root/swe_openhands_setup
r2e_root=$harness_root/swe_r2e_gym_setup
offline_wheels=/opt/swe-runner/offline-wheels
[[ -f $offline_wheels/SHA256SUMS ]] || die 'Offline evaluator wheels are missing; rebuild the runner.'
(cd "$offline_wheels" && sha256sum --check --strict SHA256SUMS)
root_python=$runner_root/.venv/bin/python
agent_python=$harness_root/.venv/bin/python
model_python=$runner_root/responses_api_models/vllm_model/.venv/bin/python

require_exec() { [[ -x $1 ]] || die "Missing baked executable: $1. Rebuild the runner image."; }
for executable in "$root_python" "$agent_python" "$model_python" \
    "$openhands_root/OpenHands/.venv/bin/python" \
    "$harness_root/swe_swebench_setup/SWE-bench/venv/bin/python" \
    "$harness_root/swe_swebench_multilingual_setup/SWE-bench_Multilingual/venv/bin/python" \
    "$r2e_root/R2E-Gym/venv/bin/python"; do
    require_exec "$executable"
done
for binary in conda mamba tmux node poetry git jq; do
    require_exec "$openhands_root/miniforge3/bin/$binary"
done
[[ -d $harness_root/swe_rebench_setup/SWE-rebench-V2 ]] || die 'SWE-rebench setup is absent; runtime startup would try to prepare it.'
[[ -f /opt/swe-runner/runtime-manifest.json ]] || die 'The image provenance manifest is missing.'

# These imports run offline; never invoke setup scripts from a preflight.
export UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RAY_USAGE_STATS_ENABLED=0 WANDB_MODE=disabled
cd "$runner_root"
timeout --kill-after=5s 60s "$root_python" -c \
    'import json, nemo_gym; json.load(open("/opt/swe-runner/runtime-manifest.json")); print("Gym and provenance: OK")'
for interpreter in "$agent_python" "$model_python"; do
    timeout --kill-after=5s 60s "$interpreter" -c 'import fastapi, openai, ray, nemo_gym'
done
timeout --kill-after=5s 60s "$openhands_root/OpenHands/.venv/bin/python" -c 'from openhands.agenthub.nemo_gym_client import NemoGymClient'
timeout --kill-after=5s 60s "$r2e_root/R2E-Gym/venv/bin/python" -c 'import r2egym'
for directory in swe_swebench_setup/SWE-bench swe_swebench_multilingual_setup/SWE-bench_Multilingual; do
    timeout --kill-after=5s 60s "$harness_root/$directory/venv/bin/python" -c 'import swebench'
done

scratch_root=${TMPDIR:-/srv/swe-runtime/tmp}
apptainer_tmp=${APPTAINER_TMPDIR:-/srv/swe-runtime/apptainer-tmp}
apptainer_cache=${APPTAINER_CACHEDIR:-/srv/swe-runtime/apptainer-cache}
for directory in "$scratch_root" "$apptainer_tmp" "$apptainer_cache" /srv/swe-results; do
    [[ -d $directory && -w $directory ]] || die "Required writable directory is missing: $directory. Use start.sh."
    scratch_probe=$(mktemp "$directory/.preflight-write.XXXXXX")
    rm -f -- "$scratch_probe"
done
export APPTAINER_TMPDIR="$apptainer_tmp" APPTAINER_CACHEDIR="$apptainer_cache"
probe_dir=$(mktemp -d "$scratch_root/mlperf-preflight.XXXXXX")
trap 'rm -rf -- "$probe_dir"' EXIT
[[ $probe_dir != *,* ]] || die 'Scratch paths must not contain commas.'
printf '127.0.0.1 localhost\n' > "$probe_dir/hosts"
marker="/mlperf-preflight-${probe_dir##*/}"

# Match _build_apptainer_command in the pinned Gym SWE runner. Bind the portable
# Python trees at their original paths, as the real harness does for its symlinks.
apptainer_args=(exec --writable-tmpfs --cleanenv --pid --no-mount home,tmp,bind-paths
    --mount "type=bind,src=$probe_dir/hosts,dst=/etc/hosts"
    --mount "type=bind,src=$probe_dir,dst=/preflight-output"
    --mount "type=bind,src=$openhands_root/OpenHands,dst=$openhands_root/OpenHands,ro"
    --mount "type=bind,src=$openhands_root/miniforge3,dst=$openhands_root/miniforge3,ro"
    --mount "type=bind,src=$r2e_root,dst=$r2e_root,ro")
cd /
apptainer --version
printf '%s\n' 'Checking a disposable writable overlay and portable harness interpreters...'
timeout --kill-after=5s 120s apptainer "${apptainer_args[@]}" "$sif_path" \
    bash -s -- "$marker" "$openhands_root" "$r2e_root" "$host_arch" <<'PROBE'
set -euo pipefail
marker=$1
openhands_root=$2
r2e_root=$3
case $(uname -m) in
    x86_64) [[ $4 == amd64 ]] ;;
    aarch64|arm64) [[ $4 == arm64 ]] ;;
    *) exit 1 ;;
esac
[[ ! -e $marker && ! -e ${marker}-hidden-tests ]]
printf 'episode-local state\n' > "$marker"
mkdir "${marker}-hidden-tests"
printf 'mock test\n' > "${marker}-hidden-tests/test.txt"
rm -rf -- "${marker}-hidden-tests"
[[ ! -e ${marker}-hidden-tests ]]
[[ $(cat "$marker") == 'episode-local state' ]]
printf '127.0.0.1 localhost preflight-disposable\n' > /etc/hosts
"$openhands_root/OpenHands/.venv/bin/python" -c 'from openhands.agenthub.nemo_gym_client import NemoGymClient'
"$r2e_root/R2E-Gym/venv/bin/python" -c 'import r2egym'
printf 'first execution finished\n' > /preflight-output/result
PROBE
[[ $(<"$probe_dir/result") == 'first execution finished' ]] || die 'Bind-mounted output was not produced.'
[[ $(<"$probe_dir/hosts") == '127.0.0.1 localhost preflight-disposable' ]] || die 'Disposable /etc/hosts binding failed.'

printf '%s\n' 'Checking the next execution has fresh writable state...'
timeout --kill-after=5s 120s apptainer "${apptainer_args[@]}" "$sif_path" \
    bash -s -- "$marker" <<'PROBE'
set -euo pipefail
[[ ! -e $1 && ! -e ${1}-hidden-tests ]]
PROBE

printf '%s\n' 'Checking offline chardet installation into the disposable task environment...'
timeout --kill-after=5s 120s apptainer "${apptainer_args[@]}" \
    --mount "type=bind,src=$offline_wheels,dst=$offline_wheels,ro" \
    --env UV_NO_INDEX=true --env "UV_FIND_LINKS=$offline_wheels" --env UV_OFFLINE=true \
    --env UV_CACHE_DIR=/tmp/mlperf-empty-uv-cache "$sif_path" bash -s <<'PROBE'
set -euo pipefail
cd /testbed
[[ -x .venv/bin/python ]] || { echo 'R2E preflight requires /testbed/.venv/bin/python' >&2; exit 1; }
# Mirror the upstream setup's target, rather than installing in the harness venv.
ln -sfn /testbed/.venv /root/.venv
# Force the baked-wheel path even when another SIF already includes chardet.
env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME uv pip install --reinstall --no-deps 'chardet==5.2.0'
env -u PYTHONPATH -u PYTHONHOME .venv/bin/python -c \
    'import chardet; assert chardet.__version__ == "5.2.0"; print("Task chardet import: OK (" + chardet.__version__ + ")")'
PROBE
printf '%s\n' 'Offline preflight passed: baked dependencies, SIF execution, writable overlay, fresh-attempt isolation, and task dependency installation.'
printf '%s\n' 'Still required: one real task, grading, model connectivity, concurrency, and cancellation validation.'
