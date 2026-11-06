#!/usr/bin/env bash
# Generate the head configuration using Python from the runner image.
set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: bash configure.sh --env-file PATH [configure.py options]

The env file is sourced as trusted Bash code on the host. See configure.env.example
for required settings and defaults. Docker runs the generator without networking;
this command does not build images or start the runner service.
SWE_SERVICE selects areal (default) or standalone (fixed model). Standalone mode
requires SWE_MODEL_BASE_URL and SWE_MODEL_NAME.
SWE_SIF_DIR, SWE_STATE_DIR and SWE_RESULTS_DIR are required host paths. The YAML
records them for start.sh --config PATH; no need to repeat them at launch.
Name, resources, preflight, role, and other launch settings are also saved. CLI
options passed to start.sh override them. SWE_ROLE=standalone omits Ray settings;
other roles require SWE_HEAD_ADDRESS. SWE_DRY_RUN must be true or false.

Additional options are passed to configure.py last, for example --check-only or
--force. Existing YAML files are preserved unless --force is explicitly supplied.
EOF
}
die() { echo "$*" >&2; exit 1; }
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
[[ ${1:-} == --env-file && $# -ge 2 && -n $2 ]] ||
    die 'Use --env-file PATH; see --help.'
config_env_file=$2
shift 2
[[ ${1:-} != -- ]] || shift
# A relative file must resolve from the current directory, never through PATH.
[[ $config_env_file == /* ]] || config_env_file=$PWD/$config_env_file
[[ -f $config_env_file && -r $config_env_file ]] || die "Cannot read env file: $config_env_file"
source "$config_env_file"

[[ -n ${SWE_DEPLOY_DIR:-} ]] || die 'SWE_DEPLOY_DIR is required.'
[[ -n ${SWE_HEAD_IP:-} ]] || die 'SWE_HEAD_IP is required.'
case ${SWE_ROLE:-} in
    standalone) ;;
    ''|head|worker)
        [[ -n ${SWE_HEAD_ADDRESS:-} ]] || die 'SWE_HEAD_ADDRESS is required (Ray head IPv4:PORT).'
        ;;
    *) die 'SWE_ROLE must be empty, standalone, head or worker.' ;;
esac
case ${SWE_DRY_RUN:-false} in
    true|false) ;;
    *) die 'SWE_DRY_RUN must be true or false.' ;;
esac
for required_path in SWE_SIF_DIR SWE_STATE_DIR SWE_RESULTS_DIR; do
    [[ -n ${!required_path:-} ]] || die "$required_path is required (absolute host path)."
    case ${!required_path} in
        /*) ;;
        *) die "$required_path must be an absolute host path." ;;
    esac
    case ${!required_path} in
        *,*|*$'\n'*|*$'\r'*|*'${'*) die "$required_path cannot contain commas, newlines or interpolation." ;;
    esac
done
[[ $SWE_DEPLOY_DIR == /* ]] || die 'SWE_DEPLOY_DIR must be absolute.'
SWE_SERVICE=${SWE_SERVICE:-areal}
case $SWE_SERVICE in
    areal) ;;
    standalone)
        [[ -n ${SWE_MODEL_BASE_URL:-} ]] || die 'SWE_MODEL_BASE_URL is required when SWE_SERVICE=standalone.'
        [[ -n ${SWE_MODEL_NAME:-} ]] || die 'SWE_MODEL_NAME is required when SWE_SERVICE=standalone.'
        ;;
    *) die 'SWE_SERVICE must be areal or standalone.' ;;
esac
SWE_CONFIG_FILE=${SWE_CONFIG_FILE:-${SWE_SERVICE}-site.yaml}
case $SWE_CONFIG_FILE in
    */*|*$'\n'*|*$'\r'*) die 'SWE_CONFIG_FILE must be a filename ending in .yaml or .yml.' ;;
    *.yaml|*.yml) ;;
    *) die 'SWE_CONFIG_FILE must be a filename ending in .yaml or .yml.' ;;
esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for mount_path in "$SWE_DEPLOY_DIR" "$script_dir"; do
    case $mount_path in
        *,*|*$'\n'*|*$'\r'*) die 'Docker mount paths cannot contain commas or newlines.' ;;
    esac
done
[[ -f $script_dir/configure.py ]] || die "Missing $script_dir/configure.py"
command -v docker >/dev/null 2>&1 || die 'Docker is required on the host.'

mkdir -p -- "$SWE_DEPLOY_DIR/config"
generator_args=(
    --service "$SWE_SERVICE"
    --output "/out/$SWE_CONFIG_FILE"
    --host-sif-dir "$SWE_SIF_DIR"
    --host-state-dir "$SWE_STATE_DIR"
    --host-results-dir "$SWE_RESULTS_DIR"
    --runner-image "${SWE_RUNNER_IMAGE:-mlperf-swe-runner:cluster}"
    --ray-worker-ports "${SWE_RAY_WORKER_PORTS:-20000-20127}"
    --object-store-memory "${SWE_OBJECT_STORE_MEMORY:-1073741824}"
    --container-name "${SWE_CONTAINER_NAME:-mlperf-swe-runner}"
    --cpus "${SWE_CPUS:-48}"
    --memory "${SWE_MEMORY:-400g}"
    --shm-size "${SWE_SHM_SIZE:-16g}"
    --preflight "${SWE_PREFLIGHT_SIF:-}"
    --port "${SWE_API_PORT:-8080}"
    --registry-port "${SWE_REGISTRY_PORT:-8081}"
    --model-port "${SWE_MODEL_PORT:-8082}"
    --concurrency "${SWE_CONCURRENCY:-4}"
    --max-turns "${SWE_MAX_TURNS:-30}"
    --agent-timeout "${SWE_AGENT_TIMEOUT:-1800}"
    --test-timeout "${SWE_TEST_TIMEOUT:-60}"
    --command-timeout "${SWE_COMMAND_TIMEOUT:-60}"
    --memory-limit-mb "${SWE_MEMORY_LIMIT_MB:-32768}"
    --chat-template-thinking "${SWE_CHAT_TEMPLATE_THINKING:-disabled}"
)
if [[ ${SWE_ROLE:-} == standalone ]]; then
    generator_args+=(--bind-address "${SWE_BIND_ADDRESS:-$SWE_HEAD_IP}")
else
    generator_args+=(--advertised-host "$SWE_HEAD_IP" --ray-head-address "$SWE_HEAD_ADDRESS")
    [[ -z ${SWE_BIND_ADDRESS:-} ]] || generator_args+=(--bind-address "$SWE_BIND_ADDRESS")
fi
[[ -z ${SWE_ROLE:-} ]] || generator_args+=(--role "$SWE_ROLE")
[[ -z ${SWE_NODE_IP:-} ]] || generator_args+=(--node-ip "$SWE_NODE_IP")
[[ -z ${SWE_ENV_FILE:-} ]] || generator_args+=(--runtime-env-file "$SWE_ENV_FILE")
if [[ ${SWE_DRY_RUN:-false} == true ]]; then
    generator_args+=(--launch-dry-run)
else
    generator_args+=(--no-launch-dry-run)
fi
if [[ $SWE_SERVICE == standalone ]]; then
    generator_args+=(--model-base-url "$SWE_MODEL_BASE_URL" --model-name "$SWE_MODEL_NAME")
else
    generator_args+=(--gym-port "${SWE_GYM_PORT:-8083}")
fi
exec docker run --rm --pull never --network none --entrypoint python \
    --mount "type=bind,src=$script_dir/configure.py,dst=/opt/swe-runner/configure.py,readonly" \
    --mount "type=bind,src=$SWE_DEPLOY_DIR/config,dst=/out" \
    "${SWE_RUNNER_IMAGE:-mlperf-swe-runner:cluster}" \
    /opt/swe-runner/configure.py "${generator_args[@]}" "$@"
