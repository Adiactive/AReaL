#!/usr/bin/env bash
# Run the prepared CPU environment service, or an offline container preflight.
set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: bash start.sh --config site.yaml [overrides]
       bash start.sh --sif-dir PATH --state-dir PATH [options]

configure.sh reads configure.env and saves launch settings in the generated YAML.
start.sh reads that YAML, not configure.env. Regenerate YAML after editing the env file.
Precedence: explicit CLI option > generated YAML > built-in default.
Service/API/Ray identity overrides must still agree with the service configuration.

Service options (configuration env variable in parentheses):
  --config PATH           YAML to read; CLI-only selection (SWE_DEPLOY_DIR/config/SWE_CONFIG_FILE)
  --role MODE             head, worker or standalone (SWE_ROLE; empty: infer)
  --service MODE          areal or fixed-model standalone (SWE_SERVICE; derived from YAML)
  --bind-address IP       Single-host published address (SWE_BIND_ADDRESS; default: head host)
  --env-file PATH         Runtime credentials file path (SWE_ENV_FILE; default: none)
                         Use --env-file '' to ignore a saved credentials path.
  --port NUMBER           Public API port (SWE_API_PORT; default: 8080)
                         Single-host: host mapping. Cluster head: must match YAML.
  --name NAME             Service container name (SWE_CONTAINER_NAME; default: mlperf-swe-runner)

Cluster options (head and worker roles):
  --node-ip IP            Local address (SWE_NODE_IP; head defaults to SWE_HEAD_IP)
                         Workers require their own address, in YAML or on the CLI.
  --ray-head-address IP:P Head address and GCS port (SWE_HEAD_ADDRESS)
  --ray-port-base PORT    GCS port from SWE_HEAD_ADDRESS, followed by 8 control ports
  --ray-worker-ports A-B  Worker port range (SWE_RAY_WORKER_PORTS; default: 20000-20127)
  --object-store-memory N Bytes per node (SWE_OBJECT_STORE_MEMORY; default: 1073741824)

Shared options:
  --image TAG             Runner image (SWE_RUNNER_IMAGE; default without YAML: mlperf-swe-runner:local)
  --sif-dir PATH          Read-only host SIF directory (SWE_SIF_DIR)
  --state-dir PATH        Node-local host state directory (SWE_STATE_DIR)
  --results-dir PATH      Writable results (SWE_RESULTS_DIR; shared across cluster nodes)
  --cpus NUMBER           Container CPU limit (SWE_CPUS; default: 48); integer for cluster roles
  --memory SIZE           Container RAM limit (SWE_MEMORY; default: 400g)
  --shm-size SIZE         Shared-memory maximum within RAM budget (SWE_SHM_SIZE; default: 16g)
  --preflight FILE        Offline check only (SWE_PREFLIGHT_SIF; empty: start service)
                         FILE is relative to the configured/explicit SIF directory.
  --no-preflight          Start the service even if YAML selects preflight
  --dry-run               Print Docker invocation only (SWE_DRY_RUN; default: false)
  --no-dry-run            Execute even if YAML selects dry-run
  --help                 Show this help; CLI-only

Apptainer receives /dev/fuse and the documented nested-Docker Ubuntu settings.
Standalone mode publishes the API port only. Cluster mode uses host networking;
allow Ray and callback ports only between trusted nodes. Docker PID isolation stays enabled.
In cluster mode --state-dir is node-local and --results-dir must be shared storage.
The script never replaces an existing container and never pulls an image.
The host, Docker daemon, and prepared image must have the same native CPU architecture.
Generated swe_runner metadata supplies launch defaults, including resource limits.
Preflight uses the image, host paths and resource limits with --network none. It uses a
temporary Docker-generated name, ignores the configured role, --name and credentials,
and never starts Ray or model services. Omit explicit --role/--service for preflight.
Workers read --config on the host only; neither it nor model credentials are mounted.
EOF
}

die() { echo "$*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && ( -n $2 || $1 == --env-file ) ]] || die "Missing value for $1"; }

image=${SWE_RUNNER_IMAGE:-mlperf-swe-runner:local}
original_args=("$@")
container_name=mlperf-swe-runner
sif_dir= state_dir= config= bind_address= env_file= preflight=
role=standalone service=standalone node_ip= ray_head_address= results_dir=
ray_port_base=16379 ray_worker_ports=20000-20127 object_store_memory=1073741824
port=8080 container_port=8080 cpus=48 memory=400g shm_size=16g dry_run=false
while (($#)); do
    case $1 in
        --help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --no-dry-run) dry_run=false; shift ;;
        --no-preflight) preflight=; shift ;;
        --sif-dir|--state-dir|--config|--bind-address|--env-file|--port|--name|--image|--cpus|--memory|--shm-size|--preflight|--role|--service|--node-ip|--ray-head-address|--results-dir|--ray-port-base|--ray-worker-ports|--object-store-memory)
            need_value "$@"
            case $1 in
                --sif-dir) sif_dir=$2 ;;
                --state-dir) state_dir=$2 ;;
                --config) config=$2 ;;
                --bind-address) bind_address=$2 ;;
                --env-file) env_file=$2 ;;
                --port) port=$2 ;;
                --name) container_name=$2 ;;
                --image) image=$2 ;;
                --cpus) cpus=$2 ;;
                --memory) memory=$2 ;;
                --shm-size) shm_size=$2 ;;
                --preflight) preflight=$2 ;;
                --role) role=$2 ;;
                --service) service=$2 ;;
                --node-ip) node_ip=$2 ;;
                --ray-head-address) ray_head_address=$2 ;;
                --results-dir) results_dir=$2 ;;
                --ray-port-base) ray_port_base=$2 ;;
                --ray-worker-ports) ray_worker_ports=$2 ;;
                --object-store-memory) object_store_memory=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

if [[ -n $config ]]; then
    command -v python3 >/dev/null || die 'Python 3 is required to read runner configuration.'
    # Keep the process-substitution PID and wait for it: mapfile alone would hide
    # helper failures. NUL pairs preserve spaces/newlines without shell evaluation.
    exec {config_fd}< <(python3 "$(dirname -- "${BASH_SOURCE[0]}")/launch-config.py" "${original_args[@]}")
    config_pid=$!
    mapfile -d '' -t config_defaults <&"$config_fd"
    exec {config_fd}<&-
    wait "$config_pid" || exit "$?"
    for ((i=0; i<${#config_defaults[@]}; i+=2)); do
        field=${config_defaults[i]}
        case $field in
            image|sif_dir|state_dir|results_dir|container_name|cpus|memory|shm_size|preflight|dry_run|bind_address|env_file|ray_worker_ports|object_store_memory|role|service|container_port|port|ray_head_address|ray_port_base|node_ip)
                printf -v "$field" '%s' "${config_defaults[i+1]}" ;;
            *) die 'Invalid host launch configuration field.' ;;
        esac
    done
fi

normalize_arch() {
    case $1 in
        amd64|x86_64) printf '%s' amd64 ;;
        arm64|aarch64) printf '%s' arm64 ;;
        *) die "Unsupported CPU architecture: $1. Use x86_64 or ARM64." ;;
    esac
}
host_arch=$(normalize_arch "$(uname -m)")

[[ $role == standalone || $role == head || $role == worker ]] || die 'Invalid --role.'
[[ $service == standalone || $service == areal ]] || die 'Invalid --service.'
[[ $role != worker || ( -z $env_file && $service == standalone ) ]] || die 'Workers run Ray only: omit --env-file and --service areal.'
[[ -z $preflight || ( $role == standalone && $service == standalone ) ]] || die 'Run --preflight separately in standalone mode.'
if [[ -z $preflight ]]; then
    [[ $container_name =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die 'Invalid --name: use letters, digits, underscores, periods or hyphens, starting with a letter or digit.'
fi
if [[ $role != standalone ]]; then
    [[ -n $node_ip && -n $ray_head_address && -n $results_dir ]] || die 'Cluster roles require --node-ip, --ray-head-address, and --results-dir.'
    [[ -z $bind_address || $bind_address == "$node_ip" ]] || die 'Cluster --bind-address, if supplied, must equal --node-ip.'
elif [[ -n $node_ip || -n $ray_head_address ]]; then
    die 'Use --role head or worker for Ray cluster options.'
fi

[[ -d $sif_dir ]] || die '--sif-dir must be an existing SIF directory.'
[[ -n $state_dir ]] || die '--state-dir is required.'
sif_dir=$(realpath -e -- "$sif_dir")
state_dir=$(realpath -m -- "$state_dir")
[[ $state_dir != / && $state_dir != "$sif_dir" ]] || die 'Use a dedicated state directory.'
[[ $state_dir != "$sif_dir/"* ]] || die 'State must be outside the SIF directory.'
runtime_dir=$(realpath -m -- "$state_dir/runtime")
[[ $runtime_dir != "$sif_dir" && $runtime_dir != "${sif_dir%/}/"* && $sif_dir != "${runtime_dir%/}/"* ]] || die 'Runtime storage and SIF directory must be separate directory trees.'
[[ $sif_dir != *,* && $state_dir != *,* ]] || die 'Docker bind paths cannot contain commas.'
[[ $cpus =~ ^[0-9]+([.][0-9]+)?$ ]] || die 'Invalid --cpus.'
[[ $cpus =~ [1-9] ]] || die '--cpus must be positive.'
[[ $memory =~ ^[0-9]+[bkmgBKMG]?$ && $shm_size =~ ^[0-9]+[bkmgBKMG]?$ ]] || die 'Invalid memory size.'
[[ $memory =~ [1-9] && $shm_size =~ [1-9] ]] || die 'Memory sizes must be positive.'
if [[ -n $results_dir ]]; then
    results_dir=$(realpath -m -- "$results_dir")
else
    results_dir=$state_dir/results
fi
[[ $results_dir != / && $results_dir != "$sif_dir" && $results_dir != "$sif_dir/"* ]] || die 'Results must be outside the SIF directory.'
[[ $sif_dir != "$results_dir/"* ]] || die 'A writable results mount must not contain the SIF directory.'
[[ $results_dir != *,* ]] || die 'Docker bind paths cannot contain commas.'
[[ $results_dir != "$state_dir" && $results_dir != "$state_dir/runtime" && $results_dir != "$state_dir/runtime/"* ]] || die 'Results must be separate from local runtime storage.'
if [[ $role != standalone ]]; then
    [[ $results_dir != "$state_dir/"* && $state_dir != "$results_dir/"* ]] || die 'Cluster shared results and node-local state must be separate directory trees.'
fi

docker_args=(run --pull never --platform "linux/$host_arch"
    --device /dev/fuse
    --security-opt apparmor=unconfined
    --security-opt seccomp=unconfined
    # Explicit flags prevent Docker client proxy defaults overriding env-file entries.
    --env HTTP_PROXY= --env HTTPS_PROXY= --env ALL_PROXY=
    --env http_proxy= --env https_proxy= --env all_proxy=
    --env 'NO_PROXY=*' --env 'no_proxy=*'
    # The sandbox already runs as mapped root; su cannot setgroups in this user namespace.
    --env APPTAINERENV_SU_TO_USER=false
    --security-opt systempaths=unconfined
    --cpus "$cpus" --memory "$memory" --memory-swap "$memory" --shm-size "$shm_size"
    --env UV_OFFLINE=1 --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1
    --env RAY_USAGE_STATS_ENABLED=0 --env WANDB_MODE=disabled
    --env APPTAINER_TMPDIR=/srv/swe-runtime/apptainer-tmp
    --env APPTAINER_CACHEDIR=/srv/swe-runtime/apptainer-cache
    --env TMPDIR=/srv/swe-runtime/tmp
    --mount "type=bind,src=$sif_dir,dst=/srv/mlperf/sif,readonly"
    --mount "type=bind,src=$results_dir,dst=/srv/swe-results"
    --mount "type=bind,src=$state_dir/runtime,dst=/srv/swe-runtime")

if [[ -n $preflight ]]; then
    [[ $preflight != /* ]] || die '--preflight is a path relative to --sif-dir.'
    preflight_host=$(realpath -e -- "$sif_dir/$preflight")
    [[ $preflight_host == "$sif_dir/"* && -f $preflight_host ]] || die 'Preflight SIF must remain inside --sif-dir.'
    preflight_relative=${preflight_host#"$sif_dir/"}
    docker_args+=(--rm --network none "$image" bash /opt/swe-runner/preflight.sh "/srv/mlperf/sif/$preflight_relative")
else
    if [[ $role != worker ]]; then
        [[ -f $config ]] || die '--config must be an existing generated configuration.'
        config=$(realpath -e -- "$config")
        [[ $config != *,* ]] || die 'Configuration path cannot contain a comma.'
        docker_args+=(--mount "type=bind,src=$config,dst=/etc/swe-runner/site.yaml,readonly")
    fi
    [[ $port =~ ^[0-9]+$ && ${#port} -le 5 ]] || die 'Invalid port.'
    ((10#$port >= 1 && 10#$port <= 65535)) || die 'Invalid port.'
    docker_args+=(--detach --name "$container_name" --restart unless-stopped)
    if [[ $role == standalone ]]; then
        [[ $bind_address =~ ^([0-9]{1,3}[.]){3}[0-9]{1,3}$ && $bind_address != 0.0.0.0 ]] || die 'Specify a host private IPv4 address with --bind-address.'
        docker_args+=(--publish "$bind_address:$port:$container_port")
    else
        docker_args+=(--network host)
    fi
    if [[ -n $env_file ]]; then
        [[ -f $env_file ]] || die 'Environment file does not exist.'
        docker_args+=(--env-file "$(realpath -e -- "$env_file")")
    fi
    if [[ $role != standalone || $service == areal ]]; then
        entry_args=(--role "$role" --service "$service" --cpus "$cpus"
            --ray-port-base "$ray_port_base" --ray-worker-ports "$ray_worker_ports"
            --object-store-memory "$object_store_memory")
        if [[ $role != standalone ]]; then
            entry_args+=(--node-ip "$node_ip" --ray-head-address "$ray_head_address" --port "$port")
        else
            entry_args+=(--port "$container_port")
        fi
        validation_args=("${entry_args[@]}")
        if [[ $role != worker ]]; then
            validation_args+=(--config "$config")
            entry_args+=(--config /etc/swe-runner/site.yaml)
        fi
        command -v python3 >/dev/null || die 'Python 3 is required to validate cluster configuration.'
        python3 "$(dirname -- "${BASH_SOURCE[0]}")/cluster-entrypoint.py" "${validation_args[@]}" --validate-only
        docker_args+=("$image" python /opt/swe-runner/cluster-entrypoint.py "${entry_args[@]}")
    else
        docker_args+=("$image")
    fi
fi

if $dry_run; then
    printf '%q ' docker "${docker_args[@]}"
    printf '\n'
    exit 0
fi

command -v docker >/dev/null || die 'Docker is required.'
daemon_arch=$(normalize_arch "$(docker info --format '{{.Architecture}}')")
[[ $daemon_arch == "$host_arch" ]] || die "Docker daemon architecture $daemon_arch differs from host $host_arch. Use a native Docker daemon."
image_platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image") || die 'Build or load the image first.'
[[ $image_platform == "linux/$host_arch" ]] || die "Runner image architecture is $image_platform; this host requires linux/$host_arch. Build or load the native runner image."
[[ -c /dev/fuse ]] || die '/dev/fuse is missing; complete host setup first.'
if [[ -z $preflight ]] && docker container inspect "$container_name" >/dev/null 2>&1; then
    die "Container $container_name already exists. Inspect it before explicitly removing/replacing it."
fi
mkdir -p -- "$results_dir" "$state_dir/runtime/apptainer-tmp" \
    "$state_dir/runtime/apptainer-cache" "$state_dir/runtime/tmp"
for private_dir in "$state_dir" "$results_dir" "$state_dir/runtime"; do
    permissions=$(stat -c %a -- "$private_dir")
    (( (8#$permissions & 077) == 0 )) || die "State directory $private_dir allows group/other access. Use a private directory or deliberately restrict its permissions before starting."
done
exec docker "${docker_args[@]}"
