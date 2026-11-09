#!/usr/bin/env bash
# Build natively on an x86_64 or ARM64 Docker host with package/source access.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=${SWE_RUNNER_IMAGE:-mlperf-swe-runner:local}
integration_dir=${SWE_AREAL_INTEGRATION_DIR:-$script_dir/..}
context_output=
build_args=()

if [[ ${1:-} == --help ]]; then
    cat <<'EOF'
Usage: bash build.sh [--integration-dir PATH] [--prepare-context PATH] [docker build arguments]

SWE_RUNNER_IMAGE sets the resulting local image tag.
--integration-dir supplies examples/swe/mlperf from the AReaL checkout. It defaults
to the parent mlperf directory, or SWE_AREAL_INTEGRATION_DIR when set.
--prepare-context writes a small, self-contained build context to a new directory
without invoking Docker. Copy that directory to an offline/internal build host.
Additional arguments can override documented Docker build arguments or mirrors.
The target platform must match the host and Docker daemon (amd64 or arm64).
Cross-architecture builds and CPU emulation are not supported by this recipe.
This downloads build dependencies; it does not start the runner or download SIFs.
EOF
    exit 0
fi

die() { echo "$*" >&2; exit 1; }
while (($#)); do
    case $1 in
        --integration-dir|--prepare-context)
            [[ $# -ge 2 && -n $2 ]] || die "Missing value for $1"
            if [[ $1 == --integration-dir ]]; then integration_dir=$2; else context_output=$2; fi
            shift 2 ;;
        *) build_args+=("$1"); shift ;;
    esac
done
set -- "${build_args[@]}"
normalize_arch() {
    case $1 in
        amd64|x86_64) printf '%s' amd64 ;;
        arm64|aarch64) printf '%s' arm64 ;;
        *) die "Unsupported CPU architecture: $1. Use x86_64 or ARM64." ;;
    esac
}
host_arch=$(normalize_arch "$(uname -m)")
platform="linux/$host_arch"
[[ -z ${DOCKER_DEFAULT_PLATFORM:-} || $DOCKER_DEFAULT_PLATFORM == "$platform" ]] ||
    die "DOCKER_DEFAULT_PLATFORM must be $platform for a native build."
expect_platform=false
for argument in "$@"; do
    if $expect_platform; then
        [[ $argument == "$platform" ]] || die "Build platform must be $platform; cross-architecture builds are unsupported."
        expect_platform=false
    else
        case $argument in
            --platform) expect_platform=true ;;
            --platform=*) [[ ${argument#--platform=} == "$platform" ]] || die "Build platform must be $platform; cross-architecture builds are unsupported." ;;
        esac
    fi
done
$expect_platform && die 'Missing value for --platform.'

adapter_files=(gym_model.py runner_bridge.py serve.py configure_runner.py)
for file in "${adapter_files[@]}"; do
    [[ -f $integration_dir/$file ]] || die "Missing $integration_dir/$file; use --integration-dir with the AReaL examples/swe/mlperf directory."
done
if [[ -n $context_output ]]; then
    [[ ! -e $context_output ]] || die '--prepare-context requires a new directory.'
    mkdir -m 700 -- "$context_output"
    context_dir=$(realpath -e -- "$context_output")
else
    context_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlperf-swe-build.XXXXXXXX")
    trap 'rm -rf -- "$context_dir"' EXIT
fi
mkdir -m 700 -- "$context_dir/runner" "$context_dir/adapter"
cp -- "$script_dir/Dockerfile" "$context_dir/Dockerfile"
# Stage explicit source files only: never include local configs, credentials,
# SIFs, logs, archives, or the rest of the training checkout in the build context.
runner_files=(build-tls.sh add-apt-repository.py runtime-pins.env prepare-runtime.sh
    patch-poetry-markers.py patch-offline-eval.py patch-sandbox-isolation.py prepare-offline-wheels.sh
    entrypoint.sh preflight.sh configure.py smoke-run.py cluster-entrypoint.py check-cluster.py)
for file in "${runner_files[@]}"; do
    [[ -f $script_dir/$file ]] || die "Runner bundle is incomplete: missing $file"
    cp -- "$script_dir/$file" "$context_dir/runner/$file"
done
for file in "${adapter_files[@]}"; do
    cp -- "$integration_dir/$file" "$context_dir/adapter/$file"
done
if [[ -n $context_output ]]; then
    printf 'Prepared unified build context: %s\n' "$context_dir"
    exit 0
fi

command -v docker >/dev/null || die 'Docker is required.'
daemon_arch=$(normalize_arch "$(docker info --format '{{.Architecture}}')")
[[ $daemon_arch == "$host_arch" ]] || die "Docker daemon architecture $daemon_arch differs from host $host_arch. Use a native Docker daemon."
docker build --platform "$platform" --tag "$image" "$@" "$context_dir"
