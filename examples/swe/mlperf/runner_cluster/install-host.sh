#!/usr/bin/env bash
# Prepare a dedicated Ubuntu host; the runner's Python/Apptainer stack is in Docker.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: sudo bash install-host.sh [--load-fuse]

Prepare Ubuntu 22.04 x86_64 or ARM64 for the local MLPerf SWE runner image:
  - Install host python3, ca-certificates, curl, kmod, and util-linux using apt.
  - Keep an existing, working Docker installation unchanged.
  - If Docker is absent, install Docker Engine and buildx from Docker's official
    Ubuntu apt repository. Refuse conflicting/partial installs and existing
    Docker repository configuration instead of deleting or replacing them.
  - Check Docker, buildx, and /dev/fuse without pulling or running an image.

--load-fuse    Load the host fuse kernel module with modprobe if /dev/fuse is absent.
--help        Show this help without changing anything.

This is an ONLINE provisioning step. For an air-gapped host, have your administrator
install these packages from internal mirrors and use a prebuilt runner image.
The script does not change Docker groups, firewall rules, sysctls, or daemon settings;
it never restarts an existing Docker daemon. Use sudo for later Docker commands if
required by your host's existing access policy.
USAGE
}

die() { echo "$*" >&2; exit 1; }
load_fuse=false
while (($#)); do
    case $1 in
        --help) usage; exit 0 ;;
        --load-fuse) load_fuse=true; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ $EUID -eq 0 ]] || die 'Run this host provisioning script with sudo.'
case $(uname -m) in
    x86_64) host_arch=amd64 ;;
    aarch64|arm64) host_arch=arm64 ;;
    *) die 'This recipe requires an x86_64 or ARM64 host.' ;;
esac
# shellcheck source=/dev/null
source /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 22.04 ]] || die 'This recipe requires Ubuntu 22.04.'
[[ $(dpkg --print-architecture) == "$host_arch" ]] || die "Expected native Ubuntu $host_arch packages."

existing_docker=false
if command -v docker >/dev/null 2>&1; then
    existing_docker=true
    docker info >/dev/null || die 'Existing Docker is inaccessible. Resolve it manually; this script will not restart or replace it.'
    docker buildx version >/dev/null || die 'Existing Docker has no working buildx. Install its matching plugin before rerunning.'
else
    conflicts=()
    for package in docker.io docker-doc docker-compose docker-compose-v2 podman-docker \
        containerd runc docker-ce docker-ce-cli containerd.io docker-buildx-plugin; do
        if [[ $(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true) == installed ]]; then
            conflicts+=("$package")
        fi
    done
    ((${#conflicts[@]} == 0)) || die "Conflicting or partial Docker installation found: ${conflicts[*]}. Review it manually; no packages were removed."
    if grep -Rqs 'download.docker.com' /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null ||
        [[ -e /etc/apt/keyrings/docker.asc || -e /etc/apt/keyrings/docker.gpg || -e /etc/apt/sources.list.d/docker.sources ]]; then
        die 'Existing Docker apt configuration found. Review it manually; this script will not overwrite it.'
    fi
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 ca-certificates curl kmod util-linux

if ! $existing_docker; then
    key_file=$(mktemp)
    trap 'rm -f -- "$key_file"' EXIT
    curl --fail --show-error --silent --location --proto '=https' --tlsv1.2 \
        https://download.docker.com/linux/ubuntu/gpg --output "$key_file"
    install -m 0755 -d /etc/apt/keyrings
    install -m 0644 "$key_file" /etc/apt/keyrings/docker.asc
    cat > /etc/apt/sources.list.d/docker.sources <<SOURCES
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: jammy
Components: stable
Architectures: $host_arch
Signed-By: /etc/apt/keyrings/docker.asc
SOURCES
    apt-get update
    apt-get install -y --no-install-recommends \
        docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

docker info >/dev/null || die 'Docker is installed but unavailable. Inspect the daemon before proceeding.'
docker buildx version
if [[ ! -c /dev/fuse ]] && $load_fuse; then
    modprobe fuse
fi
[[ -c /dev/fuse ]] || die '/dev/fuse is missing. Rerun with --load-fuse to request modprobe fuse, or ask the host administrator to expose it.'
if [[ -r /proc/sys/user/max_user_namespaces ]] && [[ $(</proc/sys/user/max_user_namespaces) == 0 ]]; then
    die 'User namespaces are disabled on this host. Ask the administrator to review this prerequisite; no sysctl was changed.'
fi
python3 --version
printf '%s\n' 'Host prerequisites are ready. Build or load the runner image, then run start.sh --preflight against a mounted SIF.'
