#!/usr/bin/env bash
# Docker RUN wrapper: temporary TLS settings are removed before a layer is saved.
set -euo pipefail

case ${SWE_BUILD_INSECURE_TLS:-0} in
    0) exec "$@" ;;
    1) ;;
    *) echo 'SWE_BUILD_INSECURE_TLS must be 0 or 1.' >&2; exit 2 ;;
esac

tls_dir=$(mktemp -d /tmp/swe-build-tls.XXXXXX)
trap 'rm -rf -- "$tls_dir"' EXIT
printf 'insecure\nproxy-insecure\n' > "$tls_dir/.curlrc"
printf 'check_certificate = off\n' > "$tls_dir/wgetrc"
cat > "$tls_dir/apt.conf" <<'APT'
Acquire::https::Verify-Peer "false";
Acquire::https::Verify-Host "false";
APT

# uv 0.9.30 and pip both split their host lists on spaces. Include GitHub
# redirect hosts used by standalone Python and release-asset downloads.
hosts='pypi.org files.pythonhosted.org pypi.python.org github.com api.github.com raw.githubusercontent.com codeload.github.com objects.githubusercontent.com release-assets.githubusercontent.com github-releases.githubusercontent.com astral.sh downloads.python.org download.pytorch.org'
hosts+=" ${SWE_BUILD_INSECURE_HOSTS:-}"
export CURL_HOME="$tls_dir" WGETRC="$tls_dir/wgetrc" APT_CONFIG="$tls_dir/apt.conf"
export GIT_SSL_NO_VERIFY=true
export UV_INSECURE_HOST="$hosts" PIP_TRUSTED_HOST="$hosts"
export CONDA_SSL_VERIFY=false MAMBA_SSL_VERIFY=false
# Poetry treats certificate environment values as paths, even the string
# "false". A TOML boolean is required to disable verification.
export POETRY_CONFIG_DIR="$tls_dir/poetry"
mkdir -p "$POETRY_CONFIG_DIR"
# Poetry matches certificate settings by host. PyPI metadata and wheels are
# served by different hosts, so register the artifact host separately.
cat > "$POETRY_CONFIG_DIR/config.toml" <<'POETRY'
[certificates.PyPI]
cert = false

[repositories.PyPIArtifacts]
url = "https://files.pythonhosted.org"

[certificates.PyPIArtifacts]
cert = false
POETRY
unset POETRY_CERTIFICATES_PYPI_CERT POETRY_CERTIFICATES_PYPI_VERIFY
export npm_config_strict_ssl=false NPM_CONFIG_STRICT_SSL=false
export NODE_TLS_REJECT_UNAUTHORIZED=0

echo 'Build TLS certificate verification disabled for supported download tools.' >&2
"$@"
