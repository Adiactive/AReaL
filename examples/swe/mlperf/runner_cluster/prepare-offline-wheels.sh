#!/usr/bin/env bash
# Build-time wheel payload, usable from either native architecture.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: prepare-offline-wheels.sh OUTPUT_DIRECTORY' >&2; exit 1; }
wheel_dir=$1
wheel_name=chardet-5.2.0-py3-none-any.whl
# PyPI 5.2.0 metadata: Python >=3.7, pure Python, no transitive dependencies.
wheel_sha256=e1cf59446890a00105fe7b7912492ea04b6e6f06d4b742b2c788469e34c82970
wheel_url=${SWE_CHARDET_WHEEL_URL:-https://files.pythonhosted.org/packages/38/6f/f5fbc992a329ee4e0f288c1fe0e2ad9485ed064cac731ed2fe47dcc38cbf/chardet-5.2.0-py3-none-any.whl}
mkdir -p -- "$wheel_dir"
wheel_tmp=$(mktemp "$wheel_dir/.download.XXXXXX")
trap 'rm -f -- "$wheel_tmp"' EXIT
curl --fail --location --retry 3 "$wheel_url" --output "$wheel_tmp"
printf '%s  %s\n' "$wheel_sha256" "$wheel_tmp" | sha256sum --check --strict
install -m 0644 "$wheel_tmp" "$wheel_dir/$wheel_name"
printf '%s  %s\n' "$wheel_sha256" "$wheel_name" > "$wheel_dir/SHA256SUMS"
