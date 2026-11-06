#!/usr/bin/env bash
set -euo pipefail

# Gym writes resolved configuration (including upstream model credentials) into
# episode artifacts. Restrict new files for all service and child processes.
umask 077
exec "$@"
