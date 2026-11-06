ARG BASE_IMAGE=registry-1.docker.io/library/ubuntu:22.04
FROM ${BASE_IMAGE}

COPY runner/build-tls.sh runner/add-apt-repository.py /opt/swe-runner/
SHELL ["/bin/bash", "/opt/swe-runner/build-tls.sh", "/bin/bash", "-euo", "pipefail", "-c"]

ARG TARGETARCH
ARG GIT_SSL_NO_VERIFY=1
ARG SWE_BUILD_INSECURE_TLS=0
ARG SWE_BUILD_INSECURE_HOSTS=
ARG APPTAINER_APT_REPOSITORY=ppa:apptainer/ppa
ARG APPTAINER_VERSION_OVERRIDE=
ARG APPTAINER_SESSIONDIR_MAX_SIZE_MB=4096
ARG GITHUB_BASE_URL=https://github.com
ARG GITHUB_DOWNLOAD_BASE_URL=https://github.com
ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG CONDA_CHANNEL_ALIAS=https://conda.anaconda.org
ARG SWE_CHARDET_WHEEL_URL=

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_CACHE_DIR=/opt/build-cache/uv \
    UV_TORCH_BACKEND=cpu \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_SYSTEM_GIT_CLIENT=true \
    INSTALL_DOCKER=0 \
    INSTALL_PLAYWRIGHT=0

COPY runner/runtime-pins.env /opt/swe-runner/runtime-pins.env

RUN case "${TARGETARCH}:$(uname -m):$(dpkg --print-architecture)" in \
        amd64:x86_64:amd64|arm64:aarch64:arm64) ;; \
        *) echo 'Use a native amd64 or arm64 build host and matching base image.' >&2; exit 1 ;; \
    esac \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash build-essential bzip2 ca-certificates curl file fuse3 git gnupg \
        libffi-dev libgl1 libglib2.0-0 libssl-dev libxml2-dev libxslt1-dev \
        locales lsb-release patch pkg-config procps python3 python3-venv \
        rsync software-properties-common squashfs-tools sudo tini tmux \
        tzdata unzip util-linux wget xz-utils zlib1g-dev \
    && python3 /opt/swe-runner/add-apt-repository.py -y "${APPTAINER_APT_REPOSITORY}" \
    && apt-get update \
    && source /opt/swe-runner/runtime-pins.env \
    && apt-get install -y --no-install-recommends \
        "apptainer=${APPTAINER_VERSION_OVERRIDE:-${APPTAINER_PACKAGE_VERSION}}" \
    && ln -sf "$(command -v apptainer)" /usr/local/bin/singularity \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# The outer Docker process supplies the namespace/FUSE permissions. Do not
# install apptainer-suid or turn on setuid in this image.
RUN python3 - "${APPTAINER_SESSIONDIR_MAX_SIZE_MB}" <<'PY'
import pathlib
import re
import sys

limit = int(sys.argv[1])
if limit < 64:
    raise SystemExit("APPTAINER_SESSIONDIR_MAX_SIZE_MB must be at least 64")
path = pathlib.Path("/etc/apptainer/apptainer.conf")
text = path.read_text()
for key, value in (("allow setuid", "no"), ("sessiondir max size", str(limit))):
    text, count = re.subn(rf"(?m)^{key}\s*=.*$", f"{key} = {value}", text)
    if count != 1:
        raise SystemExit(f"Expected exactly one Apptainer config setting: {key}")
path.write_text(text)
PY

RUN source /opt/swe-runner/runtime-pins.env \
    && case "${TARGETARCH}" in \
        amd64) uv_machine=x86_64; uv_sha256=${UV_ARCHIVE_SHA256_AMD64} ;; \
        arm64) uv_machine=aarch64; uv_sha256=${UV_ARCHIVE_SHA256_ARM64} ;; \
        *) echo "Unsupported target architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl --fail --location --retry 3 \
        "${GITHUB_DOWNLOAD_BASE_URL%/}/astral-sh/uv/releases/download/${UV_VERSION}/uv-${uv_machine}-unknown-linux-gnu.tar.gz" \
        --output /tmp/uv.tar.gz \
    && printf '%s  %s\n' "${uv_sha256}" /tmp/uv.tar.gz | sha256sum --check --strict \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install -m 0755 "/tmp/uv-${uv_machine}-unknown-linux-gnu/uv" /usr/local/bin/uv \
    && install -m 0755 "/tmp/uv-${uv_machine}-unknown-linux-gnu/uvx" /usr/local/bin/uvx \
    && rm -rf /tmp/uv.tar.gz "/tmp/uv-${uv_machine}-unknown-linux-gnu" \
    && uv python install "${PYTHON_VERSION}"

COPY runner/prepare-runtime.sh runner/patch-poetry-markers.py runner/patch-offline-eval.py runner/prepare-offline-wheels.sh /opt/swe-runner/
RUN SWE_RUNNER_BUILD=1 GITHUB_BASE_URL="${GITHUB_BASE_URL}" \
    GITHUB_DOWNLOAD_BASE_URL="${GITHUB_DOWNLOAD_BASE_URL}" \
    UV_INDEX_URL="${PYPI_INDEX_URL}" PIP_INDEX_URL="${PYPI_INDEX_URL}" \
    CONDA_CHANNEL_ALIAS="${CONDA_CHANNEL_ALIAS}" \
    bash /opt/swe-runner/prepare-runtime.sh

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
COPY runner/ /opt/swe-runner/

ENV PATH=/opt/nemo-gym/.venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/nemo-gym \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    UV_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    APPTAINERENV_LITELLM_LOCAL_MODEL_COST_MAP=True \
    APPTAINERENV_HF_HUB_OFFLINE=1 \
    APPTAINERENV_HF_DATASETS_OFFLINE=1 \
    APPTAINERENV_HF_HUB_DISABLE_TELEMETRY=1 \
    APPTAINER_CACHEDIR=/srv/swe-runtime/apptainer-cache \
    APPTAINER_TMPDIR=/srv/swe-runtime/apptainer-tmp \
    TMPDIR=/srv/swe-runtime/tmp

WORKDIR /opt/nemo-gym
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/bin/bash", "/opt/swe-runner/entrypoint.sh"]
CMD ["gym", "env", "start", "+config_paths=[/etc/swe-runner/site.yaml]"]

# Keep adapter updates last so they reuse the dependency and runtime layers.
COPY adapter/gym_model.py /opt/nemo-gym/responses_api_models/vllm_model/areal_app.py
COPY adapter/runner_bridge.py adapter/serve.py adapter/configure_runner.py /opt/areal-gym/

RUN /opt/nemo-gym/.venv/bin/python -c 'import fastapi, httpx, uvicorn' \
    && /opt/nemo-gym/.venv/bin/python -m compileall -q /opt/areal-gym \
    && sha256sum /opt/areal-gym/*.py \
        /opt/nemo-gym/responses_api_models/vllm_model/areal_app.py \
        > /opt/swe-runner/provenance/areal-adapter.sha256
