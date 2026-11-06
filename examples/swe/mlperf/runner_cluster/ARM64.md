# Ubuntu 22.04 ARM64 installation

For a two-node deployment with the unified AReaL image, use [CLUSTER.md](CLUSTER.md).

Use this bundle on a **64-bit ARM Ubuntu 22.04 host**: `uname -m` must report `aarch64`,
and `dpkg --print-architecture` must report `arm64`. The same scripts also support
x86_64; no separate source tree is needed. AReaL and model inference remain on your
other machines.

Your mounted task images must be the **aarch64 SIF collection**. Already downloaded
x86_64 SIFs cannot run natively here. MLCommons lists separate architecture bundles in
its
[benchmark instructions](https://github.com/mlcommons/training/tree/master/llm_post_training).
The ARM collection metadata is
[`r2egym-sifs-aarch64.uri`](https://training.mlcommons-storage.org/metadata/r2egym-sifs-aarch64.uri).
The installation does not download these images.

## Install and build

Copy the entire `runner_cluster` directory onto your ARM host, then:

```bash
cd ~/runner_cluster
uname -m
dpkg --print-architecture
sudo bash install-host.sh --load-fuse
sudo bash build.sh --integration-dir /path/to/AReaL/examples/swe/mlperf
```

The build selects `linux/arm64`, ARM64 uv, ARM64 Miniforge, ARM64 jq, and native Python
environments. It keeps the same pinned benchmark code and patches as the x86 version. It
installs Apptainer inside the runner's Docker image.

Building requires source/package access, either directly or through your internal
mirrors. If the sandbox host is disconnected, build on another ARM64 host and transfer
the saved image as described in the [full guide](README.md). The deployed service uses
your mounted SIFs and internal model endpoint.

If your build proxy presents an untrusted certificate, the optional build-only bypass
is:

```bash
sudo bash build.sh --integration-dir /path/to/AReaL/examples/swe/mlperf \
  --build-arg SWE_BUILD_INSECURE_TLS=1
```

See the [TLS build option](README.md#build-behind-a-proxy-with-an-untrusted-certificate)
for its scope and additional mirror hosts. Package signatures and pinned checksums are
still verified.

## Configure and check one SIF

Replace these values with your actual paths and addresses:

```bash
export SWE_SIF_DIR=/absolute/path/to/mounted/aarch64/images
export SWE_STATE_DIR=/srv/mlperf-swe
export SWE_BIND_ADDRESS=YOUR_SANDBOX_PRIVATE_IPV4
export SWE_MODEL_BASE_URL=http://YOUR_MODEL_HOST:PORT/v1
export SWE_MODEL_NAME=YOUR_SERVED_MODEL_NAME

python3 configure.py \
  --output site.yaml \
  --model-base-url "$SWE_MODEL_BASE_URL" \
  --model-name "$SWE_MODEL_NAME" \
  --concurrency 4

# Replace task.sif with an existing file path relative to SWE_SIF_DIR.
sudo bash start.sh \
  --sif-dir "$SWE_SIF_DIR" \
  --state-dir "$SWE_STATE_DIR" \
  --preflight task.sif
```

The preflight runs without network access or a model call. It checks the selected SIF's
architecture, nested Apptainer execution, writable state, and imports in the mounted
agent/evaluator environments. It checks one selected SIF, not the whole image catalog.

## Start the service

```bash
sudo bash start.sh \
  --sif-dir "$SWE_SIF_DIR" \
  --state-dir "$SWE_STATE_DIR" \
  --config "$PWD/site.yaml" \
  --bind-address "$SWE_BIND_ADDRESS"
```

If the model requires a credential, prepare `model.env` and add
`--env-file "$PWD/model.env"` as described in the [full guide](README.md). That guide
also covers a real task smoke test, logs, resource limits, and the AReaL integration
boundary. Keep the initial concurrency at four until you measure your workloads; ARM
support does not establish a throughput estimate.

**Validation:** script tests and source checks run on the development machine. An ARM64
Docker build and an actual SIF episode still need to pass on your host. This archive
contains installation source files, not a prebuilt Docker image.
