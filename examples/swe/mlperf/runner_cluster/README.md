# MLPerf SWE runner: standalone and AReaL clusters

This directory builds one image containing the MLPerf reference SWE environment and the
AReaL bridge, for **x86_64 or ARM64 (aarch64) hosts** with Docker. It supports a
standalone host or a Ray cluster. AReaL training and model inference run elsewhere. Your
**matching-architecture** SIF collection is mounted read-only. No model or SIF download
is performed by these helpers.

Build the unified CPU image once, then select standalone, head, or worker mode. Building
needs access to source/package repositories or their internal mirrors. For a host
without internet access, build on another machine of the same CPU architecture that you
control and transfer the resulting Docker image. Serving uses your own infrastructure.
See [ARM64 quickstart](ARM64.md) for the ARM host commands.

For a head and one or more workers, follow the
[two-node setup, testing, and service guide](CLUSTER.md). The instructions below retain
standalone fixed-model qualification. AReaL mode uses the bundled bridge and per-episode
inference sessions; it does not require a fixed model URL.

## Files and prerequisites

The core source pins match the `llm_post_training/RL` gitlink at `mlcommons/training`
commit `e309244f7f81ecb00da434cae98355a73fc50ff0`: NeMo-RL
`f987c0596af2bd6cfb9958e768deb8df951d7504`, Gym
`610a08ab5fe9f8f5fb5fff36b170429ea67f0f92`, OpenHands
`0d766ad06b2be64a42e6f0175b9ebcc4a06599d9`, and R2E-Gym
`6823e64f94ae645f5265c03af0eb2e8523530a0d`. The build applies the reference Gym and R2E
patches. Matching these sources does not make this CPU-only Docker packaging identical
to the full GPU reference image.

The build also disables Apptainer's implicit current-working-directory bind for agent
and evaluation containers. This prevents a writable parent mount from hiding the
explicit read-only OpenHands dependency mounts. Preflight runs from the service working
directory and checks both dependency aliases and the writable episode subdirectories.
Rebuild and recreate workers to apply this protection; restarting a container does not
restore shared dependencies that an earlier task deleted.

The evaluator has two Python environments inside Apptainer: its mounted Python 3.12
harness and the task image's `/testbed/.venv`. The upstream R2E setup runs
`uv pip install chardet` in the task environment. The build downloads the
checksum-pinned, Python 3.7+ compatible `chardet==5.2.0` wheel into
`/opt/swe-runner/offline-wheels`. A guarded build-time Gym patch exposes that directory
read-only to **R2E evaluation only**, and supplies `UV_NO_INDEX=true`, `UV_FIND_LINKS`,
and `UV_OFFLINE=true` through Apptainer `--env`. The upstream installation command,
tests, and grading logic are unchanged. No extra host mount or modified SIF is required.
The agent receives neither this mount nor these additional uv settings.

The offline preflight installs/imports `chardet` in a disposable task environment with
an empty uv cache. Run it on an R2E SIF that provides `/testbed/.venv`. This wheel
directory supplies this dependency only; it is not a general package mirror. An internal
build mirror may be supplied with Docker build argument `SWE_CHARDET_WHEEL_URL`; the
wheel must match the same SHA-256.

After updating these build files, build a new tag to preserve the working image:

```bash
SWE_RUNNER_IMAGE=mlperf-swe-runner:offline-eval bash build.sh \
  --integration-dir /path/to/AReaL/examples/swe/mlperf
```

Retain any build proxy/TLS arguments needed at your site. Then run `start.sh` with
`--image mlperf-swe-runner:offline-eval --preflight TASK.sif` and your usual SIF/state
arguments. Once it passes, recreate the service with the same image option and rerun one
task. Updating files alone does not update a running image.

| File                                                   | Purpose                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `install-host.sh`                                      | Prepare a fresh Ubuntu host, preserving an existing Docker installation           |
| `Dockerfile`, `prepare-runtime.sh`, `runtime-pins.env` | Build the pinned CPU runner, benchmark patches, and harness environments          |
| `build.sh`                                             | Stage the runner and AReaL adapter sources, then build one image                  |
| `configure.py`                                         | Generate the Gym configuration without installing Python dependencies on the host |
| `configure.sh`, `configure.env.example`                | Generate either service mode's head YAML from a host env file                     |
| `start.sh`                                             | Mount images/state and start either the service or an offline preflight           |
| `launch-config.py`                                     | Read generated launch settings using the host's standard-library Python           |
| `preflight.sh`                                         | Check nested Apptainer execution and independent writable overlays                |
| `smoke-run.py`                                         | Submit one existing MLPerf JSONL task and save its result                         |
| `cluster-entrypoint.py`                                | Supervise a dedicated Ray head/worker and head services                           |
| `check-cluster.py`                                     | Pin a diagnostic to each node and check callbacks/shared storage                  |
| `CLUSTER.md`                                           | Prepare, deploy, test, and operate the two-node cluster                           |

You need SIFs matching the sandbox host and a matching benchmark JSONL file. Standalone
testing also needs a running internal model endpoint, reachable from Docker on the
sandbox host. AReaL/other clients must be able to reach the sandbox host's chosen
private TCP port. Do not use `localhost` as the remote model hostname: inside the runner
it refers to the runner container.

| Sandbox host (`uname -m`) | Docker/Ubuntu architecture | Required SIF collection |
| ------------------------- | -------------------------- | ----------------------- |
| `x86_64`                  | `amd64`                    | `r2egym-sifs-x86_64`    |
| `aarch64`                 | `arm64`                    | `r2egym-sifs-aarch64`   |

The host architecture selects Docker packages and the uv, Miniforge, jq, and Python
binaries. The SIF architecture must match because the runner binds these Python
environments into the task containers. An x86 SIF collection cannot be reused natively
on ARM. These helpers do not configure QEMU emulation or cross-architecture builds. The
model server can use a different CPU architecture because it communicates with the
runner over HTTP.

The intended host has 64 CPUs and 512 GB RAM. The launcher initially caps the whole
service at 48 CPUs and 400 GiB, with four admitted episodes. These are starting limits,
not measured throughput guarantees. Each episode starts an agent and an evaluator
container. Keep the SIF collection on its existing mount; the 1 TB local disk is for
dependencies, build layers, results, and scratch.

## 1. Copy the bundle and prepare the host

Copy this directory to the sandbox host, for example as `~/runner_cluster`. Building
additionally takes the adapter source from `examples/swe/mlperf` in the AReaL checkout.
Neither serving nor worker execution requires the training package.

```bash
cd ~/runner_cluster
bash install-host.sh --help
sudo bash install-host.sh --load-fuse
```

The host installer checks Ubuntu 22.04 and either x86_64/amd64 or aarch64/arm64. It
preserves an existing Docker installation and stops if it finds a conflict or an
unavailable daemon. It does not uninstall another container runtime, change firewall
rules, or grant users Docker-group access. `--load-fuse` explicitly allows loading the
kernel FUSE module if `/dev/fuse` is missing.

Docker commands below use `sudo`; omit it if your account already has appropriate Docker
access. Do not run the host installer inside a container.

## 2. Build the CPU runner image

```bash
sudo bash build.sh --integration-dir /path/to/AReaL/examples/swe/mlperf
```

The default local tag is `mlperf-swe-runner:local`. There is one Dockerfile and one
final image; the former integration-only Dockerfile is no longer used.
`--integration-dir` defaults to the parent `mlperf` directory, or the directory set by
`SWE_AREAL_INTEGRATION_DIR`. This is a substantial build: it prepares OpenHands and all
evaluator environments expected by the pinned Gym startup, as well as Gym's per-server
virtual environments. It applies the benchmark's Gym and R2E patches. It does not
install the NeMo-RL training stack or start a GPU vLLM server. The Gym module named
`vllm_model` is an HTTP client wrapper around your remote model endpoint. The same image
also includes the AReaL model adapter, bridge, and supervisor. Adapter source is copied
after dependency installation so changes to it can reuse cached dependency layers.

`build.sh` prepares a small build context containing only required scripts and the four
adapter Python files. To build on another host without copying the training checkout,
export that context first:

```bash
bash build.sh --integration-dir /path/to/AReaL/examples/swe/mlperf \
  --prepare-context /tmp/mlperf-swe-context
# After transferring the prepared directory to a native build host:
docker build -t mlperf-swe-runner:local /path/to/mlperf-swe-context
```

The output directory must not already exist. Do not run `docker build .` on this source
directory: the unified Dockerfile expects the staged `runner/` and `adapter/` inputs. No
private site configuration or image data is staged.

On ARM64, the same command builds `linux/arm64`; on x86_64 it builds `linux/amd64`.
Build on the native target architecture with a local Docker daemon. Image archives are
architecture-specific: transfer an ARM64 build to an ARM64 sandbox host, or an amd64
build to an x86_64 host.

The source revisions are recorded in `runtime-pins.env`. Build provenance and
installed-package information are saved in the image at
`/opt/swe-runner/runtime-manifest.json`. Preserve the built image/digest for repeatable
deployments: pinned Git revisions alone do not freeze every OS, Conda, or package
repository used by the upstream setup scripts.

The build also applies a narrow Poetry marker fix: conjunctions evaluate OS guards
before `platform_release` comparisons. This avoids parsing vendor Linux kernel strings
while excluding macOS-only locked dependencies. It preserves the OpenHands lock and
kernel information. The applied source patch and Poetry core version are recorded under
`/opt/swe-runner/provenance/poetry-core-markers.*`. The upstream `packaging==26.0`
downgrade is omitted in the Conda build-tool environment, where it conflicts with
current Conda/virtualenv requirements; the reference pin in OpenHands' separate virtual
environment is retained.

Build arguments can be passed through `build.sh`; consult `Dockerfile` for supported
source and package settings. Upstream setup scripts also download dependencies. For a
fully isolated runtime host, transferring a built image is simpler than configuring
every upstream download to use an internal mirror.

On your build host:

```bash
sudo docker save -o mlperf-swe-runner.tar mlperf-swe-runner:local
sha256sum mlperf-swe-runner.tar > mlperf-swe-runner.tar.sha256
```

Transfer both files through your internal network, then on the sandbox host:

```bash
sha256sum -c mlperf-swe-runner.tar.sha256
sudo docker load -i mlperf-swe-runner.tar
```

Reserve disk space for the build cache and image archive. Do not copy the whole SIF
bundle into Docker's build context. After a build, inspect usage with
`sudo docker system df`; any cleanup of shared Docker storage should be a separate
deliberate operation.

### Build behind a proxy with an untrusted certificate

If a build fails with `CERTIFICATE_VERIFY_FAILED` and you intentionally want to skip TLS
certificate checks for this internal build, use:

```bash
sudo bash build.sh --integration-dir /path/to/AReaL/examples/swe/mlperf \
  --build-arg SWE_BUILD_INSECURE_TLS=1
```

This opt-in flag covers the Launchpad client used by `add-apt-repository`, APT HTTPS,
curl, wget, Git, Conda/Mamba, PyPI through pip/uv/Poetry, and Node/npm downloads. It
disables verification of download-server certificates, so use it only on the network
where you intend to trust those downloads. APT repository signatures and the pinned
SHA256 checks remain enabled. It cannot fix HTTP 407 proxy authentication failures or
proxy block pages.

For pip and uv, the bypass covers the standard upstream download hosts in
`build-tls.sh`. Add any internal mirror or additional redirect host explicitly:

```bash
sudo bash build.sh \
  --integration-dir /path/to/AReaL/examples/swe/mlperf \
  --build-arg SWE_BUILD_INSECURE_TLS=1 \
  --build-arg 'SWE_BUILD_INSECURE_HOSTS=packages.internal mirror.internal:8443'
```

The helper uses temporary configuration files and per-build-process environment
variables. It does not disable TLS verification in the running Gym/OpenHands service or
modify your host's TLS settings. Docker registry pulls and the BuildKit frontend run
outside this helper; their TLS configuration is separate. Arbitrary Python download code
and custom Poetry repositories may need their own configuration. The manifest records
whether the build used the bypass.

For Poetry, the helper writes `cert = false` as a Boolean in a temporary `config.toml`.
Do not replace this with `POETRY_CERTIFICATES_PYPI_CERT=false`: Poetry 2.5.1 interprets
that environment value as a certificate filename.

## 3. Set your paths and model endpoint

Replace the following example values with your own paths and internal addresses:

```bash
# Use the aarch64 collection on ARM64, or x86_64 on an x86 host.
export SWE_SIF_DIR=/absolute/path/to/mounted/aarch64/images
export SWE_STATE_DIR=/srv/mlperf-swe
export SWE_BIND_ADDRESS=YOUR_SANDBOX_PRIVATE_IPV4
export SWE_MODEL_BASE_URL=http://YOUR_MODEL_HOST:PORT/v1
export SWE_MODEL_NAME=YOUR_SERVED_MODEL_NAME
export SWE_JSONL=/absolute/path/to/benchmark_r2e_gym_easy_train.jsonl
```

The SIF mount must be readable by the Docker worker. With NFS root squashing, verify
file/directory permissions rather than changing the shared dataset.

Generate the standalone fixed-model configuration with one `configure.py` invocation:

```bash
python3 configure.py \
  --output site.yaml \
  --model-base-url "$SWE_MODEL_BASE_URL" \
  --model-name "$SWE_MODEL_NAME" \
  --concurrency 4
```

The generated file is JSON, which Gym's YAML loader accepts. It uses fixed **container
paths**: `/srv/mlperf/sif` for images and `/srv/swe-results` for results. `start.sh`
maps your host paths to those locations. Do not substitute host paths for these
container paths.

For a cluster in either service mode, keep configuration variables in one env file. From
the deployment directory, create and edit it once, then generate the complete head YAML
with one command:

```bash
umask 077
cp -n runner/configure.env.example configure.env
${EDITOR:-vi} configure.env
bash runner/configure.sh --env-file configure.env
```

Keep an existing `configure.env` instead of replacing it. The wrapper reads trusted Bash
assignments and runs `configure.py` inside the image. Quotes and variable expansion
work; no `export` is needed. `SWE_SERVICE=areal` is the default; that mode calls the
adapter helper internally and requires no fixed model URL/name. To test a fixed model
instead, set these values in the file before generating it:

```bash
SWE_SERVICE=standalone
SWE_CONFIG_FILE="${SWE_SERVICE}-site.yaml"
SWE_MODEL_BASE_URL=http://YOUR_MODEL_HOST:PORT/v1
SWE_MODEL_NAME=YOUR_SERVED_MODEL_NAME
```

Set `SWE_SIF_DIR`, `SWE_STATE_DIR`, and `SWE_RESULTS_DIR` in the env file along with the
head address. Set the name and CPU/RAM budgets for your host in the same file:

```bash
SWE_CONTAINER_NAME=mlperf-swe-runner
SWE_CPUS=28
SWE_MEMORY=112g
SWE_SHM_SIZE=8g
SWE_PREFLIGHT_SIF=
```

The wrapper writes these settings, host paths, image, Ray worker ports, and object-store
size into the generated file's `swe_runner` section. After generating the YAML, start
the head from the deployment directory with:

```bash
bash runner/start.sh --config config/areal-site.yaml
```

Use the selected output filename if it differs from `areal-site.yaml`. The launcher
reads the service mode, head address, and API/Ray ports from that same config. Copy it
to each worker and add `--role worker --node-ip WORKER_IPV4` to the launch command; the
worker's host launcher reads it without passing it into the Ray container. Use the same
node-local state path on both hosts, or override `--state-dir` on the worker. The
configuration env file is separate from the model credentials file supplied to
`start.sh --env-file`.

`SWE_CONTAINER_NAME` defaults to `mlperf-swe-runner`; `--name NAME` overrides it. All
launch settings can be stored in the env file and generated YAML. Explicit launch flags
override saved values. `--no-preflight` starts the service when a preflight is saved,
`--no-dry-run` enables execution when dry-run is saved, and `--env-file ''` clears a
saved credentials path. Offline preflight uses a disposable Docker-generated name and
ignores service names, roles, and credentials.

See [the env-file workflow](CLUSTER.md#generate-from-a-configuration-env-file) and
[complete input reference](CLUSTER.md#configuration-inputs-and-environment-variables),
including the single-head port `18080` setup. The direct `configure.py` CLI remains
available. Its `--port`, `--gym-port`, `--registry-port`, and `--model-port` configure
all four AReaL service ports; there is no need to repeat them at head startup. Explicit
head network arguments must agree with the generated configuration. Image, path, and
resource options remain available as launch overrides.

Regenerate older service-only YAML to include `swe_runner` for this short launch
command. Existing configs still work with explicit launch paths and options. Updating
`configure.py`, `configure.sh`, `start.sh`, and `launch-config.py` requires no image
rebuild for this configuration workflow or config-based preflight.

For the original `start.sh --role standalone` deployment below, the generated API port
defaults to `8080`. That launcher's `--port` changes the host's published port while
forwarding to the API port in the config; it does not change the service's listening
port.

The Python generator takes CLI arguments; the shell wrapper maps env-file variables to
those arguments. Variables such as `SWE_MODEL_BASE_URL` in these examples are expanded
by the shell into those arguments. The direct standalone `configure.py` example above
does not include host launch metadata and keeps using explicit paths at startup. The
wrapper requires the host paths and stores them in `swe_runner` along with Docker
CPU/RAM limits. JSONL remains a task-submission input. `SWE_SERVICE=standalone` selects
a fixed model endpoint; `SWE_ROLE=standalone` separately selects a single-host launch
without a Ray cluster and makes `SWE_HEAD_ADDRESS` unnecessary.

The generator sets the pinned OpenHands revision, reference prompts, 30 turns,
1,800-second agent timeout, and 60-second command/test timeouts. For Qwen's benchmark
behavior it sends `enable_thinking: false`. For a generic endpoint that rejects that
extension, regenerate with `--chat-template-thinking server-default`; that is a
smoke-test configuration change, not a claim of MLPerf equivalence. Use `--force` to
deliberately replace an existing config.

If the model endpoint needs a credential, create an owner-readable environment file and
edit its contents locally:

```bash
install -m 600 /dev/null model.env
${EDITOR:-vi} model.env
```

The file contains one line, `MLPERF_MODEL_API_KEY=your_internal_model_key`. Do not
commit it or put the key in the model URL. The generated `site.yaml` stores an
environment-variable reference, not the credential. Gym later saves resolved
configuration in episode artifacts, which can include this key. The service uses a
restrictive file-creation mask and requires private state directories. Protect those
artifacts when copying or sharing them. If authentication is not needed, omit
`--env-file` from the service command below.

## 4. Run the offline container preflight

To save the offline check as the launch mode, set these values in `configure.env` before
generating the YAML:

```bash
SWE_PREFLIGHT_SIF=RELATIVE.sif
SWE_CPUS=4
SWE_MEMORY=16g
SWE_SHM_SIZE=2g
```

Then run from the deployment directory:

```bash
bash runner/start.sh --config config/areal-site.yaml
```

For a one-time preflight without changing the saved mode or budgets, override them:

```bash
bash runner/start.sh --config config/areal-site.yaml \
  --cpus 4 --memory 16g --shm-size 2g --preflight RELATIVE.sif
```

Replace `RELATIVE.sif` with a known task image's path **relative to** the configured SIF
directory. The host launcher reads the image, resource limits, and SIF/state/results
paths from `swe_runner`; explicit options override these values. It does not start Ray,
Gym, the bridge, or any model service, and does not mount the config into the preflight
container. No model credentials are needed. After a saved preflight passes, clear
`SWE_PREFLIGHT_SIF` and regenerate with the intended service budgets, or use
`--no-preflight` and explicit budget overrides.

For a config without host launch settings, or before generating one, keep using explicit
paths:

```bash
sudo bash start.sh \
  --sif-dir "$SWE_SIF_DIR" \
  --state-dir "$SWE_STATE_DIR" \
  --preflight 'RELATIVE_TASK_IMAGE.sif'
```

Both forms use Docker's `--network none` and a disposable Docker-generated container
name. Preflight verifies the prepared runtime, the read-only SIF mount, writable
temporary overlays, required filesystem permissions, and that a second invocation cannot
see a first invocation's edits. It neither contacts a model nor runs a benchmark grader.

The launcher uses these settings to support nested Apptainer containers:

```text
--device /dev/fuse
--security-opt apparmor=unconfined
--security-opt seccomp=unconfined
--security-opt systempaths=unconfined
```

Docker's default seccomp filter can block the user namespace creation required by
Apptainer. Disabling that filter applies only to this runner container and its children;
it does not change host-wide settings.

The launcher also sets `APPTAINERENV_SU_TO_USER=false`, which passes `SU_TO_USER=false`
through Apptainer's clean environment. OpenHands then starts Bash directly as the
existing sandbox user. Using `su root -` would fail with
`cannot set groups: Operation not permitted` in the unprivileged user namespace and
prevent the OpenHands action server from becoming ready.

Apptainer is configured without its setuid helper. The launcher does not enable
`--privileged`, mount the Docker socket, or mount arbitrary host roots. These settings
still relax isolation inside this dedicated worker; validate them on the actual host. If
Docker rejects `systempaths=unconfined` or user namespaces are disabled, resolve that
host/runtime prerequisite instead of skipping the preflight or automatically widening
permissions.

## 5. Start the service

```bash
sudo bash start.sh \
  --sif-dir "$SWE_SIF_DIR" \
  --state-dir "$SWE_STATE_DIR" \
  --config "$PWD/site.yaml" \
  --bind-address "$SWE_BIND_ADDRESS" \
  --env-file "$PWD/model.env"
```

Only the episode endpoint is published, at `http://YOUR_SANDBOX_PRIVATE_IPV4:8080/run`.
Gym's head service, model wrapper, and Ray ports remain internal. The launcher refuses
to replace an existing container. `--dry-run` prints its invocation without starting
Docker containers or creating state directories.

Inspect startup:

```bash
sudo docker logs --tail 150 mlperf-swe-runner
sudo docker stats --no-stream mlperf-swe-runner
```

The service enables offline modes for uv and Hugging Face and disables Ray usage
reporting. This prevents their normal runtime downloads; it is **not** a network
firewall. If public egress must be impossible, use your instance/network policy to allow
only the required internal model and client connections. Gym's endpoint is for your
private network, not public exposure.

## 6. Submit one actual task

Validate the input offline first:

```bash
python3 smoke-run.py \
  --jsonl "$SWE_JSONL" --row 0 \
  --model "$SWE_MODEL_NAME" --check-only
```

Then run one attempt:

```bash
python3 smoke-run.py \
  --jsonl "$SWE_JSONL" --row 0 \
  --model "$SWE_MODEL_NAME" \
  --runner-url "http://$SWE_BIND_ADDRESS:8080" \
  --output first-result.json
```

The client preserves the task's metadata and supplied generation parameters, filling
only missing generation defaults. It does not reconstruct the task from the issue text.
It sends one request and never retries automatically, because a retry could launch a
second attempt while the original is still running.

Exit code 0 means a completed, unmasked attempt; either reward 0 or reward 1 can be
valid. Exit code 2 means input/transport failure, an incomplete result, masking,
timeout, or an agent error. Review the owner-readable result file and corresponding
runner logs. Full results may contain prompts, generated code, and configuration; the
terminal summary intentionally contains only selected scalar fields.

After a transport timeout, inspect the service before submitting another attempt.
Disconnecting an HTTP client is not proof that its remote processes were killed.

## Operation and resource limits

The examples in this section describe one service container. In cluster mode, Docker
limits apply independently on each node; the head service's concurrency limit covers
attempts dispatched across the cluster. See [CLUSTER.md](CLUSTER.md).

### Set concurrency and resource budgets

There are separate controls for the number of active attempts and the resources shared
by the whole service. Defaults are fixed in the scripts; they are not calculated from
the host's capacity.

| Control                          | Default | Scope and meaning                                                                                                                                               |
| -------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configure.py --concurrency`     | `4`     | Maximum active attempts admitted by one head, including generation and grading across its connected workers. Separate head services have independent limits.    |
| `start.sh --cpus`                | `48`    | Docker CPU-time quota shared by the service and all its children. Does not reserve or pin particular cores.                                                     |
| `start.sh --memory`              | `400g`  | Docker aggregate RAM limit (400 GiB), including agents and evaluators. The script also sets `--memory-swap` to the same value, allowing no additional swap.     |
| `start.sh --shm-size`            | `16g`   | Maximum `/dev/shm` size. Memory used here counts toward the service memory budget; this is not an additional allocation.                                        |
| `configure.py --memory-limit-mb` | `32768` | Sets the reference per-process virtual-memory limit and OpenHands command-tree RSS monitor. It is not a hard aggregate RAM quota or reservation for an attempt. |

For example, configure four attempts sharing a 48-CPU / 400-GiB service budget:

```bash
# --force replaces site.yaml: preserve any custom settings before regenerating.
python3 configure.py \
  --output site.yaml --force \
  --model-base-url "$SWE_MODEL_BASE_URL" \
  --model-name "$SWE_MODEL_NAME" \
  --concurrency 4 \
  --memory-limit-mb 32768

# If a runner already exists, follow the stop/remove procedure below first.
bash start.sh \
  --image "$SWE_RUNNER_IMAGE" \
  --sif-dir "$SWE_SIF_DIR" \
  --state-dir "$SWE_STATE_DIR" \
  --config "$PWD/site.yaml" \
  --bind-address "$SWE_BIND_ADDRESS" \
  --env-file "$PWD/model.env" \
  --cpus 48 --memory 400g --shm-size 16g
```

Set `SWE_RUNNER_IMAGE` to your built image tag; omit `--env-file` if authentication is
not needed. Changes require recreating the service container; editing the configuration
does not update a running service automatically.

Four attempts do **not** each receive 12 CPUs and 100 GiB. They share the Docker budget,
and one attempt can consume a disproportionate share. The reference starts an agent and
an evaluator Apptainer container per attempt. Its Ray CPU request is a scheduling hint,
not CPU enforcement. This bundle does not provide a hard per-attempt CPU/RAM cgroup
covering both containers. Exceeding the shared memory budget can cause an OOM kill and
affect other attempts.

The OpenHands RSS monitor can kill command subprocesses above its threshold; it does not
cover the separate evaluator as one combined attempt, and usage can overshoot between
checks. In cluster mode, `--cpus` must be a whole number because the same value is used
for Ray's integer CPU resource count.

### Monitor and size the service

```bash
sudo docker stats mlperf-swe-runner
sudo docker inspect --format \
  'NanoCPUs={{.HostConfig.NanoCpus}} Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}} ShmSize={{.HostConfig.ShmSize}}' \
  mlperf-swe-runner
```

`docker stats` reports aggregate service usage. CPU usage of 100% means roughly one CPU
core; a 48-CPU quota permits roughly 4800%. In `docker inspect`, divide `NanoCPUs` by
one billion to obtain CPU equivalents; memory fields are bytes. These commands verify
the outer service limits, not per-attempt isolation.

- Start with two concurrent smoke requests, then measure four and eight. Track peak
  memory, command/test latency, disk growth, and shared-image I/O. Leave capacity for
  the host and other services; do not infer safe concurrency by dividing total RAM by
  `--memory-limit-mb`.
- `--writable-tmpfs` uses memory for writes. The image configures an explicit Apptainer
  session size; setting `APPTAINER_TMPDIR` to disk does not turn that writable layer
  into a disk overlay. Size it based on observed task behavior.
- Results and logs accumulate under `$SWE_STATE_DIR/results`; define a retention policy
  for completed jobs. Keep free space for temporary files and Ray state.

Stop the whole service and its child processes:

```bash
sudo docker stop --time 60 mlperf-swe-runner
```

To change the configuration or image, stop it, inspect any unfinished work, then
explicitly remove the stopped container and rerun `start.sh`:

```bash
sudo docker rm mlperf-swe-runner
```

The mounted SIFs and host state remain. Standalone mode has no per-episode cancellation
API. AReaL mode revokes model routes on cancellation, but this is not proof that all
task subprocesses were reaped. Cluster startup owns only its container's processes and
never invokes host-wide Ray cleanup.

## AReaL integration boundary

The unified image contains the AReaL bridge from the integration source passed to
`build.sh`. Generate a configuration with `SWE_SERVICE=areal` through `configure.sh`,
then launch with `start.sh --config config/areal-site.yaml`. The bridge exposes one
endpoint to AReaL and forwards each attempt's model calls to its corresponding inference
proxy session. Ray placement is invisible to the training workflow. No training engine,
accelerator runtime, or AReaL package is installed on the sandbox workers.

The integration README in `examples/swe/mlperf` documents the training entrypoint and
proxy configuration. [CLUSTER.md](CLUSTER.md) documents the head/worker configuration,
ports, image distribution, and validation sequence. Standalone `smoke-run.py` is for a
fixed model and cannot authenticate as an AReaL episode.

## Sources and validation status

- [MLPerf reference and pinned stack](https://github.com/mlcommons/training/tree/master/llm_post_training)
- [Reference environment preparation and patches](https://github.com/NVIDIA-NeMo/RL/blob/f987c0596af2bd6cfb9958e768deb8df951d7504/docker/mlperf/Dockerfile#L433)
- [Pinned Gym SWE runner](https://github.com/NVIDIA-NeMo/Gym/tree/610a08ab5fe9f8f5fb5fff36b170429ea67f0f92/responses_api_agents/swe_agents)
- [Apptainer inside Docker](https://apptainer.org/docs/admin/main/installation.html#running-inside-docker)
- [Docker installation on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [uv release and architecture-specific checksums](https://github.com/astral-sh/uv/releases/tag/0.9.30)
- [Miniforge release and architecture-specific checksums](https://github.com/conda-forge/miniforge/releases/tag/26.9.4-1)
- [jq release and architecture-specific checksums](https://github.com/jqlang/jq/releases/tag/jq-1.8.1)

Follow the live qualification steps in [CLUSTER.md](CLUSTER.md) after building. Local
source and configuration checks do not establish Docker image compatibility, multi-node
Ray execution, or AReaL training correctness. A successful offline preflight establishes
runtime prerequisites, not benchmark correctness or capacity.
