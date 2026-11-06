# Two-node SWE sandbox cluster

This guide uses a head (`HEAD_NODE_IP`) and a worker (`WORKER_NODE_IP`). Replace these
placeholders and all `/path/to/...` values with your deployment settings, including the
SSH aliases `HEAD_SSH_HOST` and `WORKER_SSH_HOST`. Both hosts run the **same
native-architecture image**, built from this directory's Dockerfile. AReaL training runs
separately and submits attempts to one address: `http://HEAD_NODE_IP:8080`. No external
sandbox or model service is required.

```text
AReaL rollout workers ── task/session ──> head:8080 bridge
         ^                                     |
         | model requests                      v
         +──────────────────────────── Gym + Ray head
                                          /       \
                               head attempts   worker attempts
                                     \             /
                                      shared results
```

The head also executes attempts. The worker starts Ray only; its host launcher reads the
same generated config, while the worker container needs no model configuration or
credentials. The head's bridge routes each model call to the submitting AReaL episode's
inference proxy. You do not set `SWE_MODEL_BASE_URL` or `SWE_MODEL_NAME` for AReaL
service startup.

The **model adapter** on port `8082` is a small CPU HTTP service inside the head
container. It exposes the model API that OpenHands expects and forwards calls; it loads
no model weights and performs no inference. With the single-node test's bridge port
`18080`, the model-call path is:

```text
OpenHands on a sandbox worker
  → head:8082 model adapter (/v1/chat/completions)
  → head:18080 bridge
  → that episode's AReaL inference proxy
  → the policy inference engine on the training infrastructure
```

`SWE_MODEL_PORT` changes the adapter's listening port, not the model's inference port.
AReaL supplies the episode's route and credentials when submitting an attempt. In
standalone mode, the adapter instead forwards to the fixed internal model URL in the
generated configuration.

## 1. Prepare the hosts and network

Use Ubuntu 22.04 or 24.04 with a functioning native Docker daemon, Python 3,
`/dev/fuse`, and the nested-Apptainer settings checked by `preflight.sh`. If host
preparation is still needed, follow [README.md](README.md) and [ARM64.md](ARM64.md). Do
not reinstall Docker on an already prepared host. The supplied fresh-host installer
targets Ubuntu 22.04; on an already prepared 24.04 host, use the Docker/FUSE checks and
offline preflight below. The image retains its Ubuntu 22.04 base independently of the
host OS. Both hosts must see the same matching-architecture SIF collection and writable
results storage. These examples use:

| Purpose                  | Host path                         | Path inside each container |
| ------------------------ | --------------------------------- | -------------------------- |
| Read-only SIFs           | `/path/to/shared/sif/images`      | `/srv/mlperf/sif`          |
| Shared scripts/results   | `/path/to/shared/swe-results`     | `/srv/swe-results`         |
| Node-local runtime/cache | `/srv/mlperf-swe-cluster/runtime` | `/srv/swe-runtime`         |

The results directory must really be shared storage: the head writes task inputs that
workers read, and workers write reports that the head reads. Ray sockets, Apptainer
temporary files, and caches must stay **node-local**. Do not put the state directory on
the shared filesystem. If NFS uses root squashing, provision the results directory with
ownership that allows the container's user to write. The launcher requires private
state/results directory permissions (no group or other access).

Cluster containers use host networking and retain their own PID namespace. Restrict
access to your private peers; the scripts do not edit firewall rules. An AReaL client
sees one endpoint, but the cluster needs these internal ports:

| Traffic                                               | Default TCP ports                             |
| ----------------------------------------------------- | --------------------------------------------- |
| AReaL rollout workers → head bridge                   | 8080                                          |
| Sandbox workers → head Gym registry and model adapter | 8081, 8082                                    |
| Raw Gym attempt API on head                           | 8083, loopback only in AReaL mode             |
| Ray traffic between both sandbox nodes                | 16379–16387 and 20000–20127                   |
| Head bridge → AReaL inference proxies                 | Each proxy's advertised IP and allocated port |

The Ray range is reserved for this sandbox deployment, separate from a training Ray
cluster. Port 16379 is the head's GCS port; 16380–16385 cover node services. 16386/16387
are explicitly assigned for Ray Client/dashboard configuration; the dashboard is
disabled. Set the GCS port in `SWE_HEAD_ADDRESS` and the worker range in
`SWE_RAY_WORKER_PORTS` before generating the configuration; the launcher reads both.
Host networking exposes listeners on the host; it is not a firewall.

## 2. Build one image and distribute it

The permanent deployment directory contains one copy of the runner source and four
adapter files:

```text
mlperf-runner-deployment/
├── runner/     # Dockerfile, build/start scripts, and documentation
└── adapter/    # gym_model.py, runner_bridge.py, serve.py, configure_runner.py
```

On the development host, stage only these source files. The explicit file lists exclude
local credentials, site configurations, datasets, SIFs, and the rest of the training
checkout:

```bash
export SWE_AREAL_CHECKOUT=/path/to/AReaL
export SWE_RUNNER_BUNDLE="$SWE_AREAL_CHECKOUT/examples/swe/mlperf/runner_cluster"
export SWE_ADAPTER="$SWE_AREAL_CHECKOUT/examples/swe/mlperf"
SWE_SYNC_DIR=$(mktemp -d /tmp/mlperf-swe-sync.XXXXXX)
mkdir "$SWE_SYNC_DIR/runner" "$SWE_SYNC_DIR/adapter"
for file in \
  Dockerfile README.md ARM64.md CLUSTER.md runtime-pins.env \
  build.sh build-tls.sh install-host.sh start.sh entrypoint.sh preflight.sh \
  prepare-runtime.sh prepare-offline-wheels.sh add-apt-repository.py \
  patch-poetry-markers.py patch-offline-eval.py configure.py configure.sh \
  configure.env.example launch-config.py smoke-run.py \
  cluster-entrypoint.py check-cluster.py; do
  cp "$SWE_RUNNER_BUNDLE/$file" "$SWE_SYNC_DIR/runner/"
done
for file in gym_model.py runner_bridge.py serve.py configure_runner.py; do
  cp "$SWE_ADAPTER/$file" "$SWE_SYNC_DIR/adapter/"
done
```

Copy into a **fresh deployment directory** on each node. The commands below
intentionally fail if that directory already exists; use a new directory name for a new
deployment instead of nesting another copy inside an existing one. Keep any previous
standalone bundle and configuration for rollback. The worker needs only the launcher
bundle because it loads the image built on the head:

```bash
ssh HEAD_SSH_HOST 'umask 077; mkdir /path/to/mlperf-runner-deployment' && \
scp -r "$SWE_SYNC_DIR/runner" "$SWE_SYNC_DIR/adapter" \
  HEAD_SSH_HOST:/path/to/mlperf-runner-deployment/
ssh WORKER_SSH_HOST 'umask 077; mkdir /path/to/mlperf-runner-deployment' && \
scp -r "$SWE_SYNC_DIR/runner" \
  WORKER_SSH_HOST:/path/to/mlperf-runner-deployment/
```

On the **head**, build natively from these sources:

```bash
export SWE_DEPLOY_DIR=/path/to/mlperf-runner-deployment
export SWE_RUNNER_IMAGE=mlperf-swe-runner:cluster
bash "$SWE_DEPLOY_DIR/runner/build.sh" \
  --integration-dir "$SWE_DEPLOY_DIR/adapter"
docker image inspect --format '{{.Id}} {{.Architecture}}' "$SWE_RUNNER_IMAGE"
```

`build.sh` creates a temporary Docker build context containing `runner/` and `adapter/`,
and removes it after the build. There is no permanent `build-context/` directory or
second integration image to maintain. Edit files in the deployment's `runner/` or
`adapter/` directory, then run the same build command again.

Retain your existing build proxy/TLS arguments if required; see the build section in
[README.md](README.md). Build-time repository access is needed, but the worker does not
download packages at startup. Do not run `docker build .` in the source bundle: use
`build.sh` to assemble the unified Dockerfile's required inputs.

On the head, save to a new archive on shared storage:

```bash
umask 077
mkdir -p /path/to/shared/runner-images
cd /path/to/shared/runner-images
docker save -o mlperf-swe-runner-cluster.tar "$SWE_RUNNER_IMAGE"
sha256sum mlperf-swe-runner-cluster.tar > mlperf-swe-runner-cluster.tar.sha256
```

On the **worker**, verify and load:

```bash
export SWE_RUNNER_IMAGE=mlperf-swe-runner:cluster
cd /path/to/shared/runner-images
sha256sum -c mlperf-swe-runner-cluster.tar.sha256
docker load -i mlperf-swe-runner-cluster.tar
docker image inspect --format '{{.Id}} {{.Architecture}}' "$SWE_RUNNER_IMAGE"
```

The image IDs and architecture must match on both nodes. Neither node needs AReaL
training dependencies or accelerator devices inside this image.

## 3. Check each host and generate the head configuration

In a shell on **each host**, set:

```bash
export SWE_DEPLOY_DIR=/path/to/mlperf-runner-deployment
export SWE_RUNNER_IMAGE=mlperf-swe-runner:cluster
export SWE_SIF_DIR=/path/to/shared/sif/images
export SWE_RESULTS_DIR=/path/to/shared/swe-results
export SWE_STATE_DIR=/srv/mlperf-swe-cluster
export SWE_HEAD_ADDRESS=HEAD_NODE_IP:16379
export SWE_PREFLIGHT_SIF=RELATIVE_TASK_IMAGE.sif
cd "$SWE_DEPLOY_DIR/runner"
```

Choose a SIF that exists in your collection. Run the existing offline preflight on
**both nodes**, before starting the cluster:

```bash
bash start.sh --image "$SWE_RUNNER_IMAGE" \
  --sif-dir "$SWE_SIF_DIR" --state-dir "$SWE_STATE_DIR" \
  --cpus 28 --memory 112g --shm-size 8g \
  --preflight "$SWE_PREFLIGHT_SIF"
```

The resource values here are illustrative budgets for a 32-CPU/128-GiB host; choose
values that leave capacity for the OS and other workloads on each node. They are not a
measurement of free resources on the worker or head. Preflight checks native SIF
execution, independent writable state, harness interpreters, and offline `chardet`
installation. It does not call a model. After generating the head YAML, the config-based
preflight command below reads these paths and the image from that file instead.

On the **head only**, generate the complete sandbox configuration with one command. The
wrapper reads your configuration variables and runs `configure.py` in the image. In
AReaL mode, it calls the adapter helper internally. The same generated YAML launches the
head alone or the head with a worker joined later; copy it to the worker for its host
launcher. AReaL's training YAML remains separate in the training checkout.

### Generate from a configuration env file

Create your local file once, then edit the deployment path, host SIF/state/results
paths, and head-address placeholders. Keep an existing `configure.env` instead of
replacing it.

```bash
cd "$SWE_DEPLOY_DIR"
umask 077
cp -n runner/configure.env.example configure.env
${EDITOR:-vi} configure.env
```

Replace these example values for the head. The other variables and defaults are listed
in the template and the table below:

```bash
SWE_DEPLOY_DIR=/path/to/mlperf-runner-deployment
SWE_SERVICE=areal
SWE_HEAD_IP=HEAD_NODE_IP
SWE_HEAD_ADDRESS="${SWE_HEAD_IP}:16379"
SWE_SIF_DIR=/path/to/shared/sif/images
SWE_RESULTS_DIR=/path/to/shared/swe-results
SWE_STATE_DIR=/srv/mlperf-swe-cluster
SWE_CONTAINER_NAME=mlperf-swe-head
SWE_CPUS=28
SWE_MEMORY=112g
SWE_SHM_SIZE=8g
SWE_PREFLIGHT_SIF=
```

For the **single-head test alongside the existing service**, also set these values in
`configure.env`:

```bash
SWE_API_PORT=18080
SWE_CONCURRENCY=2
SWE_CONFIG_FILE=areal-site-single-head.yaml
SWE_STATE_DIR=/srv/mlperf-swe-cluster-single-head
SWE_RESULTS_DIR=/path/to/shared/swe-single-head-results
SWE_CONTAINER_NAME=mlperf-swe-head-test
```

Then generate the complete head YAML with:

```bash
bash runner/configure.sh --env-file configure.env
```

The file contains trusted **Bash assignments**: quotes and `${SWE_HEAD_IP}` expansion
work, and `export` is unnecessary. The wrapper sources it in its own shell; it is not
Docker's env-file format. Do not pass this file to `start.sh --env-file`, which is for
runtime model credentials. The wrapper supports both `SWE_SERVICE=areal` (the default)
and `SWE_SERVICE=standalone`.

For a **standalone fixed-model cluster**, set these values in the env file before
running the same generation command:

```bash
SWE_SERVICE=standalone
SWE_CONFIG_FILE="${SWE_SERVICE}-site.yaml"
SWE_MODEL_BASE_URL=http://YOUR_MODEL_HOST:PORT/v1
SWE_MODEL_NAME=YOUR_SERVED_MODEL_NAME
```

The wrapper supplies the model URL/name only in standalone mode and omits `--gym-port`
there. The head still uses the same Ray cluster and shared SIF/results paths; standalone
selects a fixed model, not a single-node deployment. An explicitly set `SWE_CONFIG_FILE`
is retained, so use a different filename when keeping both service configurations.

Append `--check-only` to validate without writing YAML, or append `--force` to
deliberately replace an existing YAML. The wrapper forwards trailing arguments to
`configure.py`. It runs the image without network access and mounts the bundle's current
`configure.py` read-only, so no image rebuild is required.

The generated file includes a `swe_runner` section for host launch settings. Keep the
updated `start.sh` and `launch-config.py` together in `runner/`; they use the host's
Python 3 to read this JSON-formatted YAML. The image receives the same runtime arguments
as before, so this launch change also needs no image rebuild. Regenerate older YAML to
use the short commands below; service-only YAML remains usable with explicit launch
paths and options.

To save offline preflight as the launch mode, set `SWE_PREFLIGHT_SIF=RELATIVE.sif` and
select `SWE_CPUS`, `SWE_MEMORY`, and `SWE_SHM_SIZE` in `configure.env` before generating
the YAML. Then the entire launch command is:

```bash
bash runner/start.sh --config config/areal-site.yaml
```

Alternatively, keep the saved service mode and override it for one preflight:

```bash
bash runner/start.sh --config config/areal-site.yaml \
  --cpus 4 --memory 16g --shm-size 2g --preflight RELATIVE.sif
```

Use your generated filename and replace `RELATIVE.sif` with a path relative to the
configured SIF directory. Preflight uses the saved image, resource limits, dry-run
setting, and SIF/state/results paths; it starts no Ray, Gym, bridge, or model service.
It uses `--network none`, does not mount the config into the container, and needs no
model credentials. The temporary container keeps a Docker-generated name, independent of
the service's `--name` option. Explicit image/path options still override the config.
Copy the YAML to a worker to reuse this command there; override `--state-dir` if its
local path differs. Updating the host launch scripts for this workflow requires no image
rebuild. Clear `SWE_PREFLIGHT_SIF` and regenerate with service budgets after
qualification, or use `--no-preflight` and explicit budget overrides to launch the
service.

### Equivalent explicit Docker command for AReaL mode

For manual invocation, load the same file in your current shell and run the generator
directly. Loading it also makes the variables available to diagnostic examples; running
the wrapper alone does not change your parent shell's variables.

```bash
source "$SWE_DEPLOY_DIR/configure.env"
umask 077
mkdir -p "$SWE_DEPLOY_DIR/config"
launch_args=(--no-launch-dry-run)
[[ ${SWE_DRY_RUN:-false} != true ]] || launch_args=(--launch-dry-run)
if [[ ${SWE_ROLE:-} == standalone ]]; then
  launch_args+=(--bind-address "${SWE_BIND_ADDRESS:-$SWE_HEAD_IP}")
else
  launch_args+=(--advertised-host "$SWE_HEAD_IP" --ray-head-address "$SWE_HEAD_ADDRESS")
  [[ -z ${SWE_BIND_ADDRESS:-} ]] || launch_args+=(--bind-address "$SWE_BIND_ADDRESS")
fi
[[ -z ${SWE_ROLE:-} ]] || launch_args+=(--role "$SWE_ROLE")
[[ -z ${SWE_NODE_IP:-} ]] || launch_args+=(--node-ip "$SWE_NODE_IP")
[[ -z ${SWE_ENV_FILE:-} ]] || launch_args+=(--runtime-env-file "$SWE_ENV_FILE")
docker run --rm --pull never --network none --entrypoint python \
  --mount "type=bind,src=$SWE_DEPLOY_DIR/runner/configure.py,dst=/opt/swe-runner/configure.py,readonly" \
  --mount "type=bind,src=$SWE_DEPLOY_DIR/config,dst=/out" \
  "${SWE_RUNNER_IMAGE:-mlperf-swe-runner:cluster}" /opt/swe-runner/configure.py \
  --service areal --output "/out/${SWE_CONFIG_FILE:-areal-site.yaml}" \
  "${launch_args[@]}" \
  --host-sif-dir "$SWE_SIF_DIR" \
  --host-state-dir "$SWE_STATE_DIR" \
  --host-results-dir "$SWE_RESULTS_DIR" \
  --runner-image "${SWE_RUNNER_IMAGE:-mlperf-swe-runner:cluster}" \
  --ray-worker-ports "${SWE_RAY_WORKER_PORTS:-20000-20127}" \
  --object-store-memory "${SWE_OBJECT_STORE_MEMORY:-1073741824}" \
  --container-name "${SWE_CONTAINER_NAME:-mlperf-swe-runner}" \
  --cpus "${SWE_CPUS:-48}" --memory "${SWE_MEMORY:-400g}" \
  --shm-size "${SWE_SHM_SIZE:-16g}" --preflight "${SWE_PREFLIGHT_SIF:-}" \
  --port "${SWE_API_PORT:-8080}" --gym-port "${SWE_GYM_PORT:-8083}" \
  --registry-port "${SWE_REGISTRY_PORT:-8081}" --model-port "${SWE_MODEL_PORT:-8082}" \
  --concurrency "${SWE_CONCURRENCY:-4}" \
  --max-turns "${SWE_MAX_TURNS:-30}" \
  --agent-timeout "${SWE_AGENT_TIMEOUT:-1800}" \
  --test-timeout "${SWE_TEST_TIMEOUT:-60}" \
  --command-timeout "${SWE_COMMAND_TIMEOUT:-60}" \
  --memory-limit-mb "${SWE_MEMORY_LIMIT_MB:-32768}" \
  --chat-template-thinking "${SWE_CHAT_TEMPLATE_THINKING:-disabled}"
```

This writes `$SWE_DEPLOY_DIR/config/$SWE_CONFIG_FILE` as JSON-formatted YAML.
Concurrency applies across the whole cluster. Reference defaults remain 30 turns,
1,800-second agent timeout, 60-second command/test settings, and a 32,768-MiB memory
setting. The helper refuses to overwrite an existing file; choose a new filename or use
`--force` deliberately. Generation makes no network requests and starts no service.
Worker callbacks obtain their routes through Gym.

### Configuration inputs and environment variables

`configure.sh` reads the env file and maps its variables to explicit CLI arguments.
`configure.py` itself does **not read configuration values from environment variables**.
No Docker `-e` options or `model.env` credentials are needed for AReaL config
generation.

These are the configuration variables accepted by the wrapper. `SWE_DEPLOY_DIR`,
`SWE_HEAD_IP`, `SWE_SIF_DIR`, `SWE_STATE_DIR`, and `SWE_RESULTS_DIR` are required in
both service modes. `SWE_HEAD_ADDRESS` is required unless `SWE_ROLE=standalone`.
Standalone service mode also requires `SWE_MODEL_BASE_URL` and `SWE_MODEL_NAME`. For the
remaining variables, the example values are the wrapper's defaults unless an override is
shown:

| Shell variable               | Example                                   | Purpose                                                                                        |
| ---------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `SWE_DEPLOY_DIR`             | `/path/to/mlperf-runner-deployment`       | Host bundle and output directory; not written into the YAML                                    |
| `SWE_RUNNER_IMAGE`           | `mlperf-swe-runner:cluster`               | Already-built image used for generation and saved for launch                                   |
| `SWE_SERVICE`                | `areal`                                   | `areal` for training sessions; `standalone` for a fixed model endpoint                         |
| `SWE_CONFIG_FILE`            | `${SWE_SERVICE}-site.yaml`                | Output filename under `config/`; must end in `.yaml` or `.yml`                                 |
| `SWE_HEAD_IP`                | `HEAD_NODE_IP`                            | Head address reachable by all sandbox workers                                                  |
| `SWE_HEAD_ADDRESS`           | `HEAD_NODE_IP:16379`                      | Ray head address and port base, read by `start.sh`                                             |
| `SWE_API_PORT`               | `8080` (`18080` for the single-node test) | AReaL bridge or standalone SWE API port, read by `start.sh`                                    |
| `SWE_SIF_DIR`                | `/path/to/shared/sif/images`              | Required absolute host SIF path, saved in `swe_runner.sif_dir`                                 |
| `SWE_STATE_DIR`              | `/srv/mlperf-swe-cluster`                 | Required absolute node-local state path, saved in `swe_runner.state_dir`                       |
| `SWE_RESULTS_DIR`            | `/path/to/shared/swe-results`             | Required absolute shared results path, saved in `swe_runner.results_dir`                       |
| `SWE_RAY_WORKER_PORTS`       | `20000-20127`                             | Ray worker port range, saved in `swe_runner.ray_worker_ports`                                  |
| `SWE_OBJECT_STORE_MEMORY`    | `1073741824`                              | Ray object-store bytes per node, saved in `swe_runner.object_store_memory`                     |
| `SWE_CONTAINER_NAME`         | `mlperf-swe-runner`                       | Service container name; `--name` overrides it; ignored by preflight                            |
| `SWE_CPUS`                   | `48` (`28` in this guide)                 | Docker CPU limit; positive number, whole CPUs for Ray service roles                            |
| `SWE_MEMORY`                 | `400g` (`112g` in this guide)             | Docker memory limit; choose a budget for the host                                              |
| `SWE_SHM_SIZE`               | `16g` (`8g` in this guide)                | Shared-memory maximum within the Docker memory budget                                          |
| `SWE_PREFLIGHT_SIF`          | Empty                                     | Relative SIF path for offline preflight; empty starts the service                              |
| `SWE_ROLE`                   | Empty                                     | Infer head from Ray config; or select `head`, `worker`, or single-host `standalone`            |
| `SWE_NODE_IP`                | Empty                                     | Head derives its IP; worker requires its own IPv4                                              |
| `SWE_BIND_ADDRESS`           | Empty                                     | Derive head IPv4 for standalone host port publishing; cluster override must match node IP      |
| `SWE_ENV_FILE`               | Empty                                     | Absolute host runtime credentials path; file contents are never read during generation         |
| `SWE_DRY_RUN`                | `false`                                   | `true` prints the Docker command without launching; boolean saved in YAML                      |
| `SWE_GYM_PORT`               | `8083`                                    | AReaL-only raw Gym API, bound to loopback; ignored in standalone mode                          |
| `SWE_REGISTRY_PORT`          | `8081`                                    | Gym registry, reachable by workers                                                             |
| `SWE_MODEL_PORT`             | `8082`                                    | Gym model adapter, reachable by workers; not an inference engine                               |
| `SWE_MODEL_BASE_URL`         | `http://YOUR_MODEL_HOST:PORT/v1`          | Required in standalone mode; fixed model API URL ending in `/v1`; ignored in AReaL mode        |
| `SWE_MODEL_NAME`             | `YOUR_SERVED_MODEL_NAME`                  | Required in standalone mode; name served by the model API; ignored in AReaL mode               |
| `SWE_CONCURRENCY`            | `4` (`2` for the single-node test)        | Maximum active attempts across the whole cluster                                               |
| `SWE_MAX_TURNS`              | `30`                                      | OpenHands turn limit; maps to `--max-turns`                                                    |
| `SWE_AGENT_TIMEOUT`          | `1800`                                    | Agent timeout in seconds; maps to `--agent-timeout`                                            |
| `SWE_TEST_TIMEOUT`           | `60`                                      | Reference test timeout in seconds; maps to `--test-timeout`                                    |
| `SWE_COMMAND_TIMEOUT`        | `60`                                      | Command timeout in seconds; maps to `--command-timeout`                                        |
| `SWE_MEMORY_LIMIT_MB`        | `32768`                                   | Reference memory setting in MiB; maps to `--memory-limit-mb`, not a whole-attempt cgroup quota |
| `SWE_CHAT_TEMPLATE_THINKING` | `disabled`                                | `disabled`, `enabled`, or `server-default`; maps to `--chat-template-thinking`                 |

The complete generator options are listed below. Only `--output` is always required; a
head joining the managed Ray cluster also needs `--advertised-host` and
`--ray-head-address`. Select `--service areal` for training or `--service standalone`
for a fixed-model test.

| CLI option                                               | Default                                                                   | Meaning                                                                                                                                   |
| -------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `--output`                                               | Required                                                                  | Output path inside the generator container, such as `/out/areal-site.yaml`                                                                |
| `--service`                                              | `standalone`                                                              | `areal` routes model calls through episode sessions; `standalone` uses a fixed model endpoint                                             |
| `--advertised-host`                                      | Unset                                                                     | Explicit head IPv4 address; required and non-loopback when `--ray-head-address` is supplied                                               |
| `--ray-head-address`                                     | Unset                                                                     | Existing Ray head `IPv4:PORT`; managed cluster launch requires it                                                                         |
| `--port`                                                 | `8080`                                                                    | Public AReaL bridge port, or standalone SWE API port                                                                                      |
| `--gym-port`                                             | `8083` in AReaL mode                                                      | Internal loopback SWE API port; do not pass in standalone mode                                                                            |
| `--registry-port`                                        | `8081`                                                                    | Gym registry port                                                                                                                         |
| `--model-port`                                           | `8082`                                                                    | Gym model adapter port                                                                                                                    |
| `--concurrency`                                          | `4`                                                                       | Maximum active attempts admitted by the head                                                                                              |
| `--max-turns`                                            | `30`                                                                      | OpenHands agent turn limit                                                                                                                |
| `--agent-timeout`                                        | `1800`                                                                    | Agent timeout, seconds                                                                                                                    |
| `--test-timeout`                                         | `60`                                                                      | Reference evaluator test timeout setting, seconds                                                                                         |
| `--command-timeout`                                      | `60`                                                                      | Agent command timeout, seconds                                                                                                            |
| `--memory-limit-mb`                                      | `32768`                                                                   | Reference memory setting in MiB; not a whole-attempt cgroup quota                                                                         |
| `--sif-dir`                                              | `/srv/mlperf/sif`                                                         | SIF path **inside** the runtime container; keep this default with `start.sh`                                                              |
| `--results-dir`                                          | `/srv/swe-results`                                                        | Result path **inside** the runtime container; keep this default with `start.sh`                                                           |
| `--host-sif-dir`                                         | Unset                                                                     | Absolute host SIF mount path saved in `swe_runner`; supply all three host path options together                                           |
| `--host-state-dir`                                       | Unset                                                                     | Absolute host node-local state path saved in `swe_runner`                                                                                 |
| `--host-results-dir`                                     | Unset                                                                     | Absolute host shared results path saved in `swe_runner`                                                                                   |
| `--runner-image`                                         | `mlperf-swe-runner:cluster` with Ray, otherwise `mlperf-swe-runner:local` | Image saved in `swe_runner` when host launch paths are supplied                                                                           |
| `--ray-worker-ports`                                     | `20000-20127`                                                             | Ray worker port range saved in `swe_runner`                                                                                               |
| `--object-store-memory`                                  | `1073741824`                                                              | Ray object-store bytes per node saved in `swe_runner`                                                                                     |
| `--container-name`                                       | `mlperf-swe-runner`                                                       | Service name saved in `swe_runner.container_name`                                                                                         |
| `--cpus`                                                 | `48`                                                                      | Docker CPU limit saved as a string                                                                                                        |
| `--memory`                                               | `400g`                                                                    | Docker memory limit; positive integer with optional `b`, `k`, `m`, or `g` suffix                                                          |
| `--shm-size`                                             | `16g`                                                                     | Shared-memory limit with the same size syntax                                                                                             |
| `--preflight`                                            | Empty                                                                     | Relative SIF path without `..`; empty selects service mode                                                                                |
| `--role`                                                 | Unset                                                                     | Saved launch role: `head`, `worker`, or `standalone`                                                                                      |
| `--node-ip`                                              | Unset                                                                     | Saved node IPv4; required for a worker; head must match advertised host                                                                   |
| `--bind-address`                                         | Unset                                                                     | Saved IPv4 for host port publishing                                                                                                       |
| `--runtime-env-file`                                     | Unset                                                                     | Absolute host runtime credentials path, not the configuration env file                                                                    |
| `--launch-dry-run` / `--no-launch-dry-run`               | False                                                                     | Save dry-run behavior for `start.sh`; distinct from generation's `--check-only`                                                           |
| `--integration-dir`                                      | `/opt/areal-gym` in the image                                             | Directory containing the AReaL adapter helper; no override needed in the image                                                            |
| `--host`                                                 | `0.0.0.0`                                                                 | Standalone SWE API bind address; `--advertised-host` takes precedence. AReaL raw Gym remains loopback-only and the bridge binds `0.0.0.0` |
| `--model-base-url`                                       | Unset                                                                     | Required only in standalone mode: internal model API URL ending in `/v1`; omit in AReaL mode                                              |
| `--model-name`                                           | Unset                                                                     | Required only in standalone mode: served model name; omit in AReaL mode                                                                   |
| `--api-key-env`                                          | `MLPERF_MODEL_API_KEY`                                                    | Standalone model credential variable **name**; only its reference is written, not its value                                               |
| `--uses-reasoning-parser` / `--no-uses-reasoning-parser` | Enabled in standalone                                                     | Whether the standalone model API separates `reasoning_content`; AReaL's adapter sets this to false internally                             |
| `--chat-template-thinking`                               | `disabled`                                                                | `disabled`, `enabled`, or `server-default`; reference Qwen behavior disables thinking                                                     |
| `--force`                                                | Off                                                                       | Replace an existing output file deliberately                                                                                              |
| `--check-only`                                           | Off                                                                       | Validate arguments and generated configuration without writing a file or contacting services                                              |

All active API ports must be distinct and within `1..65535`. Leave the reference turn,
timeout, memory, and thinking defaults unchanged when matching the baseline. The
thinking setting must also match the AReaL inference setup. Configure AReaL's
tool/reasoning parsers in its training YAML separately.

`start.sh --config` selects the head role when the config contains a Ray head address.
It derives service mode, head IP, API port, Ray head address, and Ray port base from the
service configuration. Explicit head network options must match these values. Image,
host path, and resource options can override generated launch settings. The original
`--role standalone` launcher instead maps its chosen host `--port` to the container API
port in the config, which defaults to `8080`.

For standalone mode, the wrapper passes `SWE_MODEL_BASE_URL` and `SWE_MODEL_NAME`
through `--model-base-url` and `--model-name`; the Python generator does not read them
automatically. If authentication is needed, the credential named by `--api-key-env` is
resolved **at service runtime**, usually from the file passed to `start.sh --env-file`.
It does not need to exist while generating YAML. AReaL mode obtains per-episode
credentials from AReaL and requires no fixed-model credential variable.

The wrapper records the host image, paths, Ray worker ports, and object-store size in
`swe_runner`. It preserves the fixed **container** SIF/results paths used by Gym.
`start.sh` reads the generated config, so you need not source `configure.env` or repeat
those settings to launch. Docker CPU/RAM/shared-memory budgets and the remaining launch
options are saved in the same section; command-line options override them. Benchmark
JSONL is read when submitting tasks or training; it is not required to generate the
sandbox YAML. Changing generated YAML does not change a running service.

### Saved launch settings and command-line overrides

Configure the values once, generate the YAML, and launch with only
`bash runner/start.sh --config config/areal-site.yaml`. The precedence is explicit
launch option, saved setting, then the launcher's default. All configurable launch
values have env-file inputs:

| `start.sh` option               | Configuration env variable                                | Saved source                                                         |
| ------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------- |
| `--image`                       | `SWE_RUNNER_IMAGE`                                        | `swe_runner.image`                                                   |
| `--name`                        | `SWE_CONTAINER_NAME`                                      | `swe_runner.container_name`                                          |
| `--sif-dir`                     | `SWE_SIF_DIR`                                             | `swe_runner.sif_dir`                                                 |
| `--state-dir`                   | `SWE_STATE_DIR`                                           | `swe_runner.state_dir`                                               |
| `--results-dir`                 | `SWE_RESULTS_DIR`                                         | `swe_runner.results_dir`                                             |
| `--cpus`                        | `SWE_CPUS`                                                | `swe_runner.cpus`                                                    |
| `--memory`                      | `SWE_MEMORY`                                              | `swe_runner.memory`                                                  |
| `--shm-size`                    | `SWE_SHM_SIZE`                                            | `swe_runner.shm_size`                                                |
| `--preflight`, `--no-preflight` | `SWE_PREFLIGHT_SIF`                                       | `swe_runner.preflight`; empty starts the service                     |
| `--role`                        | `SWE_ROLE`                                                | `swe_runner.role`; empty infers head with Ray, otherwise standalone  |
| `--node-ip`                     | `SWE_NODE_IP`                                             | `swe_runner.node_ip`; head defaults to `SWE_HEAD_IP`                 |
| `--bind-address`                | `SWE_BIND_ADDRESS`                                        | `swe_runner.bind_address`; standalone defaults to `SWE_HEAD_IP`      |
| `--env-file`                    | `SWE_ENV_FILE`                                            | `swe_runner.env_file`; only the path is stored                       |
| `--dry-run`, `--no-dry-run`     | `SWE_DRY_RUN`                                             | `swe_runner.dry_run` boolean                                         |
| `--ray-worker-ports`            | `SWE_RAY_WORKER_PORTS`                                    | `swe_runner.ray_worker_ports`                                        |
| `--object-store-memory`         | `SWE_OBJECT_STORE_MEMORY`                                 | `swe_runner.object_store_memory`                                     |
| `--service`                     | `SWE_SERVICE`                                             | Derived from the generated service configuration                     |
| `--port`                        | `SWE_API_PORT`                                            | Generated public service port; standalone may override the host port |
| `--ray-head-address`            | `SWE_HEAD_ADDRESS`                                        | Generated `ray_head_node_address`                                    |
| `--ray-port-base`               | Port suffix of `SWE_HEAD_ADDRESS`                         | Derived from the same Ray address                                    |
| `--config`                      | `SWE_CONFIG_FILE` selects the generator's output filename | Pass the resulting YAML path on the command line                     |
| `--help`                        | None                                                      | Command-line help only                                               |

Service identity and network values stay in the existing service config rather than
being duplicated in `swe_runner`. Explicit head addresses, API ports, and Ray settings
must agree with that config. `SWE_ROLE=standalone` selects a single-host deployment and
omits Ray configuration; it is independent of `SWE_SERVICE=standalone`, which selects a
fixed model endpoint and can also run on a Ray cluster. For `SWE_ROLE=standalone`, the
wrapper saves `SWE_BIND_ADDRESS` or its default `SWE_HEAD_IP` as the host publish
address, and keeps the internal registry/model binds on loopback. Direct `configure.py`
calls use `--bind-address` and omit `--advertised-host` for this role.

Set `SWE_PREFLIGHT_SIF` to save an offline check, or leave it empty to start the
service. Preflight applies saved resources and dry-run behavior while ignoring saved
service names, roles, and credentials. `--no-preflight` clears a saved check,
`--no-dry-run` clears saved dry-run mode, and `--env-file ''` clears a saved
credential-file path. `SWE_ENV_FILE` must name a separate runtime credentials file;
generation never reads its contents. A worker ignores a head's saved credentials.

For a worker with no launch flags beyond `--config`, generate a separate YAML with
`SWE_ROLE=worker`, its own `SWE_NODE_IP`, `SWE_CONTAINER_NAME`, resource budgets, and
local state path. Keep the head address, service settings, and shared paths unchanged.
Alternatively, reuse the head YAML and supply the worker overrides shown below.

### Single-head qualification

Generate the YAML above with port `18080`, concurrency `2`, filename
`areal-site-single-head.yaml`, and the separate state/results paths shown in the
env-file example. Then start only this head, leaving the existing standalone service
running:

```bash
cd "$SWE_DEPLOY_DIR"
bash runner/start.sh --config config/areal-site-single-head.yaml

# Load variables for the diagnostic commands, not for start.sh.
source configure.env
curl --fail --noproxy '*' "http://$SWE_HEAD_IP:$SWE_API_PORT/health"
docker exec mlperf-swe-head-test python /opt/swe-runner/check-cluster.py \
  --ray-address "$SWE_HEAD_ADDRESS" --expected-node "$SWE_HEAD_IP" \
  --sif "$SWE_PREFLIGHT_SIF" \
  --head-url "http://$SWE_HEAD_IP:$SWE_REGISTRY_PORT" \
  --model-url "http://$SWE_HEAD_IP:$SWE_MODEL_PORT" --preflight \
  --output "/srv/swe-results/single-head-check-$(date +%Y%m%d-%H%M%S).json"
```

The launcher refuses to replace an existing container. Once the head is ready, this
diagnostic verifies exactly one Ray node and real offline SIF execution; it does not
generate model responses or grade a task. Adding a worker later uses the same image and
a copy of this generated YAML. Keep the head's endpoint at `18080` if retaining this
configuration, and set `AREAL_SWE_RUNNER_URL` accordingly. Do not start a second Ray
head on the same ports; sections below describe the separate default-port two-node
deployment.

## 4. Start the cluster

The existing standalone service may already own the head's port 8080. Inspect its active
requests and finish/drain them before stopping it. For the previous container name, the
deliberate cutover command is:

```bash
# head, only after its current attempts have finished:
docker stop --time 60 mlperf-swe-runner
```

Keep the stopped container for rollback. The new launcher neither replaces existing
containers nor stops other Ray/training services. Running with `--dry-run` validates
paths/configuration and prints the Docker command without creating directories or
containers; actual startup also checks occupied ports.

For the two-node examples below, generate `config/areal-site.yaml` with the default API
port `8080` and the shared results path. On the **head**:

```bash
cd "$SWE_DEPLOY_DIR"
bash runner/start.sh --config config/areal-site.yaml
docker logs --tail 100 -f mlperf-swe-head
```

These examples choose `mlperf-swe-head` and `mlperf-swe-worker` for convenient
monitoring. The head name and budgets are saved using the env values above. `--name`
overrides the saved name; when no name was saved, it defaults to `mlperf-swe-runner`.
Use the actual name in logs, monitoring, and restart commands. A standalone fixed-model
cluster uses the same command with `config/standalone-site.yaml`. No `--service`,
address, or port flags are needed for either mode.

Copy the generated file to the worker before launching its worker. From the **head**:

```bash
ssh WORKER_SSH_HOST 'umask 077; mkdir -p /path/to/mlperf-runner-deployment/config'
scp config/areal-site.yaml \
  WORKER_SSH_HOST:/path/to/mlperf-runner-deployment/config/areal-site.yaml
```

Use the corresponding filename when deploying a standalone config. Once head services
are ready, on the **worker**:

```bash
cd "$SWE_DEPLOY_DIR"
bash runner/start.sh --config config/areal-site.yaml \
  --role worker --node-ip WORKER_NODE_IP \
  --cpus 28 --memory 112g --shm-size 8g --name mlperf-swe-worker
docker logs --tail 100 -f mlperf-swe-worker
```

The worker launcher reads its image, paths, and Ray settings from the same config; it
does not mount or pass that file into the worker container. Workers omit `--service` and
`--env-file` in both modes because their role runs only Ray. The state path refers to
each host's local disk; add `--state-dir /different/local/path` if the worker uses a
different path. Both nodes must still see the same shared SIF/results storage. A worker
must register within the launcher's startup timeout. Both containers restart on failure
unless explicitly stopped.

## 5. Test the cluster before training

From a training node, check the one client endpoint (`/health` in AReaL mode, `/` in
standalone mode). Source `configure.env` in the shell running these examples to obtain
the address variables; launching with `--config` does not set shell variables:

```bash
if [[ ${SWE_SERVICE:-areal} == standalone ]]; then
  curl --fail --noproxy '*' "http://$SWE_HEAD_IP:$SWE_API_PORT/"
else
  curl --fail --noproxy '*' "http://$SWE_HEAD_IP:$SWE_API_PORT/health"
fi
```

On the **head**, force a diagnostic onto each node:

```bash
docker exec mlperf-swe-head python /opt/swe-runner/check-cluster.py \
  --ray-address "$SWE_HEAD_ADDRESS" \
  --expected-node HEAD_NODE_IP --expected-node WORKER_NODE_IP \
  --sif "$SWE_PREFLIGHT_SIF" \
  --head-url "http://$SWE_HEAD_IP:$SWE_REGISTRY_PORT" \
  --model-url "http://$SWE_HEAD_IP:$SWE_MODEL_PORT" \
  --preflight --output /srv/swe-results/cluster-check.json
```

The diagnostic checks exactly the expected live nodes, identical runtime/adapter
provenance and Python/Ray versions, worker imports, selected SIF architecture,
shared-filesystem round trips, and worker-to-head callbacks. `--preflight` also executes
the offline SIF checks on each node. The JSON report lands in the shared results
directory. Choose a new report filename for subsequent checks. This is stronger than
seeing two Ray nodes, but still does not run a real agent or evaluator workload.

For a real episode **with AReaL**, continue to the next section. Do not point
`smoke-run.py` at the AReaL bridge: that tool lacks an AReaL episode session.

For optional independent qualification with a fixed internal model, use the standalone
env-file settings above and generate a second config with the same wrapper and image.
Retain the head address and shared paths. Start the head with
`--config config/standalone-site.yaml`, then use the existing `smoke-run.py` examples in
[README.md](README.md) against `http://$SWE_HEAD_IP:$SWE_API_PORT`. Supply `--env-file`
to the head if that model requires authentication. Use a currently running internal
model server; the sandbox does not launch one. Changing head mode requires
draining/stopping the cluster and restarting its head and worker as described below. The
worker image and command stay the same.

## 6. Connect AReaL and run a small training test

Use the same AReaL revision on all training and proxy workers. Follow the Qwen3.8-27B
trial instructions in `examples/swe/mlperf/README.md` in that checkout. Use its complete
`qwen3_8_27b_npu.yaml` recipe, which includes the model, optimizer, scheduler, weight
synchronization, and backend settings. For another model or hardware layout, adapt those
settings and the policy parsers.

Inside the **AReaL runtime**, set the model, native benchmark JSONL, output path, runner
URL, and proxy admin key using that guide's environment setup. Then run:

```bash
python -m examples.swe.mlperf.train \
  --config examples/swe/mlperf/qwen3_8_27b_npu.yaml --check-only
python -m examples.swe.mlperf.train \
  --config examples/swe/mlperf/qwen3_8_27b_npu.yaml
```

No fixed model endpoint goes into the sandbox config. The admin key stays in AReaL; the
bridge receives only per-episode credentials.

Use the direct module entrypoint and single-controller mode, with `AREAL_SPMD_MODE`
unset or `0`. Do not invoke the deprecated local launcher, which forces SPMD. The
integration requires v1 actor/rollout engines, inline proxy mode, HF chat templates, and
individual-turn export. Group reward normalization is supported; group advantage
normalization is rejected. See the integration README for these existing constraints.

The Qwen recipe starts with one rollout group, four samples, and two training steps. PP2
requires at least four exported sequences, even for one-turn attempts. The potential
episode count is `max_concurrent_rollouts × n_samples`; keep it within the head's
concurrency while qualifying. Confirm:

1. Requests reach the bridge and OpenHands executes in a SIF.
1. The evaluator applies the patch and produces a grading report. Zero reward is
   acceptable for an unsolved task; transport/setup errors are not success.
1. AReaL exports nonempty original token IDs, log probabilities, and weight versions
   from its proxy, then completes a training update.
1. A subsequent rollout uses the updated policy, and both nodes can perform real
   attempts before increasing concurrency. Ray spreads work but a small sample does not
   guarantee both nodes will receive an agent attempt.

Also test an intentional cancellation during a disposable attempt and inspect
Ray/Apptainer processes on both hosts. Revoking a bridge route does not prove that all
underlying sandbox processes have exited. Do not qualify cancellation using HTTP success
alone.

## 7. Resource limits, monitoring, and restart

| Setting                              | Scope and effect                                                                                                                    |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `start.sh --cpus 28 --memory 112g`   | Hard Docker budget for that node's entire service and child containers/processes                                                    |
| `--shm-size 8g`                      | Maximum `/dev/shm` size within that memory budget                                                                                   |
| `SWE_OBJECT_STORE_MEMORY=1073741824` | Generated Ray object-store size per node; counts toward that node's memory budget; `start.sh --object-store-memory` can override it |
| `configure.py --concurrency 4`       | Four active attempts across the cluster; not four on each node                                                                      |
| Ray CPU resources                    | Scheduling accounting; the reference task requests 0.1 CPU and does not enforce a per-attempt CPU cap                               |
| `--memory-limit-mb 32768`            | Reference Apptainer per-process address-space limit and OpenHands command-tree RSS monitor; not a hard total-attempt cgroup quota   |

The OpenHands RSS monitor targets command subprocesses and can overshoot between checks.
It does not cover the separate evaluator as one combined attempt. CPU/RAM budgets are
enforced separately on each node; idle memory on another node does not prevent a local
OOM. Keep headroom for Ray, Gym, and grading. Change concurrency by generating a
replacement head config and restarting after draining. Change Docker budgets by
explicitly recreating the affected container.

Monitor on each host:

```bash
# head
docker stats mlperf-swe-head
curl --fail http://HEAD_NODE_IP:8080/health
# worker
docker stats mlperf-swe-worker
```

`/health` reports active/queued attempts and whether admission is open. A bridge
transport error or unknown job outcome can close admission; inspect logs and drain/stop
the remaining jobs before restarting. Do not blindly retry episode submissions. Runtime
logs live under each host's local state directory, while episode artifacts are in shared
results.

To stop, stop submissions first and wait for active jobs to finish. Then:

```bash
# worker
docker stop --time 60 mlperf-swe-worker
# head
docker stop --time 60 mlperf-swe-head
```

After stopping, verify there are no remaining jobs before reusing results or restarting
a training client. For an unchanged image/config, start the head with
`docker start mlperf-swe-head`, wait for readiness, then start the worker with
`docker start mlperf-swe-worker` on the worker. If the head restarted after a failure,
restart the worker to rejoin the new Ray session and rerun diagnostics. For new
image/flags, remove only these stopped containers and rerun `start.sh`. Never use
host-wide `ray stop --force` on nodes shared with training.

For rollback, stop both new containers and start the preserved `mlperf-swe-runner` on
the head. The shared result files are retained; switching back does not resume
interrupted attempts.

## Validation boundary

Local CPU tests exercise configuration, build-context staging, launch arguments,
diagnostic checks, adapter routing, and AReaL proxy recording. They do not build the
Docker image or establish live cross-node Ray/Apptainer execution. Complete the checks
above before treating the cluster as ready for training.
