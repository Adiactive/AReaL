# MLPerf SWE with AReaL's proxy

This integration runs the reference NeMo Gym/OpenHands agent on separate CPU sandbox
hosts. AReaL owns inference, session recording, rewards, and training. No hosted model
or sandbox service is used. Sandbox hosts need neither AReaL training dependencies nor
an accelerator runtime installed.

```text
AReaL rollout worker ── task + episode credentials ──► sandbox bridge /run
       ▲                                                   │
       │                                                   ▼
       │                                       reference Gym /run
       │                                                   │
       │                                       OpenHands in task.sif
       │                                                   │
       │                                       Gym model adapter
       │                                                   │
       └── AReaL proxy /chat/completions ◄── sandbox bridge ◄┘
                       │
                  inference engine
```

The bridge keeps each episode's proxy URL/key in memory. Gym and OpenHands see an opaque
model identifier instead of the credentials. A small subclass of Gym's model adapter
preserves this identifier; the bridge restores the requested model and supplies the
correct episode key for each call. OpenHands code is unchanged.

AReaL captures the inference engine's original token IDs, log probabilities, weight
versions, and prompt/completion masks. Gym supplies the evaluator's reward; Gym's
reconstructed transcript is not converted into training tokens.

Training and validation both use `MLPerfSWEWorkflow` in `workflow.py`, which wraps
AReaL's proxy and the existing `NeMoGymSWEAgent.run()` interface. It handles an attempt
through proxy export, then returns training trajectories or validation metric rows. Both
modes preserve recoverable failed attempts as zero-reward slots.

The bridge enforces each task's `max_output_tokens` as a per-turn completion cap,
including when OpenHands supplies no limit; a stricter agent limit is retained. AReaL's
HTTP proxy preserves Gym's chat-template options. Gym's separate reasoning conversion is
disabled because AReaL returns reasoning inline in message content.

## Sandbox cluster and unified image

Build one image from [runner_cluster/Dockerfile](runner_cluster/Dockerfile). The bundle
installs pinned Python/Apptainer/Gym/OpenHands/evaluator dependencies and copies this
directory's bridge, model adapter, configuration helper, and supervisor into the final
layers. There is no second Dockerfile or integration image to build. Neither the
training package nor accelerator dependencies are installed in this CPU image.

From the root of this AReaL checkout:

```bash
export SWE_RUNNER_BUNDLE="$PWD/examples/swe/mlperf/runner_cluster"
SWE_RUNNER_IMAGE=mlperf-swe-runner:cluster \
  bash "$SWE_RUNNER_BUNDLE/build.sh" \
  --integration-dir "$PWD/examples/swe/mlperf"
```

Build-time package/source access is required. Subsequent changes to the adapter reuse
Docker's cached dependency layers. Use the exact same image on every sandbox node; it
contains both the standalone fixed-model mode and AReaL mode. The reference pins remain
Gym `610a08ab5fe9f8f5fb5fff36b170429ea67f0f92` and OpenHands
`0d766ad06b2be64a42e6f0175b9ebcc4a06599d9`, with the MLPerf patches applied by the
bundle.

See [runner_cluster/CLUSTER.md](runner_cluster/CLUSTER.md) for the head/worker setup,
shared mount layout, image distribution, port requirements, node-pinned SIF tests,
service launch, and training validation. In that setup:

- The head starts `start.sh --role head --service areal`: Ray head, Gym, the bridge, and
  local sandbox tasks run in one container.
- Each worker starts `start.sh --role worker`: Ray workers and assigned sandbox tasks
  run in the same image. This node does not start another Gym or bridge service.
- SIFs are read-only at the same container path on both nodes. Attempt scripts and
  results use a shared writable directory; Ray and Apptainer temporary files stay
  node-local.
- AReaL uses one URL, the bridge on the head. It does not connect to each Ray worker.

Generate the complete sandbox head YAML with the runner bundle's
`configure.py --service areal`. See its `CLUSTER.md` for the single-command container
recipe and complete configuration input reference. The generator calls the adapter
internally, including custom ports; workers need no YAML. No fixed model endpoint is
needed when starting AReaL mode.

Only when migrating an existing **JSON-formatted** standalone `site.yaml`, use the
converter directly. The registry and model-adapter address must be reachable from every
sandbox worker:

```bash
python3 examples/swe/mlperf/configure_runner.py \
  --input "$STANDALONE_SITE_CONFIG" --output "$AREAL_SITE_CONFIG" \
  --advertised-host "$SANDBOX_PRIVATE_IP" \
  --ray-head-address "$RAY_HEAD_ADDRESS"
```

`SANDBOX_PRIVATE_IP` is the head's actual IPv4, and `RAY_HEAD_ADDRESS` is its Ray GCS
`IPv4:port` address from the cluster launch. For a legacy single-host runner, omitting
these options retains loopback callbacks and preserves any Ray address already in the
input. The adapter leaves SIF paths, shared result paths, concurrency and benchmark
budgets intact. It refuses to overwrite an existing output file. Default service ports
are:

| Port on the head | Service                                                   | Network access                            |
| ---------------- | --------------------------------------------------------- | ----------------------------------------- |
| 8080             | Bridge: `/run`, `/health`, cancellation, model forwarding | Private training network                  |
| 8083             | Reference SWE attempt API                                 | Loopback only                             |
| 8082             | Gym model adapter                                         | Head IPv4; reachable from sandbox workers |
| 8081             | Gym registry                                              | Head IPv4; reachable from sandbox workers |

The adapter forwards model calls to the local bridge at `http://127.0.0.1:8080/v1`;
making worker callbacks routable does not expose the raw Gym attempt API. `serve.py`
checks each Gym service's configured address before starting the bridge. Run exactly one
bridge process: its route registry and concurrency counter are in memory. The bridge
binds all interfaces inside the container, so the cluster's host-network mode requires
host/network firewall rules restricting the service and Ray ports to the intended
private peers.

```bash
curl --fail "http://$SANDBOX_PRIVATE_IP:8080/health"
```

`/health` reports bridge admission state and active/queued attempts. It does not execute
a SIF or verify inference. Run the cluster guide's node-pinned SIF probes and an AReaL
episode separately. The standalone `smoke-run.py` uses a fixed model endpoint and cannot
call the authenticated bridge without an AReaL session.

No packages or images are fetched by this integration at runtime. The image sets offline
Hugging Face and LiteLLM cost-map flags, including inside Apptainer's clean environment.
Use internal-only network routing/egress rules when requiring that generated shell
commands also cannot reach the Internet.

## AReaL runtime container

Use this checkout's code inside the runtime container, preserving the existing
accelerator dependencies and hardware configuration. No Gym, OpenHands, Apptainer, or
SIFs are needed there. The adapter uses AReaL's existing `httpx` dependency. The CPU
runner uses the existing Gym FastAPI/httpx/uvicorn stack.

Both directions must be reachable:

- AReaL rollout workers → sandbox host TCP 8080.
- Sandbox host → every AReaL proxy worker's advertised IP and dynamically allocated
  proxy port.

The proxy supplies a base URL such as `http://worker-ip:port`, **without `/v1`**. The
bridge forwards to its `/chat/completions` endpoint. Do not configure a separate fixed
inference URL or put the proxy admin key into Gym.

Host networking in the AReaL runtime container is the simplest way to expose these
dynamically allocated ports. Docker bridge networking requires routable advertised
addresses/ports, not just access to the training controller. Deploy the same source
revision on all training/proxy workers before starting the job.

Use [qwen3_8_27b_npu.yaml](qwen3_8_27b_npu.yaml) as the complete training configuration.
The [Qwen3.8-27B trial instructions](#qwen38-27b-npu-integration-trial) below list its
environment variables and launch commands. For another model or hardware layout, adapt
the model, optimizer, engine allocation, scheduler, and weight synchronization settings,
including the tool/reasoning parsers.

The admin key is used only by AReaL; each attempt receives a different generated session
key. Ensure internal host addresses are covered by `NO_PROXY` if the runtime environment
configures an HTTP proxy.

Use the native benchmark JSONL with `responses_create_params`, including
`metadata.instance_dict` as its original JSON string. The loader preserves every row
without Arrow field coercion. Optional validation data uses `valid_dataset.path`, also
pointing to native JSONL. Set `valid_dataset: null` to disable validation. For training,
task-provided generation settings take precedence; omitted
model/temperature/top-p/token-limit fields use the training configuration. Validation
overrides temperature and top-p with the settings below.

Check from the repository root in the AReaL runtime container before launching:

```bash
python -m examples.swe.mlperf.train --config "$MLPERF_AREAL_CONFIG" --check-only
```

This validates the configuration and native task records, queries the bridge's
`/health`, and reports the configured training and validation episode demand. It starts
no accelerator workers and submits no sandbox attempts. The same checks run before
normal training. An unhealthy or unreachable bridge fails before `PPOTrainer` starts.
Passing this check does not verify the reverse callback connection, SIF execution,
grading, or policy updates.

Then launch with the same configuration:

```bash
python -m examples.swe.mlperf.train --config "$MLPERF_AREAL_CONFIG"
```

Invoke the module directly with your existing single-controller `scheduler.type`
configuration. The deprecated `areal.infra.launcher.local` entrypoint forces SPMD mode
and cannot launch this integration.

Use single-controller mode (`AREAL_SPMD_MODE` unset or `0`). The entrypoint requires v1
actor/rollout engines, inline proxy mode, HF chat templates, and individual turn export.
It preserves group-level **reward** normalization, which uses episode metadata in this
checkout. Group-level **advantage** normalization is rejected because it cannot handle
variable numbers of exported turns; batch-level normalization or `null` is supported. It
does not silently change your training objective.

The Qwen recipe starts with one rollout group and four attempts per task; the head needs
capacity for at least four to execute them together. The head also runs sandbox tasks,
so no separate Ray worker container is needed for this test. Training and inference stay
on the AReaL nodes. Keep the downloaded task order with `shuffle: false`.

`max_concurrent_rollouts` counts groups. Training can submit up to that limit times
`gconfig.n_samples` episodes; validation uses `eval_gconfig.n_samples`. The bridge
queues excess attempts. Validation requires `rollout.max_head_offpolicyness: 0`, which
is the default, to keep each sweep on fixed weights. Nonzero asynchronous prefetch is
rejected: training attempts could otherwise hold sandbox slots while their proxies are
paused for evaluation.

The example client timeout is 7200s, outer rollout request timeout is 7500s, and proxy
inactivity timeout is 10800s. The outer timeout must exceed the client deadline plus its
cancellation allowance, with time left for trajectory export. Large queues can expire
idle proxy sessions. The bridge's started-job timeout defaults to 3600s and must cover
the configured agent and evaluator lifetimes. These are deployment limits; they do not
change the benchmark's turn, command, or test budgets.

## Qwen3.8-27B NPU integration trial

[qwen3_8_27b_npu.yaml](qwen3_8_27b_npu.yaml) is a complete two-step recipe for a single
16-device Ascend node. It allocates eight devices to the Megatron actor (TP4, PP2) and
four to one vLLM replica (TP4). The sandbox head runs separately. The checkpoint must
identify as the dense `qwen3_5` architecture; the selected Qwen3.8-27B checkpoint uses
that model family.

The recipe uses one task per batch and four attempts per task. PP2 requires at least
four exported training sequences, including when each attempt finishes in one turn. It
runs validation after each completed optimizer update, starting at update 1, and
disables MTP training, speculative decoding, and checkpoint saving. It uses a
32,768-token context and a 2,048-token per-turn output cap; these are integration-test
settings, not the full benchmark recipe. Actor microbatch capacity matches the context
limit. Full recomputation, optimizer swapping, and chunked LM-head loss reduce training
memory usage; actual memory fit still needs a training run.

After copying this integration into the AReaL container, set the container-visible paths
and the sandbox head URL:

```bash
export AREAL_SWE_MODEL_PATH=/path/to/Qwen3.8-27B
export AREAL_SWE_TRAIN_JSONL=/path/to/benchmark_r2e_gym_easy_train.jsonl
export AREAL_SWE_VALID_JSONL=/path/to/benchmark_r2e_gym_easy_val.jsonl
export AREAL_SWE_RUNNER_URL=http://SANDBOX_HEAD_IP:8080
export AREAL_SWE_OUTPUT_DIR=/path/to/writable/areal-training
export AREAL_SWE_TRIAL="qwen38-smoke-$(date +%Y%m%d-%H%M%S)"
export AREAL_PROXY_ADMIN_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export NO_PROXY='*' no_proxy='*'
export LITELLM_LOCAL_MODEL_COST_MAP=True
export HF_HUB_OFFLINE=1
unset AREAL_SPMD_MODE
umask 077

python -m examples.swe.mlperf.train \
  --config examples/swe/mlperf/qwen3_8_27b_npu.yaml --check-only
python -m examples.swe.mlperf.train \
  --config examples/swe/mlperf/qwen3_8_27b_npu.yaml
```

The worker environment retains `VLLM_WORKER_MULTIPROC_METHOD=spawn` for accelerator
subprocesses, `LITELLM_LOCAL_MODEL_COST_MAP=True` to avoid LiteLLM's cost-map download,
and `HF_HUB_OFFLINE=1` for local model/data access. Export the offline flags before
starting Python so they cover controller imports too. On the inspected runtime, Datasets
inherits Hub offline mode; the separate `HF_DATASETS_OFFLINE` and `TRANSFORMERS_OFFLINE`
flags are redundant.

OpenMP thread counts follow the launch environment or AReaL's CPU scheduling
specification. The inspected container sets `OMP_NUM_THREADS=1`; the recipe does not
override it. Current vLLM-Ascend ignores the old `VLLM_ASCEND_ENABLE_FUSED_MC2` and
`VLLM_ASCEND_ENABLE_NZ` variables. Those options now use `additional_config`
(`enable_fused_mc2` and `weight_nz_mode`); this recipe leaves their runtime defaults
unchanged.

Run from the AReaL checkout root. The training controller starts its own proxy and
inference processes; no fixed model endpoint is required on the sandbox host. Use a
fresh trial name for each run. Logs and rollout records use `AREAL_SWE_OUTPUT_DIR`. A
completed update with all-equal group rewards does not establish a useful policy
gradient; inspect rewards and exported trajectories.

## Benchmark validation

The example YAML enables validation with the fields below. Set `AREAL_SWE_VALID_JSONL`
to the native validation JSONL path as shown in the launch commands above. Set
`valid_dataset: null` to disable validation; otherwise every scheduled sweep evaluates
the full supplied split.

```yaml
valid_dataset:
  path: ${oc.env:AREAL_SWE_VALID_JSONL}

eval_gconfig:
  n_samples: 4
  temperature: 0.1
  top_p: 0.95
  max_tokens: ${gconfig.max_tokens}
  max_new_tokens: ${gconfig.max_new_tokens}

validation:
  start_step: 1  # Exercise validation during the two-step integration trial.
```

Preflight requires `rollout.max_head_offpolicyness: 0`, `dynamic_bs: false`, and
positive, equal values for `rollout.consumer_batch_size` and `train_dataset.batch_size`.
The example already satisfies these conditions. They let training attempts drain before
evaluation, so paused training jobs do not occupy the shared runner's sandbox slots
during validation. Actor and rollout must also use separate scheduling: colocated
rollout onload can resume training admission before validation.

Keep non-greedy sampling and the validation dataset's default `shuffle: false` and
`drop_last: false`. The token limits above inherit the selected training recipe; the
shorter smoke limits do not reproduce the full reference recipe. Validation overrides
task temperature/top-p and caps per-turn output at `eval_gconfig.max_new_tokens`,
retaining any stricter task cap.

The default schedule matches the reference: start after completed optimizer update
**20** and evaluate every update (`validation.interval: 1`). Each complete sweep reports
whether **pass@4 >= 0.7** (`validation.target: 0.7`). Training continues to its
configured step or epoch limit even when the target is reached. Remove the
`start_step: 1` override for that schedule. A two-step trial would otherwise finish
without validation. There is no initial or forced final evaluation; the schedule uses
`validation.*`, independently of the generic `evaluator.*` settings. Set
`validation.target: null` to disable target comparison while retaining scores.

Every sweep visits the entire supplied validation dataset, with exactly four attempts
per task. The official 251-task JSONL therefore runs **1,004 attempts**;
`valid_dataset.batch_size` does not limit the sweep. A task passes when at least one of
its four attempts resolves the issue. Reported `pass_at_4` is that observed fraction of
tasks; `pass_at_1` is the fraction of successful attempts, and `pass_all_4` is the
fraction of tasks with all four attempts successful.

Expected runner/proxy failures and empty or invalid results count as failed, zero-reward
attempts. Their slots remain in the denominator and their siblings continue. The shared
workflow returns one metric row per validation attempt; validation does not need
placeholder tokens. Exceptions outside its recovery classes, direct cancellation,
missing groups, or duplicate slots abort the sweep rather than publishing a partial
score. The remaining core exception-handling limitations are listed below.

The trial's log directory contains `validation/step_000020.json` (one file per completed
sweep) with aggregate scores and every task/attempt outcome, plus
`validation/status.json` with the final status: `target_reached`, `target_not_reached`,
`not_evaluated`, `completed` when the target is disabled, or `aborted`. Metrics are also
logged under `mlperf_validation/`, including `pass_at_4`, `pass_at_1`, `pass_all_4`,
`failed_attempts`, task/attempt counts, and sweep duration. The final target status
reflects the latest completed sweep; earlier target-reaching results remain in their
step files even if a later score drops below the target.

## Failure behavior

- Context exhaustion is returned by AReaL as HTTP 400 with a structured
  `context_length_exceeded` error. The bridge preserves this specific error so
  Gym/OpenHands terminates the attempt instead of retrying a generic HTTP 502. Gym still
  grades the existing patch and reports its masking metadata. The terminal signal adds
  no synthetic tokens to AReaL's training records.
- Valid completed Gym results retain their binary reward, including results with
  `mask_sample=true`, a reported agent/evaluator timeout, or an agent error kind. The
  pinned
  [NeMo-RL result conversion](https://github.com/NVIDIA-NeMo/RL/blob/f987c0596af2bd6cfb9958e768deb8df951d7504/nemo_rl/experience/rollouts.py#L1624)
  consumes Gym's reward without filtering on this metadata. One such result does not
  reject its siblings. This matches reference result handling; it does not implement the
  reference trainer's separate token-length/logprob filtering.
- The shared workflow recovers `NeMoGymSWEError`, empty proxy exports, and
  request/timeout errors. Training retains a failed attempt as a reward-zero placeholder
  with every token loss mask set to zero. Its reward still enters group normalization,
  and valid siblings retain their trajectories. Cached partial interactions from a
  failed attempt are not trained. Validation retains the same reward-zero slot in its
  metric denominator.
- A completed patch that fails tests is different: its reward is zero, but its real
  model interactions remain trainable. A valid completed result carrying
  `mask_sample=true` is not automatically converted into a placeholder.
- Each completed or recovered attempt records `mlperf_attempt_reward` and
  `mlperf_attempt_failed` statistics, with `mlperf_attempts` and
  `mlperf_failed_attempts` counters. Recovery also logs the error class and task/sample
  identity. AReaL's trajectory dumper omits zero-loss placeholders, so use these
  statistics/logs to observe failed training attempts.
- Cancellation revokes the episode route and cancels outstanding model requests. It does
  **not** prove that Gym's Ray job or Apptainer containers stopped. Started jobs retain
  their slot until the reference runner returns; queued jobs cancel.
- An unknown job outcome (transport failure, timeout, or Gym 5xx) closes new admission.
  Inspect/drain the sandbox jobs before restarting the whole runner service. Restarting
  only the bridge loses the in-memory registry.
- Neither the client nor bridge retries episode submissions. Keep caller retries
  disabled to avoid duplicate sandbox jobs.

This change is limited to the MLPerf workflow and examples. These core limitations
remain:

- Megatron rejects an optimizer batch with no trainable tokens. A mixed batch can
  contain failed slots, but a batch consisting entirely of placeholders cannot yet
  complete the reference's zero-loss optimizer update.
- `NeMoGymSWEAgent` still wraps some unexpected request exceptions and invalid JSON as
  `NeMoGymSWEError`. The shared wrapper recovers these too; exact parity with the
  reference's exception classification needs a separate core fix.
- A proxy cleanup failure can replace an earlier cancellation or unexpected exception
  with a recoverable HTTP error. Direct cancellation and unexpected errors reaching the
  shared wrapper still propagate, but preserving the original error through cleanup
  needs a separate core fix.

## Validation boundary

CPU tests cover task and credential forwarding, concurrent routes, cancellation, error
handling, configuration, and the real AReaL proxy's token/logprob/version capture with a
stub inference engine. Run them from the repository root:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest -q \
  tests/test_nemo_gym_swe.py tests/test_mlperf_runner_bridge.py \
  tests/test_mlperf_swe_configuration.py tests/test_mlperf_preflight.py \
  tests/test_mlperf_proxy_roundtrip.py \
  tests/test_mlperf_validation.py tests/test_mlperf_workflow.py \
  tests/test_mlperf_validation_trainer.py
```

These tests do not qualify the Docker build, real OpenHands/Apptainer execution, network
callbacks across sandbox and training hosts, or an accelerator training update. Run a
small real job after deployment and check nonempty proxy exports, evaluator rewards, and
completed training updates before increasing concurrency. This is an internal AReaL
integration, not a claim of an MLPerf-compliant submission.
