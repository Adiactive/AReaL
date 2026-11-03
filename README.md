# Qwen3.5-2B MTP training on Geometry3K

Figures and extracted metrics for AReaL PR #1870, from `geometry3k-grpo/trial4`.

- Model: Qwen3.5-2B.
- Training: Megatron-Bridge, `enable_mtp_training: true`,
  `mtp_loss_scaling_factor: 0.1`, learning rate `5e-7`.
- Weight updates: `xccl`, `use_bridge_for_update_weights: true`.
- Rollout: vLLM MTP speculative decoding with four proposed tokens per draft.
- Parallel configuration: actor `megatron:d8p1t1`, rollout `vllm:d8p1t1`.
- Data: 342 consecutive completed training steps from the saved `main.log` tables.
  Values retain the precision printed in those tables.
- Source timestamps: 2026-10-24 21:38:45 through 2026-10-25 03:01:15, as recorded in the
  log; its timezone is not specified.
- Run metadata records branch `ascend` and Git commit
  `98598d07c120f3ad05c7a3794d461c5bef7809bf`. This development run predates the rebased
  PR head; the metadata does not capture local source modifications.

The figures show every recorded point and a trailing arithmetic mean over 20 training
steps. The first and last 20-step summary windows are steps 1–20 and 323–342. Acceptance
is the logged ratio of accepted draft tokens to proposed draft tokens. The window
summaries average the logged per-step rates.

| Metric                  | First 20 steps | Last 20 steps |                    Change |
| ----------------------- | -------------: | ------------: | ------------------------: |
| MTP loss                |      0.7677865 |     0.5416615 |                   -29.45% |
| Rollout acceptance rate |      42.74485% |     51.34835% | +8.6035 percentage points |

`trial4-metrics.csv` contains the unsmoothed series, including mean accepted draft
length as a cross-check. `summary.json` records the summary values and SHA-256 hashes of
the source log and config. The original log and config are not included.
