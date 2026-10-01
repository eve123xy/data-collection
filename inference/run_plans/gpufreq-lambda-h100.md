# GPU Frequency sheet — 54 cells on Lambda H100 SXM5

Status: **awaiting operator approval of this plan.** Box
`f03e13d3a7ee407ebdbaf76ebdca723d` is launched and booting (approved
separately); it is assigned batch 1 below.

## What we are trying to learn

J/token as a function of SM clock, across three models and three workload
regimes. This is the only sheet that touches the SM-clock axis, and the only
one that runs on Lambda — vast.ai blocks `nvidia-smi -lgc`.

The specific question the ladder has to answer is **whether J/token vs clock is
U-shaped**. At low clock the fixed ~80 W idle floor is amortised over collapsing
throughput, so efficiency should get *worse* below some minimum. The original
tracker ladder stopped at 600 MHz and could have reported the right half of a U
as a clean monotonic trend. The 345 MHz rung (added 2026-09-07, operator
decision) puts the hardware floor in the sample so the minimum is bracketed
rather than assumed.

## The ladder

`1980 / 1605 / 1200 / 900 / 600 / 345` MHz.

**1605, not 1600.** 1600 MHz is not an offered SM clock on this part — the H100
SXM5 exposes 110 clocks on a 15 MHz grid from 1980 down to 345. The cells keep
`1600mhz` in their `cell_id` so the ledger and tracker do not fracture, and
carry `clock_mhz_actual: 1605` (0.31 % off intended). **Released metadata must
say 1605.** Every frequency cell asserts its target is in `SUPPORTED_CLOCKS`
before pinning, so an unsupported value stops the run instead of silently
landing on a neighbour.

## Grouping, and why

6 clocks x 3 models x 3 blocks = 54 cells. Batched **model-major**: one model,
one block, all six clocks per box.

| batch | model | block | cells | budget h |
|---|---|---|---|---|
| 1 | Qwen3-8B | b32 | 6 | 1.63 |
| 2 | Qwen3-8B | work | 6 | 2.63 |
| 3 | Qwen3-8B | burst | 6 | 2.63 |
| 4 | Llama-3.1-8B-Instruct | b32 | 6 | 1.63 |
| 5 | Llama-3.1-8B-Instruct | work | 6 | 2.63 |
| 6 | Llama-3.1-8B-Instruct | burst | 6 | 2.63 |
| 7 | Qwen3-30B-A3B | b32 | 6 | 1.63 |
| 8 | Qwen3-30B-A3B | work | 6 | 2.63 |
| 9 | Qwen3-30B-A3B | burst | 6 | 2.63 |

Budgets are `provision.py budget_hours()` output, not restated from memory.
Every batch is under the 4.0 h `pins.MAX_INSTANCE_HOURS` cap; the whole sheet on
one box would be 18.0 h and `launch` would refuse it.

**Why model-major and not clock-major.** Clock-major (one clock, all 3 models,
all 3 blocks) would need only 6 boxes instead of 9 — but each box would download
all three models, 92 GB, versus one model per box here. 9 downloads total
against 18. The 20-minute fixed allowance per box does not cover 92 GB.

Within a batch, cells run **descending clock, 1980 first**. The top rung is the
reference point, so a broken box is discovered on the cell we can most easily
recognise as wrong.

## Cost

Summed budget **20.7 h at $4.29/hr = ~$89**. That is 18.0 h of actual window
plus 3.0 h of per-box fixed overhead (image pull, model load, vLLM start), which
is the price of the 4 h cap and is not recoverable without raising it.

Serial, that is ~21 h wall clock. **Requesting up to 3 concurrent Lambda boxes**,
which brings it to ~7 h at the same total cost. The standing 5-provision grant
explicitly excludes Lambda, so this needs its own yes.

## Disk

Per box: image 20 GB + weights (16 GB for the 8B models, 60 GB for
Qwen3-30B-A3B) + artifacts ~25 GB. Worst case **105 GB**, on batches 7-9.
Lambda's `gpu_1x_h100_sxm5` ships a fixed local disk well above that; verified
per box at setup rather than assumed.

## Preflight (run 2026-09-08, before this plan was shown)

- `HF_TOKEN` authenticates as `<hf-owner>`, role **write**;
  `<hf-owner>/LLM-Power-Runs-Main` exists.
- Google service-account credential loads; campaign Shared Drive visible.
- `status.py` reports no runs-repo warning: 265 cells, 161 ok, 57 pending.
- No `nvidia-smi topo -m` check: every cell here is TP1, single GPU.

## The fabric gate

`lambda_provision.py check` runs before any cell is dispatched and **refuses a
box whose `Fabric / State` reads `In Progress`**. On 2026-09-07 a box came up
with `nvidia-smi` entirely healthy — right part, right memory, all six rungs in
`SUPPORTED_CLOCKS` — and could not run CUDA at all (`error 802: system not yet
initialized`), because the H100 was passed through from an HGX baseboard without
its NVSwitches. It billed ~20 minutes and ran nothing. See OPERATIONAL_LEARNINGS
2.48.

## What "done" looks like

54 ledger records in `<hf-owner>/LLM-Power-Runs-Main`, each with
`clock_mhz_actual` and per-GPU `mean/min/max_mem_clock_mhz`, and 54 status cells
in the Drive tracker across columns 3-7 and 9 of the GPU Frequency sheet.

A cell that fails a gate is a **result** and is recorded as one. The one failure
that is an alert, not a result, is a clock that did not pin — that invalidates
the axis rather than the cell.

## Outcome

_Appended after the run._

- `gpufreq_h100-lambda_qwen3-8b_1980mhz_b32` — `run_failed`, 2026-09-07, on a
  prior box: `cuda_error_802_system_not_yet_initialized`. Host fabric fault, not
  a cell result. Re-runs as part of batch 1.
