# Run plan — mass launch, closed/forced path

**Status:** running under the operator's standing grant of 2026-09-03
**Cells:** 111 pending of the 149 on this path · **Ceiling:** 5 concurrent provisions, $22/hr

## Why this path can launch and the others cannot

`closed/forced` is the only path that is both pinned and unaffected by the two
defects the driver smoke found:

| Defect | Effect | Does it touch closed/forced? |
|---|---|---|
| httpx `max_connections=100` caps "unbounded" in-flight client-side | the 81 unbounded cells queue in the client pool, not in vLLM, so server-side saturation is never observed | **No.** These cells cap at 32 in-flight, far below 100 |
| `--prompts` read unconditionally before the replay branch | a replay cell dies with `FileNotFoundError` on a clean box | **No.** These cells use the prompts file, so it is always present |

Both are frozen instrument code and both are with the operator. The 84 open-loop
cells wait; these 111 do not.

## What validated this path

Six cells across three commits, and the agreement between them is the evidence
that matters more than any single run:

| Run | J/token | achieved batch |
|---|---|---|
| `kvquant_h100_qwen3-8b_bf16_b32` #1 | 0.13667 | 31.581 |
| #2 (after six collection fixes) | 0.13485 | 31.684 |
| #3 (after the scraper moved out of process) | 0.14037 | 31.552 |

Achieved batch agrees to **0.4%** across all three — and achieved batch is the
quantity the scraper change actually touches. The #3 power drift is a
contaminated idle baseline on that rental, diagnosed and recorded, not an
instrument change.

`tp_h100_qwen3-14b_tp2_b32` additionally validated the multi-GPU path: two cards
agreeing to 1.7%, `nvlink_tx/rx` non-zero on 99.4% of in-window samples, four
populated telemetry CSVs.

## Grouping

Cells share an instance only when they share GPU type **and** count — the rule
`provision.py launch` enforces. Within a group they are ordered by model so one
weight download serves several cells.

| Group | Cells | Provisions |
|---|---|---|
| H200 x1 | 39 | 4 |
| A100 x1 | 24 | 3 |
| H100 x1 | 20 | 2 |
| H200 x2 | 9 | 1 |
| A100 x2 / x4 | 8 | 2 |
| H100 x2 / x4 | 7 | 2 |
| H200 x4 | 4 | 1 |

Ten cells per provision is ~2.6 h against the 3 h hard cap. That headroom is
deliberate: two groups this session overran their budget because of blockers
found mid-run, and the cap is enforced by the orchestrator rather than a timer.

**Single-GPU first.** 83 of the 111 cells need one GPU, and single-GPU is the
most-validated shape in the campaign. Multi-GPU follows once the first wave
lands. A100 multi-GPU is ordered **last** of all: Ampere cannot serve the DCP
fields, so `nvlink_tx/rx` — the mechanism the TP knob exists to measure — is
unobtainable there, and A100 multi-GPU has never run at all.

## Per-provision requirements

Every box, before an agent touches it:

- offers filtered on `cuda_max_good>=13.0` and driver >= 580.65, or the image
  cannot run CUDA at all (OPERATIONAL_LEARNINGS 2.16)
- `torch.cuda.init()` preflight, which catches that in seconds rather than after
  the image pull, the weight download and a vLLM start
- ssh key injected at boot by `provision.py launch`; `vastai attach ssh` has
  failed on 3 of the last 8 boxes
- `push-creds` verified: scripts count, `env`, `cred`, `commit`, `manifest`
- one agent per provision, its own scratchpad directory, the `run-cell` skill
  invoked first

## What "done" looks like

- every cell recorded in the ledger with a non-null `git_commit`, and **all
  cells within one provision on the same commit**
- tracker synced and count-checked after every cell, never batched to the end
- artifacts in both sinks, checksum-verified
- instances destroyed immediately on completion, `sweep` clean

## Known caveats carried into analysis

- `nvlink_rx` is ~1.75x `nvlink_tx` on both GPUs of a TP2 pair. With two peers
  GPU0 tx should equal GPU1 rx, so these are **not a matched pair** — rx counts
  bytes tx does not. Usable as a relative TP-traffic signal only.
- A100 cells carry 13 CORE fields, Hopper cells 23. Cross-architecture claims
  may use only the CORE set; anything resting on `tensor_active` / `dram_active`
  is Hopper-only.
- `run_meta.max_model_len` and `in_flight` record pins/argparse defaults rather
  than what applied. Correctable post-hoc from `engine_args_raw`, which is
  recorded per cell.
- The idle-baseline validity gate and the per-GPU idle-card gate are both
  proposed and unapproved. Until they land, a contaminated idle baseline can
  produce a negative `energy_above_idle_j` silently, and a dead card in a TP
  pair can pass the idle-card gate by averaging.

## Outcome

*(appended as waves complete)*
