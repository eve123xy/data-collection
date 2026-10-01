# Run plan — wave 5, first round under the 4 h cap

**Status:** approved by the operator 2026-09-04 ("lets do it, you can up the
budget to 4 hours for this round")
**Instances:** 3 · **Cells:** 21 · **Burn:** $16.24/hr · **Cap:** 4.0 h

## What this is trying to learn

Three independent axes, none of which touches Llama-3.1-70B — those are **held**
pending the investigation of the client-vs-server token divergence (below).

| Box | Offer | Hardware | $/hr | Cells | Budget |
|---|---|---|---|---|---|
| 49860546 | 29522440 | 1x H200 SXM 141GB | 4.011 | 10 closed-loop TP1 | 2.5 h |
| 49860554 | 48539238 | 2x H200 SXM 141GB | 8.895 | 7 Poisson Qwen3-32B | 3.0 h |
| 49860564 | 47141476 | 2x A100 SXM4 80GB | 3.398 | 4 A100 TP2 | 1.2 h |

**Box 1 — closed-loop TP1, 10 cells.** Only two models (Qwen3-8B, Qwen3-32B)
across three sheets, so one weight download serves five cells each and the
server is reloaded only on a flag change. This is the grouping the 3 h cap could
not hold: at 2.5 h it fits the new 4 h cap with margin.

KV Quant: `qwen3-8b_bf16`, `qwen3-8b_fp8-e4m3`, `qwen3-32b_bf16`,
`qwen3-32b_fp8-e4m3`. TP Number: `qwen3-8b_tp1`, `qwen3-32b_tp1`. Weight Quant:
`qwen3-8b_bf16`, `qwen3-8b_fp8`, `qwen3-32b_bf16`, `qwen3-32b_fp8`.

**Box 2 — Poisson Qwen3-32B, 7 cells.** The full arrival-rate ladder on one
model: r0.25, r0.5, r1, r2, r4, then r1-think, then r8 **last** — the tracker
reconciliation ordered 8 req/s last so the lower rates reveal saturation before
the rate most likely to be past it. 1200 s windows, so seven cells is the most
that fits.

**Box 3 — A100 TP2, 4 cells.** The Ampere half of the TP axis, against the
Hopper TP2 pairs already recorded. Qwen3-8B, 14B, 32B, 30B-A3B.

**This box carries a mandatory extra gate.** The previous A100 TP2 rental was a
2x 40GB box (2.19) and had no NVLink (`SYS`). This offer reports 80 GB/gpu, but
the agent must assert `nvidia-smi topo -m` shows an `NV#` pair before running a
cell, and stop if it does not — Ampere cannot measure nvlink traffic at all, so
a PCIe pair would produce cells that cannot be read against the Hopper
reference.

## Not in this wave

- **All 10 remaining Llama-3.1-70B cells** — held pending investigation.
- **TP4 (4 cells).** The only 4x H200 offer is $23.46/hr, which breaches the
  $22/hr cap on its own. Not a judgement call; the cap refuses it.

## What "done" looks like

21 x `summary.json` with `fatal: []`, both sinks verified per cell, ledger and
tracker synced after every cell.

## Outcome

*(appended after the run)*
