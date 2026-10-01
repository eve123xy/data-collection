# Run plan — TP2 multi-GPU smoke (H100 SXM5)

**Status:** plan APPROVED 2026-09-02; re-pointed from A100 to H100 the same day
**Cells:** 1 · **Budget:** 0.65 h of the 3 h cap · **Est. cost:** ~$1.45 at $2.19/hr

## What this is trying to learn

The one thing no single-GPU cell can tell us: **does the pipeline work when vLLM
is sharded across two GPUs.** Until a TP≥2 cell has run, 24 of the 36 TP Number
cells — plus the TP2 rows on KV Quant and Weight Quant — sit on a code path that
has never executed.

Three things are new here and nowhere else:

1. **Tensor-parallel launch.** `--tensor-parallel-size 2`, two GPU workers, and
   a startup sequence that can fail in ways TP1 never shows.
2. **`nvlink_tx_bytes` / `nvlink_rx_bytes` become real.** On the TP1 smoke they
   read 0, which is *correct* — no NVLink peer exists. At TP2 they carry the
   activation traffic between shards, and they are the mechanism behind the
   whole TP knob. This is the first cell that can tell us whether we are
   actually capturing them.
3. **Per-GPU telemetry, doubled.** `dcgm_capture.py` writes one CSV per GPU.
   Two GPUs is the first test that the slicing, gating and power summation
   handle more than one device — and power is the campaign's dependent
   variable, so summing it wrongly is a silent, campaign-wide error.

## The cell

| # | cell_id | model | GPU | tp |
|---|---|---|---|---|
| 1 | `tp_h100_qwen3-14b_tp2_b32` | Qwen3-14B | 2× H100 SXM5 80GB | 2 |

**Why only one cell.** `provision.py launch` refuses a grouping whose cells do
not share GPU type *and count*, so the natural companion — `tp_a100_qwen3-14b_tp1_b32`
at 1 GPU — cannot ride along. That refusal is correct and is not worked around.
TP1 is already covered by the KV Quant smoke, so one TP2 cell is the whole of
what is new.

**Why H100, not A100.** The original plan chose A100 as the cheapest 2-GPU
rental. That is now impossible: the ten DCP profiling fields are unobtainable on
Ampere in an unprivileged container (OPERATIONAL_LEARNINGS 2.15), and
`nvlink_tx/rx` are among them — so an A100 TP2 cell cannot measure the very
mechanism this run exists to validate. H100 keeps the scarce H200 supply for the
130 cells that require it.

## Hardware

2× **H100 SXM5 80GB**, vast.ai, **150 GB disk** (9.6 GB image + ~28 GB weights +
compile cache + artifacts ≈ 45 GB; 150 GB is the smallest comfortable ask).

Prefer a high-bandwidth host: at ~800 Mbps the image and weights take ~6 min,
at ~7 Gbps under a minute. The difference is larger than the price difference
between offers.

## Budget

1 cell × (60 s warm-up + 600 s window + 120 s drain) = 13 min, plus ~25 min for
image pull, weight download and a two-worker vLLM start = **0.65 h**.

## What "done" looks like

- `summary.json` with `fatal: []`
- **two** per-GPU telemetry CSVs, both populated
- `nvlink_tx_bytes` / `nvlink_rx_bytes` **non-zero** on both GPUs — if they are
  zero here, either the capture is wrong or the shards are not talking, and
  either answer is worth the rental
- power summed across both GPUs, with the per-GPU values also retained
- a written answer to: does anything in the summarise/gate path assume one GPU?

## Known risks

| Risk | Handling |
|---|---|
| The rented pair is not NVLink-connected (PCIe-only host) | check `nvidia-smi topo -m` before the cell; if there is no NV link, nvlink fields read 0 for a *topological* reason and that must be recorded, not treated as a capture bug |
| Gates assume a single GPU | that is exactly what this run is for; a failure here is a result |
| Two-worker start is slower than budgeted | 3 h hard cap still binds |

## Outcome

*(appended after the run)*
