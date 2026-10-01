# Run plan — KV Quant H100, 2×2 smoke group

**Status:** plan APPROVED by operator 2026-09-02 — provisioning pending
**Cells:** 4 · **Budget:** 1.20 h of the 3 h cap · **Est. cost:** ~$4 at $3.14/hr

## What this is trying to learn

Two things at once, and the second is why it goes first in the campaign.

**Scientific:** a complete 2×2 of KV-cache dtype × model size on one machine —
`{Qwen3-8B, Qwen3-14B} × {bf16, fp8_e4m3}`, H100 SXM5, TP1, max-num-seqs 32,
bucket-S prompts, OSL forced 512. Running all four on one instance removes
machine-to-machine variance from the comparison, which is the point of rule 10's
grouping.

**Operational:** this is the first real cell the pipeline has ever run. Four
version-sensitive unknowns resolve here, and every one of them fails *silently* —
the cell completes and writes nulls:

1. the `/metrics` names `vllm_metrics.py` targets, under vLLM 0.28.0 (the
   archive ran 0.21.0)
2. the streaming reasoning delta field — `reasoning_content` or `reasoning`
3. the startup-log format for engine args and KV-cache-as-allocated
4. **whether `nv-hostengine` and the DCP profiling fields work in a rented
   container** — the real risk. If they do not, we lose `tensor_active`,
   `dram_active` and `nvlink_tx/rx` campaign-wide, and the compute-vs-memory-bound
   discriminator with them. `bootstrap.py` fails loudly rather than producing 248
   cells of empty columns, and the fallback is a decision to take once.

## The cells

| # | cell_id | model | kv-cache-dtype |
|---|---|---|---|
| 1 | `kvquant_h100_qwen3-8b_bf16_b32` | Qwen3-8B | auto (bf16) |
| 2 | `kvquant_h100_qwen3-8b_fp8-e4m3_b32` | Qwen3-8B | fp8_e4m3 |
| 3 | `kvquant_h100_qwen3-14b_bf16_b32` | Qwen3-14B | auto (bf16) |
| 4 | `kvquant_h100_qwen3-14b_fp8-e4m3_b32` | Qwen3-14B | fp8_e4m3 |

Ordered 8B first: it is the cheaper model to discover a problem on, and cells 3–4
are worth nothing if cells 1–2 reveal the instrument is broken.

## Hardware

1× **H100 SXM5 80GB**, vast.ai, **150 GB disk**.

Disk is sized deliberately: 31.6 GB image + 16 GB (8B) + 28 GB (14B) + artifacts
≈ 78 GB. The default vast allocation would not hold the image alone.

## Budget

4 cells × (60 s warm-up + 600 s window + 120 s drain) = 52 min, plus a 20 min
allowance for image pull, two model downloads and vLLM startups = **1.20 h**,
against the 3 h hard cap. `provision.py launch` refuses the grouping if this is
wrong.

## What "done" looks like

- 4 × `summary.json`, each with `fatal: []`
- 4 × `power_profile_<cell>.png` showing real serving power, not a flat idle trace
- `run_record.json` and `timeline_<cell>.csv` per cell for the data team
- artifacts in HF and Drive under `artifacts/KV Quant/H100/...`, checksums verified
- 4 ledger records, tracker status copy updated
- **and a written answer to each of the four unknowns above**

## Known risks

| Risk | Handling |
|---|---|
| `nv-hostengine` blocked in an unprivileged container | `bootstrap.py` exits non-zero before any cell runs; report and stop |
| Metric names moved in 0.28.0 | Step 6 greps `/metrics` and reports actual names |
| SSH unavailable (team-account key restriction) | try `vastai attach ssh`, then `--onstart-cmd` key injection |
| Instance hangs | watchdog every 5 min; 3 h hard cap destroys it regardless |
| A gate fails | that is a *result*, not a failure of the run — upload anyway, record, report |

## Outcome

*(appended after the run)*
