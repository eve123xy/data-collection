# Run plan — driver-mode smoke, free generation and arrival processes (H200)

**Status:** awaiting operator approval
**Cells:** 4 · **Budget:** 1.55 h of the 3 h cap · **Est. cost:** ~$6.20 at $3.98/hr

## What this is trying to learn

**Correction to the earlier path taxonomy, which this plan is built on.** I had
described the campaign as closed-loop versus "open-loop arrival rate". Reading
`plan/cells_v1.json` rather than the prose, there are **five** driver modes, and
the KV Quant smoke exercised exactly one of them:

| `(mode, osl)` | Cells | Status |
|---|---|---|
| `closed / forced` | 149 | smoked (KV Quant 2×2) |
| `replay / free` | 63 | **never run** |
| `poisson / free` | 18 | **never run** |
| `closed / free` | 9 | **never run** |
| `closed / thinking` | 6 | **never run** |
| `poisson / thinking` | 3 | **never run** |

That is **99 cells on four unexercised driver behaviours** — 40% of v1. This
group runs the cheapest representative of each, on one machine, so that all four
are validated before any of them mass-launches.

What is new in each, and nowhere else:

1. **`closed / free`** — `ignore_eos=False` and `max_tokens` 2048. The server
   decides when a request ends. Every length covariate in the law fits comes
   from realized token counts, so this is the first test that we record them
   correctly when generation is not forced to 512.
2. **`closed / thinking`** — thinking mode at `MAX_MODEL_LEN` 16384. This is
   where the **streaming reasoning delta field** matters (`reasoning_content`
   vs `reasoning`), which was listed as unknown #2 for the KV Quant smoke but
   **could not be answered there**, because every KV Quant cell runs
   `thinking: False`. It is answerable only here.
3. **`poisson / free`** — an actual arrival process with **unbounded in-flight**.
   Nothing so far has ever offered load the server could not immediately absorb.
   Queueing, admission and saturation behaviour all appear for the first time.
4. **`replay / free`** — BurstGPT trace replay. Also the driver half of the
   **GPU Frequency** sheet, so smoking it here de-risks 30 of those 45 cells in
   advance, leaving only clock pinning genuinely new when Lambda is ready.

## The cells

All Qwen3-8B on 1× H200, so one weight download serves all four.

| # | cell_id | mode | window |
|---|---|---|---|
| 1 | `islosl_h200_qwen3-8b_chat-sharegpt_b32` | `closed / free` | 600 s |
| 2 | `islosl_h200_qwen3-8b_chat-sharegpt` | `closed / thinking` | 600 s |
| 3 | `poisson_h200_qwen3-8b_r0.25_unbounded` | `poisson / free` | 1200 s |
| 4 | `burstgpt_h200_qwen3-8b_burst_unbounded` | `replay / free` | 1200 s |

Ordered cheapest-and-simplest first: the two 600 s closed-loop cells before the
two 1200 s unbounded ones, so a broken free-generation path is discovered in ten
minutes rather than forty.

## Hardware

1× **H200 SXM 141GB**, vast.ai, **150 GB disk**.

H200 is not a choice — every Poisson, BurstGPT and ISL_OSL cell in the manifest
is pinned to H200 SXM 141GB, so there is no cheaper variant that validates this
path. Supply is thin (2 offers at the time of writing, $3.98 and $5.00/hr),
which is the main scheduling risk.

## Budget

Windows 600 + 600 + 1200 + 1200 = 60 min, plus 4 × (60 s warm-up + 120 s drain)
= 12 min, plus ~20 min for image pull, one weight download and four vLLM starts
= **1.55 h** against the 3 h cap.

## What "done" looks like

- 4 × `summary.json`, each with `fatal: []`
- **realized ISL/OSL token counts recorded per request** on cells 1–4, since
  free generation is exactly where nominal config values stop being usable
- a written answer to the streaming reasoning delta field, from cell 2
- for cells 3–4: evidence the driver actually offered the intended load —
  achieved arrival rate against nominal, and queue depth over time
- confirmation that an unbounded-in-flight cell terminates cleanly rather than
  running the server out of KV cache

## Known risks

| Risk | Handling |
|---|---|
| H200 supply is thin | if no offer is available at approval time, this group waits; it does not move to a different variant |
| Unbounded load saturates the server | that is the phenomenon under study, not a failure — record it |
| Thinking mode blows past `MAX_MODEL_LEN` | gate should catch and report; a truncation rate is a result |
| `poisson` at r=0.25 is too gentle to be interesting | it is chosen to be *safe*, not interesting — the point is that the driver works |

## Outcome

*(appended after the run)*
