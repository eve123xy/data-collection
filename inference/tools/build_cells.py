"""LOCAL, run once. Turn the tracker into plan/cells_v1.json.

Every axis gets exactly one definition here (PIPELINE_POSTMORTEM R1), and every
row is validated before anything is provisioned (R2).

    python tools/build_cells.py
"""

import collections
import copy
import json
import re
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pins

# Tracker clock value -> clock actually pinned. Only 1600 differs: it is not an
# offered SM clock on H100 SXM5. Measured on the device, not assumed.
CLOCK_SUBSTITUTIONS = {1600: 1605}

# Rungs the operator's tracker does not have a column for, synthesised here so a
# regeneration reproduces the full grid without anyone hand-editing the sheet.
#
# 345 MHz is the H100 SXM5 hardware floor. The tracker's original ladder
# (1980/1600/1200/900/600) covered 78% of the supported range and left the bottom
# 255 MHz unsampled with no recorded rationale. DESIGN_LIMITATIONS asks for an
# explicit per-axis range decision; J/token vs clock is expected to be U-shaped,
# so a ladder stopping at 600 could miss the minimum and report the right half of
# a U as a clean monotonic trend. Operator decision 2026-09-07.
#
# `col` is the grid column the synthesised cells claim. It must not collide with
# a column the tracker already uses -- validate() enforces that.
# col 9, not 8: column 8 on this sheet carries the operator's per-block notes
# ("Closed loop, max in flight = 32", ...) on every data row, and claiming it
# would have overwritten them. _assert_column_free enforces that now.
EXTRA_RUNGS = {"GPU Frequency": [{"axis_value": "345", "col": 9}]}


def _assert_column_free(ws, sheet, col, rows, what):
    """A synthesised cell may only claim a grid column the tracker is not using.

    The tracker is the operator's hand-maintained document; writing a run status
    into a column that already holds their notes destroys them silently.
    """
    clashes = [r["tracker_row"] if "tracker_row" in r else r["tracker_ref"]["row"]
               for r in rows]
    busy = [rr for rr in sorted(set(clashes)) if ws.cell(rr, col).value is not None]
    if busy:
        raise SystemExit(
            f"[FATAL] {what} wants column {col} on '{sheet}' but the tracker "
            f"already has content there at row(s) {busy}: "
            f"{ws.cell(busy[0], col).value!r}. Pick an empty column.")


def synthesise_extra_rungs(rows, sheet, ws):
    """Clone each existing rung's rows onto the extra axis values for `sheet`.

    Takes the LOWEST existing rung as the template so block/gpu/model structure
    is inherited exactly, then overrides the axis value and grid column. Returns
    only the new raw records; the caller appends them.
    """
    extras = EXTRA_RUNGS.get(sheet) or []
    if not extras or not rows:
        return []
    used_cols = {r["tracker_col"] for r in rows}
    lowest = min(rows, key=lambda r: int(r["axis_value"]))["axis_value"]
    template = [r for r in rows if r["axis_value"] == lowest]
    out = []
    for spec in extras:
        if spec["col"] in used_cols:
            raise SystemExit(f"[FATAL] EXTRA_RUNGS column {spec['col']} for {sheet} "
                             f"is already used by the tracker")
        _assert_column_free(ws, sheet, spec["col"], template,
                            f"extra rung {spec['axis_value']}")
        for r in template:
            n = dict(r)
            n["axis_value"] = spec["axis_value"]
            n["tracker_col"] = spec["col"]
            n["synthesised"] = True
            n["rung_note"] = EXTRA_RUNG_NOTE.get((sheet, spec["axis_value"]))
            n["ref_note"] = (f"added rung: {spec['axis_value']} MHz hardware floor "
                             f"(new column {spec['col']})")
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# TP4/TP8 capacity blocks (operator decision 2026-09-05).
#
# Four open-loop BurstGPT cells offer more load than a TP2 server can absorb
# (11.66 and 10.28 req/s against a measured 7.2 for Qwen3-32B and 6.6 for
# Llama-70B), so the client backlog diverges and the driver deadlocks mid-window.
# The TP2 rows are RETAINED -- "TP2 is not enough for this trace" is a capacity
# result and is reported as one -- and these rows are ADDED so the capacity axis
# is measured rather than worked around.
#
# These lived as hand-edits in cells_v1.json until 2026-09-08, which meant a
# regeneration silently dropped all eight and the manifest had to be restored
# from git. They are declared here so the builder reproduces all 265 cells.
#
# TP8 returns despite MAX_TP_V1=4 for a different purpose than the TP sweep:
# not a point on the TP axis, but the hardware needed to make a fixed offered
# load servable. Only these named cells are exempt from the TP cap.
CAPACITY_BASES = [
    "burstgpt_h200_qwen3-32b_peakarrival_unbounded",
    "burstgpt_h200_qwen3-32b_c2_scaling",
    "burstgpt_h200_llama-3.1-70b-instruct_peakarrival_unbounded",
    "burstgpt_h200_llama-3.1-70b-instruct_c2_scaling",
]
CAPACITY_TPS = [(4, 8), (8, 9)]     # (tensor-parallel size, grid column)
CAPACITY_NOTE = ("2026-09-05: TP2 exceeded server capacity for this trace; "
                 "TP2 row RETAINED as the capacity result, this row ADDED to "
                 "reach it.")


def synthesise_capacity_blocks(rows, wb):
    """Clone each over-capacity TP2 cell up to TP4 and TP8. Returns new rows."""
    by_id = {r["cell_id"]: r for r in rows}
    missing = [b for b in CAPACITY_BASES if b not in by_id]
    if missing:
        raise SystemExit(f"[FATAL] capacity-block base cell(s) absent from the "
                         f"tracker: {', '.join(missing)}")
    for tp, col in CAPACITY_TPS:
        _assert_column_free(wb["BurstGPT"], "BurstGPT", col,
                            [by_id[b] for b in CAPACITY_BASES], f"capacity block TP{tp}")
    out = []
    for base_id in CAPACITY_BASES:
        base = by_id[base_id]
        for tp, col in CAPACITY_TPS:
            n = copy.deepcopy(base)
            n["cell_id"] = f"{base_id}_tp{tp}"
            n["tp"] = n["expected_gpus"] = tp
            n["block"] = f"capacity-tp{tp}"
            n["added"] = CAPACITY_NOTE
            a = n["vllm_args"]
            a[a.index("--tensor-parallel-size") + 1] = str(tp)
            # Its own grid column, so it does not overwrite the TP2 row's
            # status. Carrying the base's tracker_ref along is exactly the bug
            # validate() now catches.
            n["tracker_ref"] = {**base["tracker_ref"], "col": col,
                                "note": f"capacity block TP{tp} (own column, "
                                        f"was colliding with the TP2 row)"}
            out.append(n)
    return out



# Operator annotations. These lived as hand-edits in cells_v1.json and were all
# wiped the first time the builder was re-run (2026-09-08). Anything that must
# survive a regeneration is declared here.
BLOCKED = {
    "Concurrency": "out of scope for this campaign run (operator, 2026-09-03) - "
                   "the Concurrency sweep will be run separately. Marked rather "
                   "than deleted so the exclusion is on the record.",
}

EXTRA_RUNG_NOTE = {
    ("GPU Frequency", "345"): (
        "2026-09-07: added so the SM-clock axis spans the FULL supported range "
        "(345-1980 on H100 SXM5). The original 1980/1600/1200/900/600 ladder "
        "left the bottom 255 MHz unsampled with no recorded rationale; J/token "
        "vs clock is expected to be U-shaped and a ladder stopping at 600 could "
        "miss the minimum entirely and report the right half as monotonic."),
}

CLOCK_NOTE = {
    1600: ("cell_id says 1600mhz; 1600 is NOT an offered SM clock on H100 SXM5 "
           "(15 MHz grid, nearest 1605/1590). Runs at 1605, 0.31% off intended. "
           "id kept stable so ledger/tracker do not fracture; released metadata "
           "must use 1605."),
}


# Qwen3 chat-template overhead, measured from realized prompt_tokens
# (target 128 -> 140, 512 -> 524, 2048 -> 2060).
CHAT_TEMPLATE_TOKENS = 12

ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "plan" / "New_Manual_Tracker_v2 (1).xlsx"
OUT = ROOT / "plan" / "cells_v1.json"
PINK = "FFEAD1DC"

SHEETS = ["Concurrency", "KV Quant", "TP Number", "Weight Quant",
          "GPU Frequency", "BurstGPT", "ISL_OSL", "Poisson"]

# Per-sheet expectations, kept here so a tracker edit fails loudly rather than
# silently changing the campaign size. Updated 2026-08-31: a Qwen3-14B
# thinking-OFF row was added to Poisson block 1 (+6 cells).
EXPECTED_PER_SHEET = {"Concurrency": 18, "KV Quant": 26, "TP Number": 36,
                      "Weight Quant": 39, "GPU Frequency": 54, "BurstGPT": 41,
                      "ISL_OSL": 30, "Poisson": 21}
EXPECTED_TOTAL = sum(EXPECTED_PER_SHEET.values())

SHEET_SLUG = {"Concurrency": "conc", "KV Quant": "kvquant", "TP Number": "tp",
              "Weight Quant": "wquant", "GPU Frequency": "gpufreq",
              "BurstGPT": "burstgpt", "ISL_OSL": "islosl", "Poisson": "poisson"}


def _txt(cell):
    v = cell.value
    return str(v).strip() if v is not None else ""


# A v1 cell is pink AND carries one of these. "To Run" is the initial state;
# the others are written back by sync_tracker as the campaign progresses.
# Matching only "To Run" would mean a regenerated manifest silently drops every
# cell that had already been run.
# Deliberately avoids "Done" and "Ran": the tracker already uses those for the
# archive's completed Exp B cells, and reusing them would pull 162 finished
# cells into the v1 manifest.
V1_STATUSES = ("To Run", "Running", "Completed", "FAILED", "Flagged", "Skipped")


def _is_pink_to_run(cell):
    return (_txt(cell).startswith(V1_STATUSES)
            and getattr(cell.fill.start_color, "rgb", None) == PINK)


def iter_blocks(ws):
    """A block is an axis header row (column B ends in '->') plus the model rows
    beneath it, up to the next header or a gap."""
    headers = [r for r in range(1, ws.max_row + 1)
               if _txt(ws.cell(r, 2)).endswith("->")]
    blocks = []
    for i, hr in enumerate(headers):
        end = headers[i + 1] - 2 if i + 1 < len(headers) else ws.max_row

        starts = [(c, _txt(ws.cell(hr - 1, c)))
                  for c in range(3, ws.max_column + 1)
                  if _txt(ws.cell(hr - 1, c))]
        spans = []
        for j, (c, label) in enumerate(starts):
            last = starts[j + 1][0] - 1 if j + 1 < len(starts) else ws.max_column
            while last >= c and not _txt(ws.cell(hr, last)):
                last -= 1
            spans.append((label, c, last))

        model_rows = []
        for r in range(hr + 1, end + 1):
            a, b = _txt(ws.cell(r, 1)), _txt(ws.cell(r, 2))
            if not a and not b:
                continue
            if b.endswith("->"):
                break
            model_rows.append(r)

        note = ""
        for r in model_rows:
            for c in range(3, ws.max_column + 1):
                t = _txt(ws.cell(r, c))
                if t and not t.startswith(("To Run", "Ran", "Done", "N/A", "<-")):
                    note = t
                    break
            if note:
                break

        blocks.append({"header_row": hr, "axis": _txt(ws.cell(hr, 2))[:-2].strip(),
                       "gpu_spans": spans, "note": note, "model_rows": model_rows})
    return blocks


def parse_sheet(ws):
    """Every pink To Run cell on this sheet, as a raw record."""
    out = []
    for blk in iter_blocks(ws):
        for r in blk["model_rows"]:
            a, b = _txt(ws.cell(r, 1)).lower(), _txt(ws.cell(r, 2)).lower()
            if (a, b) not in pins.MODELS:
                continue
            for gpu_label, lo, hi in blk["gpu_spans"]:
                for c in range(lo, hi + 1):
                    cell = ws.cell(r, c)
                    if not _is_pink_to_run(cell):
                        continue
                    out.append({
                        # Exact tracker coordinates, so status can be written
                        # back to the right cell without re-deriving anything.
                        "tracker_row": r, "tracker_col": c,
                        "sheet": ws.title, "block_note": blk["note"],
                        "axis": blk["axis"],
                        "axis_value": _txt(ws.cell(blk["header_row"], c)),
                        "gpu_label": gpu_label, "model_key": (a, b),
                        "label_text": _txt(cell),
                    })
    return out


def slug(s):
    s = str(s).strip().lower().replace("(", "").replace(")", "")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9.]+", "-", s)).strip("-")


def _block_kind(note):
    """Map a block's margin note to its load regime."""
    n = note.lower()
    if "unbounded" in n and "burstgpt" not in n:
        return "unbounded"
    if "= 32" in n or "max in flight = 32" in n:
        return "b32"
    if "= 16" in n:
        return "b16"
    if "= 128" in n:
        return "b128"
    if "burst window" in n:
        return "burst"
    if "work window" in n and "scaling" not in n:
        return "work"
    if "scaling" in n:
        return "scaling"
    return "default"


def derive(raw):
    """One raw tracker cell -> one fully-resolved manifest row."""
    sheet, axis_v = raw["sheet"], raw["axis_value"]
    clock_actual = None
    m = pins.MODELS[raw["model_key"]]
    gpu = pins.GPUS[raw["gpu_label"]]
    kind = _block_kind(raw["block_note"])
    hf_id = m["hf_id"]

    tp = 1
    in_flight, mode, osl, thinking = 32, "closed", "forced", False
    duration = pins.DURATION_S["knob"]
    dataset = {"kind": "smallprompts"}
    extra, pre, post, workload_extra = [], [], [], {}
    missing_ckpt = None
    knob = slug(axis_v)

    if sheet == "Concurrency":
        in_flight = int(axis_v)
        knob = f"c{axis_v}"
        if m["size_b"] >= 70:
            tp = 2
    elif sheet == "KV Quant":
        extra += ["--kv-cache-dtype", "auto" if axis_v == "bf16" else axis_v]
        if m["size_b"] >= 70:
            tp = 2
    elif sheet == "TP Number":
        tp = int(axis_v)
        knob = f"tp{axis_v}"
    elif sheet == "Weight Quant":
        if axis_v == "fp8":
            extra += ["--quantization", "fp8"]
        elif axis_v.startswith("int4"):
            # ASSETS.md section 2 verified checkpoints for the three pilot
            # models only. A missing one is a validation error, never a guess:
            # the archive already burned a slot on a repo whose config.json
            # turned out to be compressed-tensors rather than GPTQ.
            base = hf_id
            hf_id = pins.GPTQ_INT4.get(base)
            if hf_id is None:
                hf_id, missing_ckpt = base, base
            extra += ["--quantization", "gptq_marlin"]
        if m["size_b"] >= 70:
            tp = 2
    elif sheet == "GPU Frequency":
        # The tracker column says 1600, but 1600 is NOT an offered SM clock on
        # H100 SXM5 -- the part exposes 110 clocks on a 15 MHz grid, 1980 down to
        # 345, measured on the device 2026-09-07. The nearest is 1605 (0.31%
        # off). The cell_id keeps the tracker's value so the ledger and grid do
        # not fracture over 5 MHz; `clock_mhz_actual` carries the truth and
        # released metadata must use it.
        target = CLOCK_SUBSTITUTIONS.get(int(axis_v), int(axis_v))
        # Assert the clock is really offered before pinning. run_plan_settings_v2
        # section 6 always required "ladder values validated against
        # SUPPORTED_CLOCKS on the actual device before launch" and nothing
        # implemented it. Without this an unsupported target silently lands on a
        # neighbouring clock, which corrupts the axis instead of stopping the run.
        guard = (r"nvidia-smi -q -d SUPPORTED_CLOCKS | grep -oP 'Graphics\s*:\s*\K[0-9]+' "
                 f"| sort -u | grep -qx {target} || {{ echo \"[FATAL] SM clock {target} MHz "
                 f"is not in SUPPORTED_CLOCKS on this device - refusing to run, the axis "
                 f"would be wrong\"; exit 1; }}")
        pre = [guard, f"nvidia-smi -lgc {target}", "nvidia-smi -q -d CLOCK"]
        post = ["nvidia-smi -rgc"]
        knob = f"{axis_v}mhz"
        clock_actual = target
        if kind in ("work", "burst"):
            mode, osl, in_flight = "replay", "free", None
            duration = pins.DURATION_S["open"]
            dataset = {"kind": "burstgpt", "window": kind, "family": m["family"]}
    elif sheet == "BurstGPT":
        # Contract section 7: 32B and 70B at TP2; MoE and 8B stay TP1.
        tp = 2 if (m["size_b"] >= 32 and not m["moe"]) else 1
        duration = pins.DURATION_S["open"]
        osl, in_flight = "free", None
        if kind == "scaling":
            mode = "replay"
            dataset = {"kind": "burstgpt", "window": "work", "family": m["family"]}
            c = float(axis_v)
            if c not in pins.BURSTGPT_C_ALLOWED:
                sys.exit(f"[FATAL] BurstGPT block 2 column {axis_v!r} is not a "
                         f"pinned compression factor {sorted(pins.BURSTGPT_C_ALLOWED)}. "
                         f"If the tracker column was relabelled back to req/s, "
                         f"update pins.BURSTGPT_C_ALLOWED deliberately - do not "
                         f"let a rate be read as a compression factor.")
            # Whole window at speed c: duration follows from c rather than
            # truncating the trace, so every rung replays identical content.
            duration = int(pins.BURSTGPT_WINDOW_S / c)
            workload_extra = {"scale": c}
            knob = f"c{c:g}"
        elif raw["axis"].startswith("TP"):
            mode, tp = "replay", int(axis_v)
            knob = f"tp{axis_v}"
            dataset = {"kind": "burstgpt", "window": "work", "family": m["family"]}
        else:
            mode = "replay"
            dataset = {"kind": "burstgpt", "window": slug(axis_v),
                       "family": m["family"]}
    elif sheet == "ISL_OSL":
        if raw["axis"].startswith("ISL/OSL"):
            isl, osl_tok = axis_v.split("/")
            workload_extra = {"forced_isl": int(isl), "forced_osl": int(osl_tok)}
        elif "Thinking" in raw["axis"]:
            osl, thinking = "thinking", True
            workload_extra = {"source": slug(axis_v)}
        else:
            osl = "free"
            workload_extra = {"source": slug(axis_v)}
    elif sheet == "Poisson":
        mode, osl, in_flight = "poisson", "free", None
        duration = pins.DURATION_S["open"]
        workload_extra = {"rate": float(axis_v)}
        knob = f"r{axis_v}"
        if "thinking" in raw["block_note"].lower():
            osl, thinking = "thinking", True
            knob = f"r{axis_v}-think"
        if m["size_b"] >= 32 and not m["moe"]:
            tp = 2

    # One rule: max_model_len must exceed longest ISL + max_tokens, or the
    # output cap does not bind and generation stops at the context boundary.
    if thinking or osl == "free":
        max_model_len = pins.MAX_MODEL_LEN["thinking"]          # 16384
    elif workload_extra.get("forced_isl"):
        # +CHAT_TEMPLATE_TOKENS: the served ISL is the artifact's ISL PLUS the
        # chat-template wrapper, measured at +12 for Qwen3. Sizing the context
        # to exactly forced_isl + forced_osl made the 2048/2048 cells
        # unrunnable -- vLLM rejected every request with "maximum context
        # length is 4096 ... total of at least 4097", the closed loop replaced
        # each failure at full speed, and the whole prompt pool was consumed in
        # ~45 s. run_plan_settings_v2 §1: max_model_len must EXCEED ISL plus
        # max_tokens, or the output cap does not bind.
        need = (workload_extra["forced_isl"] + CHAT_TEMPLATE_TOKENS
                + workload_extra["forced_osl"])
        # Smallest power of two that clears `need`. This was a two-rung
        # ladder (4096 if need > 2048 else 2048) which could not express
        # anything above 4096, so the 2048/2048 cells were emitted at 4096
        # against a need of 4108 -- the cap did not bind and the manifest had
        # to be hand-corrected to 8192. Fixed 2026-09-08; the three affected
        # cells ran AFTER that hand-correction, so their data is unaffected.
        max_model_len = 2048
        while max_model_len < need:
            max_model_len *= 2
    else:
        max_model_len = 2048

    args = ["--model", hf_id, "--tensor-parallel-size", str(tp),
            "--max-model-len", str(max_model_len), "--dtype", "auto",
            "--enable-chunked-prefill", "--enable-prefix-caching"]
    if in_flight is not None:
        args += ["--max-num-seqs", str(in_flight)]
    args += extra
    if thinking:
        args += ["--reasoning-parser", pins.REASONING_PARSER]

    block_suffix = f"_{kind}" if kind != "default" else ""
    cell_id = "_".join([SHEET_SLUG[sheet], slug(raw["gpu_label"]),
                        slug(hf_id.split("/")[-1]), knob]) + block_suffix

    return {
        "cell_id": cell_id, "sheet": sheet, "block": kind,
        "tracker_ref": {"sheet": sheet, "row": raw["tracker_row"],
                        "col": raw["tracker_col"],
                        **({"note": raw["ref_note"]} if raw.get("ref_note") else {})},
        "gpu": raw["gpu_label"], "gpu_variant": gpu["variant"],
        "expected_gpus": tp, "provider": gpu["provider"],
        "model": hf_id, "family": m["family"], "tp": tp,
        "vllm_args": args, "pre_launch": pre, "post_run": post,
        "workload": {"mode": mode, "osl": osl, "in_flight": in_flight,
                     "duration": duration, "thinking": thinking, **workload_extra},
        "dataset": dataset,
        "missing_checkpoint": missing_ckpt,
        "blocked": BLOCKED.get(sheet),
        **({"clock_mhz_actual": clock_actual} if clock_actual else {}),
        **({"clock_note": CLOCK_NOTE[int(axis_v)]}
           if sheet == "GPU Frequency" and int(axis_v) in CLOCK_NOTE else {}),
        **({"added": raw["rung_note"]} if raw.get("rung_note") else {}),
    }


def validate(rows):
    errs = []
    seen = set()
    # tracker_ref must be unique too, not just cell_id. sync_tracker writes each
    # cell's outcome into its grid coordinate, so two cells sharing one ref means
    # the second silently overwrites the first and the grid reports the wrong
    # result for a cell that actually ran. It happened TWICE, both times from
    # cloning a cell to make a new one and carrying its tracker_ref along: the
    # TP4/TP8 capacity blocks (2026-09-05) and the 345 MHz rung (2026-09-07).
    # Neither was caught because validate() only ever checked cell_id.
    refs = {}
    for r in rows:
        cid = r["cell_id"]
        if cid in seen:
            errs.append(f"{cid}: cell_id is not unique")
        seen.add(cid)

        tr = r.get("tracker_ref") or {}
        key = (tr.get("sheet"), tr.get("row"), tr.get("col"))
        if all(k is not None for k in key):
            if key in refs:
                errs.append(f"{cid}: tracker_ref {key} already used by {refs[key]} - "
                            f"the grid would show only one of them")
            refs[key] = cid

        a = r["vllm_args"]
        mml = int(a[a.index("--max-model-len") + 1]) if "--max-model-len" in a else 0
        w = r["workload"]
        cap = w.get("forced_osl") or pins.OSL.get(w["osl"], 0)
        # Longest ISL actually seen, per dataset. Measured 2026-08-31:
        # bucket-S tops out near 410 tokens after the chat template; the
        # BurstGPT windows reach 11,536.
        # + chat-template overhead: the server sees the artifact's ISL plus
        # the template's tokens, so checking the nominal ISL let a too-small
        # max_model_len through (see the 2048/2048 cells, fixed 2026-09-08).
        longest_isl = (w["forced_isl"] + CHAT_TEMPLATE_TOKENS if w.get("forced_isl")
                       else {"smallprompts": 512, "burstgpt": 11536,
                             "islosl": 2048}[r["dataset"]["kind"]])
        if mml < longest_isl + cap:
            errs.append(f"{cid}: max_model_len {mml} < ISL {longest_isl} + "
                        f"max_tokens {cap}; the cap would not bind")

        if r["tp"] != r["expected_gpus"]:
            errs.append(f"{cid}: tp {r['tp']} != expected_gpus {r['expected_gpus']}")

        need = pins.MODEL_MEM_GB.get(r["model"], 0)
        have = pins.GPUS[r["gpu"]]["mem_gb"] * r["tp"]
        if need and need > 0.85 * have:
            errs.append(f"{cid}: {r['model']} needs ~{need}GB but {r['tp']}x"
                        f"{r['gpu']} gives {have}GB - will not fit")

        if w["mode"] not in ("closed", "replay", "poisson"):
            errs.append(f"{cid}: unknown workload mode {w['mode']}")
        if w["osl"] not in pins.OSL:
            errs.append(f"{cid}: unknown osl kind {w['osl']}")
        if r["dataset"]["kind"] not in ("smallprompts", "burstgpt", "islosl"):
            errs.append(f"{cid}: no prepare script for dataset {r['dataset']['kind']}")
        if "--reasoning-parser" in a and r["family"] != "qwen3":
            errs.append(f"{cid}: --reasoning-parser on a non-Qwen3 model")
        if w["thinking"] and r["family"] != "qwen3":
            errs.append(f"{cid}: thinking enabled on a non-Qwen3 model")
        if r.get("missing_checkpoint"):
            errs.append(f"{cid}: no verified INT4-GPTQ checkpoint for "
                        f"{r['missing_checkpoint']} - ASSETS.md section 2 covers "
                        f"the three pilot models only")
    return errs


def build():
    wb = openpyxl.load_workbook(TRACKER, data_only=True)
    raw_rows = []
    for name in SHEETS:
        got = parse_sheet(wb[name])
        got += synthesise_extra_rungs(got, name, wb[name])
        raw_rows += got
    rows = [derive(raw) for raw in raw_rows]
    dropped = [r["cell_id"] for r in rows if r["tp"] > pins.MAX_TP_V1]
    if dropped:
        rows = [r for r in rows if r["tp"] <= pins.MAX_TP_V1]
        print(f"[..] excluded {len(dropped)} cell(s) above TP{pins.MAX_TP_V1}: "
              + ", ".join(dropped))
    rows += synthesise_capacity_blocks(rows, wb)
    # Poisson 8 req/s runs last within its sheet, so lower rates reveal
    # saturation before those two cells are spent.
    for i, r in enumerate(rows):
        last = r["sheet"] == "Poisson" and r["workload"].get("rate") == 8.0
        r["order"] = i + (10_000 if last else 0)
    rows.sort(key=lambda r: r["order"])
    for i, r in enumerate(rows):
        r["order"] = i
    return rows


def main():
    rows = build()
    errs = validate(rows)
    if errs:
        print(f"[FATAL] {len(errs)} validation errors:")
        for e in errs:
            print("   ", e)
        sys.exit(1)
    # Per-sheet, not just the total: two sheets drifting in opposite directions
    # cancel out in the sum and the campaign silently changes shape.
    got = collections.Counter(r["sheet"] for r in rows)
    bad = [f"{k} {got.get(k, 0)}!={v}" for k, v in EXPECTED_PER_SHEET.items()
           if got.get(k, 0) != v]
    if bad or len(rows) != EXPECTED_TOTAL:
        sys.exit(f"[FATAL] emitted {len(rows)} cells, expected {EXPECTED_TOTAL}"
                 + (f"; per-sheet: {', '.join(bad)}" if bad else ""))
    OUT.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"[OK] {len(rows)} cells -> {OUT}")


if __name__ == "__main__":
    main()
