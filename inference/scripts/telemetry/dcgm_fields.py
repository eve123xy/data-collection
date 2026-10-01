"""DCGM field table and `dcgmi dmon` output parsing.

`dcgmi dmon` emits whitespace-aligned text, not CSV:

    #Entity   POWER    TOTEC     SMCLK
    GPU 0     120.5    1234567   1410

Field ids follow plan/logging_metrics.md section 2. Note 150 is GPU temperature
and 140 is MEMORY temperature - the archive's capture had these swapped, so its
"gpu_temp_c" column actually held HBM temperature.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

# (field_id, column_name)
CORE_FIELDS = [
    (155, "power_w"),
    (156, "total_energy_mj"),
    (100, "sm_clock_mhz"),
    (101, "mem_clock_mhz"),
    (150, "gpu_temp_c"),
    (140, "memory_temp_c"),
    (203, "gpu_util_pct"),
    (204, "mem_copy_util_pct"),
    (252, "fb_used_mib"),
    (251, "fb_free_mib"),
    (112, "clock_throttle_reasons"),
    # 240 power_violation_us was pinned here and is DROPPED campaign-wide
    # (2026-09-02, operator's standing rule on non-core fields). On H200 it
    # returns a static 1141477044024480 -- not DCGM's blank code (8.11x it), not
    # incrementing, and no unit reading makes it real: 13,211 days as us, 13.2
    # days as ns against a 100-day host uptime. Its sibling 241
    # thermal_violation_us is sane on the same card and stays. It was consumed
    # by nothing -- no gate, no summary field -- and 112 above already answers
    # the question it was there for: whether the run was clock-capped, power
    # cap included. Dropping it keeps ONE telemetry contract across all 248
    # cells, which matters more than a column nothing reads: H200 carries every
    # Poisson, BurstGPT and ISL_OSL cell, so keeping it would have meant
    # "populated on H100, absent on H200" -- the absent-vs-zero ambiguity we
    # just spent a commit eliminating.
    (241, "thermal_violation_us"),
    (190, "pstate"),
]

# DCP profiling fields: these separate compute-bound from memory-bound directly.
# tensor_active is the prefill signature, dram_active the decode signature, and
# nvlink_tx/rx is the mechanism behind the TP knob.
DCP_FIELDS = [
    (1002, "sm_active"),
    (1003, "sm_occupancy"),
    (1004, "tensor_active"),
    (1005, "dram_active"),
    (1007, "fp32_active"),
    (1008, "fp16_active"),
    (1009, "pcie_tx_bytes"),
    (1010, "pcie_rx_bytes"),
    (1011, "nvlink_tx_bytes"),
    (1012, "nvlink_rx_bytes"),
]

ALL_FIELDS = CORE_FIELDS + DCP_FIELDS
FIELD_NAMES = [n for _, n in ALL_FIELDS]
FIELD_IDS = [str(i) for i, _ in ALL_FIELDS]

# --- architecture-conditional field set --------------------------------------
# The DCP fields are unobtainable on Ampere in an unprivileged container, and
# this is a property of the architecture, not of any particular rental.
# Measured 2026-09-02 on two boxes with IDENTICAL restrictions
# (RmProfilingAdminOnly: 1, CapEff a80405fb, no CAP_SYS_ADMIN):
#   H200, compute capability 9.0  -> all 10 DCP fields resolve
#   A100, compute capability 8.0  -> all 10 refuse: "requires the host engine
#                                    to be running as root"
# Hopper serves them through NVML GPM, which bypasses the perfworks path that
# RmProfilingAdminOnly gates; Ampere has no GPM and falls back to that path.
#
# This matters beyond the gate: dcgm_capture issues ONE `dcgmi dmon` for all
# field ids, so a single unwatchable field kills the entire stream. An A100 cell
# asking for the DCP fields writes ZERO rows, not partial columns.
#
# Cross-architecture claims may therefore use only the CORE fields. Any claim
# resting on tensor_active / dram_active -- the compute-vs-memory-bound
# discriminator -- is Hopper-only and must be labelled as such.
DCP_MIN_COMPUTE_CAP = 9.0


def fields_for(compute_cap):
    """(fields, names, ids) expected on a GPU of this compute capability.

    `compute_cap` may be a float (9.0) or the string nvidia-smi prints ("9.0").
    Unknown or unparseable capability is treated as Hopper-or-better, so the
    gate stays strict by default and a silent downgrade is impossible.
    """
    try:
        cap = float(compute_cap)
    except (TypeError, ValueError):
        cap = DCP_MIN_COMPUTE_CAP
    fields = CORE_FIELDS + (DCP_FIELDS if cap >= DCP_MIN_COMPUTE_CAP else [])
    return fields, [n for _, n in fields], [str(i) for i, _ in fields]


def compute_cap_of_gpu0():
    """Compute capability of GPU 0 via nvidia-smi, or None if unavailable."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60)
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return None

# DCGM reports an unavailable value as a very large integer. Read as a number it
# would silently poison every mean and every counter delta.
# DCGM_FP64_BLANK, from dcgm_fields.h. The old guard was 1e15 -- seven times
# larger than the sentinel it was meant to catch, so a raw sentinel would have
# been kept as a reading of 1.4e14. It never fired, because `dcgmi dmon` renders
# blanks as "N/A" and the float() failure path handles those. A backstop that
# was the wrong size, now the right size. NOT_FOUND/NOT_SUPPORTED/NOT_PERMISSIONED
# are BLANK+1/+2/+3, hence >= rather than >.
SENTINEL_MIN = 140737488355328.0

_ENTITIES = ("GPU", "GPU-I", "GPU-CI")


def parse_dmon_line(line, field_names):
    """One `dcgmi dmon` data line -> dict, or None if it is not a data line."""
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("Id"):
        return None
    parts = line.split()
    if len(parts) < 2 + len(field_names):
        return None
    if parts[0] not in _ENTITIES:
        return None

    row = {"entity_type": parts[0], "gpu_id": parts[1]}
    for name, raw in zip(field_names, parts[2:]):
        try:
            v = float(raw)
        except ValueError:
            row[name] = None
            continue
        row[name] = None if v >= SENTINEL_MIN else v
    return row
