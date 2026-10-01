"""ON INSTANCE, first thing. Start nv-hostengine, prove the instrument works,
record the hardware facts.

    python scripts/bootstrap.py --out /data/run

A pinned field that does not resolve is FATAL here. The archive's capture
silently skipped unsupported fields, which makes the field set vary per cell -
plan/logging_metrics.md forbids exactly that, because profiling overhead then
becomes a confounder instead of a constant. Failing here costs a minute; failing
silently costs a 20-minute cell that writes a column of nulls.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dcgm_fields import (ALL_FIELDS, compute_cap_of_gpu0, fields_for,
                         parse_dmon_line)


def unsupported_fields(probe, expected):
    """Return the expected fields that do not yield a real value.

    `probe(field_id)` returns one dmon data line, or None if the field errored.
    `expected` is the architecture's field set from dcgm_fields.fields_for().
    """
    bad = []
    for fid, name in expected:
        line = probe(fid)
        if not line:
            bad.append((fid, name))
            continue
        row = parse_dmon_line(line, [name])
        if row is None or row.get(name) is None:
            bad.append((fid, name))
    return bad


def _probe(fid):
    """Return the LAST dmon data line for one field, or None.

    Deliberately several samples, not one, AT THE CAPTURE'S OWN INTERVAL.
    On Hopper, DCGM 3.3.x serves the DCP profiling fields (1001-1012) out of
    NVML GPM, which computes each value from a PAIR of samples -- so the
    leading rows of any dmon run are blank ("N/A") no matter how healthy the
    field is. A `-c 1` probe therefore reported all ten DCP fields as
    unsupported on a box where they work perfectly (verified on H100 SXM5,
    driver 580.95.05, DCGM 3.3.9, 2026-09-02).

    -d 100 mirrors dcgm_capture.py's --interval-ms default exactly: the gate
    must certify the fields resolve at the interval the campaign actually
    samples at, not at some slower one they might only work at. -c 5 clears
    the cold-hostengine warm-up -- measured at TWO blank rows at 100 ms with
    no warm watch, zero with one -- and still costs under a second.
    """
    try:
        out = subprocess.run(
            ["dcgmi", "dmon", "-e", str(fid), "-d", "100", "-c", "5"],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    last = None
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.startswith(("GPU ", "GPU-I ", "GPU-CI ")):
            last = s
    return last


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=60).stdout
    except Exception as e:
        return f"<{type(e).__name__}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/run")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run(["nv-hostengine"], capture_output=True, text=True)

    # The DCP fields exist only on Hopper+ in an unprivileged container, so the
    # gate is set by the architecture rather than by one global list. Unknown
    # capability falls back to the full set, so the gate stays strict.
    cap = compute_cap_of_gpu0()
    expected, _, _ = fields_for(cap)

    bad = unsupported_fields(_probe, expected)
    if bad:
        print(f"[FATAL] {len(bad)} pinned DCGM field(s) do not resolve:")
        for fid, name in bad:
            print(f"    {fid:>5}  {name}")
        print(f"Do not run campaign cells on this instance. GPU compute "
              f"capability {cap}; {len(expected)} field(s) expected on this "
              f"architecture. DCP fields (1002-1012) are expected only at "
              f"compute capability >= 9.0 -- if those are the failures on a "
              f"Hopper box, that is a campaign-wide decision, not a per-cell "
              f"one. A CORE field failing is always a hardware problem.")
        sys.exit(1)
    print(f"[OK] all {len(expected)} pinned DCGM fields resolve "
          f"(compute capability {cap}{' - DCP fields not expected on this '
                                     'architecture' if len(expected) < len(ALL_FIELDS) else ''})")

    facts = {
        "nvidia_smi_q": _run(["nvidia-smi", "-q"]),
        "supported_clocks": _run(["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"]),
        "topology": _run(["nvidia-smi", "topo", "-m"]),
        "gpu_csv": _run(["nvidia-smi",
                         "--query-gpu=name,memory.total,driver_version,vbios_version,"
                         "power.limit,clocks.default_applications.graphics,uuid,serial",
                         "--format=csv,noheader"]),
        "compute_cap": cap,
        "dcgm_fields_enabled": [{"id": i, "name": n} for i, n in expected],
    }
    (out / "instance_facts.json").write_text(json.dumps(facts, indent=1) + "\n")
    print(f"[OK] instance facts -> {out / 'instance_facts.json'}")


if __name__ == "__main__":
    main()
