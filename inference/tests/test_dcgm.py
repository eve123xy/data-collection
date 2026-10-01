import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _sub in ("", "dataset", "workload", "telemetry", "cells", "upload"):
    sys.path.insert(0, str(ROOT / "scripts" / _sub))

from dcgm_fields import (ALL_FIELDS, CORE_FIELDS, DCP_FIELDS, FIELD_NAMES,
                         SENTINEL_MIN, parse_dmon_line)

NAMES = ["power_w", "total_energy_mj", "sm_clock_mhz"]


def test_gpu_temp_is_150_and_memory_temp_is_140():
    """The archive mislabelled field 140 as gpu_temp_c; 140 is HBM temp."""
    by_name = {n: i for i, n in ALL_FIELDS}
    assert by_name["gpu_temp_c"] == 150
    assert by_name["memory_temp_c"] == 140


def test_core_fields_cover_the_logging_contract():
    names = {n for _, n in CORE_FIELDS}
    for required in ("power_w", "total_energy_mj", "sm_clock_mhz", "mem_clock_mhz",
                     "gpu_temp_c", "memory_temp_c", "gpu_util_pct",
                     "mem_copy_util_pct", "fb_used_mib", "fb_free_mib",
                     "clock_throttle_reasons",
                     # power_violation_us (240) was DROPPED campaign-wide on
                     # 2026-09-03: it returned a static non-blank junk value on
                     # H200 and was consumed by nothing. 112
                     # clock_throttle_reasons already answers what it was for.
                     "thermal_violation_us", "pstate"):
        assert required in names


def test_dcp_fields_cover_the_compute_vs_memory_discriminators():
    names = {n for _, n in DCP_FIELDS}
    for required in ("sm_active", "sm_occupancy", "tensor_active", "dram_active",
                     "fp16_active", "fp32_active", "pcie_tx_bytes",
                     "pcie_rx_bytes", "nvlink_tx_bytes", "nvlink_rx_bytes"):
        assert required in names


def test_parse_reads_a_gpu_data_line():
    row = parse_dmon_line("GPU 0    120.5   1234567   1410", NAMES)
    assert row == {"entity_type": "GPU", "gpu_id": "0", "power_w": 120.5,
                   "total_energy_mj": 1234567.0, "sm_clock_mhz": 1410.0}


def test_parse_skips_headers_and_blanks():
    for line in ("#Entity  POWER  TOTEC  SMCLK", "Id  foo", "", "   "):
        assert parse_dmon_line(line, NAMES) is None


def test_parse_maps_the_dcgm_sentinel_to_none_not_a_reading():
    """DCGM signals 'unavailable' with a huge integer. Treated as a number it
    would silently poison every mean and every counter delta."""
    row = parse_dmon_line(f"GPU 0    120.5   {int(SENTINEL_MIN)}   1410", NAMES)
    assert row["total_energy_mj"] is None
    assert row["power_w"] == 120.5


def test_parse_rejects_a_short_line():
    assert parse_dmon_line("GPU 0    120.5", NAMES) is None


def test_parse_accepts_mig_entities():
    assert parse_dmon_line("GPU-I 3    1.0 2.0 3.0", NAMES)["gpu_id"] == "3"


# --- capture ---

from dcgm_capture import CSV_COLUMNS, capture_stream, clear_output_dir


def _lines():
    return iter([
        "#Entity   POWER   TOTEC   SMCLK",
        "GPU 0    120.5   1000   1410",
        "GPU 1    130.5   2000   1400",
        "GPU 0    121.5   1100   1410",
        "GPU 1    131.5   2100   1400",
    ])


def test_capture_writes_one_file_per_gpu(tmp_path):
    counts = capture_stream(_lines(), tmp_path,
                            ["power_w", "total_energy_mj", "sm_clock_mhz"], "smoke")
    assert counts == {"0": 2, "1": 2}
    assert (tmp_path / "dcgm_smoke_gpu0.csv").exists()
    assert (tmp_path / "dcgm_smoke_gpu1.csv").exists()


def test_each_row_carries_an_epoch_timestamp(tmp_path):
    capture_stream(_lines(), tmp_path,
                   ["power_w", "total_energy_mj", "sm_clock_mhz"], "smoke")
    rows = list(csv.DictReader(open(tmp_path / "dcgm_smoke_gpu0.csv")))
    assert rows[0]["ts"]
    assert float(rows[1]["ts"]) >= float(rows[0]["ts"])
    assert rows[0]["power_w"] == "120.5"


def test_csv_columns_lead_with_ts_and_gpu_id():
    assert CSV_COLUMNS[:3] == ["ts", "entity_type", "gpu_id"]


def test_clear_output_dir_removes_stale_captures(tmp_path):
    """Section 3.1: retrieval never cleared the folder, so a fresh results file
    was paired with a capture from a HUNG run - 117 W instead of 599 W."""
    (tmp_path / "dcgm_old_gpu0.csv").write_text("stale\n")
    (tmp_path / "run_meta.json").write_text("{}")
    clear_output_dir(tmp_path, "newcell")
    assert not (tmp_path / "dcgm_old_gpu0.csv").exists()
    assert not (tmp_path / "run_meta.json").exists()


# --- bootstrap ---

from bootstrap import unsupported_fields


def test_every_field_supported_returns_empty():
    assert unsupported_fields(lambda fid: "GPU 0    1.0", ALL_FIELDS) == []


def test_a_field_returning_nothing_is_reported():
    bad = {1004, 1005}
    out = unsupported_fields(lambda fid: None if fid in bad else "GPU 0    1.0",
                             ALL_FIELDS)
    assert {i for i, _ in out} == bad


def test_a_field_returning_only_the_sentinel_is_unsupported():
    """A field that resolves but always reports 'unavailable' is not usable, and
    must fail at bootstrap rather than produce a column of nulls for 20 minutes."""
    out = unsupported_fields(
        lambda fid: "GPU 0    9223372036854775794" if fid == 1011 else "GPU 0    1.0",
        ALL_FIELDS)
    assert [i for i, _ in out] == [1011]


# --- slice and aggregate ---

from summarise_run import (IDLE_CARD_W, aggregate, load_capture, per_gpu,
                           run_gates, slice_window)


def _cap(tmp_path, name, rows):
    p = tmp_path / name
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ts", "entity_type", "gpu_id",
                                           "power_w", "total_energy_mj", "sm_clock_mhz"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _rows(gpu, n, p0, e0, clock=1410):
    return [{"ts": 100.0 + i, "entity_type": "GPU", "gpu_id": gpu,
             "power_w": p0, "total_energy_mj": e0 + i * p0 * 1000,
             "sm_clock_mhz": clock} for i in range(n)]


def test_slice_keeps_only_rows_inside_the_window():
    rows = [{"ts": t} for t in (1.0, 5.0, 10.0, 15.0)]
    assert [r["ts"] for r in slice_window(rows, 4.0, 11.0)] == [5.0, 10.0]


def test_energy_and_power_are_summed_per_gpu_not_across_rows(tmp_path):
    """Section 3.3 calls the ungrouped form a gate on running ANY multi-GPU
    sweep: ungrouped it yields an invalid energy-per-token with no error and a
    plausible magnitude, and the defect scales with GPU count."""
    rows = (_rows("0", 11, 300.0, 0) + _rows("1", 11, 400.0, 5_000_000)
            + _rows("2", 11, 350.0, 9_000_000) + _rows("3", 11, 250.0, 1_000_000))
    agg = aggregate(rows, t0=100.0, t1=110.0)

    assert agg["n_gpus"] == 4
    assert agg["mean_power_w"] == pytest.approx(300 + 400 + 350 + 250)
    assert agg["energy_j"] == pytest.approx((300 + 400 + 350 + 250) * 10)

    naive = max(r["total_energy_mj"] for r in rows) - min(r["total_energy_mj"] for r in rows)
    assert naive / 1000.0 != pytest.approx(agg["energy_j"])


def test_per_gpu_reports_each_card_separately():
    rows = _rows("0", 11, 300.0, 0) + _rows("1", 11, 400.0, 0)
    g = per_gpu(slice_window(rows, 100.0, 110.0))
    assert set(g) == {"0", "1"}
    assert g["0"]["mean_power_w"] == pytest.approx(300.0)
    assert g["1"]["mean_power_w"] == pytest.approx(400.0)


def test_load_capture_reads_numbers_and_blank_as_none(tmp_path):
    p = _cap(tmp_path, "dcgm_x_gpu0.csv",
             [{"ts": 1.0, "entity_type": "GPU", "gpu_id": "0", "power_w": 120.5,
               "total_energy_mj": "", "sm_clock_mhz": 1410}])
    rows = load_capture(p)
    assert rows[0]["power_w"] == 120.5
    assert rows[0]["total_energy_mj"] is None
    assert rows[0]["gpu_id"] == "0"


def test_aggregate_reports_sample_gaps():
    rows = [{"ts": 100.0, "gpu_id": "0", "power_w": 300.0, "total_energy_mj": 0},
            {"ts": 100.1, "gpu_id": "0", "power_w": 300.0, "total_energy_mj": 30000},
            {"ts": 100.8, "gpu_id": "0", "power_w": 300.0, "total_energy_mj": 240000},
            {"ts": 100.9, "gpu_id": "0", "power_w": 300.0, "total_energy_mj": 270000}]
    agg = aggregate(rows, t0=100.0, t1=101.0)
    assert agg["max_sample_gap_s"] == pytest.approx(0.7)


# --- gates ---

CELL = {"cell_id": "c", "sheet": "KV Quant", "expected_gpus": 1,
        "workload": {"duration": 600}, "pre_launch": []}
FREQ_CELL = {"cell_id": "f", "sheet": "GPU Frequency", "expected_gpus": 1,
             "workload": {"duration": 600},
             "pre_launch": ["nvidia-smi -lgc 1200", "nvidia-smi -q -d CLOCK"]}


def _agg(power, energy, gap=0.1, clocks=None, n=1):
    return {"n_gpus": n, "samples": 6000, "mean_power_w": power,
            "mean_power_per_gpu_w": power / n, "energy_j": energy,
            "max_sample_gap_s": gap, "window_s": 600.0,
            "_per_gpu_full": {"0": {"sm_clocks": clocks or [1410] * 100,
                                    "powers": [power] * 100}}}


def test_an_idle_card_fails_loudly():
    """The 117 W signature: 19 of 93 archive cells paired a working run's tokens
    with a hung run's power trace, and it inverted a TP conclusion."""
    fatal, _ = run_gates(_agg(117.0, 117.0 * 600), CELL)
    assert any("idle" in f.lower() or str(int(IDLE_CARD_W)) in f for f in fatal)


def test_a_real_serving_power_passes():
    fatal, flags = run_gates(_agg(599.0, 599.0 * 600), CELL)
    assert fatal == [] and flags == []


def test_energy_counter_disagreeing_with_the_power_integral_fails():
    fatal, _ = run_gates(_agg(599.0, 599.0 * 600 * 1.03), CELL)
    assert any("energy" in f.lower() for f in fatal)


def test_a_clock_off_its_lock_fails_only_on_frequency_cells():
    drifted = [1200] * 97 + [1400] * 3
    fatal, _ = run_gates(_agg(599.0, 599.0 * 600, clocks=drifted), FREQ_CELL)
    assert any("clock" in f.lower() for f in fatal)
    fatal2, _ = run_gates(_agg(599.0, 599.0 * 600, clocks=drifted), CELL)
    assert not any("clock" in f.lower() for f in fatal2)


def test_a_clock_holding_its_lock_passes():
    held = [1200] * 995 + [1195] * 5
    fatal, _ = run_gates(_agg(599.0, 599.0 * 600, clocks=held), FREQ_CELL)
    assert fatal == []


def test_a_long_sample_gap_flags_but_does_not_fail():
    fatal, flags = run_gates(_agg(599.0, 599.0 * 600, gap=0.7), CELL)
    assert fatal == []
    assert any("gap" in f.lower() for f in flags)


def test_client_and_server_token_disagreement_flags():
    _, flags = run_gates(_agg(599.0, 599.0 * 600), CELL,
                         client_tokens=1000, server_tokens=1200)
    assert any("token" in f.lower() for f in flags)


def test_the_idle_gate_is_per_gpu_not_per_cell():
    """A 4-GPU cell at 500 W total is 125 W per card - idle, despite the total
    looking healthy."""
    fatal, _ = run_gates(_agg(500.0, 500.0 * 600, n=4), dict(CELL, expected_gpus=4))
    assert any("idle" in f.lower() or "per GPU" in f for f in fatal)


# --- derived metrics ---

from summarise_run import derive_metrics, power_percentiles, ramp_rates


def test_joules_per_token_uses_server_token_counts():
    agg = _agg(600.0, 600.0 * 600)
    d = derive_metrics(agg, idle_agg=None, server_tokens=1_200_000)
    assert d["energy_j"] == pytest.approx(360_000)
    assert d["j_per_token"] == pytest.approx(0.3)


def test_joules_per_token_is_none_without_token_counts():
    """Never estimate a denominator we did not measure."""
    d = derive_metrics(_agg(600.0, 600.0 * 600), None, server_tokens=None)
    assert d["j_per_token"] is None


def test_idle_baseline_is_subtracted_when_present():
    agg = _agg(600.0, 600.0 * 600)
    idle = _agg(100.0, 100.0 * 30)
    d = derive_metrics(agg, idle, server_tokens=1_200_000)
    assert d["idle_power_w"] == pytest.approx(100.0)
    assert d["energy_above_idle_j"] == pytest.approx(300_000)


def test_power_percentiles_come_from_the_per_gpu_series():
    agg = {"_per_gpu_full": {"0": {"powers": [100.0] * 90 + [500.0] * 10}}}
    p = power_percentiles(agg)
    assert p["p99_power_w"] == pytest.approx(500.0)
    assert p["p50_power_w"] == pytest.approx(100.0)


def test_ramp_rate_distribution_is_computed_per_gpu():
    """dP/dt over the 100 ms series, per Dr Chen's metric list."""
    rows = [{"ts": 100.0 + i * 0.1, "gpu_id": "0",
             "power_w": 300.0 + (100.0 if i == 5 else 0.0),
             "total_energy_mj": i} for i in range(20)]
    r = ramp_rates(rows, 100.0, 102.0)
    assert r["max_abs_dpdt_w_per_s"] == pytest.approx(1000.0, rel=0.01)


# --- data-team exports ---

from export_record import TIMELINE_COLUMNS, build_run_record, build_timeline

RUN_META = {"cell_id": "c", "window_start_ts_ns": 100_000_000_000,
            "window_end_ts_ns": 700_000_000_000, "vllm_version": "0.28.0",
            "requests_total": 12, "server": {"generation_tokens": 5000,
                                             "prompt_tokens": 3000}}


def test_run_record_uses_the_data_team_vocabulary():
    row = {"cell_id": "c", "model": "Qwen/Qwen3-32B", "gpu": "H100", "tp": 2,
           "sheet": "KV Quant", "pre_launch": [],
           "vllm_args": ["--kv-cache-dtype", "fp8_e4m3"],
           "workload": {"mode": "closed", "in_flight": 32, "osl": "forced",
                        "thinking": False, "duration": 600}}
    rec = build_run_record(row, RUN_META, _agg(599.0, 599.0 * 600))
    assert rec["model"] == "Qwen/Qwen3-32B"
    assert rec["gpu_type"] == "H100"
    assert rec["tp_size"] == 2
    assert rec["kv_cache_quant"] == "fp8_e4m3"
    assert rec["serving_engine"] == "vLLM 0.28.0"
    assert rec["workload_pattern"] == "closed"
    assert rec["concurrency"] == 32


def test_inapplicable_fields_are_null_never_estimated():
    """Their instruction: if a value was not recorded or does not apply, leave
    it blank/null rather than estimating it."""
    row = {"cell_id": "c", "model": "Qwen/Qwen3-8B", "gpu": "H200", "tp": 1,
           "sheet": "Poisson", "pre_launch": [], "vllm_args": [],
           "workload": {"mode": "poisson", "in_flight": None, "rate": 2.0,
                        "osl": "free", "thinking": False, "duration": 1200}}
    rec = build_run_record(row, RUN_META, _agg(599.0, 599.0 * 600))
    assert rec["concurrency"] is None
    assert rec["arrival_rate"] == 2.0
    assert rec["gpu_frequency"] is None
    assert rec["weight_quant"] is None
    assert rec["kv_cache_quant"] is None


def test_gpu_frequency_cells_report_their_locked_clock():
    row = {"cell_id": "f", "model": "Qwen/Qwen3-8B", "gpu": "H100 (Lambda)", "tp": 1,
           "sheet": "GPU Frequency", "pre_launch": ["nvidia-smi -lgc 1200"],
           "vllm_args": [], "workload": {"mode": "closed", "in_flight": 32,
                                         "osl": "forced", "thinking": False,
                                         "duration": 600}}
    assert build_run_record(row, RUN_META, _agg(599.0, 599.0 * 600))["gpu_frequency"] == 1200


def test_timeline_bins_at_one_second_on_the_relative_axis():
    reqs = [{"send_ts": 100.5, "prompt_tokens": 300, "completion_tokens": 500},
            {"send_ts": 100.9, "prompt_tokens": 320, "completion_tokens": 480},
            {"send_ts": 102.2, "prompt_tokens": 310, "completion_tokens": 490}]
    samples = [{"ts": 100.0, "vllm:num_requests_running": 4},
               {"ts": 101.0, "vllm:num_requests_running": 8},
               {"ts": 102.0, "vllm:num_requests_running": 6}]
    tl = build_timeline(reqs, samples, t0=100.0, t1=103.0, warmup_s=0)
    assert TIMELINE_COLUMNS[0] == "time_relative_s"
    assert tl[0]["time_relative_s"] == 0.0
    assert tl[0]["requests_arrived"] == 2
    assert tl[0]["active_requests"] == 4
    assert tl[0]["mean_prompt_tokens"] == pytest.approx(310.0)
    assert tl[2]["requests_arrived"] == 1


def test_timeline_covers_warmup_with_negative_relative_time():
    """Warm-up rows are negative so their visualization shows the boundary
    rather than hiding it."""
    reqs = [{"send_ts": 95.0, "prompt_tokens": 300, "completion_tokens": 100}]
    tl = build_timeline(reqs, [], t0=100.0, t1=101.0, warmup_s=10)
    assert tl[0]["time_relative_s"] == -10.0
    assert any(r["time_relative_s"] == -5.0 and r["requests_arrived"] == 1 for r in tl)


def test_a_bin_with_no_requests_reports_none_not_zero_for_means():
    tl = build_timeline([], [], t0=100.0, t1=102.0, warmup_s=0)
    assert tl[0]["requests_arrived"] == 0
    assert tl[0]["mean_prompt_tokens"] is None


# --- credential helpers ---

import common


def test_drive_root_accepts_either_variable_name(monkeypatch):
    """The .env uses DRIVE_ROOT_FOLDER; earlier docs said DRIVE_ROOT_FOLDER_ID.
    Accept both rather than make the operator edit their environment."""
    monkeypatch.setattr(common, "load_env", lambda: None)
    for var in ("DRIVE_ROOT_FOLDER_ID", "DRIVE_ROOT_FOLDER"):
        monkeypatch.delenv("DRIVE_ROOT_FOLDER_ID", raising=False)
        monkeypatch.delenv("DRIVE_ROOT_FOLDER", raising=False)
        monkeypatch.setenv(var, "folder123")
        assert common.drive_root_folder_id() == "folder123"


def test_drive_root_falls_back_to_the_drive_id(monkeypatch):
    """A Shared Drive's root folder id is its drive id."""
    monkeypatch.setattr(common, "load_env", lambda: None)
    monkeypatch.delenv("DRIVE_ROOT_FOLDER_ID", raising=False)
    monkeypatch.delenv("DRIVE_ROOT_FOLDER", raising=False)
    monkeypatch.setenv("DRIVE_ID", "drive456")
    assert common.drive_root_folder_id() == "drive456"
