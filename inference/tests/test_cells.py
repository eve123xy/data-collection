import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))              # for `tools`
for _sub in ("", "dataset", "workload", "telemetry", "cells", "upload"):
    sys.path.insert(0, str(ROOT / "scripts" / _sub))  # for `pins`, `cell`, ...

import pins


def test_model_table_resolves_the_tracker_labels():
    assert pins.MODELS[("qwen3", "8b dense")]["hf_id"] == "Qwen/Qwen3-8B"
    assert pins.MODELS[("qwen3", "0.6b")]["hf_id"] == "Qwen/Qwen3-0.6B"
    assert pins.MODELS[("qwen3-moe-30b-a3b", "30b 3b moe")]["hf_id"] == "Qwen/Qwen3-30B-A3B"
    assert pins.MODELS[("llama3.1", "70b dense")]["family"] == "llama31"


def test_moe_model_is_flagged_as_moe():
    assert pins.MODELS[("qwen3-moe-30b-a3b", "30b 3b moe")]["moe"] is True
    assert pins.MODELS[("qwen3", "32b dense")]["moe"] is False


def test_gpu_table_carries_variant_and_provider():
    assert pins.GPUS["H100"]["variant"] == "SXM5 80GB"
    assert pins.GPUS["H100"]["provider"] == "vast.ai"
    assert pins.GPUS["H100 (Lambda)"]["provider"] == "Lambda"
    assert pins.GPUS["H200"]["mem_gb"] == 141


def test_int4_checkpoints_exist_for_every_model_that_needs_one():
    for hf in ("Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B"):
        assert hf in pins.GPTQ_INT4


import openpyxl
from tools.build_cells import (iter_blocks, parse_sheet, TRACKER,
                               EXPECTED_PER_SHEET, EXPECTED_TOTAL)

WB = openpyxl.load_workbook(TRACKER, data_only=True)


def test_blocks_are_found_by_their_axis_header():
    blocks = iter_blocks(WB["KV Quant"])
    assert len(blocks) == 4
    assert blocks[1]["note"] == "Max in flight requests = 32"
    assert blocks[0]["axis"] == "KV Quant"


def test_gpu_spans_cover_the_right_columns():
    b = iter_blocks(WB["KV Quant"])[1]
    assert b["gpu_spans"] == [("A100", 3, 5), ("H100", 6, 8), ("H200", 9, 11)]


def test_gpu_frequency_has_three_blocks_including_the_burst_window():
    notes = [b["note"] for b in iter_blocks(WB["GPU Frequency"])]
    assert any("Closed loop" in n for n in notes)
    assert any("Work window" in n for n in notes)
    assert any("Burst window" in n for n in notes)


def test_parse_sheet_returns_only_pink_to_run_cells():
    cells = parse_sheet(WB["Concurrency"])
    assert len(cells) == 18
    assert {c["gpu_label"] for c in cells} == {"H100", "H200"}
    assert {c["axis_value"] for c in cells} == {"1","2","4","8","16","32","64","128","256"}


def test_parse_sheet_totals_match_the_reconciled_counts():
    got = {name: len(parse_sheet(WB[name])) for name in EXPECTED_PER_SHEET}
    # parse_sheet reads the tracker as-is; the manifest additionally drops the
    # two TP8 cells, so BurstGPT differs by 2 here.
    expected_in_tracker = dict(EXPECTED_PER_SHEET, BurstGPT=EXPECTED_PER_SHEET["BurstGPT"] + 2)
    assert got == expected_in_tracker


import json
from tools.build_cells import build, derive, validate, slug

ROWS = build()


def test_manifest_has_exactly_the_expected_unique_cells():
    assert len(ROWS) == EXPECTED_TOTAL
    assert len({r["cell_id"] for r in ROWS}) == EXPECTED_TOTAL


def test_manifest_has_no_validation_errors():
    assert validate(ROWS) == []


def test_per_sheet_counts_match_the_reconciliation():
    from collections import Counter
    assert Counter(r["sheet"] for r in ROWS) == EXPECTED_PER_SHEET


def test_the_four_resolved_conflicts_are_all_present():
    assert any(r["sheet"] == "GPU Frequency" and r["block"] == "burst" for r in ROWS)
    assert any(r["sheet"] == "ISL_OSL" and "14b" in r["cell_id"] for r in ROWS)
    assert any(r["sheet"] == "Poisson" and r["workload"].get("rate") == 8.0
               for r in ROWS)
    assert sum(1 for r in ROWS if r["sheet"] in ("KV Quant", "Weight Quant")
               and "70b" in r["cell_id"]) == 5


def test_kv_quant_row_carries_the_right_launch_flags():
    r = next(x for x in ROWS if x["cell_id"]
             == "kvquant_h100_qwen3-32b_fp8-e4m3_b32")
    a = r["vllm_args"]
    assert a[a.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert a[a.index("--max-num-seqs") + 1] == "32"
    assert "--enable-chunked-prefill" in a and "--enable-prefix-caching" in a
    assert r["workload"] == {"mode": "closed", "osl": "forced",
                             "in_flight": 32, "duration": 600, "thinking": False}


def test_burstgpt_rows_have_room_for_the_longest_real_prompt():
    """Measured: BurstGPT prompts reach 11,536 tokens. At max_model_len 2048
    vLLM rejects 313 of 22,090 requests outright, deleting part of the arrival
    process, and the output cap binds for none of them."""
    for r in ROWS:
        if r["dataset"]["kind"] == "burstgpt":
            a = r["vllm_args"]
            assert int(a[a.index("--max-model-len") + 1]) >= 11536 + 2048


def test_every_free_generation_row_lets_its_output_cap_bind():
    for r in ROWS:
        if r["workload"]["osl"] in ("free", "thinking"):
            a = r["vllm_args"]
            assert int(a[a.index("--max-model-len") + 1]) == 16384


def test_isl_osl_forced_pairs_get_4096_where_they_need_it():
    for r in ROWS:
        fi, fo = r["workload"].get("forced_isl"), r["workload"].get("forced_osl")
        if fi and fi + fo > 2048:
            a = r["vllm_args"]
            assert int(a[a.index("--max-model-len") + 1]) == 4096


def test_thinking_rows_use_the_contract_caps_not_the_stale_sheet_note():
    r = next(x for x in ROWS if x["sheet"] == "ISL_OSL" and x["workload"]["thinking"])
    a = r["vllm_args"]
    assert a[a.index("--max-model-len") + 1] == "16384"
    assert a[a.index("--reasoning-parser") + 1] == "qwen3"
    assert r["workload"]["osl"] == "thinking"


def test_llama_rows_never_carry_a_reasoning_parser():
    for r in ROWS:
        if r["family"] == "llama31":
            assert "--reasoning-parser" not in r["vllm_args"]
            assert r["workload"]["thinking"] is False


def test_gpu_frequency_rows_lock_and_restore_the_clock():
    r = next(x for x in ROWS if x["sheet"] == "GPU Frequency")
    assert any("-lgc" in c for c in r["pre_launch"])
    assert any("-rgc" in c for c in r["post_run"])
    assert r["provider"] == "Lambda"


def test_no_cell_is_blocked():
    assert [r["cell_id"] for r in ROWS if r.get("blocked")] == []


def test_burstgpt_block2_replays_the_whole_window_at_each_scale():
    """Duration follows from c, so every rung replays the identical 6,171
    requests instead of a different slice of the same window."""
    b2 = [r for r in ROWS if r["workload"].get("scale")]
    assert len(b2) == 16
    assert {r["workload"]["scale"] for r in b2} == {2.0, 1.0, 0.5, 0.25}
    for r in b2:
        assert r["workload"]["duration"] == int(1200 / r["workload"]["scale"])
        assert r["dataset"]["window"] == "work"


def test_every_int4_row_uses_a_verified_checkpoint():
    for r in ROWS:
        if "gptq_marlin" in r["vllm_args"]:
            assert r["missing_checkpoint"] is None
            assert r["model"] in pins.GPTQ_INT4.values()


def test_poisson_8_reqs_sorts_last_within_its_sheet():
    p = sorted((r for r in ROWS if r["sheet"] == "Poisson"), key=lambda r: r["order"])
    assert p[-1]["workload"]["rate"] == 8.0


def test_validate_rejects_a_cap_that_cannot_bind():
    bad = dict(ROWS[0])
    bad["vllm_args"] = ["--max-model-len", "2048"]
    bad["workload"] = dict(bad["workload"], osl="thinking")
    assert any("max_model_len" in e for e in validate([bad]))


def test_validate_rejects_tp_that_does_not_divide_the_gpu_count():
    assert any("tp" in e.lower() for e in validate([dict(ROWS[0], tp=4,
                                                         expected_gpus=1)]))


def test_validate_rejects_a_model_that_cannot_fit():
    bad = dict(ROWS[0], model="meta-llama/Llama-3.1-70B-Instruct",
               gpu="H100", tp=1, expected_gpus=1)
    assert any("fit" in e for e in validate([bad]))


def test_validate_rejects_duplicate_cell_ids():
    assert any("unique" in e for e in validate([ROWS[0], dict(ROWS[0])]))


def test_no_v1_cell_exceeds_tp4():
    """TP8 is excluded from v1: 8xH200 is the scarcest market segment, and 70B
    at TP8 spreads a 140 GB model across 1,128 GB of HBM."""
    assert max(r["tp"] for r in ROWS) <= pins.MAX_TP_V1
    assert not any("tp8" in r["cell_id"] for r in ROWS)


def test_slug_is_filesystem_safe():
    assert slug("fp8_e4m3") == "fp8-e4m3"
    assert slug("2048/2048") == "2048-2048"
    assert slug("H100 (Lambda)") == "h100-lambda"


from cell import get, launch_cmd, load_cells, workload_cmd, check_gpu
from record_run import build_record, OUTCOMES
from status import summarise_records


def test_launch_command_is_a_complete_vllm_serve_line():
    cmd = launch_cmd(get("kvquant_h100_qwen3-32b_fp8-e4m3_b32"))
    assert cmd.startswith("vllm serve ")
    assert "--kv-cache-dtype fp8_e4m3" in cmd
    assert "--max-num-seqs 32" in cmd


def test_workload_command_carries_cell_id_and_mode():
    cmd = workload_cmd(get("kvquant_h100_qwen3-32b_fp8-e4m3_b32"))
    assert "--mode closed" in cmd
    assert "--cell-id kvquant_h100_qwen3-32b_fp8-e4m3_b32" in cmd
    assert "--osl forced" in cmd and "--in-flight 32" in cmd


def test_burstgpt_workload_command_points_at_a_replay_file():
    row = next(r for r in load_cells() if r["dataset"]["kind"] == "burstgpt")
    assert "--replay" in workload_cmd(row)


def test_scaled_burstgpt_command_carries_its_compression_factor():
    row = next(r for r in load_cells() if r["workload"].get("scale") == 0.25)
    cmd = workload_cmd(row)
    assert "--scale 0.25" in cmd
    assert "--duration 4800" in cmd


def test_check_gpu_accepts_a_matching_smi_report():
    row = get("kvquant_h100_qwen3-32b_fp8-e4m3_b32")
    assert check_gpu(row, "NVIDIA H100 80GB HBM3, 81559 MiB\n") == []


def test_check_gpu_rejects_the_wrong_part():
    row = get("kvquant_h100_qwen3-32b_fp8-e4m3_b32")
    errs = check_gpu(row, "NVIDIA A100-PCIE-40GB, 40960 MiB\n")
    assert errs and any("H100" in e for e in errs)


def test_check_gpu_rejects_the_wrong_gpu_count():
    row = next(r for r in load_cells() if r["expected_gpus"] == 2)
    assert check_gpu(row, "NVIDIA H200, 143771 MiB\n")


def test_record_carries_outcome_and_provenance():
    r = build_record("conc_h100_qwen3-30b-a3b_c32", "ok",
                     instance="vast-1", flags=[], artifacts="conc/...")
    assert r["cell_id"] == "conc_h100_qwen3-30b-a3b_c32"
    assert r["outcome"] == "ok"
    assert r["vllm_version"] == pins.VLLM_VERSION
    assert r["dataset_revision"] == pins.DATASETS_REVISION
    assert r["ts"].endswith("Z")


def test_every_failure_mode_is_a_valid_outcome():
    for o in ("ok", "flagged", "run_failed", "upload_failed", "skipped"):
        assert o in OUTCOMES
    with pytest.raises(ValueError):
        build_record("x", "finished-ish")


def test_status_counts_and_picks_the_next_runnable_cell():
    cells = [{"cell_id": "a", "order": 0}, {"cell_id": "b", "order": 1},
             {"cell_id": "c", "order": 2, "blocked": "rescaling"}]
    records = [{"cell_id": "a", "outcome": "ok", "flags": []}]
    s = summarise_records(cells, records)
    assert s["counts"]["ok"] == 1
    assert s["counts"]["pending"] == 1
    assert s["counts"]["blocked"] == 1
    assert s["next"] == "b"


def test_status_separates_a_flagged_run_from_a_clean_one():
    cells = [{"cell_id": "a", "order": 0}]
    records = [{"cell_id": "a", "outcome": "flagged",
                "flags": ["prefix_hit_pct=9.0 exceeds 2.0"]}]
    s = summarise_records(cells, records)
    assert s["counts"]["flagged"] == 1 and "ok" not in s["counts"]
    assert s["flagged"][0][0] == "a"


def test_a_failed_run_stays_pending_for_rerun():
    cells = [{"cell_id": "a", "order": 0}]
    records = [{"cell_id": "a", "outcome": "run_failed", "flags": []}]
    s = summarise_records(cells, records)
    assert s["counts"]["run_failed"] == 1
    assert s["next"] == "a"


def test_campaign_image_is_pinned_by_digest_not_tag():
    """A tag is mutable: two cells provisioned weeks apart under :v1 could run
    different builds, which is an uncontrolled variable across 248 cells."""
    assert pins.CAMPAIGN_IMAGE is not None, "publish the image, then pin it"
    assert "@sha256:" in pins.CAMPAIGN_IMAGE
    assert len(pins.CAMPAIGN_IMAGE.split("@sha256:")[1]) == 64
