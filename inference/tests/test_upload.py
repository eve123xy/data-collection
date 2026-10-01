import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _sub in ("", "dataset", "workload", "telemetry", "cells", "upload"):
    sys.path.insert(0, str(ROOT / "scripts" / _sub))

from upload_run import (CHECKSUMS_NAME, dest_prefix, file_checksums,
                        mark_incomplete, upload, verify_remote, write_checksums)

CELL = {"cell_id": "kvquant_h100_qwen3-32b_fp8-e4m3_b32", "sheet": "KV Quant",
        "gpu": "H100", "model": "Qwen/Qwen3-32B", "tp": 1,
        "workload": {"duration": 600}, "pre_launch": []}


# --- checksums and paths ---

def test_destination_path_mirrors_the_tracker_shape():
    assert dest_prefix(CELL) == (
        "artifacts/KV Quant/H100/Qwen3-32B/kvquant_h100_qwen3-32b_fp8-e4m3_b32")


def test_destination_uses_the_model_leaf_not_the_full_hf_id():
    """A '/' in the model id would silently create an extra folder level."""
    assert "Qwen/Qwen3-32B" not in dest_prefix(CELL)
    assert dest_prefix(CELL).count("/") == 4


def test_checksums_cover_every_file_with_size_and_sha(tmp_path):
    (tmp_path / "run_meta.json").write_text('{"a": 1}')
    (tmp_path / "dcgm_x_gpu0.csv").write_text("ts,power_w\n1,300\n")
    got = file_checksums(tmp_path)
    assert set(got) == {"run_meta.json", "dcgm_x_gpu0.csv"}
    assert got["run_meta.json"]["size"] == 8
    assert len(got["run_meta.json"]["sha256"]) == 64


def test_checksums_file_excludes_itself(tmp_path):
    (tmp_path / "run_meta.json").write_text("{}")
    write_checksums(tmp_path, CELL)
    assert CHECKSUMS_NAME not in file_checksums(tmp_path)


def test_written_checksums_carry_the_cell_and_destination(tmp_path):
    (tmp_path / "run_meta.json").write_text("{}")
    manifest = write_checksums(tmp_path, CELL)
    on_disk = json.loads((tmp_path / CHECKSUMS_NAME).read_text())
    assert on_disk == manifest
    assert manifest["cell_id"] == CELL["cell_id"]
    assert manifest["dest"] == dest_prefix(CELL)
    assert manifest["incomplete"] == []
    assert "run_meta.json" in manifest["files"]


# --- Drive client ---

from drive import ensure_folder, ensure_path, find_file, list_folder, put_file

FOLDER_MIME = "application/vnd.google-apps.folder"


class FakeFiles:
    """Records every call so the tests can assert on the API contract."""

    def __init__(self, store):
        self.store = store
        self.calls = []
        self._n = 0

    def _exec(self, value):
        return type("R", (), {"execute": lambda _self: value})()

    def list(self, **kw):
        self.calls.append(("list", kw))
        parent = kw["q"].split("'")[1]
        name = kw["q"].split("name='")[1].split("'")[0] if "name='" in kw["q"] else None
        hits = [{"id": i, "name": v["name"], "mimeType": v["mimeType"],
                 "size": str(v.get("size", 1))}
                for i, v in self.store.items()
                if v["parent"] == parent and (name is None or v["name"] == name)]
        return self._exec({"files": hits})

    def create(self, **kw):
        self.calls.append(("create", kw))
        self._n += 1
        fid = f"id{self._n}"
        body = kw["body"]
        self.store[fid] = {"name": body["name"], "parent": body["parents"][0],
                           "mimeType": body.get("mimeType", "text/plain"), "size": 1}
        return self._exec({"id": fid, "name": body["name"]})

    def update(self, **kw):
        self.calls.append(("update", kw))
        return self._exec({"id": kw["fileId"], "name": self.store[kw["fileId"]]["name"]})

    def delete(self, **kw):
        raise AssertionError("delete must never be called - canDelete is False")


class FakeSvc:
    def __init__(self, store=None):
        self._files = FakeFiles(store or {})

    def files(self):
        return self._files


def test_every_call_declares_shared_drive_support():
    """A Shared Drive is invisible without supportsAllDrives, and its contents
    without includeItemsFromAllDrives."""
    svc = FakeSvc()
    ensure_folder(svc, "drv", "root", "KV Quant")
    for name, kw in svc.files().calls:
        assert kw.get("supportsAllDrives") is True
        if name == "list":
            assert kw.get("includeItemsFromAllDrives") is True
            assert kw.get("driveId") == "drv"
            assert kw.get("corpora") == "drive"


def test_ensure_folder_creates_once_then_reuses():
    svc = FakeSvc()
    a = ensure_folder(svc, "drv", "root", "KV Quant")
    b = ensure_folder(svc, "drv", "root", "KV Quant")
    assert a == b
    assert sum(1 for n, _ in svc.files().calls if n == "create") == 1


def test_ensure_path_builds_the_whole_tree():
    svc = FakeSvc()
    assert ensure_path(svc, "drv", "root", ["artifacts", "KV Quant", "H100"])
    created = [kw["body"]["name"] for n, kw in svc.files().calls if n == "create"]
    assert created == ["artifacts", "KV Quant", "H100"]


def test_put_file_updates_in_place_when_the_name_exists(tmp_path, monkeypatch):
    """canDelete is False, so overwrite MUST be find-then-update. That is also
    the idempotent form R15 asks for: re-uploading a cell never duplicates."""
    import drive as drv
    monkeypatch.setattr(drv, "MediaFileUpload", None, raising=False)
    p = tmp_path / "run_meta.json"
    p.write_text("{}")
    store = {"existing": {"name": "run_meta.json", "parent": "fold",
                          "mimeType": "application/json", "size": 2}}
    svc = FakeSvc(store)
    put_file(svc, "drv", "fold", p)
    kinds = [n for n, _ in svc.files().calls]
    assert "update" in kinds and "create" not in kinds


def test_put_file_creates_when_absent(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text("{}")
    svc = FakeSvc()
    put_file(svc, "drv", "fold", p)
    kinds = [n for n, _ in svc.files().calls]
    assert "create" in kinds and "update" not in kinds


def test_find_file_returns_none_when_absent():
    assert find_file(FakeSvc(), "drv", "fold", "nope.json") is None


# --- upload, verify, retry ---

def _manifest(files):
    return {"cell_id": "c", "dest": "artifacts/x", "incomplete": [],
            "files": {n: {"sha256": "s" * 64, "size": sz} for n, sz in files.items()}}


def test_verify_passes_when_every_file_arrived_at_the_right_size():
    m = _manifest({"a.json": 10, "b.csv": 20})
    assert verify_remote(m, [{"name": "a.json", "size": "10"},
                             {"name": "b.csv", "size": "20"}]) == []


def test_verify_catches_a_missing_file():
    m = _manifest({"a.json": 10, "b.csv": 20})
    assert verify_remote(m, [{"name": "a.json", "size": "10"}]) == ["b.csv"]


def test_verify_catches_a_truncated_file():
    """Section 2.3 is an entry about artifacts silently not arriving; a short
    file is the version of that which looks like success."""
    m = _manifest({"a.json": 10})
    assert verify_remote(m, [{"name": "a.json", "size": "3"}]) == ["a.json"]


def test_mark_incomplete_records_the_gap_structurally(tmp_path):
    """R13: partial results are marked as partial, never left to be inferred."""
    (tmp_path / "run_meta.json").write_text("{}")
    m = _manifest({"a.json": 10, "b.csv": 20})
    out = mark_incomplete(tmp_path, m, ["b.csv"])
    assert out["incomplete"] == ["b.csv"]
    assert json.loads((tmp_path / CHECKSUMS_NAME).read_text())["incomplete"] == ["b.csv"]


def _fixture_run(tmp_path):
    (tmp_path / "run_meta.json").write_text('{"cell_id": "c"}')
    (tmp_path / "summary.json").write_text('{"fatal": []}')
    (tmp_path / "dcgm_c_gpu0.csv").write_text("ts,power_w\n1,300\n")
    return tmp_path


def test_upload_sends_every_file_to_both_sinks(tmp_path):
    run = _fixture_run(tmp_path)
    hf, dv, remote = [], [], {}
    res = upload(run, CELL,
                 lambda p, rp: hf.append(rp),
                 lambda rd, pfx, names: [hf.append(f"{pfx}/{n}") for n in names],
                 lambda p: (dv.append(p.name),
                            remote.__setitem__(p.name, p.stat().st_size)),
                 lambda: [{"name": n, "size": str(s)} for n, s in remote.items()])
    assert res["ok"] is True and res["missing"] == []
    assert "checksums.json" in dv
    assert any(p.endswith("/checksums.json") for p in hf)
    assert len(hf) == len(dv) == 4


def test_upload_retries_only_the_missing_files(tmp_path):
    run = _fixture_run(tmp_path)
    remote, attempts = {}, []
    drop_once = {"dcgm_c_gpu0.csv"}

    def drive_put(path):
        attempts.append(path.name)
        if path.name in drop_once:
            drop_once.discard(path.name)
            return
        remote[path.name] = path.stat().st_size

    res = upload(run, CELL, lambda p, r: None, lambda rd, pfx, n: None, drive_put,
                 lambda: [{"name": n, "size": str(s)} for n, s in remote.items()])
    assert res["ok"] is True
    assert attempts.count("dcgm_c_gpu0.csv") == 2
    assert attempts.count("run_meta.json") == 1


def test_a_persistently_missing_file_is_reported_and_marked(tmp_path):
    run = _fixture_run(tmp_path)
    remote = {}

    def drive_put(path):
        if path.name != "dcgm_c_gpu0.csv":
            remote[path.name] = path.stat().st_size

    res = upload(run, CELL, lambda p, r: None, lambda rd, pfx, n: None, drive_put,
                 lambda: [{"name": n, "size": str(s)} for n, s in remote.items()])
    assert res["ok"] is False
    assert res["missing"] == ["dcgm_c_gpu0.csv"]
    assert json.loads((run / CHECKSUMS_NAME).read_text())["incomplete"] == ["dcgm_c_gpu0.csv"]


def test_a_failed_cell_still_uploads(tmp_path):
    """R14: publishing is decoupled from run success. The failure is evidence."""
    run = _fixture_run(tmp_path)
    (run / "summary.json").write_text('{"fatal": ["idle card"]}')
    remote = {}
    res = upload(run, CELL, lambda p, r: None, lambda rd, pfx, n: None,
                 lambda p: remote.__setitem__(p.name, p.stat().st_size),
                 lambda: [{"name": n, "size": str(s)} for n, s in remote.items()])
    assert res["ok"] is True and "summary.json" in remote


# --- the power profile figure ---

import csv as _csv
from plot_run import SERIES_COLORS, TOTAL_INK, panel_count, plot_run

FREQ_CELL = {"cell_id": "gpufreq_h100-lambda_qwen3-8b_1200mhz_b32",
             "sheet": "GPU Frequency", "gpu": "H100 (Lambda)",
             "model": "Qwen/Qwen3-8B", "tp": 1, "workload": {"duration": 600},
             "pre_launch": ["nvidia-smi -lgc 1200"]}


def test_palette_is_the_validated_categorical_order():
    assert SERIES_COLORS[:4] == ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    assert len(SERIES_COLORS) == 8
    assert TOTAL_INK not in SERIES_COLORS


def test_gpu_frequency_cells_get_a_fourth_panel_not_a_second_axis():
    """A dual-axis chart is ruled out; SM clock is its own panel."""
    assert panel_count(CELL) == 3
    assert panel_count(FREQ_CELL) == 4


def _write_run(tmp_path, cell=None, gpus=("0",), t0=1000.0, dur=60.0):
    cid = (cell or CELL)["cell_id"]
    for g in gpus:
        with open(tmp_path / f"dcgm_{cid}_gpu{g}.csv", "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=["ts", "entity_type", "gpu_id",
                                                "power_w", "total_energy_mj",
                                                "sm_clock_mhz"])
            w.writeheader()
            for i in range(int((dur + 20) * 10)):
                w.writerow({"ts": t0 - 10 + i * 0.1, "entity_type": "GPU",
                            "gpu_id": g, "power_w": 300 + 10 * int(g),
                            "total_energy_mj": i * 1000, "sm_clock_mhz": 1200})
    with open(tmp_path / f"timeline_{cid}.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["time_relative_s", "requests_arrived",
                                            "active_requests", "mean_prompt_tokens",
                                            "mean_output_tokens"])
        w.writeheader()
        for s in range(-10, int(dur)):
            w.writerow({"time_relative_s": s, "requests_arrived": 4,
                        "active_requests": 30, "mean_prompt_tokens": 310,
                        "mean_output_tokens": 500})
    return {"window_start_ts_ns": int(t0 * 1e9),
            "window_end_ts_ns": int((t0 + dur) * 1e9),
            "server": {"generation_tokens": 1000}}


def test_plot_renders_a_single_gpu_cell(tmp_path):
    meta = _write_run(tmp_path)
    out = plot_run(tmp_path, CELL, meta, tmp_path / "p.png")
    assert out.exists() and out.stat().st_size > 5000


def test_plot_renders_a_four_gpu_cell(tmp_path):
    """Per-GPU lines are the point on TP cells: an imbalanced card is obvious
    in the trace and invisible in a summed mean."""
    meta = _write_run(tmp_path, gpus=("0", "1", "2", "3"))
    out = plot_run(tmp_path, dict(CELL, tp=4), meta, tmp_path / "p4.png")
    assert out.exists() and out.stat().st_size > 5000


def test_plot_renders_the_fourth_panel_on_a_frequency_cell(tmp_path):
    meta = _write_run(tmp_path, cell=FREQ_CELL)
    out = plot_run(tmp_path, FREQ_CELL, meta, tmp_path / "pf.png")
    assert out.exists() and out.stat().st_size > 5000


def test_plot_survives_a_missing_timeline(tmp_path):
    """A cell can fail before the timeline is written; the power panel is still
    the most useful thing to look at."""
    meta = _write_run(tmp_path)
    (tmp_path / f"timeline_{CELL['cell_id']}.csv").unlink()
    assert plot_run(tmp_path, CELL, meta, tmp_path / "p.png").exists()


# --- the local pull ---

from pull_runs import pull, verify_local


def test_verify_local_accepts_matching_files(tmp_path):
    import hashlib
    (tmp_path / "a.json").write_text("hello")
    m = {"files": {"a.json": {"sha256": hashlib.sha256(b"hello").hexdigest(),
                              "size": 5}}}
    assert verify_local(tmp_path, m) == []


def test_verify_local_rejects_a_corrupted_file(tmp_path):
    """Refuse a bad remote copy rather than overwrite a good local one."""
    (tmp_path / "a.json").write_text("tampered")
    m = {"files": {"a.json": {"sha256": "0" * 64, "size": 8}}}
    assert verify_local(tmp_path, m) == ["a.json"]


def test_verify_local_reports_a_file_that_never_arrived(tmp_path):
    m = {"files": {"missing.json": {"sha256": "0" * 64, "size": 3}}}
    assert verify_local(tmp_path, m) == ["missing.json"]


def test_pull_writes_files_and_verifies_them(tmp_path):
    import hashlib
    payload = {"run_meta.json": b"{}", "summary.json": b'{"fatal": []}'}
    manifest = {"cell_id": CELL["cell_id"], "dest": dest_prefix(CELL), "incomplete": [],
                "files": {n: {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
                          for n, b in payload.items()}}

    def fetch(repo_path):
        name = repo_path.rsplit("/", 1)[-1]
        return json.dumps(manifest).encode() if name == CHECKSUMS_NAME else payload[name]

    res = pull(CELL, tmp_path, fetch)
    assert res["ok"] is True and res["bad"] == []
    assert (tmp_path / dest_prefix(CELL) / "run_meta.json").read_bytes() == b"{}"


def test_pull_refuses_a_file_whose_hash_does_not_match(tmp_path):
    manifest = {"cell_id": CELL["cell_id"], "dest": dest_prefix(CELL), "incomplete": [],
                "files": {"a.json": {"sha256": "0" * 64, "size": 2}}}

    def fetch(repo_path):
        name = repo_path.rsplit("/", 1)[-1]
        return json.dumps(manifest).encode() if name == CHECKSUMS_NAME else b"{}"

    res = pull(CELL, tmp_path, fetch)
    assert res["ok"] is False and res["bad"] == ["a.json"]
    assert not (tmp_path / dest_prefix(CELL) / "a.json").exists()


def test_pull_surfaces_an_upload_that_was_marked_incomplete(tmp_path):
    """R13's marking is only useful if the reader acts on it."""
    manifest = {"cell_id": CELL["cell_id"], "dest": dest_prefix(CELL),
                "incomplete": ["dcgm_c_gpu0.csv"], "files": {}}
    res = pull(CELL, tmp_path, lambda rp: json.dumps(manifest).encode())
    assert res["incomplete"] == ["dcgm_c_gpu0.csv"] and res["ok"] is False
