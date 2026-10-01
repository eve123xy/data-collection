"""Small shared helpers. Nothing clever."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path):
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_meta(path, meta):
    Path(path).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_meta(path):
    return json.loads(Path(path).read_text())


def load_env():
    """Load scripts/.env into os.environ. Values are never printed.

    scripts/.env is AUTHORITATIVE and overrides anything already exported in
    the shell. This used to use os.environ.setdefault(), which meant a stale
    HF_TOKEN exported in a developer shell silently beat the rotated one in
    .env -- every script then authenticated with a revoked credential and the
    failure surfaced as "Invalid user token", pointing at the wrong file.
    That is PIPELINE_POSTMORTEM R1 (configuration with no single definition)
    recurring, so the credential file wins.
    """
    env = Path(__file__).with_name(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")


def drive_root_folder_id():
    """Target folder for run uploads.

    Accepts DRIVE_ROOT_FOLDER_ID or the shorter DRIVE_ROOT_FOLDER, and falls
    back to DRIVE_ID (a Shared Drive's root folder id IS its drive id).
    """
    load_env()
    return (os.environ.get("DRIVE_ROOT_FOLDER_ID")
            or os.environ.get("DRIVE_ROOT_FOLDER")
            or os.environ.get("DRIVE_ID"))


def token():
    load_env()
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("[FATAL] HF_TOKEN not found in environment or scripts/.env")
    return tok


def download(repo, filename, revision, dest_dir):
    """Download one file at a PINNED revision into dest_dir."""
    from huggingface_hub import hf_hub_download
    if not revision:
        sys.exit("[FATAL] pins.DATASETS_REVISION is not set. Publish first, "
                 "record the printed sha in scripts/pins.py, and commit.")
    cached = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                             repo_type="dataset", token=token())
    out = Path(dest_dir) / Path(filename).name
    shutil.copyfile(cached, out)
    return out


def verify(path, expected_sha, expected_rows):
    """Exit non-zero unless the file matches both expectations."""
    rows = count_rows(path)
    if rows != expected_rows:
        sys.exit(f"[FATAL] {path}: {rows:,} rows, expected {expected_rows:,}")
    sha = sha256_file(path)
    if sha != expected_sha:
        sys.exit(f"[FATAL] {path}: sha256 {sha} != expected {expected_sha}")


def publish(repo, folder):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    res = api.upload_folder(repo_id=repo, folder_path=str(folder),
                            repo_type="dataset", commit_message="publish artifacts")
    print(f'[OK] published. Set pins.DATASETS_REVISION = "{res.oid}" and commit.')
    return res.oid
