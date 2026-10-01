#!/usr/bin/env python3
"""
build_prompt_bank.py

Build reusable offline prompt banks keyed by a tokenizer fingerprint.

What it does:
- Loads a dataset (non-streaming)
- Tokenizes text into a token stream
- For each requested max_prompt_len, slices fixed-size chunks and saves:
    prompts_max{L}_ids.npy with shape [num_prompts, L]
- Writes manifest.json for reproducibility
- Skips existing banks unless --force is set
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def sha256_file(path: str) -> str:
    """Compute SHA256 for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc_iso() -> str:
    """UTC timestamp for manifest."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def tokenizer_fingerprint(tok, work_dir: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Compute a tokenizer fingerprint by hashing core tokenizer files.

    Notes:
    - We save the tokenizer into work_dir to ensure stable local files.
    - Prefer tokenizer.json if present; otherwise use tokenizer.model or vocab/merges.
    """
    os.makedirs(work_dir, exist_ok=True)
    tok.save_pretrained(work_dir)

    core_files: List[str] = []

    tj = os.path.join(work_dir, "tokenizer.json")
    tm = os.path.join(work_dir, "tokenizer.model")
    vj = os.path.join(work_dir, "vocab.json")
    mt = os.path.join(work_dir, "merges.txt")

    if os.path.exists(tj):
        core_files = [tj]
    elif os.path.exists(tm):
        core_files = [tm]
    elif os.path.exists(vj) and os.path.exists(mt):
        core_files = [vj, mt]
    else:
        raise RuntimeError(f"Cannot find core tokenizer files under: {work_dir}")

    file_hashes: List[Tuple[str, str]] = []
    combo = hashlib.sha256()
    for p in sorted(core_files):
        name = os.path.basename(p)
        h = sha256_file(p)
        file_hashes.append((name, h))
        combo.update(name.encode("utf-8"))
        combo.update(h.encode("utf-8"))

    return combo.hexdigest(), file_hashes


def main():
    ap = argparse.ArgumentParser()

    # Tokenizer source
    ap.add_argument("--model", required=True, help="Local model path or HF id (download may occur here)")
    ap.add_argument("--trust_remote_code", action="store_true")

    # Dataset (non-streaming)
    ap.add_argument("--dataset", default=os.environ.get("PROMPT_DATASET", "<hf-dataset-id>"), help="HF dataset name")
    ap.add_argument("--subset", default=os.environ.get("PROMPT_DATASET_SUBSET"), help="HF dataset subset/config")
    ap.add_argument("--split", default="train", help="Dataset split")
    ap.add_argument("--text_col", default="text", help="Text column name")

    # Banks to build
    ap.add_argument("--max_prompt_lens", type=int, nargs="+", default=[1024, 2048, 4096],
                    help="List of max prompt lengths (tokens) to build in one run")
    ap.add_argument("--num_prompts", type=int, default=50000, help="Number of prompts per max length")

    # Output and behavior
    ap.add_argument("--out_root", default="prompt_banks", help="Root directory for prompt banks")
    ap.add_argument("--force", action="store_true", help="Rebuild even if outputs already exist")
    ap.add_argument("--shuffle_examples", action="store_true", help="Shuffle dataset once (uses --seed)")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Normalize and validate lengths.
    lens = sorted(set(int(x) for x in args.max_prompt_lens))
    if any(x <= 0 for x in lens):
        raise ValueError("--max_prompt_lens must be positive integers")

    # Load tokenizer.
    tok = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Compute tokenizer fingerprint.
    tmp_dir = os.path.join(args.out_root, "_tokenizer_snapshots", "tmp")
    fp, fp_files = tokenizer_fingerprint(tok, tmp_dir)

    bank_dir = os.path.join(args.out_root, fp)
    os.makedirs(bank_dir, exist_ok=True)

    # Save a tokenizer snapshot for auditing.
    snap_dir = os.path.join(bank_dir, "tokenizer_snapshot")
    if args.force or (not os.path.isdir(snap_dir)):
        os.makedirs(snap_dir, exist_ok=True)
        tok.save_pretrained(snap_dir)

    # Decide which banks still need to be built.
    outputs: Dict[int, str] = {}
    need_build: List[int] = []
    for L in lens:
        path = os.path.join(bank_dir, f"prompts_max{L}_ids.npy")
        outputs[L] = path
        if args.force or (not os.path.exists(path)):
            need_build.append(L)

    if not need_build:
        print(f"[SKIP] All requested banks already exist for tokenizer fingerprint: {fp}")
        for L in lens:
            print(f"       max_len={L}  file={outputs[L]}")
        print(f"       manifest={os.path.join(bank_dir, 'manifest.json')}")
        return

    # Load dataset (non-streaming). This may download once if not cached.
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if args.shuffle_examples:
        ds = ds.shuffle(seed=args.seed)

    # Independent buffers per length (simple and robust).
    bufs: Dict[int, List[int]] = {L: [] for L in need_build}
    banks: Dict[int, List[List[int]]] = {L: [] for L in need_build}

    def try_cut(L: int) -> None:
        """Slice as many fixed chunks of length L as possible."""
        buf = bufs[L]
        while len(buf) >= L and len(banks[L]) < args.num_prompts:
            banks[L].append(buf[:L])
            del buf[:L]

    # Stream through dataset examples and fill all requested banks.
    for ex in ds:
        text = ex.get(args.text_col, "")
        if not text:
            continue

        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue

        # Add EOS between examples to avoid unnatural boundary merges.
        ids = ids + [tok.eos_token_id]

        for L in need_build:
            if len(banks[L]) >= args.num_prompts:
                continue
            bufs[L].extend(ids)
            try_cut(L)

        if all(len(banks[L]) >= args.num_prompts for L in need_build):
            break

    # Validate and save outputs.
    built = {}
    for L in need_build:
        if len(banks[L]) < args.num_prompts:
            raise RuntimeError(
                f"Insufficient data to build max_len={L}: "
                f"generated {len(banks[L])}, expected {args.num_prompts}. "
                f"Use a larger dataset/split or reduce --num_prompts."
            )

        arr = np.asarray(banks[L], dtype=np.int32)
        np.save(outputs[L], arr)
        built[L] = {"path": os.path.basename(outputs[L]), "shape": list(arr.shape), "dtype": str(arr.dtype)}

    # Write/overwrite manifest.json.
    manifest_path = os.path.join(bank_dir, "manifest.json")
    manifest = {
        "created_utc": now_utc_iso(),
        "tokenizer": {
            "fingerprint_sha256": fp,
            "fingerprinted_files": [{"name": n, "sha256": h} for n, h in fp_files],
            "snapshot_dir": "tokenizer_snapshot",
            "model_arg": args.model,
            "trust_remote_code": bool(args.trust_remote_code),
        },
        "dataset": {
            "name": args.dataset,
            "subset": args.subset,
            "split": args.split,
            "text_col": args.text_col,
            "shuffle_examples": bool(args.shuffle_examples),
            "seed": int(args.seed),
        },
        "banks": {
            "num_prompts_per_len": int(args.num_prompts),
            "built_or_existing": {str(L): os.path.basename(outputs[L]) for L in lens},
        },
        "built_this_run": built,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Tokenizer fingerprint: {fp}")
    for L in lens:
        status = "BUILT" if L in need_build else "EXISTS"
        print(f"[{status}] max_len={L}  file={outputs[L]}")
    print(f"[OK] manifest={manifest_path}")


if __name__ == "__main__":
    main()
