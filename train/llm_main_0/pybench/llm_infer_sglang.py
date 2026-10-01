#!/usr/bin/env python3
"""
llm_infer_sglang.py

Offline inference burn loop for GPU power profiling using SGLang Engine.

Default behavior:
- prompt_source=synthetic (repetitive text) for simple burn/stress runs.

Also supports:
- prompt_source=realistic (dataset-based prompt bank)
  - Auto-selects a local prompt bank by tokenizer fingerprint
  - Chooses the smallest prompts_max{L}_ids.npy with L >= prompt_len
  - Loads the bank with NumPy mmap (read-only) to avoid large RAM usage
  - Internal one-time shuffle for better mixing (no CLI flag)
  - Optional nonce to reduce prefix-cache reuse

Logs:
- Clear, concise English logs that state which mode is used and which bank file is selected.
"""

import os
import re
import time
import argparse
import random
import hashlib
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

import sglang as sgl


# ----------------------------
# Logging
# ----------------------------

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(tag: str, msg: str = "") -> None:
    print(f"[{ts()}][{tag}] {msg}", flush=True)


# ----------------------------
# Tokenizer fingerprint + bank selection
# ----------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tokenizer_fingerprint(tok, tmp_dir: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Hash core tokenizer files to get a stable fingerprint.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    tok.save_pretrained(tmp_dir)

    tj = os.path.join(tmp_dir, "tokenizer.json")
    tm = os.path.join(tmp_dir, "tokenizer.model")
    vj = os.path.join(tmp_dir, "vocab.json")
    mt = os.path.join(tmp_dir, "merges.txt")

    if os.path.exists(tj):
        core_files = [tj]
    elif os.path.exists(tm):
        core_files = [tm]
    elif os.path.exists(vj) and os.path.exists(mt):
        core_files = [vj, mt]
    else:
        raise RuntimeError(f"Cannot find core tokenizer files under: {tmp_dir}")

    file_hashes: List[Tuple[str, str]] = []
    combo = hashlib.sha256()
    for p in sorted(core_files):
        name = os.path.basename(p)
        h = _sha256_file(p)
        file_hashes.append((name, h))
        combo.update(name.encode("utf-8"))
        combo.update(h.encode("utf-8"))

    return combo.hexdigest(), file_hashes


def _select_prompt_bank(prompt_bank_root: str, fingerprint: str, prompt_len: int) -> Tuple[str, int]:
    """
    Select the smallest prompts_max{L}_ids.npy with L >= prompt_len.
    Returns (bank_path, bank_max_len).
    """
    bank_dir = os.path.join(prompt_bank_root, fingerprint)
    if not os.path.isdir(bank_dir):
        raise FileNotFoundError(
            "Prompt bank not found for this tokenizer.\n"
            f"Expected directory: {bank_dir}\n"
            "Please run build_prompt_bank.py with the same tokenizer."
        )

    pat = re.compile(r"^prompts_max(\d+)_ids\.npy$")
    candidates: List[Tuple[int, str]] = []
    for fn in os.listdir(bank_dir):
        m = pat.match(fn)
        if m:
            L = int(m.group(1))
            candidates.append((L, os.path.join(bank_dir, fn)))

    if not candidates:
        raise FileNotFoundError(
            "Prompt bank not found for this tokenizer.\n"
            f"Expected files like prompts_maxXXXX_ids.npy under: {bank_dir}\n"
            "Please run build_prompt_bank.py first."
        )

    candidates.sort(key=lambda x: x[0])
    for L, path in candidates:
        if L >= prompt_len:
            return path, L

    max_available = candidates[-1][0]
    raise FileNotFoundError(
        "Prompt bank not found for this tokenizer.\n"
        f"Requested prompt_len={prompt_len}, but largest available bank is max_len={max_available}.\n"
        f"Please rebuild with a larger --max_prompt_len (>= {prompt_len})."
    )


# ----------------------------
# Prompt builders
# ----------------------------

def make_synthetic_prompt_text(tokenizer, prompt_len: int, base: str, repeat: int) -> str:
    """
    Build a synthetic prompt whose tokenized length is exactly prompt_len.
    """
    raw = (base * repeat).strip()
    enc = tokenizer(
        raw,
        truncation=True,
        max_length=prompt_len,
        padding="max_length",
        return_tensors="pt",
    )
    ids = enc["input_ids"][0]
    return tokenizer.decode(ids, skip_special_tokens=True)


def apply_nonce_prefix(ids_1d: np.ndarray, k: int, tok, rng: random.Random) -> np.ndarray:
    """Replace the k-th token (1-based) with a random token (k<=0 disables)."""
    if k <= 0:
        return ids_1d

    ids = ids_1d.copy()
    n = len(ids)
    if n == 0:
        return ids

    # k is 1-based; clamp to valid range.
    i = k - 1
    if i >= n:
        i = n - 1

    vocab = tok.vocab_size
    bad = {x for x in (tok.eos_token_id, tok.bos_token_id, tok.pad_token_id, tok.unk_token_id) if x is not None}

    t = rng.randrange(vocab)
    while t in bad:
        t = rng.randrange(vocab)

    ids[i] = t
    return ids


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()

    # Model / engine
    p.add_argument("--model", type=str, required=True, help="Model id or local path")
    p.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--dtype", type=str, default="fp16", help="auto | fp16/f16 | bf16 | f32")
    p.add_argument("--mem_fraction_static", type=float, default=0.85,
                   help="Static GPU memory fraction for engine KV/cache allocation")

    # Workload shape
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--prompt_len", type=int, default=512, help="Runtime prompt length (tokens)")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--min_new_tokens", type=int, default=0)

    # Prompt source selection (default: synthetic)
    p.add_argument("--prompt_source", type=str, default="synthetic",
                   choices=["synthetic", "realistic"],
                   help="synthetic: repetitive text; realistic: dataset prompt bank")

    # Synthetic prompt config
    p.add_argument("--synthetic_base", type=str, default="Hello world. ",
                   help="Base text used for synthetic prompt")
    p.add_argument("--synthetic_repeat", type=int, default=10000,
                   help="Repeat count for synthetic_base")

    # Realistic prompt bank config (auto-selected by tokenizer fingerprint)
    p.add_argument("--prompt_bank_root", type=str, default="prompt_banks",
                   help="Root directory containing prompt banks keyed by tokenizer fingerprint")
    p.add_argument("--nonce_tokens", type=int, default=0,
                   help="Replace the k-th token (1-based) in prompts (0 disables)")
    p.add_argument("--verify_prompt_len", action="store_true",
                   help="Re-tokenize decoded prompts to verify token length (slower)")

    # Decoding
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)

    # Measurement controls
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--sync_each_iter", action="store_true")
    p.add_argument("--count_total_tokens", action="store_true",
                   help="Report (prompt+gen) tokens/s in addition to gen tokens/s")

    # Optional plateaus
    p.add_argument("--sleep_every", type=int, default=0)
    p.add_argument("--sleep_sec", type=float, default=0.0)

    args = p.parse_args()

    # Offline mode.
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log("INFO", f"model={args.model} tp_size={args.tp_size} dtype={args.dtype}")
    log("INFO", f"prompt_source={args.prompt_source} batch_size={args.batch_size} prompt_len={args.prompt_len} max_new_tokens={args.max_new_tokens}")

    # Tokenizer.
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    rng = random.Random(args.seed)

    # Build prompt batch function.
    if args.prompt_source == "synthetic":
        # Build fixed-length ids once, then apply nonce per batch element.
        raw = (args.synthetic_base * args.synthetic_repeat).strip()
        enc = tok(
            raw,
            truncation=True,
            max_length=args.prompt_len,
            padding="max_length",
            return_tensors="pt",
        )
        synthetic_ids = enc["input_ids"][0].cpu().numpy().astype(np.int32)

        log("INFO", f"Synthetic prompt ready (len={args.prompt_len}). nonce_tokens={args.nonce_tokens}")

        def build_prompts_batch() -> List[str]:
            out = []
            for _ in range(args.batch_size):
                ids = apply_nonce_prefix(synthetic_ids, args.nonce_tokens, tok, rng)
                out.append(tok.decode(ids.tolist(), skip_special_tokens=True))
            return out

    else:
        tmp_dir = os.path.join(args.prompt_bank_root, "_tokenizer_snapshots", "tmp")
        fingerprint, _ = _tokenizer_fingerprint(tok, tmp_dir)
        bank_path, bank_max_len = _select_prompt_bank(args.prompt_bank_root, fingerprint, args.prompt_len)

        prompt_bank = np.load(bank_path, mmap_mode="r")
        if prompt_bank.ndim != 2:
            raise ValueError(f"prompt bank must be 2D, got shape={prompt_bank.shape}")

        N, L = prompt_bank.shape
        if L != bank_max_len:
            log("WARN", f"Bank file name max_len={bank_max_len} but array shape max_len={L}")
            bank_max_len = L

        indices = list(range(N))
        rng.shuffle(indices)
        ptr = 0

        log("INFO", f"Tokenizer fingerprint: {fingerprint}")
        log("INFO", f"Selected prompt bank: {bank_path}")
        log("INFO", f"Bank shape={prompt_bank.shape} (mmap=read-only)")
        log("INFO", f"nonce_tokens={args.nonce_tokens} (replace k-th token; 0 disables)")
        log("INFO", f"Requested prompt_len={args.prompt_len}, selected bank_max_len={bank_max_len}")

        def build_prompts_batch() -> List[str]:
            """Sample rows, slice to prompt_len, apply nonce, decode to text."""
            nonlocal ptr
            out = []
            for _ in range(args.batch_size):
                idx = indices[ptr]
                ptr = (ptr + 1) % N

                ids = np.asarray(prompt_bank[idx][:args.prompt_len], dtype=np.int32)
                ids = apply_nonce_prefix(ids, args.nonce_tokens, tok, rng)
                text = tok.decode(ids.tolist(), skip_special_tokens=True)

                if args.verify_prompt_len:
                    v = tok(text, add_special_tokens=False)["input_ids"]
                    if len(v) != args.prompt_len:
                        raise RuntimeError(f"Token length mismatch: expected {args.prompt_len}, got {len(v)}")

                out.append(text)
            return out

    # Engine.
    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        mem_fraction_static=args.mem_fraction_static,
        disable_radix_cache=False,
    )

    # Sampling params.
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
    }
    if args.do_sample:
        sampling_params.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k if args.top_k > 0 else -1,
        })
    else:
        sampling_params.update({"temperature": 0.0})

    log("MARK", "WARMUP_BEGIN")
    for _ in range(args.warmup_steps):
        prompts = build_prompts_batch()
        _ = engine.generate(prompts, sampling_params)
        if args.sync_each_iter and torch.cuda.is_available():
            torch.cuda.synchronize()
    log("MARK", "WARMUP_END")

    log("MARK", "STEADY_BEGIN")
    t0 = time.time()
    gen_tokens = 0
    total_tokens = 0

    for step in range(1, args.steps + 1):
        prompts = build_prompts_batch()

        t1 = time.perf_counter()
        _ = engine.generate(prompts, sampling_params)
        if args.sync_each_iter and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t1

        gen_tokens += args.batch_size * max(1, args.max_new_tokens)
        if args.count_total_tokens:
            total_tokens += args.batch_size * (args.prompt_len + max(1, args.max_new_tokens))

        if step % args.print_every == 0:
            elapsed = time.time() - t0
            gen_tput = gen_tokens / max(elapsed, 1e-9)
            if args.count_total_tokens:
                total_tput = total_tokens / max(elapsed, 1e-9)
                log("STEP", f"{step:05d} step_time={dt:.3f}s gen_tok/s={gen_tput:.1f} total_tok/s={total_tput:.1f}")
            else:
                log("STEP", f"{step:05d} step_time={dt:.3f}s gen_tok/s={gen_tput:.1f}")

        if args.sleep_every > 0 and (step % args.sleep_every == 0):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            log("SLEEP", f"begin {args.sleep_sec}s at step={step}")
            time.sleep(args.sleep_sec)
            log("SLEEP", "end")

    log("MARK", "STEADY_END")


if __name__ == "__main__":
    main()
