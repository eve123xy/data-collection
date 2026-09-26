#!/usr/bin/env python3
"""
sglang_burn.py

Quick SGLang offline-engine burn test (no HTTP server).
- Fixed prompt length via HF tokenizer -> stable workload shape
- Warmup + steady loop
- Prints tokens/s for generated tokens (decode-heavy friendly)

This is intended for interactive srun sanity runs and later power profiling.
"""

import time
import argparse
from datetime import datetime

import torch
from transformers import AutoTokenizer
import sglang as sgl


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(tag: str, msg: str = "") -> None:
    print(f"[{ts()}][{tag}] {msg}", flush=True)


def make_fixed_prompt_text(tokenizer, prompt_len: int) -> str:
    """Create a deterministic prompt whose tokenized length is ~prompt_len."""
    base = ("Hello world. " * 10000).strip()
    enc = tokenizer(
        base,
        truncation=True,
        max_length=prompt_len,
        padding="max_length",
        return_tensors="pt",
    )
    ids = enc["input_ids"][0]
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()

    # Engine / model
    p.add_argument("--model", type=str, required=True, help="HF repo id or local path")
    p.add_argument("--tp_size", type=int, default=1, help="Tensor-parallel size on this node")
    p.add_argument("--dtype", type=str, default="fp16", help="Data type: auto | fp16/f16 | bf16 | f32")
    p.add_argument("--mem_fraction_static", type=float, default=0.85)

    # Workload shape
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--prompt_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--min_new_tokens", type=int, default=0)

    # Decoding
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)

    # Loop
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--print_every", type=int, default=5)
    p.add_argument("--sync_each_iter", action="store_true")

    args = p.parse_args()

    # Basic sanity
    assert torch.cuda.is_available(), "CUDA is not available"
    log("ENV", f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    log("CFG", f"model={args.model} tp_size={args.tp_size} dtype={args.dtype}")

    # Tokenizer only used to build a stable-length prompt
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    prompt = make_fixed_prompt_text(tok, args.prompt_len)
    prompts = [prompt for _ in range(args.batch_size)]

    # Create offline engine
    # Note: some versions use tp instead of tp_size.
    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp_size,
        dtype=args.dtype,
        mem_fraction_static=args.mem_fraction_static,
        trust_remote_code=True,
    )

    # Sampling params (dict-based)
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

    # Warmup
    log("MARK", "WARMUP_BEGIN")
    for _ in range(args.warmup_steps):
        _ = engine.generate(prompts, sampling_params)
        if args.sync_each_iter:
            torch.cuda.synchronize()
    log("MARK", "WARMUP_END")

    # Steady loop
    log("MARK", "STEADY_BEGIN")
    t0 = time.time()
    tokens = 0

    for step in range(1, args.steps + 1):
        t1 = time.perf_counter()
        _ = engine.generate(prompts, sampling_params)
        if args.sync_each_iter:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t1

        tokens += args.batch_size * max(1, args.max_new_tokens)

        if step % args.print_every == 0:
            elapsed = time.time() - t0
            tput = tokens / max(elapsed, 1e-9)
            log("STEP", f"{step:05d} step_time={dt:.3f}s gen_tok/s={tput:.1f}")

    log("MARK", "STEADY_END")


if __name__ == "__main__":
    main()
