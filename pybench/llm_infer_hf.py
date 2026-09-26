#!/usr/bin/env python3
"""
infer_power_burn.py

Minimal inference burn loop for GPU power profiling.
- Fixed shape workload: batch, prompt_len, max_new_tokens
- Warmup + steady-state loop
- Optional idle phases to create clear plateaus in power logs
"""

import os
import time
import argparse
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(tag: str, msg: str = ""):
    print(f"[{ts()}][{tag}] {msg}", flush=True)


def pick_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    raise ValueError("dtype must be bf16|fp16")


def get_device_map(mode: str):
    # tp: one process sees multiple GPUs -> device_map="auto"
    # dp: one process per GPU -> map to LOCAL_RANK
    if mode == "tp":
        return "auto"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return {"": local_rank}


def make_fixed_prompt(tokenizer, prompt_len: int, batch_size: int):
    # Build a deterministic text and truncate to prompt_len tokens.
    base = ("Hello world. " * 10000).strip()
    enc = tokenizer(
        base,
        truncation=True,
        max_length=prompt_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].repeat(batch_size, 1)
    attn_mask = enc["attention_mask"].repeat(batch_size, 1)
    return input_ids, attn_mask


def main():
    p = argparse.ArgumentParser()

    # Model loading
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-chat-hf", help="HF repo id")
    p.add_argument("--parallel_mode", type=str, default="tp", choices=["tp", "dp"])
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--load_in_4bit", action="store_true", help="4-bit quantized weights")
    p.add_argument("--attn_impl", type=str, default="eager",
                   choices=["eager", "sdpa", "flash_attention_2"])

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
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--use_cache", action="store_true")

    # Measurement controls
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--sync_each_iter", action="store_true")
    p.add_argument("--sleep_every", type=int, default=0)
    p.add_argument("--sleep_sec", type=float, default=0.0)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = pick_dtype(args.dtype)
    device_map = get_device_map(args.parallel_mode)

    log("INFO", f"loading model={args.model} dtype={args.dtype} 4bit={args.load_in_4bit} attn={args.attn_impl}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    common = dict(
        pretrained_model_name_or_path=args.model,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_impl,
    )

    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(**common, quantization_config=bnb)
    else:
        model = AutoModelForCausalLM.from_pretrained(**common, dtype=dtype)

    model.eval()

    input_ids, attn_mask = make_fixed_prompt(tok, args.prompt_len, args.batch_size)

    # Move inputs only for single-device (dp) to avoid multi-GPU device_map issues.
    if isinstance(device_map, dict):
        dev = torch.device("cuda")
        input_ids = input_ids.to(dev, non_blocking=True)
        attn_mask = attn_mask.to(dev, non_blocking=True)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping if args.num_beams > 1 else None,
        use_cache=args.use_cache,
        pad_token_id=tok.eos_token_id,
    )
    # Remove None to keep HF generate clean.
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    log("MARK", "WARMUP_BEGIN")
    with torch.inference_mode():
        for _ in range(args.warmup_steps):
            _ = model.generate(input_ids=input_ids, attention_mask=attn_mask, **gen_kwargs)
            if args.sync_each_iter and torch.cuda.is_available():
                torch.cuda.synchronize()
    log("MARK", "WARMUP_END")

    log("MARK", "STEADY_BEGIN")
    t0 = time.time()
    tokens = 0

    with torch.inference_mode():
        for step in range(1, args.steps + 1):
            t1 = time.perf_counter()
            _ = model.generate(input_ids=input_ids, attention_mask=attn_mask, **gen_kwargs)
            if args.sync_each_iter and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t1

            tokens += args.batch_size * max(1, args.max_new_tokens)

            if step % args.print_every == 0:
                elapsed = time.time() - t0
                tput = tokens / max(elapsed, 1e-9)
                log("STEP", f"{step:05d} step_time={dt:.3f}s gen_tok/s={tput:.1f}")

            if args.sleep_every > 0 and (step % args.sleep_every == 0):
                log("SLEEP", f"begin {args.sleep_sec}s at step={step}")
                time.sleep(args.sleep_sec)
                log("SLEEP", "end")

    log("MARK", "STEADY_END")


if __name__ == "__main__":
    main()
