#!/usr/bin/env python3
"""
prefetch_llms_ft.py
Pre-download (cache) popular open LLMs **and their tokenizers** using
Transformers' `from_pretrained()` so later training/inference reuses the same cache.

Design:
- Simple: no CLI args. Edit model lists below as needed.
- Safe: loads on CPU, frees memory immediately after download.
- Skips ultra-large models by default (uncomment if you really need them).
"""

import os
import argparse
import gc
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===============================================================
# 1) SMALL_FULL
#    Full-parameter fine-tuning of ALL weights in fp16/bf16.
#    Typical model size: ≤ ~9B dense.
#
#    Hardware guidance:
#      - 4×RTX8000 (48GB): OK if you keep batch size small.
#      - 4×A100 80GB / H100 80GB: very comfortable.
# ===============================================================
SMALL_FULL = [
    # ----- Meta (LLaMA small) -----
    "meta-llama/Llama-3.2-3B-Instruct",          # ~3B dense
    "meta-llama/Llama-2-7b-chat-hf",             # ~7B dense
    "meta-llama/Meta-Llama-3-8B-Instruct",       # ~8B dense

    # ----- Mistral -----
    "mistralai/Mistral-7B-Instruct-v0.3",        # ~7B dense

    # ----- DeepSeek distilled small -----
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   # ~7B dense; distilled reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # ~8B dense; distilled reasoning

    # ----- Alibaba (Qwen 2.x / 3.x small) -----
    "Qwen/Qwen2.5-3B-Instruct",                  # ~3B dense
    "Qwen/Qwen3-4B-Instruct-2507",               # ~4B dense; instruction-tuned
    "Qwen/Qwen3-4B-Thinking-2507",               # ~4B dense; reasoning-tuned
    "Qwen/Qwen2-7B-Instruct",                    # ~7B dense; bilingual

    # ----- Microsoft (Phi mini) -----
    "microsoft/Phi-3.5-mini-instruct",           # ~4B dense
    # "microsoft/Phi-4-mini-instruct",             # ~4B dense; LossKwargs removed from transformers==4.54.0

    # ----- Google (Gemma small-ish) -----
    "google/gemma-2-9b-it",                      # ~9B dense
]

# =====================================================================
# 2) LORA_FP16
#    LoRA adapters on fp16/bf16 backbone.
#    Typical model size: ~13B–15B dense or MoE with ≤ ~15B active params.
#
#    Hardware guidance:
#      • 4×RTX8000 (48GB): LoRA strongly recommended.
#      • 4×A100 80GB / H100 80GB: full fine-tune optionally feasible.
# =====================================================================
LORA_FP16 = [
    # ----- Meta (Llama mid) -----
    "meta-llama/Llama-2-13b-chat-hf",              # ~13B dense; standard LoRA baseline

    # ----- Mistral sparse mid (active params small enough for LoRA in fp16) -----
    "mistralai/Mixtral-8x7B-Instruct-v0.1",        # MoE 8×7B (~56B total, 12~13B active)

    # ----- Alibaba (Qwen mid) -----
    "Qwen/Qwen2.5-14B-Instruct",                   # ~15B dense

    # ----- Microsoft (Phi mid) -----
    "microsoft/Phi-3-medium-4k-instruct",          # ~14B dense
    "microsoft/phi-4",                             # ~15B dense
]

# ===============================================================
# 3) QLORA_4BIT
#    4-bit (NF4/AWQ) quantized backbone + LoRA adapters.
#    Typical model size: ~20B–70B dense or MoE with moderate active params.
#
#    Hardware guidance:
#      - 4×RTX8000: practical for QLoRA.
#      - 4×A100/H100: standard for QLoRA, allows larger batch sizes.
# ===============================================================
QLORA_4BIT = [
    # ----- OpenAI (mid-large) -----
    "openai/gpt-oss-20b",                          # ~22B dense

    # ----- DeepSeek -----
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",    # ~33B dense

    # ----- Alibaba (Qwen 30B+ dense / A3B MoE) -----
    "Qwen/Qwen2.5-32B-Instruct",                   # ~33B dense
    # "Qwen/Qwen2.5-32B-Instruct-AWQ",               # ~33B dense (4-bit AWQ); AutoAWQ is officially deprecated after Transformers==4.51.3.
    "Qwen/Qwen3-30B-A3B-Instruct-2507",            # ~30B total / ~3B active
    "Qwen/Qwen3-32B",                              # ~32B dense

    # ----- Google (Gemma large) -----
    "google/gemma-2-27b-it",                       # ~27B dense
]

# ===============================================================
# 4) FRONTIER_ADAPT
#    Frontier-scale adaptation (large QLoRA / distributed MoE).
#    Typical model size: ~50B–70B dense or very large MoE/A3B.
#
#    Hardware guidance:
#      • 4×RTX8000: borderline for QLoRA, requires careful tuning.
#      • 4×A100 / H100: QLoRA strongly recommended.
# ===============================================================
FRONTIER_ADAPT = [
    # ----- Meta (LLaMA 70B family) -----
    "meta-llama/Llama-2-70b-chat-hf",              # ~70B dense
    "meta-llama/Meta-Llama-3-70B-Instruct",        # ~70B dense
    "meta-llama/Llama-3.1-70B-Instruct",           # ~70B dense
    "meta-llama/Llama-3.3-70B-Instruct",           # ~70B dense

    # ----- Mistral (dense + sparse MoE) -----
    "mistralai/Mistral-Large-Instruct-2407",       # ~123B dense
    "mistralai/Mixtral-8x22B-Instruct-v0.1",       # MoE (8×22B total ~176B, ~39B active)

    # ----- Alibaba (Qwen2 / Qwen3 large-scale) -----
    "Qwen/Qwen2-72B-Instruct",                     # ~72B dense
    "Qwen/Qwen2.5-72B-Instruct",                   # ~73B dense
    "Qwen/Qwen3-Next-80B-A3B-Instruct",            # MoE (~80B total, ~3B active)
    "Qwen/Qwen3-Next-80B-A3B-Thinking",            # MoE (~80B total, ~3B active)
]

# ===============================================================
# 5) RESEARCH_SCALE
#    Ultra-large / frontier-tier models (≥ ~50B dense or MoE with ~37B active).
#
#    Hardware guidance:
#      - 4×RTX8000: inference only.
#      - 4×A100/H100: may support QLoRA training.
# ===============================================================
RESEARCH_SCALE = [
    # ----- OpenAI (GPT-OSS large) -----
    "openai/gpt-oss-120b",                         # ~120B dense; not runnable on RTX8000 (48G)

    # ----- DeepSeek frontier MoE -----
    "deepseek-ai/DeepSeek-R1",                     # MoE (~685B total; ~37B active); needs Hopper/Ada (H100/RTX4090)
    "deepseek-ai/DeepSeek-V3.1",                   # MoE (~685B total; ~37B active); needs Hopper/Ada (H100/RTX4090)
]

# Combine all
# MODELS = SMALL_FULL + LORA_FP16 + QLORA_4BIT + FRONTIER_ADAPT
# MODELS = SMALL_FULL + LORA_FP16 + QLORA_4BIT
# MODELS = FRONTIER_ADAPT
# MODELS = RESEARCH_SCALE
MODELS = ["openai/gpt-oss-20b", "meta-llama/Llama-2-13b-chat-hf"]

def pick_dtype_for_this_node() -> torch.dtype:
    """
    Hardware-aware dtype selector.
    - Use bf16 only on Ampere+ (SM >= 8.0).
    - Otherwise use fp16 (faster and real Tensor Core support on Turing/V100).
    - On CPU, fallback to fp16 to reduce memory.
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float16

def info_env():
    """Print basic runtime and cache environment info."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    has_token = bool(os.environ.get("HUGGINGFACE_HUB_TOKEN"))

    print("========== ENV INFO ==========")
    print(f"[INFO] Python version          : {platform.python_version()}")
    print(f"[INFO] PyTorch version         : {torch.__version__}")
    print(f"[INFO] HF_HOME                 : {hf_home}")
    print(f"[INFO] HF token detected       : {'YES' if has_token else 'NO'}")
    

    # GPU / CUDA info
    print(f"[INFO] torch.cuda.is_available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] bfloat16 supported      : {torch.cuda.is_bf16_supported()}")
        print(f"[INFO] cuda.device_count       : {torch.cuda.device_count()}")
        print(f"[INFO] cuda.device_name        : {torch.cuda.get_device_name(0)}")
        print(f"[INFO] cuda.device_memory      : {torch.cuda.get_device_properties(0).total_memory / (1024**3)}")
    dtype = pick_dtype_for_this_node()
    print(f"[INFO] chosen dtype            : {dtype}")
    print("================================")

def cache_tokenizer(repo_id: str):
    """
    Download tokenizer files into local Hugging Face cache.
    We don't keep the object; we just trigger the download.
    """
    tok = AutoTokenizer.from_pretrained(
        repo_id,
        use_fast=True,
        trust_remote_code=True,
    )
    del tok
    gc.collect()

def cache_model(repo_id: str, method: str):
    """
    Download model weights into cache via `from_pretrained`.
    - CPU only to avoid allocating GPU memory.
    - `low_cpu_mem_usage=True` streams weights in chunks to reduce RAM spikes.
    - Immediately free references after caching.
    """
    
    common_kwargs = dict(
        pretrained_model_name_or_path=repo_id,
        device_map="auto",                # enable HF tensor-parallel
        attn_implementation="eager",      # safest across envs
        low_cpu_mem_usage=True,           # stream weights on CPU side
        trust_remote_code=True,
    )

     # ---- branch by training method ----
    if method in {"full", "lora"}:
        print(f"[INFO] Prefetching {repo_id} ({method}, FP32 base weights).")
        model = AutoModelForCausalLM.from_pretrained(
            **common_kwargs,
            dtype=torch.float32,
        )

    elif method == "qlora":
        print(f"[INFO] Prefetching {repo_id} (QLoRA, 4-bit quant).")
        if repo_id.startswith("openai"):
            model = AutoModelForCausalLM.from_pretrained(
                **common_kwargs,
            )
        else:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=pick_dtype_for_this_node(),
            )
            model = AutoModelForCausalLM.from_pretrained(
                **common_kwargs,
                quantization_config=bnb_cfg,
            )

    else:
        raise ValueError(f"Unknown method={method}. Use 'full' | 'lora' | 'qlora'.")
    
    # Check model's dtype
    first_param = next(model.parameters())
    print(f"[INFO] Model dtype: {first_param.dtype}")

    # Immediately drop reference, so tensors become collectible
    del model
    # Aggressive cleanup to avoid Slurm OOM at exit
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def prefetch(repo_id: str, method: str):
    """
    Try to cache tokenizer + model for one repo.
    Return (ok: bool, err_msg: str|None)
    """
    print(f"[...] {repo_id}")
    try:
        # 1) Tokenizer first (small, fast)
        cache_tokenizer(repo_id)

        # 2) Model weights (large; should resume if interrupted)
        cache_model(repo_id, method)

        print(f"[OK]  Cached: {repo_id}\n")
        return True, None
    except Exception as e:
        err_msg = str(e)
        print(f"[FAIL] {repo_id}: {err_msg}\n")
        return False, err_msg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch tokenizer/model artifacts for a list of repos."
    )
    parser.add_argument(
        "-m", "--method",
        default="full",
        choices=["full", "lora", "qlora"],
        help="Loading strategy to use (default: %(default)s)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    method = args.method

    info_env()
    print("\n[INFO] Starting prefetch (tokenizer + model). Large models may take time.\n")

    successes, failures = [], []
    for repo in MODELS:
        ok, err = prefetch(repo, method)
        if ok:
            successes.append(repo)
        else:
            failures.append((repo, err))
    
    # Summary
    print("\n========== Summary ==========")
    # Print successes only
    print(f"Downloaded successfully: {len(successes)}")
    for s in successes:
        print(f"  - {s}")
    # Print failures only
    if failures:
        print(f"\nFailed: {len(failures)}")
        for rid, msg in failures:
            print(f"  - {rid}: {msg}")
    print("=============================\n")
