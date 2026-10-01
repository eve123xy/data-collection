#!/usr/bin/env python3
import os
import platform
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    hf_home = os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    has_token = bool(os.environ.get("HUGGINGFACE_HUB_TOKEN"))

    print("========== ENV INFO ==========")
    print(f"[INFO] Python version          : {platform.python_version()}")
    print(f"[INFO] PyTorch version         : {torch.__version__}")
    print(f"[INFO] HF_HOME                 : {hf_home}")
    print(f"[INFO] HF token detected       : {'YES' if has_token else 'NO'}")

    # GPU / CUDA info
    print(f"[INFO] torch.cuda.is_available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(
            f"[INFO] bfloat16 supported      : {torch.cuda.is_bf16_supported()}")
        print(f"[INFO] cuda.device_count       : {torch.cuda.device_count()}")
        print(
            f"[INFO] cuda.device_name        : {torch.cuda.get_device_name(0)}")
        print(
            f"[INFO] cuda.device_memory      : {torch.cuda.get_device_properties(0).total_memory / (1024**3)}")
    dtype = pick_dtype_for_this_node()
    print(f"[INFO] chosen dtype            : {dtype}")
    print("================================")


if __name__ == "__main__":
    info_env()

    # Read model name from command line
    repo_id = sys.argv[1]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        # dtype=pick_dtype_for_this_node(),
        dtype="auto",
        device_map="auto",            # spread across all GPUs / CPU
        attn_implementation="eager",  # force PyTorch attention
        low_cpu_mem_usage=True,       # stream weights; safer on login nodes
        trust_remote_code=True,    # uncomment if you want to forbid custom code
    )
    model.eval()

    # Simple prompt
    prompt = "Who are you? Answer in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    # Decode and print
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)
