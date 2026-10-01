"""
Distributed burn-in training loop for power profiling.

Key features:
- Hugging Face Accelerate for distributed execution
- full / LoRA / QLoRA (with optional PEFT + bitsandbytes)
- sync_mode: async | sync | local
- Optional idle sleeps for power profiling
- Checkpointing (two independent knobs):
  1) model_save: full | lora
  2) include_optimizer: 0/1  (also saves scheduler state)
- Retention: keep only last K checkpoints (delete older in background)

Checkpoint outputs:
- model_save=full:
    step_XXXXXX.full.pt
- model_save=lora:
    step_XXXXXX.lora/              (PEFT adapter dir)
      - adapter_model.*            (PEFT artifact)
      - adapter_config.json        (PEFT artifact)
      - meta.json                  (our metadata)
      - train_state.pt             (optional: optimizer/scheduler/global_step)

Notes:
- For large models, full checkpoints can be huge and CPU-memory heavy.
- LoRA checkpoints are typically small and much faster.
"""

import argparse
import json
import os
import platform
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import cycle
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator

# Optional PEFT imports (for LoRA / QLoRA)
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft import PeftModel  # type: ignore
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
    PeftModel = None  # type: ignore


# =================================================
# Small helpers
# =================================================

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(stage: str, state: str, msg: str = "") -> None:
    print(f"[{_ts()}][{stage.upper()}][{state.upper()}] {msg}", flush=True)


def info_env() -> None:
    print("========== ENV INFO ==========")
    print(f"[INFO] Python version          : {platform.python_version()}")
    print(f"[INFO] PyTorch version         : {torch.__version__}")
    print(f"[INFO] torch.cuda.is_available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] cuda.device_count       : {torch.cuda.device_count()}")
        print(f"[INFO] cuda.device_name        : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] cuda.total_memory(GB)   : {props.total_memory / (1024**3):.1f}")
        print(f"[INFO] bf16 supported          : {torch.cuda.is_bf16_supported()}")
    print("================================")


def normpath(p: str) -> str:
    # Expand "~" and make it absolute.
    return os.path.abspath(os.path.expanduser(p))


def is_peft_model(m) -> bool:
    if not PEFT_AVAILABLE:
        return False
    if PeftModel is not None and isinstance(m, PeftModel):
        return True
    # Heuristic fallback for PEFT wrappers.
    return hasattr(m, "peft_config") and hasattr(m, "save_pretrained")


def atomic_replace_dir(tmp_dir: str, final_dir: str) -> None:
    # Directory "atomic replace": remove final then rename tmp -> final.
    shutil.rmtree(final_dir, ignore_errors=True)
    os.replace(tmp_dir, final_dir)


def atomic_replace_file(tmp_path: str, final_path: str) -> None:
    os.replace(tmp_path, final_path)


def async_delete_items(items: List[Tuple[str, str]]) -> None:
    # Best-effort cleanup; never fail training because cleanup fails.
    for path, kind in items:
        try:
            if kind == "file":
                os.remove(path)
            else:
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


# =================================================
# Model loading utilities
# =================================================

def get_device_map():
    # TP: single process sees multiple GPUs -> shard with "auto"
    # DP: one process per GPU -> pin to local_rank
    mode = os.getenv("PARALLEL_MODE", "tp").lower()
    if mode == "tp":
        return "auto", None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return {"": local_rank}, local_rank


def pick_dtype_for_this_node() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float16


def apply_lora(model, method: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    # LoRA rank controls trainable params and adapter ckpt size.
    if method == "full":
        print("[INFO] method=full, training all params (no LoRA).", flush=True)
        return model

    if not PEFT_AVAILABLE:
        print("[WARN] PEFT not available; falling back to full fine-tune.", flush=True)
        return model

    print(
        f"[INFO] method={method}, applying LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}).",
        flush=True,
    )

    if method == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def load_model_and_tokenizer(
    repo_id: str,
    method: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device_map, _ = get_device_map()
    dtype = pick_dtype_for_this_node()

    common_kwargs = dict(
        pretrained_model_name_or_path=repo_id,
        device_map=device_map,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"[INFO] loading {repo_id} with method={method}", flush=True)

    if method in {"full", "lora"}:
        model = AutoModelForCausalLM.from_pretrained(**common_kwargs, dtype=dtype)
    elif method == "qlora":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(**common_kwargs, quantization_config=bnb_cfg)
    else:
        raise ValueError("method must be one of: full | lora | qlora")

    model = apply_lora(model, method, lora_r, lora_alpha, lora_dropout)

    first_param = next(model.parameters())
    print(f"[INFO] Model dtype: {first_param.dtype}", flush=True)
    return model, tokenizer


# =================================================
# Dataset / DataLoader
# =================================================

class InstructionDataset(Dataset):
    def __init__(self, tokenizer, seq_len: int, split="train", size_limit=128):
        raw = load_dataset(os.environ.get("TRAIN_DATASET", "<hf-dataset-id>"))[split]
        raw = raw.select(range(min(size_limit, len(raw))))

        self.samples = []
        for ex in raw:
            instr, inp, out = ex["instruction"], ex["input"], ex["output"]
            if inp.strip():
                text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"

            tok = tokenizer(
                text,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.samples.append((input_ids, attention_mask, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_loader(tokenizer, seq_len: int, batch_size: int, dataset_size_limit: int, pin_memory: bool):
    ds = InstructionDataset(tokenizer, seq_len=seq_len, size_limit=dataset_size_limit)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_memory,
    )


# =================================================
# Checkpoint policy
# =================================================

@dataclass(frozen=True)
class CkptPolicy:
    ckpt_dir: str
    ckpt_every: int
    model_save: str              # "full" | "lora"
    include_optimizer: bool
    keep_last: int


def resolve_ckpt_root(policy: CkptPolicy) -> str:
    root = normpath(policy.ckpt_dir)
    os.makedirs(root, exist_ok=True)
    return root


def retention_add(
    accelerator: Accelerator,
    history: List[Tuple[str, str]],
    item: Tuple[str, str],
    keep_last: int,
) -> None:
    # Keep only on main process.
    if not accelerator.is_main_process:
        return

    history.append(item)
    if keep_last <= 0:
        return

    to_delete = []
    while len(history) > keep_last:
        to_delete.append(history.pop(0))

    if to_delete:
        threading.Thread(target=async_delete_items, args=(to_delete,), daemon=True).start()


# =================================================
# Training loop
# =================================================

def run_burn(
    accelerator: Accelerator,
    model,
    loader,
    total_steps: int,
    accum_steps: int,
    lr: float,
    warmup_steps: int,
    print_every: int,
    seq_len: int,
    batch_size: int,
    sleep_every: int,
    sleep_sec: float,
    sync_mode: str,
    policy: CkptPolicy,
):
    """
    model_save:
      - full: save full state dict (big)
      - lora: save PEFT adapter dir (small)

    include_optimizer:
      - if true, also save optimizer + scheduler states for resuming training
    """

    model.train()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(1.0, s / max(1, warmup_steps))
    )

    ddp_model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    data_iter = cycle(loader)

    running_loss = 0.0
    wall_start = time.time()

    ckpt_root = resolve_ckpt_root(policy)

    # Track saved checkpoints for retention.
    # Each entry is (path, "file"|"dir").
    ckpt_history: List[Tuple[str, str]] = []

    if accelerator.is_main_process:
        print(f"[INFO] run_burn start sync_mode={sync_mode}", flush=True)
        print(f"[INFO] gradient_accumulation steps = {accum_steps}", flush=True)
        print(f"[INFO] ckpt_dir = {ckpt_root}", flush=True)
        print(f"[INFO] ckpt_every = {policy.ckpt_every}", flush=True)
        print(f"[INFO] model_save = {policy.model_save}", flush=True)
        print(f"[INFO] include_optimizer = {int(policy.include_optimizer)}", flush=True)
        print(f"[INFO] keep_last = {policy.keep_last}", flush=True)

    def log_and_maybe_sleep(step_idx: int, effective_accum: int):
        nonlocal running_loss, wall_start

        if accelerator.is_main_process and print_every > 0 and (step_idx % print_every == 0):
            elapsed = time.time() - wall_start
            avg_loss = running_loss / print_every
            tokens = seq_len * batch_size * effective_accum * print_every
            tput = tokens / max(elapsed, 1e-6)
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[step {step_idx:06d}] loss={avg_loss:.4f} lr={lr_now:.2e} tokens/s={tput:.1f}",
                flush=True,
            )
            running_loss = 0.0
            wall_start = time.time()

        if sleep_every > 0 and (step_idx % sleep_every == 0):
            accelerator.wait_for_everyone()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            log("SLEEP", "begin", f"sleeping {sleep_sec}s at step {step_idx}")
            t0 = time.perf_counter()
            time.sleep(sleep_sec)
            log("SLEEP", "end", f"step={step_idx} | actual={time.perf_counter() - t0:.3f}s")

    def save_checkpoint(global_step: int, last_loss_value: Optional[float]):
        # Save on both full and LoRA runs.
        if policy.ckpt_every <= 0:
            return
        if global_step % policy.ckpt_every != 0:
            return

        step_name = f"step_{global_step:06d}"
        t0 = time.perf_counter()

        if accelerator.is_main_process:
            log(
                "CKPT",
                "begin",
                f"step={global_step} | model_save={policy.model_save} | opt={int(policy.include_optimizer)} | dir={ckpt_root}",
            )

        # Common metadata.
        meta = {
            "global_step": int(global_step),
            "loss": float(last_loss_value) if last_loss_value is not None else None,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "accum_steps": int(accum_steps),
            "sync_mode": str(sync_mode),
            "model_save": policy.model_save,
            "include_optimizer": bool(policy.include_optimizer),
        }

        if policy.model_save == "lora":
            # LoRA checkpoints require PEFT adapters.
            if not is_peft_model(ddp_model):
                if accelerator.is_main_process:
                    log("CKPT", "warn", "model_save=lora but model is not PEFT; skipping")
                return

            tmp_dir = os.path.join(ckpt_root, step_name + ".lora.tmp")
            final_dir = os.path.join(ckpt_root, step_name + ".lora")

            if accelerator.is_main_process:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                os.makedirs(tmp_dir, exist_ok=True)

                # Save adapter weights/config.
                ddp_model.save_pretrained(tmp_dir)

                # Save metadata.
                with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)

                # Optional training state for resuming (still small for LoRA).
                if policy.include_optimizer:
                    train_state = {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "meta": meta,
                    }
                    torch.save(train_state, os.path.join(tmp_dir, "train_state.pt"))

                atomic_replace_dir(tmp_dir, final_dir)

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                dt = time.perf_counter() - t0
                log("CKPT", "end", f"saved={final_dir} | duration={dt:.3f}s")
                retention_add(accelerator, ckpt_history, (final_dir, "dir"), policy.keep_last)

            return

        if policy.model_save != "full":
            raise ValueError("model_save must be 'full' or 'lora'")

        # Full checkpoint is a single .pt file. Use tmp then atomic rename.
        tmp_path = os.path.join(ckpt_root, step_name + ".full.pt.tmp")
        final_path = os.path.join(ckpt_root, step_name + ".full.pt")

        # This can be CPU-memory heavy for large models.
        model_state = accelerator.get_state_dict(ddp_model)

        state = {
            "model": model_state,
            "meta": meta,
        }
        if policy.include_optimizer:
            state["optimizer"] = optimizer.state_dict()
            state["scheduler"] = scheduler.state_dict()

        accelerator.save(state, tmp_path)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            atomic_replace_file(tmp_path, final_path)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            dt = time.perf_counter() - t0
            log("CKPT", "end", f"saved={final_path} | duration={dt:.3f}s")
            retention_add(accelerator, ckpt_history, (final_path, "file"), policy.keep_last)

    # -------------------------
    # Training modes
    # -------------------------

    if sync_mode == "async":
        global_step = 0  # optimizer steps

        for _micro_step in range(1, total_steps + 1):
            with accelerator.accumulate(ddp_model):
                input_ids, attn, labels = next(data_iter)
                out = ddp_model(input_ids=input_ids, attention_mask=attn, labels=labels)
                accelerator.backward(out.loss)

                if accelerator.is_main_process:
                    running_loss += float(out.loss.item())

                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    global_step += 1
                    save_checkpoint(global_step, float(out.loss.item()))

                    if accelerator.is_main_process:
                        log_and_maybe_sleep(
                            step_idx=global_step,
                            effective_accum=accelerator.gradient_accumulation_steps,
                        )

                    accelerator.wait_for_everyone()

    elif sync_mode == "sync":
        accum_count = 0
        global_step = 0

        while global_step < total_steps:
            input_ids, attn, labels = next(data_iter)
            out = ddp_model(input_ids=input_ids, attention_mask=attn, labels=labels)

            loss = out.loss / accum_steps
            accelerator.backward(loss)

            if accelerator.is_main_process:
                running_loss += float(loss.item())

            accum_count += 1
            if accum_count == accum_steps:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in ddp_model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                accum_count = 0

                save_checkpoint(global_step, float(out.loss.item()))

                if accelerator.is_main_process:
                    log_and_maybe_sleep(
                        step_idx=global_step,
                        effective_accum=accelerator.gradient_accumulation_steps,
                    )

                accelerator.wait_for_everyone()

    elif sync_mode == "local":
        accum_count = 0
        global_step = 0

        while global_step < total_steps:
            input_ids, attn, labels = next(data_iter)
            out = ddp_model(input_ids=input_ids, attention_mask=attn, labels=labels)

            loss = out.loss / accum_steps
            with accelerator.no_sync(ddp_model):
                loss.backward()

            if accelerator.is_main_process:
                running_loss += float(loss.item())

            accum_count += 1
            if accum_count == accum_steps:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in ddp_model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                accum_count = 0

                save_checkpoint(global_step, float(out.loss.item()))

                if accelerator.is_main_process:
                    log_and_maybe_sleep(step_idx=global_step, effective_accum=accum_steps)

                accelerator.wait_for_everyone()
    else:
        raise ValueError("sync_mode must be async | sync | local")


# =================================================
# Argparse
# =================================================

def parse_args():
    p = argparse.ArgumentParser()

    # Model / method
    p.add_argument("--model", type=str, required=True,
                   help="Hugging Face model ID, e.g. meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--method", type=str, default="full",
                   choices=["full", "lora", "qlora"],
                   help="Training mode: full | lora | qlora")
    p.add_argument("--sync_mode", type=str, default="async",
                   choices=["async", "sync", "local"],
                   help="async=delayed grad sync; sync=sync every microbatch; local=no cross-rank sync")

    # Data / shape
    p.add_argument("--seq_len", type=int, default=1024, help="Tokenized sequence length")
    p.add_argument("--batch_size", type=int, default=1, help="Microbatch size per process")
    p.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")

    # LoRA hyperparameters
    p.add_argument("--lora_r", type=int, default=64,
                   help="LoRA rank (higher r = more trainable params and larger adapter ckpt)")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Schedule
    p.add_argument("--lr", type=float, default=1e-6, help="Base learning rate")
    p.add_argument("--warmup_steps", type=int, default=10, help="Linear warmup steps")
    p.add_argument("--total_steps", type=int, default=999999, help="Number of optimizer steps to run")
    p.add_argument("--dataset_size", type=int, default=52000, help="How many instruction-tuning samples to load")
    p.add_argument("--pin_memory", action="store_true",
                   help="Use pinned CPU memory for faster H2D copies")

    # Sleep
    p.add_argument("--print_every", type=int, default=1, help="Print interval in optimizer steps")
    p.add_argument("--sleep_every", type=int, default=0, help="Sleep every N optimizer steps (0 disables)")
    p.add_argument("--sleep_sec", type=float, default=0.0, help="Seconds to sleep when triggered")
    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default="llm_example/ckpts",
                   help="Checkpoint root directory (expanded + absolutized)")
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="Save checkpoint every N optimizer steps (0 disables)")
    p.add_argument("--model_save", type=str, default="lora",
                   choices=["full", "lora"],
                   help="Which weights to save: full model weights or LoRA adapters")
    p.add_argument("--include_optimizer", type=int, default=0,
                   help="1 saves optimizer+scheduler state for resuming; 0 saves weights only")
    p.add_argument("--keep_last", type=int, default=1,
                   help="Keep only last K checkpoints (0 keeps all)")

    return p.parse_args()


# =================================================
# Main
# =================================================

def main():
    # info_env()
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(
        repo_id=args.model,
        method=args.method,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    loader = make_loader(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dataset_size_limit=args.dataset_size,
        pin_memory=args.pin_memory,
    )

    accelerator = Accelerator(gradient_accumulation_steps=args.accum_steps)

    policy = CkptPolicy(
        ckpt_dir=args.ckpt_dir,
        ckpt_every=args.ckpt_every,
        model_save=args.model_save,
        include_optimizer=bool(args.include_optimizer),
        keep_last=args.keep_last,
    )

    run_burn(
        accelerator=accelerator,
        model=model,
        loader=loader,
        total_steps=args.total_steps,
        accum_steps=args.accum_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        print_every=args.print_every,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        sleep_every=args.sleep_every,
        sleep_sec=args.sleep_sec,
        sync_mode=args.sync_mode,
        policy=policy,
    )


if __name__ == "__main__":
    main()
