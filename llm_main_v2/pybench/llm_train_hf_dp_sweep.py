#!/usr/bin/env python3
"""
Single-group DP workload sweep for GPU power profiling.

Each candidate configuration runs exactly one optimizer step made of
`accum_steps` microsteps (default: 4). Candidates are tested from small
to large. If CUDA OOM occurs, this process exits non-zero immediately;
the outer Slurm script catches that and moves to the next group.

No checkpoints are written.

Optional PyTorch Profiler support records one selected microstep per
configuration (rank 0 by default), exports a Chrome/Perfetto trace, and
writes an operator summary. Attention and MLP forward regions are also
annotated with record_function ranges so they are easy to locate.
"""

import argparse
import csv
import os
import platform
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def ts_ms() -> str:
    # Match nvidia-smi's millisecond-style wall-clock timestamp closely.
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def log(msg: str) -> None:
    print(f"[{ts_ms()}] {msg}", flush=True)


def append_event(path: str, row: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path)
    fields = [
        "timestamp", "event", "group", "sweep_type", "batch_size",
        "seq_len", "config_index", "status"
    ]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})
        f.flush()


def pick_dtype_for_this_node() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float16


def infer_lora_target_modules(model) -> List[str]:
    preferred = ["q_proj", "v_proj"]
    fallback = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]

    found = set()
    for name, _module in model.named_modules():
        short = name.split(".")[-1]
        if short in preferred:
            found.add(short)
    if found:
        return sorted(found)

    for name, _module in model.named_modules():
        short = name.split(".")[-1]
        if short in fallback:
            found.add(short)
    if found:
        return sorted(found)

    raise ValueError("Could not infer LoRA target_modules from model structure.")


def apply_lora(model, method: str, r: int, alpha: int, dropout: float):
    if method == "full":
        return model
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT is required for LoRA/QLoRA but is not available.")

    if method == "qlora":
        model = prepare_model_for_kbit_training(model)

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    try:
        return get_peft_model(model, cfg)
    except ValueError as e:
        msg = str(e)
        if "target_modules" not in msg and "target_parameters" not in msg:
            raise
        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=infer_lora_target_modules(model),
        )
        return get_peft_model(model, cfg)


def load_model_and_tokenizer(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = pick_dtype_for_this_node()
    common = dict(
        pretrained_model_name_or_path=args.model,
        device_map={"": local_rank},
        # Keep the original experiment behavior. Eager attention is also useful
        # here because the profiler exposes the decomposed attention operations.
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if args.method in {"full", "lora"}:
        model = AutoModelForCausalLM.from_pretrained(**common, dtype=dtype)
    elif args.method == "qlora":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            **common, quantization_config=bnb_cfg
        )
    else:
        raise ValueError("method must be full | lora | qlora")

    model = apply_lora(
        model, args.method, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    return model, tokenizer, dtype


class BatchPool:
    """Small pretokenized CPU pool reused across all configs in one group."""

    def __init__(self, tokenizer, max_seq_len: int, pool_size: int, pin_memory: bool):
        raw = load_dataset(os.environ.get("TRAIN_DATASET", "<hf-dataset-id>"), split="train")
        raw = raw.select(range(min(pool_size, len(raw))))

        texts = []
        for ex in raw:
            instr, inp, out = ex["instruction"], ex["input"], ex["output"]
            if inp.strip():
                text = (
                    f"### Instruction:\n{instr}\n\n"
                    f"### Input:\n{inp}\n\n"
                    f"### Response:\n{out}"
                )
            else:
                text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
            texts.append(text)

        tok = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = tok["input_ids"]
        self.attention_mask = tok["attention_mask"]
        self.labels = self.input_ids.clone()
        self.labels[self.attention_mask == 0] = -100

        if pin_memory:
            self.input_ids = self.input_ids.pin_memory()
            self.attention_mask = self.attention_mask.pin_memory()
            self.labels = self.labels.pin_memory()

        self.n = self.input_ids.size(0)
        self.ptr = 0

    def next_batch(self, batch_size: int, seq_len: int, device: torch.device):
        # Wrap around the small pool if necessary.
        idx = [(self.ptr + i) % self.n for i in range(batch_size)]
        self.ptr = (self.ptr + batch_size) % self.n
        idx = torch.tensor(idx, dtype=torch.long)

        input_ids = self.input_ids.index_select(0, idx)[:, :seq_len].contiguous()
        attn = self.attention_mask.index_select(0, idx)[:, :seq_len].contiguous()
        labels = self.labels.index_select(0, idx)[:, :seq_len].contiguous()

        non_blocking = self.input_ids.is_pinned()
        return (
            input_ids.to(device, non_blocking=non_blocking),
            attn.to(device, non_blocking=non_blocking),
            labels.to(device, non_blocking=non_blocking),
        )


def parse_values(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("--values cannot be empty")
    return vals


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="full", choices=["full", "lora", "qlora"])

    p.add_argument("--group", required=True)
    p.add_argument("--sweep_type", required=True, choices=["fixed_seq", "fixed_batch"])
    p.add_argument("--fixed_value", type=int, required=True)
    p.add_argument("--values", required=True, help="comma-separated ascending candidate values")

    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--pool_size", type=int, default=512)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--event_log", type=str, default="")

    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Profiler is optional so the same code can still be used for a clean power run.
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profiler_dir", type=str, default="")
    p.add_argument(
        "--profile_microstep",
        type=int,
        default=1,
        help="1-based microstep index to profile in each config; default=1",
    )
    p.add_argument(
        "--profile_all_ranks",
        action="store_true",
        help="Profile every DP rank. Default is rank 0 only to reduce overhead/output size.",
    )
    return p.parse_args()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def should_profile_this_rank(args, accelerator: Accelerator) -> bool:
    if not args.profile:
        return False
    return args.profile_all_ranks or accelerator.is_main_process


def profiler_activities():
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    return acts


@contextmanager
def annotate_attention_and_mlp(model):
    """
    Add profiler ranges around top-level attention and MLP module forwards.

    The labels are deliberately shared across layers (ATTENTION_FORWARD / MLP_FORWARD)
    so key_averages() aggregates them, while the Chrome trace still shows every call.
    """
    handles = []
    active = {}

    def make_pre(label):
        def _pre(module, _inputs):
            ctx = torch.profiler.record_function(label)
            ctx.__enter__()
            active[id(module)] = ctx
        return _pre

    def _post(module, _inputs, _output):
        ctx = active.pop(id(module), None)
        if ctx is not None:
            ctx.__exit__(None, None, None)

    for name, module in model.named_modules():
        lname = name.lower()
        cls = module.__class__.__name__.lower()

        is_attn = (
            lname.endswith("self_attn")
            or lname.endswith(".attn")
            or cls.endswith("attention")
        )
        is_mlp = lname.endswith(".mlp") or cls.endswith("mlp")

        if is_attn:
            handles.append(module.register_forward_pre_hook(make_pre("ATTENTION_FORWARD")))
            handles.append(module.register_forward_hook(_post))
        elif is_mlp:
            handles.append(module.register_forward_pre_hook(make_pre("MLP_FORWARD")))
            handles.append(module.register_forward_hook(_post))

    try:
        yield
    finally:
        # Close any range left open because a forward raised an exception.
        for ctx in list(active.values()):
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        active.clear()
        for h in handles:
            h.remove()


def save_profiler_outputs(prof, out_dir: str, rank: int):
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, f"rank{rank}_trace.json")
    summary_path = os.path.join(out_dir, f"rank{rank}_summary.txt")

    prof.export_chrome_trace(trace_path)

    # self_cuda_time_total is the documented CUDA sort key. Fall back to CPU if
    # the build/device does not expose it.
    try:
        table = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=200
        )
    except Exception:
        table = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=200
        )

    with open(summary_path, "w") as f:
        f.write(table)
        f.write("\n")

    return trace_path, summary_path


def main():
    args = parse_args()
    values = parse_values(args.values)

    if args.accum_steps != 4:
        raise ValueError(
            "This sweep is intended for exactly 4 microsteps per optimizer step; "
            "use --accum_steps 4."
        )
    if not 1 <= args.profile_microstep <= args.accum_steps:
        raise ValueError("--profile_microstep must be between 1 and accum_steps")
    if args.profile and not args.profiler_dir:
        raise ValueError("--profiler_dir is required when --profile is enabled")

    if args.sweep_type == "fixed_seq":
        max_seq_len = args.fixed_value
        max_batch = max(values)
    else:
        max_seq_len = max(values)
        max_batch = args.fixed_value

    # We only need enough unique examples for the largest candidate's 4 microsteps.
    required_pool = max_batch * args.accum_steps
    pool_size = max(args.pool_size, required_pool)

    # Some HF/PEFT models can have trainable parameters that are not used in
    # every text-only forward pass. DDP must explicitly detect such parameters;
    # otherwise it raises "Expected to have finished reduction..." on the
    # next forward. This is especially important for LoRA/QLoRA sweeps.
    """ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )"""
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_steps
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    profile_rank = should_profile_this_rank(args, accelerator)

    if accelerator.is_main_process:
        log(
            f"GROUP_START group={args.group} type={args.sweep_type} "
            f"fixed={args.fixed_value} values={values} accum={args.accum_steps}"
        )
        append_event(args.event_log, {
            "timestamp": ts_ms(), "event": "GROUP_START", "group": args.group,
            "sweep_type": args.sweep_type, "status": "start"
        })
        print(
            f"[INFO] Python={platform.python_version()} PyTorch={torch.__version__} "
            f"world_size={world_size}",
            flush=True,
        )
        print(
            f"[INFO] profiler={'on' if args.profile else 'off'} "
            f"profile_microstep={args.profile_microstep} "
            f"profile_all_ranks={int(args.profile_all_ranks)}",
            flush=True,
        )

    if accelerator.is_main_process:
        log(f"[INFO] loading {args.model} with method={args.method}")

    model, tokenizer, model_dtype = load_model_and_tokenizer(args)
    model.train()

    total_params, trainable_params = count_parameters(model)
    if accelerator.is_main_process:
        if args.method == "full":
            log("[INFO] method=full, training all parameters.")
        else:
            log(
                f"[INFO] method={args.method}, trainable parameters="
                f"{trainable_params:,}/{total_params:,} "
                f"({100.0 * trainable_params / max(1, total_params):.4f}%)."
            )
        log(f"[INFO] Model dtype: {model_dtype}")
        log("[INFO] attention implementation: eager")
        log(f"[INFO] gradient_accumulation steps = {args.accum_steps}")
        log(f"[INFO] data pool size = {pool_size}")

    pool = BatchPool(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        pool_size=pool_size,
        pin_memory=args.pin_memory,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, s / max(1, args.warmup_steps)),
    )

    ddp_model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    optimizer.zero_grad(set_to_none=True)

    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for i, v in enumerate(values):
        if args.sweep_type == "fixed_seq":
            seq_len = args.fixed_value
            batch_size = v
        else:
            batch_size = args.fixed_value
            seq_len = v

        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        config_t0 = time.perf_counter()

        if accelerator.is_main_process:
            log(
                f"CONFIG_START group={args.group} idx={i} "
                f"batch_size={batch_size} seq_len={seq_len}"
            )
            append_event(args.event_log, {
                "timestamp": ts_ms(), "event": "CONFIG_START", "group": args.group,
                "sweep_type": args.sweep_type, "batch_size": batch_size,
                "seq_len": seq_len, "config_index": i, "status": "start"
            })

        try:
            last_loss = None
            optimizer.zero_grad(set_to_none=True)
            profiler_saved = ""

            # Exactly 4 microsteps -> exactly one optimizer update.
            for micro_idx in range(args.accum_steps):
                is_selected_profile_step = (
                    profile_rank and (micro_idx + 1 == args.profile_microstep)
                )

                prof = None
                profile_out_dir = None
                if is_selected_profile_step:
                    profile_out_dir = os.path.join(
                        args.profiler_dir,
                        args.group,
                        f"cfg{i:02d}_bs{batch_size}_seq{seq_len}",
                    )
                    os.makedirs(profile_out_dir, exist_ok=True)
                    prof = torch.profiler.profile(
                        activities=profiler_activities(),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=False,
                        with_flops=True,
                    )

                profile_ctx = prof if prof is not None else nullcontext()
                module_ctx = annotate_attention_and_mlp(ddp_model) if prof is not None else nullcontext()

                if prof is not None:
                    log(
                        f"PROFILER_START group={args.group} idx={i} rank={rank} "
                        f"microstep={micro_idx + 1} batch_size={batch_size} seq_len={seq_len}"
                    )

                with profile_ctx:
                    with module_ctx:
                        with accelerator.accumulate(ddp_model):
                            rf = torch.profiler.record_function if prof is not None else None

                            with (rf("H2D_BATCH") if rf else nullcontext()):
                                input_ids, attn, labels = pool.next_batch(
                                    batch_size=batch_size,
                                    seq_len=seq_len,
                                    device=accelerator.device,
                                )

                            with (rf("MODEL_FORWARD") if rf else nullcontext()):
                                out = ddp_model(
                                    input_ids=input_ids,
                                    attention_mask=attn,
                                    labels=labels,
                                )
                            last_loss = out.loss

                            with (rf("MODEL_BACKWARD") if rf else nullcontext()):
                                accelerator.backward(out.loss)

                            if accelerator.sync_gradients:
                                with (rf("GRAD_CLIP") if rf else nullcontext()):
                                    torch.nn.utils.clip_grad_norm_(
                                        [p for p in ddp_model.parameters() if p.requires_grad],
                                        1.0,
                                    )
                                with (rf("OPTIMIZER_STEP") if rf else nullcontext()):
                                    optimizer.step()
                                    optimizer.zero_grad(set_to_none=True)
                                    scheduler.step()

                if prof is not None:
                    trace_path, summary_path = save_profiler_outputs(
                        prof, profile_out_dir, rank
                    )
                    profiler_saved = trace_path
                    log(
                        f"PROFILER_END group={args.group} idx={i} rank={rank} "
                        f"trace={trace_path} summary={summary_path}"
                    )

            accelerator.wait_for_everyone()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - config_t0

            if accelerator.is_main_process:
                loss_value = (
                    float(last_loss.detach().item())
                    if last_loss is not None else float("nan")
                )
                current_lr = optimizer.param_groups[0]["lr"]
                global_tokens = batch_size * seq_len * args.accum_steps * world_size
                tokens_per_s = global_tokens / elapsed if elapsed > 0 else float("nan")

                if torch.cuda.is_available():
                    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                else:
                    peak_alloc_gb = 0.0
                    peak_reserved_gb = 0.0

                log(
                    f"CONFIG_END group={args.group} idx={i} batch_size={batch_size} "
                    f"seq_len={seq_len} loss={loss_value:.6f} lr={current_lr:.3e} "
                    f"time={elapsed:.3f}s tokens/s={tokens_per_s:.1f} "
                    f"peak_alloc={peak_alloc_gb:.2f}GiB "
                    f"peak_reserved={peak_reserved_gb:.2f}GiB status=success"
                )
                print(
                    f"[config {i:02d}] batch={batch_size} seq={seq_len} "
                    f"loss={loss_value:.6f} lr={current_lr:.3e} "
                    f"time={elapsed:.3f}s tokens/s={tokens_per_s:.1f}",
                    flush=True,
                )
                if args.profile:
                    print(
                        f"[profiler] config={i:02d} batch={batch_size} seq={seq_len} "
                        f"rank0_trace={profiler_saved or 'see rank-specific profiler directory'}",
                        flush=True,
                    )

                append_event(args.event_log, {
                    "timestamp": ts_ms(), "event": "CONFIG_END", "group": args.group,
                    "sweep_type": args.sweep_type, "batch_size": batch_size,
                    "seq_len": seq_len, "config_index": i, "status": "success"
                })

        except torch.cuda.OutOfMemoryError:
            # Do NOT try to recover this distributed process. Record and re-raise;
            # the outer Slurm shell will move on to the next group with a clean launch.
            if accelerator.is_main_process:
                log(
                    f"CONFIG_OOM group={args.group} idx={i} "
                    f"batch_size={batch_size} seq_len={seq_len}"
                )
                append_event(args.event_log, {
                    "timestamp": ts_ms(), "event": "CONFIG_OOM", "group": args.group,
                    "sweep_type": args.sweep_type, "batch_size": batch_size,
                    "seq_len": seq_len, "config_index": i, "status": "oom"
                })
            raise

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        log(f"GROUP_END group={args.group} status=success")
        append_event(args.event_log, {
            "timestamp": ts_ms(), "event": "GROUP_END", "group": args.group,
            "sweep_type": args.sweep_type, "status": "success"
        })


if __name__ == "__main__":
    main()
