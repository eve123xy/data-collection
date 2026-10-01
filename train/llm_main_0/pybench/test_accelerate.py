import os
import socket
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from accelerate import Accelerator


class TinyNet(nn.Module):
    # simple 2-layer MLP for demo
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_dataloader(batch_size=4, num_batches=16, in_dim=16, num_classes=4):
    # make a tiny random dataset
    x = torch.randn(num_batches * batch_size, in_dim)
    y = torch.randint(0, num_classes, (num_batches * batch_size,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def param_checksum(model: nn.Module) -> float:
    """
    Lightweight signature of current model weights.
    Sum of all params (on CPU, in float32).
    If different ranks print different numbers, params have diverged.
    """
    with torch.no_grad():
        total = 0.0
        for p in model.parameters():
            total += p.detach().float().sum().item()
    return total


if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device

    # Training config
    lr = 1e-3
    accum_steps = 2  # every 2 steps: 1 no_sync step, then 1 sync step

    # Build model / optimizer / loss / data
    model = TinyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataloader = get_dataloader()

    # Prepare for distributed
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader)

    # Basic runtime metadata
    hostname = socket.gethostname()
    rank = accelerator.process_index         # global rank (0..world_size-1)
    local_rank = accelerator.local_process_index  # local index on this node
    world_size = accelerator.num_processes   # total processes
    device_str = str(device)

    # Print startup info from every process (not just rank 0)
    print(f"[START] host={hostname} rank={rank}/{world_size} "
          f"local_rank={local_rank} device={device_str} "
          f"accum_steps={accum_steps}", flush=True)

    model.train()

    # Training loop
    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Decide whether this step is "accumulation-only" (no_sync) or "sync/update"
        is_accum_phase = (step % accum_steps) != (accum_steps - 1)

        if is_accum_phase:
            # --------------------------------------
            # ACCUMULATION PHASE (no_sync region)
            # --------------------------------------
            with accelerator.no_sync(model):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                accelerator.backward(loss)

                # local-only optimizer step without gradient sync
                optimizer.step()
                optimizer.zero_grad()

            # We print from every rank here (on purpose)
            checksum_now = param_checksum(model)
            print(
                f"[ACCUM] step={step} host={hostname} "
                f"rank={rank}/{world_size} "
                f"loss={loss.item():.2f} "
                f"checksum={checksum_now:.3f} "
                f"(no_sync, local update applied)",
                flush=True,
            )

        else:
            # --------------------------------------
            # SYNC PHASE (DDP will all-reduce grads)
            # --------------------------------------
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            checksum_now = param_checksum(model)
            print(
                f"[SYNC ] step={step} host={hostname} "
                f"rank={rank}/{world_size} "
                f"loss={loss.item():.2f} "
                f"checksum={checksum_now:.3f} "
                f"(synced grads before step)",
                flush=True,
            )

    # finished loop
    print(
        f"[END  ] host={hostname} rank={rank}/{world_size} finished training loop.",
        flush=True,
    )

    # ---- graceful shutdown ----
    accelerator.wait_for_everyone()
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        # don't crash job on cleanup issues
        print(f"[CLEANUP] rank={rank} destroy_process_group exception: {e}", flush=True)
    accelerator.end_training()
