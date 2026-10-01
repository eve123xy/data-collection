#!/usr/bin/env bash
# Run a campaign command inside the pinned image on a Lambda box.
#
# Lambda is a bare VM, not vast's container-as-machine, so campaign code needs
# an explicit container. Three things this has to get right, each learned the
# expensive way:
#
#  1. The image's ENTRYPOINT is `vllm`, so a bare `docker run <img> python ...`
#     is parsed as vllm arguments and dies with a usage message.
#     -> --entrypoint bash
#
#  2. A fresh `--rm` container per command CANNOT run the cell flow. The
#     telemetry capture and the vLLM server are started in the background and
#     have to outlive the command that launched them, and a later step has to
#     reach the server on localhost:8000. With one container per command the
#     background processes die at command exit and the network namespace is
#     gone. -> ONE PERSISTENT container, exec into it, --network host.
#
#  3. Clock control does NOT belong in here. `nvidia-smi -lgc/-rgc` runs on the
#     HOST with sudo. Clocks are device-level and cross the container boundary
#     unchanged (verified 2026-09-07), so pinning outside is correct.
#
# usage: camp.sh "<command>"      run a command in the persistent container
#        camp.sh --reset          tear the container down and start clean
set -euo pipefail

IMAGE="<dockerhub-account>/llm-power-main-runs@sha256:dcd7cbef41a899660ec989c54c38c626e44791ace67f5424308ac95b6424e891"
NAME="camp-$(hostname | tr -cd 'a-z0-9' | tail -c 8)"

if [ "${1:-}" = "--reset" ]; then
  sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "[OK] $NAME removed"
  exit 0
fi

if ! sudo docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; then
  sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
  sudo docker run -d --name "$NAME" \
    --gpus all --ipc=host --network host --entrypoint bash \
    -v /opt/campaign:/opt/campaign -v /data:/data -w /opt/campaign \
    --env-file /opt/campaign/scripts/.env \
    "$IMAGE" -lc 'sleep infinity' >/dev/null
  echo "[camp] started persistent container $NAME" >&2
fi

exec sudo docker exec -w /opt/campaign "$NAME" bash -lc "$*"
