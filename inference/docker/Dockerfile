# Built once and pinned BY DIGEST in scripts/pins.py.
#
# This image exists because OPERATIONAL_LEARNINGS section 2.11 records the
# archive re-downloading 911 MB of DCGM on every provision, which became a
# 2h43m billing incident. A per-provision install is simultaneously a standing
# time cost, a standing money cost, and a new network failure surface on every
# run.
FROM vllm/vllm-openai:v0.28.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends datacenter-gpu-manager \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "httpx>=0.27" "huggingface-hub>=0.24" \
        google-api-python-client google-auth matplotlib

# The base image ships python3 with no `python` alias, but every command the
# agent is handed starts with `python`. Without this the skill file's steps fail
# on the first line.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# NOTE: scripts/ is deliberately NOT copied in. On 2026-09-02 `COPY scripts/`
# with no .dockerignore baked scripts/.env and the Google service-account key
# into a published layer, and the same image shipped scripts that predated the
# stage-folder reorg. Both failures had one root cause: baking fast-moving code
# into a slow-moving image. scripts/ is now delivered at run time by
#     python scripts/cells/provision.py push-creds --instance <id>
# which rsyncs the current tree and scp's the secrets. This image carries only
# the slow, heavy dependencies, so a script edit never needs a rebuild.
RUN mkdir -p /opt/campaign/scripts
WORKDIR /opt/campaign
# Each stage folder is on the path so a script can be run directly and still
# import its siblings; scripts/ itself carries the shared pins and common.
ENV PYTHONPATH=/opt/campaign/scripts:/opt/campaign/scripts/dataset:/opt/campaign/scripts/workload:/opt/campaign/scripts/telemetry:/opt/campaign/scripts/cells:/opt/campaign/scripts/upload
