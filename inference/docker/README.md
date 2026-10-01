# Campaign image

Built once, pinned by digest, used on both vast.ai and Lambda so every cell runs
one vLLM build and one DCGM version.

## Build and push

Run from the repository root, not this directory:

    docker build -f docker/Dockerfile -t <account>/llm-power:v1 .
    docker push <account>/llm-power:v1
    docker inspect --format='{{index .RepoDigests 0}}' <account>/llm-power:v1

Copy the printed `name@sha256:...` into `CAMPAIGN_IMAGE` in `scripts/pins.py`.

## Why a digest and not a tag

A tag is mutable. Two cells provisioned weeks apart under the same tag could run
different builds, which is an uncontrolled variable across 250 cells — the same
reason vLLM is pinned to `v0.28.0` rather than `latest`.

## What is deliberately NOT in the image

The prompt artifacts. They are ~150 MB, already verified by sha256 on arrival,
and versioned separately by HF revision — baking them in would couple a dataset
revision to an image rebuild.
