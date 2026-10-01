# Data Collection

Scripts used to collect the LLM GPU power traces used in this paper, released
for double-blind review. The repository is split by workload type:

- [`train/`](train/) — Slurm launch scripts and training harnesses that
  produced the training-side power traces, on HPC clusters.
- [`inference/`](inference/) — the serving-campaign pipeline (dataset
  preparation, GPU provisioning, vLLM launch, telemetry capture, upload) that
  produced the inference-side power traces, on rented GPU providers.

Each part has its own `README.md` with more detail. Both have been anonymized
for review: account names, personal notification endpoints, cluster hostnames,
Hugging Face/Docker Hub usernames, and dataset/storage identifiers have been
replaced with placeholders (e.g. `<slurm-account>`, `<hf-owner>`,
`<dockerhub-account>`). Set the indicated environment variables before running
any script.
