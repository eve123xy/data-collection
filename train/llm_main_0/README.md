# Data Collection

A lightweight framework for running LLM training and inference workloads on Slurm-based HPC clusters.

## Structure

- `env.sh`  
  Environment setup script.

- `launch/`  
  Entry scripts for submitting jobs (`sbatch` / `srun`), grouped by workload type (train / infer) and backend.

- `templates/`  
  Slurm job templates (resource config + unified logging + optional tracing), invoked by scripts in `launch/`.

- `pybench/`  
  Python workload implementations (training / inference / small tests).

- `logs_train/`  
  Training run outputs (e.g., `stdout.log`, `stderr.log`, `train_runtime.log`, `power_trace.csv`).

- `logs_infer/`  
  Inference run outputs (e.g., `stdout.log`, `stderr.log`, `power_trace.csv`).


## Anonymization note

This repository is released for double-blind review. Slurm account names, notification e-mails, cluster hostnames, user names and Hugging Face dataset identifiers have been replaced by placeholders (`<slurm-account>`, `<user>`, `<cluster>`, `<hf-dataset-id>`). Set `TRAIN_DATASET` / `PROMPT_DATASET` in the environment before running the training or prompt-bank scripts.
