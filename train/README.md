# Training-side collection

Two successive drops of the Slurm training harness used to collect the
training power traces. Both are anonymized for double-blind review (see each
drop's own `README.md`).

- [`llm_main_0/`](llm_main_0/) — the original harness: `env.sh`, `launch/`
  (sbatch/srun entry scripts for training and inference), `templates/` (Slurm
  job templates), `pybench/` (Python training/inference workloads), and a
  sample `logs_train/` run.
- [`llm_main_v2/`](llm_main_v2/) — an expanded set of per-model launch
  templates (Gemma, GLM, GPT-OSS, Kimi, Llama 4, Mistral, Qwen1/2/3) plus a
  data-parallel sweep and MoE training variant.

Both read the fine-tuning dataset from the `TRAIN_DATASET` environment
variable (a Hugging Face dataset id), rather than a hardcoded value.
