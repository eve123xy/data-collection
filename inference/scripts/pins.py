"""Every pinned constant for the dataset stage."""

SMALLPROMPTS_SOURCE = "<hf-owner>/SmallPromptsDataset"
SMALLPROMPTS_FILE = "prompts_S.jsonl"
SMALLPROMPTS_ROWS = 116_981
SHUFFLE_SEED = 20260830

DATASETS_REPO = "<hf-owner>/LLM-Power-Datasets-Main"   # prompt artifacts
RUNS_REPO = "<hf-owner>/LLM-Power-Runs-Main"           # run outputs (subsystem 5)
DATASETS_REVISION = "28e70a75d28de8786f8cd867e2b9801ebb182ee8"

# Exact-ISL prompt artifacts for the ISL_OSL grid, published 2026-09-03.
# Pinned SEPARATELY from DATASETS_REVISION so publishing them cannot move the
# revision the bucket-S artifact is pinned at -- the smallprompts cells must
# keep reading byte-identical prompts to the ones already run.
# islprompts/isl<N>/prompts.jsonl holds prompts truncated to exactly N tokens
# with the Qwen3 tokenizer; 100% of a 200-row sample re-tokenises to exactly N.
ISLPROMPTS_REVISION = "bbec762ca9aa41436c62365c2adcdf6ce832454c"
ISLPROMPTS_ROWS = {128: 155067, 512: 24801, 2048: 4966}

# A LARGER isl2048 pool, for the ONE cell that exhausted the original.
# islprompts/isl2048 is unchanged and remains the provenance for the four 2048
# cells already completed -- this is a separate artifact, not a replacement.
# Built the same way (no dedup) so the cell stays methodologically consistent
# with its siblings: the original is 27% duplicates and they drew 4-26% repeats.
ISL2048_EXPANDED_REVISION = "ab3a29bc74582e9705f7f50558a9f2ed2a3322db"
ISL2048_EXPANDED_ROWS = 6819
ISL2048_EXPANDED_FOR = ("islosl_h200_qwen3-8b_2048-128_b32",)

# Free-generation SOURCE pools for ISL_OSL blocks 2-3. Separate revision from
# DATASETS_REVISION and ISLPROMPTS_REVISION so publishing one cannot move the
# revision another artifact is pinned at.
SOURCEPROMPTS_REVISION = "bc3ad3b3a57a929e886ae83c85acb721723f7d4f"
SOURCEPROMPTS_ROWS = {"chat-sharegpt": 175782,
                      "code-instructcoder": 114099,
                      "math-comp": 178209}
# The manifest's source label -> the artifact that serves it. Only `math-aime`
# is not an identity: every AIME problem set between 1983 and 2024 is 933
# prompts, about a seventh of ONE cell's demand, and the driver treats a short
# pool as a hard error rather than recycling (a recycled prompt is a prefix-cache
# hit, so the cell would measure the cache). On the operator's decision
# (2026-09-04) the source is widened to competition mathematics as a class,
# with all 933 AIME problems kept as its core. These cells are competition
# maths, NOT AIME, and released metadata must say so.
SOURCE_ARTIFACT = {"chat-sharegpt": "chat-sharegpt",
                   "code-instructcoder": "code-instructcoder",
                   "math-aime": "math-comp"}

BURSTGPT_TRACE_ROWS = 4_956_058
WINDOW_S = 1200
RATE_TOLERANCE = 0.02
# The archive searched on a coarser grid than 1 s: re-deriving lands 14-30 s
# from the recorded offsets (>97% window overlap). Verified 2026-08-30 against
# BurstGPT_without_fails_3.csv. We assert the criterion reproduces the window,
# then slice at the RECORDED offset for continuity with the archive.
OFFSET_TOLERANCE_S = 60

POOLS = {
    "qwen3": "<hf-owner>/ShareGPTPairsQwen3",
    "llama31": "<hf-owner>/ShareGPTPairsLlama31",
}

# Burstiness is peak-requests-in-any-minute over mean-requests-per-minute.
# Unconstrained, its argmax is degenerate: a near-empty window at a traffic
# gap puts all its traffic in one minute and scores the ceiling of W/60 = 20.
# Constraining the search to windows sustaining at least this rate makes it
# meaningful, and at exactly 1.0 req/s it reproduces the recorded Burst
# offset with zero drift (verified 2026-08-30).
BURSTINESS_MIN_RATE = 1.0

# All three windows are DERIVED. offset_s and req_per_s are assertions: the
# criterion's argmax must land within OFFSET_TOLERANCE_S of offset_s, and the
# measured rate there must match req_per_s.
# `argmax_offset_s` is where the criterion's argmax lands; `offset_s` is where
# we actually slice. They differ only for burst - see shift_reason.
WINDOWS = {
    "peakarrival": {"offset_s": 21_447_110, "req_per_s": 11.66,
                    "criterion": "request_count"},
    "burst":       {"offset_s": 22_263_530, "req_per_s": 1.60,
                    "criterion": "burstiness",
                    "argmax_offset_s": 22_263_230, "shift_s": 300,
                    "shift_reason":
                        "The argmax window ends mid-spike, capturing 1,058 of "
                        "1,817 burst requests (58%). The metric selects for "
                        "this: including the rest raises the mean and lowers "
                        "burstiness. Shifting +300s puts the whole 3-minute "
                        "spike inside with ~4 min of drain after it, at the "
                        "cost of burstiness 15.2 -> 9.6."},
    "work":        {"offset_s": 28_141_910, "req_per_s": 5.14,
                    "criterion": "decode_work"},
}

# --- serving stack -----------------------------------------------------------
# vast.ai's vLLM template ships `vllm/vllm-openai:latest`, which is a MOVING tag:
# two cells provisioned weeks apart could run different builds, an uncontrolled
# variable across the campaign. Pin the concrete tag `latest` resolved to on
# 2026-08-30 and use it on Lambda too, so every cell runs one version.
VLLM_VERSION = "0.28.0"
VLLM_IMAGE = "vllm/vllm-openai:v0.28.0"

# --- workload generator ------------------------------------------------------

CHAT_ENDPOINT = "/v1/chat/completions"   # never /v1/completions: the raw
                                         # endpoint bypasses the chat template,
                                         # so a reasoning model never reasons
METRICS_PATH = "/metrics"
# Client-side connection pool for UNBOUNDED cells. httpx defaults to 100, which
# silently capped "unbounded in-flight" at 100 and meant vLLM never queued --
# the open-loop sheets were measuring our own connection pool. Set high enough
# that the client can never be the binding constraint; the limit in force is
# recorded per run and flagged if achieved concurrency approaches it.
CLIENT_MAX_CONNECTIONS = 1024   # run_plan_settings_v2 §1: unbounded blocks hold 1024 in flight

SCRAPE_INTERVAL_S = 1.0

# top_k is "off", which means the key is absent, not zero.
SAMPLING = {"temperature": 0.7, "top_p": 0.9}

WARMUP_S = 60
DURATION_S = {"knob": 600, "open": 1200}

OSL = {"forced": 512, "free": 2048, "thinking": 8192}
# run_plan_settings_v2 §1: "All free-generation and thinking cells: 16384."
# "free" read 2048 -- the OSL value pasted into the context-window slot, which is
# internally impossible: a 2048 context cannot hold a 2048-token output plus any
# prompt at all. The manifest always passed --max-model-len 16384, so the SERVER
# was correct; only the recorded metadata was wrong, on ~90 cells.
MAX_MODEL_LEN = {"forced": 2048, "free": 16384, "thinking": 16384}

# Set on thinking-ON cells only, and never for Llama-3.1.
REASONING_PARSER = "qwen3"

# A closed-loop cell serving distinct prompts should hit ~0. Above this, flag.
# Retired 2026-09-02. prefix_hit_pct is still measured and recorded on every
# cell; it is no longer a flag condition. The 2.0 threshold assumed distinct
# prompts imply a near-zero hit rate, but every request shares the same Qwen3
# chat-template preamble, so the real rate is 15-18% structurally and the flag
# fired on every cell. Kept as a named constant rather than deleted, so the
# number that was in force is on the record.
PREFIX_HIT_FLAG_PCT_RETIRED = 2.0

# --- model and hardware tables ------------------------------------------------
# Keys are the tracker's own (column A, column B) labels, normalised to
# lowercase. Keeping the tracker's vocabulary here means the sheet is parsed in
# one place and every downstream consumer sees resolved values instead.

def _m(hf_id, family, size_b, moe=False):
    return {"hf_id": hf_id, "family": family, "size_b": size_b, "moe": moe}


MODELS = {
    ("qwen3", "0.6b"): _m("Qwen/Qwen3-0.6B", "qwen3", 0.6),
    ("qwen3", "1.7b"): _m("Qwen/Qwen3-1.7B", "qwen3", 1.7),
    ("qwen3", "4b"): _m("Qwen/Qwen3-4B", "qwen3", 4),
    ("qwen3", "8b"): _m("Qwen/Qwen3-8B", "qwen3", 8),
    ("qwen3", "14b"): _m("Qwen/Qwen3-14B", "qwen3", 14),
    ("qwen3", "32b"): _m("Qwen/Qwen3-32B", "qwen3", 32),
    ("qwen3", "8b dense"): _m("Qwen/Qwen3-8B", "qwen3", 8),
    ("qwen3", "14b dense"): _m("Qwen/Qwen3-14B", "qwen3", 14),
    ("qwen3", "32b dense"): _m("Qwen/Qwen3-32B", "qwen3", 32),
    ("qwen3-moe-30b-a3b", "30b 3b moe"): _m("Qwen/Qwen3-30B-A3B", "qwen3", 30, moe=True),
    ("qwen3-moe-30b-a3b", "30ba3b"): _m("Qwen/Qwen3-30B-A3B", "qwen3", 30, moe=True),
    ("llama3.1", "8b dense"): _m("meta-llama/Llama-3.1-8B-Instruct", "llama31", 8),
    ("llama3.1-8b", "8b dense"): _m("meta-llama/Llama-3.1-8B-Instruct", "llama31", 8),
    ("llama3.1", "70b"): _m("meta-llama/Llama-3.1-70B-Instruct", "llama31", 70),
    ("llama3.1", "70b dense"): _m("meta-llama/Llama-3.1-70B-Instruct", "llama31", 70),
    ("qwen3-32b", "32b dense"): _m("Qwen/Qwen3-32B", "qwen3", 32),
}

GPUS = {
    "A100": {"name": "A100-SXM4-80GB", "variant": "SXM4 80GB",
             "mem_gb": 80, "provider": "vast.ai"},
    "H100": {"name": "H100 80GB HBM3", "variant": "SXM5 80GB",
             "mem_gb": 80, "provider": "vast.ai"},
    "H200": {"name": "H200", "variant": "SXM 141GB",
             "mem_gb": 141, "provider": "vast.ai"},
    "H100 (Lambda)": {"name": "H100 80GB HBM3", "variant": "SXM5 80GB",
                      "mem_gb": 80, "provider": "Lambda"},
}

# Every entry verified by reading the repo's real config.json and asserting
# quantization_config.quant_method == "gptq" and bits == 4. Naming is not
# evidence: two Qwen3-14B repos named "gptq-int4" are compressed-tensors, and
# the top-download "Qwen3-8B GPTQ Int4" hit is a vision-language model.
# Publishers are kept consistent within a family (JunHowie across Qwen3,
# hugging-quants across Llama-3.1) so the weight-quant knob is not confounded
# by differences in quantiser or calibration set.
GPTQ_INT4 = {
    "Qwen/Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",   # official
    "Qwen/Qwen3-32B": "JunHowie/Qwen3-32B-GPTQ-Int4",
    "Qwen/Qwen3-14B": "JunHowie/Qwen3-14B-GPTQ-Int4",       # verified 2026-08-31
    "Qwen/Qwen3-8B": "JunHowie/Qwen3-8B-GPTQ-Int4",         # verified 2026-08-31
    "meta-llama/Llama-3.1-8B-Instruct":
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    "meta-llama/Llama-3.1-70B-Instruct":                     # verified 2026-08-31
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
}

# bf16 weight footprint, GB. Used only to reject rows where the model cannot
# fit the GPU at the stated TP - a check that otherwise fails on a rented box.
MODEL_MEM_GB = {
    "Qwen/Qwen3-0.6B": 1.2, "Qwen/Qwen3-1.7B": 3.4, "Qwen/Qwen3-4B": 8,
    "Qwen/Qwen3-8B": 16, "Qwen/Qwen3-14B": 28, "Qwen/Qwen3-32B": 64,
    "Qwen/Qwen3-30B-A3B": 60,
    "meta-llama/Llama-3.1-8B-Instruct": 16,
    "meta-llama/Llama-3.1-70B-Instruct": 140,
}

# BurstGPT Block 2's tracker column holds compression factors c directly
# (relabelled in the tracker 2026-08-31; it previously held target rates).
# Replaying the whole 1200 s window at c gives 5.1425*c req/s.
# The set is pinned so that if the column's MEANING ever changes again - back to
# rates, say - generation fails loudly rather than silently reading 10 as c=10
# (51 req/s) instead of c=2.
# TP8 is excluded from v1 (decided 2026-08-31). 8xH200 is the scarcest segment
# on the market - OPERATIONAL_LEARNINGS section 4.2 is an entry on sequencing
# those cells alone - and 70B at TP8 spreads a 140 GB model across 1,128 GB of
# HBM, which measures interconnect overhead more than serving behaviour. The
# tracker still marks the two cells pink; the exclusion lives here so generation
# is deterministic without waiting on a manual tracker edit.
MAX_TP_V1 = 4

BURSTGPT_C_ALLOWED = {2.0, 1.0, 0.5, 0.25}
BURSTGPT_WINDOW_S = 1200
BURSTGPT_WORK_NATIVE_REQ_S = 5.1425

# Built from docker/Dockerfile, pushed 2026-09-02. Pinned BY DIGEST, not by
# tag: a tag is mutable, so two cells provisioned weeks apart under one tag
# could run different builds - an uncontrolled variable across 248 cells.
#
# :v2 replaces the withdrawn :v1, which baked scripts/.env and the Google
# service-account key into a published layer (OPERATIONAL_LEARNINGS 2.13).
# :v2 carries NO scripts/ at all - /opt/campaign/scripts is an empty directory
# and the tree is delivered per-run by `provision.py push-creds`.
# Verified before push: no .env / credential JSON / key material anywhere in
# the image; x86_64; nv-hostengine + dcgmi present; vLLM 0.28.0; python alias.
CAMPAIGN_IMAGE = ("<dockerhub-account>/llm-power-main-runs@sha256:"
                  "dcd7cbef41a899660ec989c54c38c626e44791ace67f5424308ac95b6424e891")
CAMPAIGN_IMAGE_TAG = "<dockerhub-account>/llm-power-main-runs:v2"   # human-readable alias

# The image is built on vllm/vllm-openai:v0.28.0, whose torch is 2.13.0+cu130.
# A cu130 build needs driver API >= 13000, i.e. NVIDIA driver >= 580.65. Rent a
# host below that and the container loads
# /usr/local/cuda-13.0/compat/libcuda.so ahead of the native lib, cuInit(0)
# returns 803 CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, and NOTHING CUDA runs -- not
# TP2, not TP1, not `import torch; torch.cuda.init()`.
#
# This is not rare: a 2x H100 box rented 2026-09-03 had driver 560.35.05 from
# October 2024, and the first H100 offer listed after it had 560.35.03. Both
# --verify-gpu and bootstrap PASS on such a box, because both use NVML/DCGM and
# never touch the CUDA driver API -- so the failure surfaces only after the
# image pull, the weight download and the vLLM start have all been paid for.
MIN_DRIVER_VERSION = 580.65
MIN_CUDA_MAX_GOOD = 13.0


# --- campaign orchestration ---------------------------------------------------
NTFY_TOPIC = "ntfy.sh/<notify-topic>"

# The tracker workbook in Drive that sync_tracker writes run status back into.
# From the sheet URL: .../spreadsheets/d/<ID>/edit
TRACKER_DRIVE_FILE_ID = "1tTgaBqxO2HpWwnZ7R4LLLrlSOHI_tuX0"

# Hard spend guards, enforced in code rather than by an agent's judgement.
# OPERATIONAL_LEARNINGS 4.1: burn rate, not total GPU-hours, is what kills a
# campaign; 2.1 records a 2h43m billing incident from a single hung step.
# Raised 3 -> 5 on 2026-09-03 for mass launch, matching the operator's ceiling:
# "don't do more than 5 provisions at a time after the smoke tests are
# completed". The skill's ceiling and this constant must agree -- whichever is
# lower silently wins, and a cap that refuses a launch you believe should
# proceed is a stop-and-ask, not something to work around.
MAX_CONCURRENT_INSTANCES = 5
# Raised 12.0 -> 22.0 on 2026-09-03, on the operator's approval, so the
# 5-instance ceiling is actually reachable: 5 concurrent H200s run ~$20.15/hr
# and the old cap refused the third. Two limits that disagree are worse than
# either, because the lower one wins silently.
#
# The number to be comfortable with is not the hourly rate but the worst case,
# which the 3 h hard cap bounds: 5 instances x 3 h x ~$4/hr is about $60 before
# everything auto-destroys.
MAX_DPH_IN_FLIGHT = 35.0        # total $/hr across all live instances
# TEMPORARILY 35.0 for the TP4 capacity block (2026-09-05, operator-approved):
# a 4x H200 is ~$16.64/hr on top of ~$17 already in flight. REVERT TO 22.0
# once those four cells are done. TP8 is NOT approved -- ask before it.
# Held at 22.0. Raised to 25.0 and then 28.0 on 2026-09-04 for two specific
# operator-approved groupings (the 4x H100 TP4 group, then a rare fresh H200
# host), and reverted here once both were done. Recorded because the pattern
# is the point: a breach is approved for a NAMED grouping and ends with it.
# Note for later: the three llama-70b TP4 cells each need a 4-GPU box, and a
# 4x H200 offer alone is ~$23.46/hr -- that one will need a fresh approval.
MAX_INSTANCE_HOURS = 4.0        # auto-destroy past this, whatever it is doing
# Raised 3.0 -> 4.0 on 2026-09-04 on the operator's instruction, to fit the
# 10-cell closed-loop groupings and the 1200 s open-loop windows in one box.
POLL_INTERVAL_S = 300           # orchestrator watchdog cadence
INSTANCE_LABEL_PREFIX = "llmpl"  # every provision is tagged
# The vast account is a TEAM account - other people's instances live there too.
# Sweeps and cost accounting therefore consider only instances carrying our
# label. Never destroy, or even flag, something we did not launch.
SWEEP_ONLY_OUR_LABEL = True
