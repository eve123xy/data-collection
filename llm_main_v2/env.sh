#!/bin/bash

unset -f which

# Conda setup
module load anaconda3/2025.06
export PATH=/scratch/$USER/datacenter/env/bin:$PATH
export PYTHONPATH=/scratch/$USER/env/bin:$PATH

# Hugging Face cache directory
export HF_HOME=/scratch/$USER/hf
mkdir -p "$HF_HOME"

# Load Hugging Face token
if [ -f /home/$USER/.hf_token ]; then
    export HUGGINGFACE_HUB_TOKEN=$(cat /home/$USER/.hf_token)
fi

# Write token for client use
### 4. If we have a token, set up both legacy and new auth stores
if [ -n "$HUGGINGFACE_HUB_TOKEN" ]; then
    # Legacy-style (older huggingface-cli expects this)
    mkdir -p "$HOME/.huggingface"
    echo -n "$HUGGINGFACE_HUB_TOKEN" > "$HOME/.huggingface/token"
    chmod 600 "$HOME/.huggingface/token"

    # New-style: explicitly log in non-interactively
    hf auth login --token "$HUGGINGFACE_HUB_TOKEN" >/dev/null 2>&1 || true
fi
