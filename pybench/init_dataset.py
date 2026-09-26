"""
Goal:
- Load an instruction-tuning dataset from Hugging Face (set TRAIN_DATASET).
- Inspect a few samples.
- Tokenize them with a Hugging Face tokenizer.
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Load the dataset (auto-download from Hugging Face Hub)
dataset = load_dataset(os.environ.get("TRAIN_DATASET", "<hf-dataset-id>"))

# 2. Print basic info
print(dataset)
print("\nExample sample:")
print(dataset["train"][0])

# 3. Load a tokenizer (you can change model name if needed)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# 4. Define a helper to format instruction + response
def format_sample(sample):
    """Combine instruction, input, and output into a single text."""
    instr = sample.get("instruction", "")
    inp = sample.get("input", "")
    out = sample.get("output", "")
    if inp.strip():
        text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    else:
        text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
    return text

# 5. Take one sample and tokenize it
sample = dataset["train"][0]
text = format_sample(sample)
tokens = tokenizer(text, truncation=True, max_length=256)

# 6. Print tokenized result
print("\nFormatted text:")
print(text)
print("\nTokenized:")
print(tokens)
print(f"\nNumber of tokens: {len(tokens['input_ids'])}")

