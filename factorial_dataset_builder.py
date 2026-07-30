import json 
from operator import index
import random
from datasets import load_dataset
from torch.special import i0e 


random.seed(42)

gsm8k_reasoning = load_dataset("44David/gsm8k-reasoning-traces", "default", split="train")
alpaca = load_dataset("tatsu-lab/alpaca",split="train")

gsm8k_samples = random.sample(range(len(gsm8k_reasoning)), 100)
alpaca_samples = random.sample(range(len(alpaca)), 100)

rows = []

for i in gsm8k_samples:
    sample = gsm8k_reasoning[i]

    rows.append({
        "id": f"gsm8k_{i}",
        "source": "gsm8k",
        "prompt": sample["problem"],
        "reference_answer": sample["reasoning"],
        "original_index": i
    })

for i in alpaca_samples:
    sample = alpaca[i]

    prompt = sample["instruction"]

    if sample["input"].strip():
        prompt += f"\n\n{sample['input']}"

    rows.append({
        "id": f"alpaca_{i}",
        "source": "alpaca",
        "prompt": prompt,
        "reference_answer": sample["output"],
        "original_index": i,
    })


with open("raw_prompt_samples.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} samples")


