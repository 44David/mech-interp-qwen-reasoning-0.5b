import argparse
import json
from pathlib import Path
import numpy as np

def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if "prompt" not in row:
                raise ValueError(f"Missing 'prompt' on line {line_number}")

            row.setdefault("id", f"sample_{line_number:04d}")
            row.setdefault("source", "unknown")
            rows.append(row)

    return rows



def get_token_sequence(tokenizer, text: str,) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def merge_and_save_npz(files: list[str]):
    merged = {}
    
    for path in files:
        data = np.load(path)
    
        for key in data.files:
            if key in merged:
                raise ValueError(f"Duplicate activation key: {key}")
    
            merged[key] = data[key]
    
    np.savez_compressed(
        "all_routing_activations.npz",
        **merged,
    )
    
    print(f"Saved {len(merged)} activation arrays.")



def create_metadata_file(files: list[str]):
    rows = []
    seen_ids = set()
    
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
    
                if row["id"] in seen_ids:
                    raise ValueError(f"Duplicate ID: {row['id']}")
    
                seen_ids.add(row["id"])
                rows.append(row)
    
    with open("all_routing_results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(rows)} rows.")


