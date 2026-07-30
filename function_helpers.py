import argparse
import json
from pathlib import Path

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



def get_token_sequence(
    tokenizer,
    text: str,
) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)