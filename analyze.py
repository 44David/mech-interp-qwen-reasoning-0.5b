import json
import pandas as pd

rows = []

with open("routing_results.jsonl", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

print(
    df.groupby("source")
    .agg(
        samples=("id", "count"),
        think_rate=("used_think", "mean"),
        mean_think_probability=("think_first_token_probability", "mean"),
        median_think_probability=("think_first_token_probability", "median"),
    )
)

cols = [
    "id",
    "source",
    "prompt",
    "think_first_token_probability",
    "used_think",
]

print("\nAlpaca prompts most likely to think:")
print(
    df[df["source"] == "alpaca_rewrite"]
    .sort_values("think_first_token_probability", ascending=False)[cols]
    .head(10)
    .to_string(index=False)
)

print("\nGSM8K prompts least likely to think:")
print(
    df[df["source"] == "gsm8k_rewrite"]
    .sort_values("think_first_token_probability")[cols]
    .head(10)
    .to_string(index=False)
)

cols = [
    "id",
    "prompt",
    "think_first_token_probability",
    "used_think",
    "generated_text",
]

print(
    df[df["source"] == "logic_no_numbers"]
    .sort_values("think_first_token_probability", ascending=False)[cols]
    .to_string(index=False)
)