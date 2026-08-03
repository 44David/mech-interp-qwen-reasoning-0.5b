import json
import math
import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# config
METADATA_PATH = Path("all_routing_results_relabelled.jsonl")
LABELS_PATH = Path("all_routing_results_relabelled.jsonl")
OUTPUT_DIR = Path("full_generation_causal_validation")

MODEL_NAME = "44David/qwen-0.5b-reasoning-v2"
DEVICE = "cpu"
INTERVENTION_LAYER = 23
DIRECTION_PATH = Path("routing_direction_analysis/layer_23_direction.npy")
INTERVENTION_ALPHAS = [0.1, 0.25, 0.5, 1.0]
MAX_NEW_TOKENS = 256
TEMPERATURE = None

PROMPTS_PER_GROUP = 10
BALANCE_TO_MIN_GROUP = False
RANDOM_SEED = 0

GROUP_SPECS = [
    {"name": "gsm8k", "sources": ["gsm8k"]},
    {"name": "direct_alpaca", "sources": ["alpaca"], "used_think": False},
    {"name": "trivial_math", "sources": ["trivial_math"]},
    {"name": "relational_logic", "sources": ["relational_logic"]},
    {"name": "factual_or_summarization", "sources": ["multihop_fact", "numbers_no_math"]},
]

CONDITIONS = [
    "normal",
    "add_direction_once",
    "subtract_direction_once",
    "random_direction_once",
]

OBJECTIVE_SOURCES = {
    "gsm8k",
    "gsm8k_rewrite",
    "trivial_math",
    "conceptual_math",
    "multihop_fact",
    "numbers_no_math",
    "relational_logic",
    "syllogistic_logic",
}


def load_label_lookup(path: Path):
    lookup = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            activation_key = row.get("activation_key") or row.get("id")
            if activation_key is None:
                raise ValueError(f"Missing activation_key/id in labels file on line {line_number}")
            if "used_think" not in row:
                raise ValueError(f"Missing used_think in labels file on line {line_number}")

            lookup[activation_key] = bool(row["used_think"])

    return lookup


def load_rows(path: Path, label_lookup: dict[str, bool]):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            row.setdefault("activation_key", row.get("id"))
            row["category"] = row.get("category") or row.get("source") or "unknown"

            if row["activation_key"] is None:
                raise ValueError(f"Missing activation_key/id on line {line_number}")
            if "prompt" not in row:
                raise ValueError(f"Missing prompt on line {line_number}")

            if "used_think" not in row:
                if row["activation_key"] not in label_lookup:
                    raise ValueError(
                        f"Missing used_think on line {line_number} and no label lookup match for "
                        f"{row['activation_key']}"
                    )
                row["used_think"] = label_lookup[row["activation_key"]]

            rows.append(row)

    return rows


def matches_group(row: dict, group_spec: dict):
    if row.get("source") not in group_spec["sources"]:
        return False
    if "used_think" in group_spec and bool(row.get("used_think")) != bool(group_spec["used_think"]):
        return False
    return True


def sample_prompt_rows(rows: list[dict]):
    rng = np.random.default_rng(RANDOM_SEED)
    grouped_rows = {}
    availability = {}

    for group_spec in GROUP_SPECS:
        group_rows = [row for row in rows if matches_group(row, group_spec)]
        availability[group_spec["name"]] = len(group_rows)
        if not group_rows:
            raise ValueError(f"No rows available for group {group_spec['name']}")
        rng.shuffle(group_rows)
        grouped_rows[group_spec["name"]] = group_rows

    if BALANCE_TO_MIN_GROUP:
        target_count = min(PROMPTS_PER_GROUP, min(availability.values()))
    else:
        target_count = PROMPTS_PER_GROUP

    selected_rows = []
    for group_spec in GROUP_SPECS:
        group_name = group_spec["name"]
        selected = grouped_rows[group_name][: min(target_count, availability[group_name])]
        for row in selected:
            selected_rows.append({**row, "prompt_group": group_name})

    return selected_rows, availability


def load_model_and_tokenizer() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE if DEVICE != "cpu" else None,
        trust_remote_code=True,
    )

    if DEVICE == "cpu":
        model = model.to("cpu")

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def get_model_backbone(model):
    for attr in ("model", "transformer"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError("Could not locate transformer backbone on model.")


def get_decoder_layers(model):
    backbone = get_model_backbone(model)
    for attr in ("layers", "h"):
        if hasattr(backbone, attr):
            return getattr(backbone, attr)
    raise ValueError("Could not locate decoder layers on model backbone.")


def format_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def intervene_on_output(output, delta: torch.Tensor):
    if isinstance(output, tuple):
        hidden_states = output[0].clone()
        hidden_states[:, -1, :] += delta.to(hidden_states.device, dtype=hidden_states.dtype)
        return (hidden_states, *output[1:])

    hidden_states = output.clone()
    hidden_states[:, -1, :] += delta.to(hidden_states.device, dtype=hidden_states.dtype)
    return hidden_states


def hook_context(model, layer_index: int, delta: np.ndarray):
    decoder_layers = get_decoder_layers(model)
    module = decoder_layers[layer_index - 1]
    model_device = get_model_device(model)
    delta_tensor = torch.tensor(delta, device=model_device, dtype=torch.float32)

    intervention_applied = False

    def apply_once(_module, _inputs, output):
        nonlocal intervention_applied

        hidden_states = output[0] if isinstance(output, tuple) else output
        if intervention_applied or hidden_states.shape[1] <= 1:
            return output

        intervention_applied = True
        return intervene_on_output(output, delta_tensor)

    hook = module.register_forward_hook(
        apply_once
    )

    class _HookContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            hook.remove()
            return False

    return _HookContext()


def generate_with_condition(tokenizer, model, prompt: str, condition: str, alpha: float, direction: np.ndarray, random_direction: np.ndarray,):
    formatted_prompt = format_prompt(tokenizer, prompt)
    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(get_model_device(model)) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])

    if condition == "normal":
        context = nullcontext()
    elif condition == "add_direction_once":
        context = hook_context(model, INTERVENTION_LAYER, direction * alpha)
    elif condition == "subtract_direction_once":
        context = hook_context(model, INTERVENTION_LAYER, -direction * alpha)
    elif condition == "random_direction_once":
        context = hook_context(model, INTERVENTION_LAYER, random_direction * alpha)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": TEMPERATURE is not None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if TEMPERATURE is not None:
        generate_kwargs["temperature"] = TEMPERATURE

    with torch.inference_mode():
        with context:
            generated_ids = model.generate(**encoded, **generate_kwargs)

    response_ids = generated_ids[0, prompt_length:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    return response_text, int(response_ids.shape[0])


def clean_special_tokens(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|im_start|>", "")
    return text.strip()


def extract_reasoning_text(text: str) -> str:
    if "<think>" not in text:
        return ""

    after_open = text.split("<think>", 1)[1]
    if "</think>" in after_open:
        return after_open.split("</think>", 1)[0].strip()
    return after_open.strip()


def extract_final_answer(text: str) -> str:
    cleaned = clean_special_tokens(text)
    boxed_match = re.findall(r"\\boxed\{([^}]*)\}", cleaned)
    if boxed_match:
        return boxed_match[-1].strip()

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    elif "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def normalize_text(text: str) -> str:
    text = clean_special_tokens(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.strip(" .,:;!?\n\t\"")
    return text


def try_parse_number(text: str):
    text = extract_final_answer(text)
    matches = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text.replace(",", ""))
    if not matches:
        return None
    token = matches[-1]
    if "/" in token and token.count("/") == 1:
        numerator, denominator = token.split("/")
        if denominator == "0":
            return None
        return float(numerator) / float(denominator)
    return float(token)


def evaluate_correctness(row: dict, generated_text: str) -> tuple[bool, bool | None]:
    if row.get("source") not in OBJECTIVE_SOURCES:
        return False, None
    if "reference_answer" not in row:
        return False, None

    generated_number = try_parse_number(generated_text)
    reference_number = try_parse_number(str(row["reference_answer"]))
    if generated_number is not None and reference_number is not None:
        return True, math.isclose(generated_number, reference_number, rel_tol=1e-9, abs_tol=1e-9)

    generated_answer = normalize_text(extract_final_answer(generated_text))
    reference_answer = normalize_text(str(row["reference_answer"]))
    return True, generated_answer == reference_answer


def detect_repetition(text: str) -> bool:
    stripped_lines = [line.strip() for line in clean_special_tokens(text).splitlines() if line.strip()]
    if stripped_lines:
        line_counts = Counter(stripped_lines)
        if max(line_counts.values()) >= 3:
            return True

    tokens = clean_special_tokens(text).split()
    if len(tokens) < 12:
        return False

    ngrams = Counter(tuple(tokens[index:index + 4]) for index in range(len(tokens) - 3))
    return bool(ngrams and max(ngrams.values()) >= 3)


def analyze_generation(row: dict, condition: str, generated_text: str, response_token_count: int) -> dict:
    cleaned_text = clean_special_tokens(generated_text)
    opens_think = cleaned_text.lstrip().startswith("<think>")
    closes_think = "</think>" in cleaned_text
    reasoning_text = extract_reasoning_text(cleaned_text)
    final_answer = extract_final_answer(cleaned_text)
    correctness_evaluable, is_correct = evaluate_correctness(row, cleaned_text)
    repetitive_generation = detect_repetition(cleaned_text)
    malformed_generation = (
        cleaned_text.count("<think>") != cleaned_text.count("</think>")
        or (opens_think and not closes_think)
    )

    return {
        "id": row["activation_key"],
        "source": row.get("source"),
        "category": row.get("category"),
        "prompt_group": row["prompt_group"],
        "used_think_label": bool(row["used_think"]),
        "condition": condition,
        "prompt": row["prompt"],
        "generated_text": generated_text,
        "opens_think": opens_think,
        "closes_think": closes_think,
        "reasoning_length_chars": len(reasoning_text),
        "reasoning_length_tokens": len(reasoning_text.split()),
        "response_token_count": response_token_count,
        "final_answer": final_answer,
        "correctness_evaluable": correctness_evaluable,
        "is_correct": is_correct,
        "malformed_generation": malformed_generation,
        "repetitive_generation": repetitive_generation,
    }


def save_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_results(results: list[dict]) -> dict:
    summary = {
        "overall": {},
        "by_group": {},
        "group_availability": {},
    }

    def aggregate(subset: list[dict]) -> dict:
        evaluable = [row for row in subset if row["correctness_evaluable"] and row["is_correct"] is not None]
        return {
            "count": len(subset),
            "opens_think_rate": float(np.mean([row["opens_think"] for row in subset])) if subset else None,
            "closes_think_rate": float(np.mean([row["closes_think"] for row in subset])) if subset else None,
            "mean_reasoning_length_tokens": float(np.mean([row["reasoning_length_tokens"] for row in subset])) if subset else None,
            "mean_response_token_count": float(np.mean([row["response_token_count"] for row in subset])) if subset else None,
            "malformed_rate": float(np.mean([row["malformed_generation"] for row in subset])) if subset else None,
            "repetitive_rate": float(np.mean([row["repetitive_generation"] for row in subset])) if subset else None,
            "accuracy_evaluable_count": len(evaluable),
            "accuracy": float(np.mean([row["is_correct"] for row in evaluable])) if evaluable else None,
        }

    condition_keys = sorted({(row["condition"], row["alpha"]) for row in results}, key=lambda item: (CONDITIONS.index(item[0]), item[1]))

    for condition, alpha in condition_keys:
        subset = [row for row in results if row["condition"] == condition and row["alpha"] == alpha]
        summary["overall"][f"{condition}@{alpha}"] = aggregate(subset)

    for group_spec in GROUP_SPECS:
        group_name = group_spec["name"]
        summary["by_group"][group_name] = {}
        for condition, alpha in condition_keys:
            subset = [
                row for row in results
                if row["prompt_group"] == group_name and row["condition"] == condition and row["alpha"] == alpha
            ]
            summary["by_group"][group_name][f"{condition}@{alpha}"] = aggregate(subset)

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_lookup = load_label_lookup(LABELS_PATH)
    rows = load_rows(METADATA_PATH, label_lookup)
    sampled_rows, availability = sample_prompt_rows(rows)

    direction = np.load(DIRECTION_PATH)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0:
        raise ValueError("Loaded direction has zero norm.")

    rng = np.random.default_rng(RANDOM_SEED)
    random_direction = rng.standard_normal(direction.shape[0])
    random_direction = random_direction / np.linalg.norm(random_direction) * direction_norm

    tokenizer, model = load_model_and_tokenizer()

    results = []
    for row in sampled_rows:
        for condition in CONDITIONS:
            condition_alphas = [0.0] if condition == "normal" else INTERVENTION_ALPHAS
            for alpha in condition_alphas:
                generated_text, response_token_count = generate_with_condition(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=row["prompt"],
                    condition=condition,
                    alpha=alpha,
                    direction=direction,
                    random_direction=random_direction,
                )
                result = analyze_generation(
                    row=row,
                    condition=condition,
                    generated_text=generated_text,
                    response_token_count=response_token_count,
                )
                result["alpha"] = alpha
                results.append(result)

    save_jsonl(results, OUTPUT_DIR / "generation_records.jsonl")

    summary = summarize_results(results)
    summary["group_availability"] = availability
    summary["config"] = {
        "intervention_layer": INTERVENTION_LAYER,
        "direction_path": str(DIRECTION_PATH),
        "intervention_alphas": INTERVENTION_ALPHAS,
        "prompts_per_group": PROMPTS_PER_GROUP,
        "balance_to_min_group": BALANCE_TO_MIN_GROUP,
        "conditions": CONDITIONS,
    }

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with (OUTPUT_DIR / "sampled_prompts.json").open("w", encoding="utf-8") as file:
        json.dump(sampled_rows, file, indent=2)

    print(f"Saved generation records to {OUTPUT_DIR / 'generation_records.jsonl'}")
    print(f"Saved summary to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
