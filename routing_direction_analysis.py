import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


# config
ACTIVATIONS_PATH = Path("all_routing_activations.npz")
METADATA_PATH = Path("all_routing_results_relabelled.jsonl")
LABELS_PATH = Path("all_routing_results_relabelled.jsonl")
OUTPUT_DIR = Path("routing_direction_analysis")

MODEL_NAME = "44David/qwen-0.5b-reasoning-v2"
DEVICE = "cpu"

TEST_SIZE = 0.25
RANDOM_SEED = 0
MAX_ITER = 2000
INTERVENTION_STRENGTH = 1.0
MAX_CAUSAL_SAMPLES_PER_CLASS = None


def load_label_lookup(path: Path) -> dict[str, bool]:
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


def load_rows(path: Path, label_lookup: dict[str, bool]) -> list[dict]:
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

            if "used_think" not in row:
                if row["activation_key"] not in label_lookup:
                    raise ValueError(
                        f"Missing used_think on line {line_number} and no label lookup match for "
                        f"{row['activation_key']}"
                    )
                row["used_think"] = label_lookup[row["activation_key"]]

            if "prompt" not in row:
                raise ValueError(f"Missing prompt on line {line_number}")

            rows.append(row)

    return rows


def collect_layer_matrix(rows: list[dict], activations_path: Path) -> tuple[list[dict], np.ndarray]:
    kept_rows = []
    vectors = []

    with np.load(activations_path) as activations:
        for row in rows:
            key = row["activation_key"]
            if key not in activations:
                continue

            kept_rows.append(row)
            vectors.append(activations[key])

    if not vectors:
        raise ValueError("No metadata rows matched activation keys in the .npz file.")

    return kept_rows, np.stack(vectors)


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=MAX_ITER,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    )
    classifier.fit(features[train_indices], labels[train_indices])

    predicted_labels = classifier.predict(features[test_indices])
    predicted_scores = classifier.predict_proba(features[test_indices])[:, 1]

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(labels[test_indices], predicted_labels)
        ),
        "auroc": float(roc_auc_score(labels[test_indices], predicted_scores)),
    }


def compute_direction(layer_matrix: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    think_mean = layer_matrix[labels == 1].mean(axis=0)
    direct_mean = layer_matrix[labels == 0].mean(axis=0)
    direction = think_mean - direct_mean

    norm = float(np.linalg.norm(direction))
    if norm == 0:
        raise ValueError("Direction norm is zero; cannot define routing direction.")

    unit_direction = direction / norm
    projections = layer_matrix @ unit_direction
    projection_auroc = float(roc_auc_score(labels, projections))

    return direction, projections, projection_auroc


def build_layer_summary(activations: np.ndarray, labels: np.ndarray) -> tuple[list[dict], int]:
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    summaries = []
    for layer in range(activations.shape[1]):
        layer_matrix = activations[:, layer, :]
        _, projections, projection_auroc = compute_direction(layer_matrix, labels)
        probe_scores = train_probe(layer_matrix, labels, train_indices, test_indices)

        summaries.append(
            {
                "layer": layer,
                "probe_balanced_accuracy": probe_scores["balanced_accuracy"],
                "probe_auroc": probe_scores["auroc"],
                "projection_auroc": projection_auroc,
                "projection_mean_think": float(projections[labels == 1].mean()),
                "projection_mean_direct": float(projections[labels == 0].mean()),
                "combined_score": float(probe_scores["auroc"] + projection_auroc),
            }
        )

    strongest_layer = max(
        summaries,
        key=lambda row: (
            row["combined_score"],
            row["probe_balanced_accuracy"],
            row["projection_auroc"],
        ),
    )["layer"]
    return summaries, strongest_layer


def save_layer_summary(summaries: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)


def plot_layer_metrics(summaries: list[dict], strongest_layer: int, output_path: Path) -> None:
    layers = [row["layer"] for row in summaries]
    probe_balanced_accuracy = [row["probe_balanced_accuracy"] for row in summaries]
    probe_auroc = [row["probe_auroc"] for row in summaries]
    projection_auroc = [row["projection_auroc"] for row in summaries]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, probe_balanced_accuracy, marker="o", label="Probe balanced accuracy")
    ax.plot(layers, probe_auroc, marker="o", label="Probe AUROC")
    ax.plot(layers, projection_auroc, marker="o", label="Projection AUROC")
    ax.axvline(strongest_layer, color="black", linestyle="--", alpha=0.5, label=f"Selected layer {strongest_layer}")
    ax.set_title("Layerwise routing separability")
    ax.set_xlabel("Saved layer index")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(layers)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_projections_by_category(
    rows: list[dict],
    projections: np.ndarray,
    strongest_layer: int,
    output_path: Path,
) -> None:
    categories = sorted({str(row["category"]) for row in rows})
    positions = {category: index for index, category in enumerate(categories)}
    rng = np.random.default_rng(RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.4), 6))

    for used_think, color, label in [(0, "tab:blue", "direct"), (1, "tab:orange", "think")]:
        xs = []
        ys = []
        for projection, row in zip(projections, rows):
            if int(bool(row["used_think"])) != used_think:
                continue
            base_x = positions[str(row["category"])]
            xs.append(base_x + rng.uniform(-0.18, 0.18))
            ys.append(projection)

        ax.scatter(xs, ys, alpha=0.75, s=28, color=color, label=label)

    ax.set_title(f"Layer {strongest_layer} projection onto routing direction by category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Projection onto think-minus-direct direction")
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_model_and_tokenizer() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
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


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def format_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def get_think_token_id(tokenizer) -> int:
    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    if not think_ids:
        raise ValueError("Tokenizer produced no token ids for <think>.")
    return int(think_ids[0])


def intervene_on_output(output, delta: torch.Tensor):
    if isinstance(output, tuple):
        hidden_states = output[0].clone()
        hidden_states[:, -1, :] += delta.to(hidden_states.device, dtype=hidden_states.dtype)
        return (hidden_states, *output[1:])

    hidden_states = output.clone()
    hidden_states[:, -1, :] += delta.to(hidden_states.device, dtype=hidden_states.dtype)
    return hidden_states


def think_probability_with_intervention(
    tokenizer,
    model,
    prompt: str,
    saved_layer_index: int,
    direction: np.ndarray,
    scale: float,
    think_token_id: int,
) -> tuple[float, float]:
    model_device = get_model_device(model)
    formatted_prompt = format_prompt(tokenizer, prompt)
    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

    with torch.inference_mode():
        baseline_outputs = model(**encoded, use_cache=False, return_dict=True)
        baseline_probs = torch.softmax(baseline_outputs.logits[:, -1, :].float(), dim=-1)
        baseline_probability = float(baseline_probs[0, think_token_id].item())

    delta = torch.tensor(direction * scale, device=model_device, dtype=torch.float32)

    if saved_layer_index == 0:
        embeddings = model.get_input_embeddings()(encoded["input_ids"]).clone()
        embeddings[:, -1, :] += delta.to(embeddings.device, dtype=embeddings.dtype)
        intervention_inputs = {
            "inputs_embeds": embeddings,
            "attention_mask": encoded.get("attention_mask"),
        }

        with torch.inference_mode():
            intervened_outputs = model(**intervention_inputs, use_cache=False, return_dict=True)
    else:
        decoder_layers = get_decoder_layers(model)
        module = decoder_layers[saved_layer_index - 1]
        hook = module.register_forward_hook(
            lambda _module, _inputs, output: intervene_on_output(output, delta)
        )
        try:
            with torch.inference_mode():
                intervened_outputs = model(**encoded, use_cache=False, return_dict=True)
        finally:
            hook.remove()

    intervened_probs = torch.softmax(intervened_outputs.logits[:, -1, :].float(), dim=-1)
    intervened_probability = float(intervened_probs[0, think_token_id].item())

    return baseline_probability, intervened_probability


def select_rows_for_causal_test(rows: list[dict]) -> list[dict]:
    direct_rows = [row for row in rows if not row["used_think"]]
    think_rows = [row for row in rows if row["used_think"]]

    if MAX_CAUSAL_SAMPLES_PER_CLASS is not None:
        direct_rows = direct_rows[:MAX_CAUSAL_SAMPLES_PER_CLASS]
        think_rows = think_rows[:MAX_CAUSAL_SAMPLES_PER_CLASS]

    return direct_rows + think_rows


def run_causal_test(rows: list[dict], strongest_layer: int, direction: np.ndarray, output_path: Path) -> list[dict]:
    tokenizer, model = load_model_and_tokenizer()
    think_token_id = get_think_token_id(tokenizer)
    selected_rows = select_rows_for_causal_test(rows)

    results = []
    for row in selected_rows:
        signed_scale = INTERVENTION_STRENGTH if not row["used_think"] else -INTERVENTION_STRENGTH
        baseline_probability, intervened_probability = think_probability_with_intervention(
            tokenizer=tokenizer,
            model=model,
            prompt=row["prompt"],
            saved_layer_index=strongest_layer,
            direction=direction,
            scale=signed_scale,
            think_token_id=think_token_id,
        )

        results.append(
            {
                "id": row["activation_key"],
                "category": row["category"],
                "used_think": bool(row["used_think"]),
                "intervention": "add" if not row["used_think"] else "subtract",
                "baseline_think_probability": baseline_probability,
                "intervened_think_probability": intervened_probability,
                "delta_think_probability": intervened_probability - baseline_probability,
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return results


def summarize_causal_results(results: list[dict], output_path: Path) -> None:
    summary = {}
    for group_name, selector in {
        "direct_prompts_add_direction": lambda row: row["intervention"] == "add",
        "think_prompts_subtract_direction": lambda row: row["intervention"] == "subtract",
        "all": lambda row: True,
    }.items():
        subset = [row for row in results if selector(row)]
        if not subset:
            continue

        baseline = [row["baseline_think_probability"] for row in subset]
        intervened = [row["intervened_think_probability"] for row in subset]
        delta = [row["delta_think_probability"] for row in subset]
        summary[group_name] = {
            "count": len(subset),
            "mean_baseline_think_probability": float(np.mean(baseline)),
            "mean_intervened_think_probability": float(np.mean(intervened)),
            "mean_delta_think_probability": float(np.mean(delta)),
        }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_lookup = load_label_lookup(LABELS_PATH)
    rows = load_rows(METADATA_PATH, label_lookup)
    rows, activations = collect_layer_matrix(rows, ACTIVATIONS_PATH)
    labels = np.array([int(bool(row["used_think"])) for row in rows], dtype=np.int64)

    if len(np.unique(labels)) < 2:
        raise ValueError("used_think must contain both classes to analyze routing.")

    layer_summaries, strongest_layer = build_layer_summary(activations, labels)
    save_layer_summary(layer_summaries, OUTPUT_DIR / "layer_summary.json")
    plot_layer_metrics(layer_summaries, strongest_layer, OUTPUT_DIR / "layerwise_metrics.png")

    strongest_layer_matrix = activations[:, strongest_layer, :]
    direction, projections, projection_auroc = compute_direction(strongest_layer_matrix, labels)
    np.save(OUTPUT_DIR / f"layer_{strongest_layer:02d}_direction.npy", direction)

    strongest_summary = {
        "strongest_layer": strongest_layer,
        "projection_auroc": projection_auroc,
        "direction_norm": float(np.linalg.norm(direction)),
        "intervention_strength": INTERVENTION_STRENGTH,
    }
    with (OUTPUT_DIR / "selected_layer.json").open("w", encoding="utf-8") as file:
        json.dump(strongest_summary, file, indent=2)

    plot_projections_by_category(
        rows=rows,
        projections=projections,
        strongest_layer=strongest_layer,
        output_path=OUTPUT_DIR / f"layer_{strongest_layer:02d}_projections_by_category.png",
    )

    causal_results = run_causal_test(
        rows=rows,
        strongest_layer=strongest_layer,
        direction=direction,
        output_path=OUTPUT_DIR / "causal_results.json",
    )
    summarize_causal_results(causal_results, OUTPUT_DIR / "causal_summary.json")

    print(f"Selected strongest layer: {strongest_layer}")
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
