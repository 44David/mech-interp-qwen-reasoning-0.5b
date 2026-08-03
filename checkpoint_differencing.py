import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM


# config
MODEL_SPECS = [
    ("base", "Qwen/Qwen2.5-0.5B"),
    ("cot", "44David/qwen-0.5b-reasoning-v1"),
    ("alpaca", "44David/qwen-0.5b-reasoning-v2"),
]
TRANSITIONS = [
    ("base_to_cot", "base", "cot"),
    ("cot_to_alpaca", "cot", "alpaca"),
]
OUTPUT_DIR = Path("checkpoint_differencing")
CSV_PATH = OUTPUT_DIR / "parameter_differences.csv"
HEATMAP_PATH = OUTPUT_DIR / "relative_change_heatmaps.png"
LINE_PLOT_PATH = OUTPUT_DIR / "per_layer_relative_change.png"

PLOT_PARAMETERS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_state_dict(model_name: str) -> dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model = model.to("cpu")
    state_dict = {
        name: tensor.detach().float().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    del model
    return state_dict


def parse_parameter_name(name: str):
    layer_match = re.search(r"\.layers\.(\d+)\.", name)
    layer = int(layer_match.group(1)) if layer_match else None

    if ".self_attn." in name:
        module = "attention"
    elif ".mlp." in name:
        module = "mlp"
    else:
        module = "other"

    for parameter in PLOT_PARAMETERS:
        if f".{parameter}." in name:
            return layer, module, parameter

    if name.endswith("embed_tokens.weight"):
        return layer, module, "embed_tokens"
    if name.endswith("norm.weight"):
        return layer, module, "norm"
    if name.endswith("lm_head.weight"):
        return layer, module, "lm_head"

    trailing = name.split(".")[-2:]
    return layer, module, ".".join(trailing)


def frobenius_norm(tensor: torch.Tensor):
    return float(torch.linalg.norm(tensor).item())


def cosine_similarity(old_tensor: torch.Tensor, new_tensor: torch.Tensor):
    old_flat = old_tensor.reshape(-1)
    new_flat = new_tensor.reshape(-1)

    old_norm = torch.linalg.norm(old_flat)
    new_norm = torch.linalg.norm(new_flat)
    if old_norm.item() == 0.0 or new_norm.item() == 0.0:
        return float("nan")

    cosine = torch.dot(old_flat, new_flat) / (old_norm * new_norm)
    return float(cosine.item())


def compare_state_dicts(transition_name: str, old_state_dict: dict[str, torch.Tensor], new_state_dict: dict[str, torch.Tensor]):
    old_keys = set(old_state_dict)
    new_keys = set(new_state_dict)
    shared_keys = sorted(old_keys & new_keys)

    missing_old = sorted(new_keys - old_keys)
    missing_new = sorted(old_keys - new_keys)
    if missing_old or missing_new:
        raise ValueError(
            f"State dict mismatch for {transition_name}: "
            f"{len(missing_old)} keys only in new, {len(missing_new)} keys only in old"
        )

    rows = []
    for name in shared_keys:
        old_tensor = old_state_dict[name]
        new_tensor = new_state_dict[name]

        if old_tensor.shape != new_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {transition_name} at {name}: "
                f"{tuple(old_tensor.shape)} vs {tuple(new_tensor.shape)}"
            )

        delta = new_tensor - old_tensor
        old_norm = frobenius_norm(old_tensor)
        delta_norm = frobenius_norm(delta)
        relative_change = delta_norm / old_norm if old_norm > 0 else float("nan")

        layer, module, parameter = parse_parameter_name(name)
        rows.append(
            {
                "transition": transition_name,
                "layer": layer,
                "module": module,
                "parameter": parameter,
                "tensor_name": name,
                "absolute_change": delta_norm,
                "relative_change": relative_change,
                "cosine_similarity": cosine_similarity(old_tensor, new_tensor),
            }
        )

    return rows


def write_csv(rows, path: Path):
    fieldnames = [
        "transition",
        "layer",
        "module",
        "parameter",
        "tensor_name",
        "absolute_change",
        "relative_change",
        "cosine_similarity",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_heatmap_values(rows, transition: str) -> tuple[list[int], list[str], np.ndarray]:
    filtered = [
        row for row in rows
        if row["transition"] == transition and row["layer"] is not None and row["module"] in {"attention", "mlp"}
    ]

    layers = sorted({int(row["layer"]) for row in filtered})
    modules = ["attention", "mlp"]
    values = np.full((len(layers), len(modules)), np.nan, dtype=np.float64)

    grouped = defaultdict(list)
    for row in filtered:
        grouped[(int(row["layer"]), row["module"])].append(float(row["relative_change"]))

    for layer_index, layer in enumerate(layers):
        for module_index, module in enumerate(modules):
            group = grouped.get((layer, module), [])
            if group:
                values[layer_index, module_index] = float(np.nanmean(group))

    return layers, modules, values


def plot_heatmaps(rows, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(TRANSITIONS), figsize=(6 * len(TRANSITIONS), 8), squeeze=False)

    for axis, (transition_name, _, _) in zip(axes[0], TRANSITIONS):
        layers, modules, values = build_heatmap_values(rows, transition_name)
        image = axis.imshow(values, aspect="auto", cmap="viridis")
        axis.set_title(transition_name.replace("_", " "))
        axis.set_xlabel("Module")
        axis.set_ylabel("Layer")
        axis.set_xticks(range(len(modules)))
        axis.set_xticklabels(modules)
        axis.set_yticks(range(len(layers)))
        axis.set_yticklabels(layers)

        for row_index in range(values.shape[0]):
            for col_index in range(values.shape[1]):
                value = values[row_index, col_index]
                if not math.isnan(value):
                    axis.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_per_layer_series(rows: list[dict], transition: str) -> tuple[list[int], list[float]]:
    filtered = [
        row for row in rows
        if row["transition"] == transition and row["layer"] is not None
    ]

    grouped = defaultdict(list)
    for row in filtered:
        grouped[int(row["layer"])] .append(float(row["relative_change"]))

    layers = sorted(grouped)
    values = [float(np.nanmean(grouped[layer])) for layer in layers]
    return layers, values


def plot_per_layer_lines(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for transition_name, _, _ in TRANSITIONS:
        layers, values = build_per_layer_series(rows, transition_name)
        ax.plot(layers, values, marker="o", linewidth=2, label=transition_name.replace("_", " "))

    ax.set_title("Per-layer relative parameter change")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean relative change")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded_state_dicts = {}
    for short_name, model_name in MODEL_SPECS:
        print(f"Loading {short_name}: {model_name}")
        loaded_state_dicts[short_name] = load_state_dict(model_name)

    all_rows = []
    for transition_name, old_name, new_name in TRANSITIONS:
        print(f"Comparing {transition_name}")
        all_rows.extend(
            compare_state_dicts(
                transition_name=transition_name,
                old_state_dict=loaded_state_dicts[old_name],
                new_state_dict=loaded_state_dicts[new_name],
            )
        )

    write_csv(all_rows, CSV_PATH)
    plot_heatmaps(all_rows, HEATMAP_PATH)
    plot_per_layer_lines(all_rows, LINE_PLOT_PATH)

    print(f"Saved CSV to {CSV_PATH}")
    print(f"Saved heatmaps to {HEATMAP_PATH}")
    print(f"Saved line plot to {LINE_PLOT_PATH}")


if __name__ == "__main__":
    main()
