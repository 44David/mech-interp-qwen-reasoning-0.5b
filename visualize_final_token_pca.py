import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# config
ACTIVATIONS_PATH = Path("all_routing_activations.npz")
METADATA_PATH = Path("all_routing_results_relabelled.jsonl")
LAYERS = [0, 12, 24]
OUTPUT_DIR = Path("pca_plots")


def load_rows(path: Path) -> list[dict]:
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

    stacked = np.stack(vectors)
    return kept_rows, stacked


def run_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vt[:2].T

    explained_variance = (singular_values ** 2) / max(centered.shape[0] - 1, 1)
    total_variance = explained_variance.sum()
    if total_variance == 0:
        explained_ratio = np.zeros(2, dtype=np.float64)
    else:
        explained_ratio = explained_variance[:2] / total_variance

    return components, explained_ratio


def plot_by_label(points: np.ndarray, labels: list[str], title: str, explained_ratio: np.ndarray,output_path: Path):
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(9, 7))

    for index, label in enumerate(unique_labels):
        mask = [value == label for value in labels]
        subset = points[mask]
        ax.scatter(
            subset[:, 0],
            subset[:, 1],
            s=28,
            alpha=0.8,
            label=label,
            color=cmap(index),
        )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% var)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows(METADATA_PATH)
    rows, activations = collect_layer_matrix(rows, ACTIVATIONS_PATH)

    num_saved_layers = activations.shape[1]
    invalid_layers = [layer for layer in LAYERS if layer < 0 or layer >= num_saved_layers]
    if invalid_layers:
        raise ValueError(
            f"Requested layers {invalid_layers} are out of range for activation stack with "
            f"{num_saved_layers} layers."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    category_labels = [str(row["category"]) for row in rows]
    think_labels = ["used_think" if row.get("used_think") else "no_think" for row in rows]

    for layer in LAYERS:
        layer_matrix = activations[:, layer, :]
        points, explained_ratio = run_pca(layer_matrix)

        plot_by_label(
            points=points,
            labels=category_labels,
            title=f"Layer {layer} final-token PCA colored by category",
            explained_ratio=explained_ratio,
            output_path=OUTPUT_DIR / f"layer_{layer:02d}_by_category.png",
        )

        plot_by_label(
            points=points,
            labels=think_labels,
            title=f"Layer {layer} final-token PCA colored by used_think",
            explained_ratio=explained_ratio,
            output_path=OUTPUT_DIR / f"layer_{layer:02d}_by_used_think.png",
        )

        print(f"Saved PCA plots for layer {layer} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
