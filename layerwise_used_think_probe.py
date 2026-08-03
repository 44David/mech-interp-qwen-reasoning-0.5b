import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# config
ACTIVATIONS_PATH = Path("all_routing_activations.npz")
METADATA_PATH = Path("rewritten_variants.jsonl")
LABELS_PATH = Path("all_routing_results_relabelled.jsonl")
OUTPUT_DIR = Path("rewritten_variants_samples_probe_plots")
PLOT_METRICS = ["balanced_accuracy", "auroc"]
TEST_SIZE = 0.25
RANDOM_SEED = 0
MAX_ITER = 2000


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

            if row["activation_key"] is None:
                raise ValueError(f"Missing activation_key/id on line {line_number}")

            if "used_think" not in row:
                if row["activation_key"] not in label_lookup:
                    raise ValueError(
                        f"Missing used_think on line {line_number} and no label lookup match for "
                        f"{row['activation_key']}"
                    )
                row["used_think"] = label_lookup[row["activation_key"]]

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


def train_probe(features: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray) -> dict[str, float]:
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


def plot_metric(layer_indices: list[int], scores: list[float], metric_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(layer_indices, scores, marker="o", linewidth=2)
    ax.set_title(f"Layerwise used_think separability ({metric_name})")
    ax.set_xlabel("Saved layer index")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_xticks(layer_indices)
    ax.grid(alpha=0.25)

    if metric_name in {"balanced_accuracy", "auroc"}:
        ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    label_lookup = load_label_lookup(LABELS_PATH)
    rows = load_rows(METADATA_PATH, label_lookup)
    rows, activations = collect_layer_matrix(rows, ACTIVATIONS_PATH)

    labels = np.array([int(bool(row["used_think"])) for row in rows], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise ValueError("used_think must contain both classes to train a probe.")

    indices = np.arange(len(rows))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    num_saved_layers = activations.shape[1]
    layer_indices = list(range(num_saved_layers))
    metric_series = {metric_name: [] for metric_name in PLOT_METRICS}

    for layer in layer_indices:
        layer_features = activations[:, layer, :]
        layer_scores = train_probe(layer_features, labels, train_indices, test_indices)

        for metric_name in PLOT_METRICS:
            if metric_name not in layer_scores:
                raise ValueError(f"Unsupported metric: {metric_name}")
            metric_series[metric_name].append(layer_scores[metric_name])

    for metric_name, scores in metric_series.items():
        plot_metric(
            layer_indices=layer_indices,
            scores=scores,
            metric_name=metric_name,
            output_path=OUTPUT_DIR / f"layerwise_{metric_name}.png",
        )
        print(f"Saved {metric_name} plot to {OUTPUT_DIR / f'layerwise_{metric_name}.png'}")


if __name__ == "__main__":
    main()
