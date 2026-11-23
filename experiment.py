"""Compare ECG morphology and RR features for offline PVC classification."""

import argparse
import csv
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import wfdb
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit


SAMPLE_RATE = 360
LEAD = "MLII"
FILTER_ORDER = 4
FILTER_BAND_HZ = (0.5, 40.0)
PRE_SAMPLES = 72
POST_SAMPLES = 144
BLOCK_SAMPLES = 12
SEED = 42
TEST_FRACTION = 0.25
TREES = 128
MIN_LEAF_SIZE = 5
MODELS = {
    "morphology": "Morphology",
    "rr": "RR intervals",
    "combined": "Morphology + RR",
}
TARGET_SYMBOLS = ("N", "V")
# Standard WFDB beat codes. Non-beat events such as rhythm/quality changes
# are excluded from both the classification target and the RR calculation.
BEAT_SYMBOLS = (
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j",
    "n", "E", "/", "f", "Q",
)
WAVEFORM_TIMES = 1000 * np.arange(-PRE_SAMPLES, POST_SAMPLES) / SAMPLE_RATE
N_COLOR = "#17679c"
V_COLOR = "#a2335f"


@dataclass(frozen=True)
class Beats:
    windows: np.ndarray
    rr_features: np.ndarray
    labels: np.ndarray
    records: np.ndarray
    samples: np.ndarray
    excluded_records: dict[str, str]
    record_counts: dict[str, dict[str, int]]

    @property
    def subject_groups(self) -> np.ndarray:
        # These two records are from the same subject (PhysioNet MIT-BIH notes).
        return np.where(np.isin(self.records, ["201", "202"]), "201/202", self.records)


def load_lead(data_dir: Path, record_id: str, channel: int) -> np.ndarray:
    record = wfdb.rdrecord(str(data_dir / record_id), channels=[channel])
    signal = record.p_signal[:, 0]
    if not np.isfinite(signal).all():
        raise ValueError(f"Record {record_id} has non-finite {LEAD} samples")
    return signal


def load_beats(data_dir: Path, sos: np.ndarray) -> Beats:
    records_path = data_dir / "RECORDS"
    if not records_path.is_file():
        raise FileNotFoundError(f"MIT-BIH RECORDS is missing at {records_path}")

    record_ids = records_path.read_text(encoding="utf-8").split()
    dataset = {"windows": [], "rr_features": [], "labels": [], "records": [], "samples": []}
    excluded_records = {}
    record_counts = {}

    for record_id in record_ids:
        record_path = data_dir / record_id
        header = wfdb.rdheader(str(record_path))
        if header.fs != SAMPLE_RATE:
            raise ValueError(f"Record {record_id} has unexpected sample rate {header.fs} Hz")
        if LEAD not in header.sig_name:
            excluded_records[record_id] = f"{LEAD} lead unavailable"
            continue

        signal = load_lead(data_dir, record_id, header.sig_name.index(LEAD))
        filtered = sosfiltfilt(sos, signal)
        ann = wfdb.rdann(str(record_path), "atr")
        symbols = np.asarray(ann.symbol)
        beat_mask = np.isin(symbols, BEAT_SYMBOLS)
        beat_samples = ann.sample[beat_mask]
        beat_symbols = symbols[beat_mask]

        # Compute RR on the chronological beat sequence before keeping N and V.
        # Endpoints and duplicate or invalid positions have no usable RR pair.
        rr_seconds = np.diff(beat_samples) / SAMPLE_RATE
        center_samples = beat_samples[1:-1]
        center_symbols = beat_symbols[1:-1]
        keep = (
            np.isin(center_symbols, TARGET_SYMBOLS)
            & (center_samples >= PRE_SAMPLES)
            & (center_samples + POST_SAMPLES <= filtered.size)
            & (rr_seconds[:-1] > 0)
            & (rr_seconds[1:] > 0)
        )
        samples = center_samples[keep]
        prev_rr = rr_seconds[:-1][keep]
        next_rr = rr_seconds[1:][keep]
        labels = (center_symbols[keep] == "V").astype(np.int8)
        if not samples.size:
            excluded_records[record_id] = "No eligible target heartbeats after window/RR checks"
            continue

        window_indices = samples[:, None] + np.arange(-PRE_SAMPLES, POST_SAMPLES)
        dataset["windows"].append(filtered[window_indices])
        dataset["rr_features"].append(np.column_stack((prev_rr, next_rr, prev_rr / next_rr)))
        dataset["labels"].append(labels)
        dataset["records"].append(np.repeat(record_id, samples.size))
        dataset["samples"].append(samples)
        record_counts[record_id] = {
            "beat_annotations": int(beat_samples.size),
            "target_annotations": int(np.isin(beat_symbols, TARGET_SYMBOLS).sum()),
            "n": int((labels == 0).sum()),
            "v": int((labels == 1).sum()),
        }

    if not dataset["windows"]:
        raise ValueError("No eligible MLII heartbeats in the selected MIT-BIH data directory")

    return Beats(
        **{name: np.concatenate(values) for name, values in dataset.items()},
        excluded_records=excluded_records,
        record_counts=record_counts,
    )


def morphology_features(windows: np.ndarray) -> np.ndarray:
    block_means = windows.reshape(windows.shape[0], -1, BLOCK_SAMPLES).mean(axis=2)
    return np.column_stack(
        (
            block_means,
            np.ptp(windows, axis=1),
            np.sqrt(np.mean(windows**2, axis=1)),
            np.mean(np.abs(windows), axis=1),
            np.mean(np.abs(np.diff(windows, axis=1)), axis=1),
        )
    )


def split_beats(beats: Beats) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=32, test_size=TEST_FRACTION, random_state=SEED)
    for train, test in splitter.split(beats.labels, groups=beats.subject_groups):
        if np.unique(beats.labels[train]).size == np.unique(beats.labels[test]).size == 2:
            return train, test
    raise ValueError("Unable to find a subject-disjoint split with both classes in each partition")


def partition_details(beats: Beats, subset: np.ndarray) -> dict:
    labels = beats.labels[subset]
    return {
        "subjects": sorted(set(beats.subject_groups[subset])),
        "records": sorted(set(beats.records[subset])),
        "n": int((labels == 0).sum()),
        "v": int((labels == 1).sum()),
    }


def train_models(
    beats: Beats, train: np.ndarray, test: np.ndarray, jobs: int
) -> tuple[dict, np.ndarray]:
    morphology = morphology_features(beats.windows)
    feature_sets = {
        "morphology": morphology,
        "rr": beats.rr_features,
        "combined": np.column_stack((morphology, beats.rr_features)),
    }
    metrics = {}
    combined_predictions: np.ndarray

    for name, features in feature_sets.items():
        print(f"Fitting {MODELS[name]} on {train.size:,} beats...", flush=True)
        classifier = RandomForestClassifier(
            n_estimators=TREES,
            min_samples_leaf=MIN_LEAF_SIZE,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=jobs,
        )
        classifier.fit(features[train], beats.labels[train])
        predicted = classifier.predict(features[test])
        if name == "combined":
            combined_predictions = predicted
        precision, recall, f1, _ = precision_recall_fscore_support(
            beats.labels[test], predicted, average="binary", zero_division=0
        )
        metrics[name] = {
            "feature_count": features.shape[1],
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy_score(beats.labels[test], predicted)),
            "confusion_matrix": confusion_matrix(
                beats.labels[test], predicted, labels=[0, 1]
            ).tolist(),
        }

    return metrics, combined_predictions


def finish_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_preprocessing(
    beats: Beats, index: int, data_dir: Path, sos: np.ndarray, output_dir: Path
) -> None:
    record_id = beats.records[index]
    center = beats.samples[index]
    channel = wfdb.rdheader(str(data_dir / record_id)).sig_name.index(LEAD)
    raw = load_lead(data_dir, record_id, channel)
    filtered = sosfiltfilt(sos, raw)
    # Include more recording context for this filter diagnostic than for one heartbeat.
    left = max(0, center - 2 * SAMPLE_RATE)
    right = min(raw.size, center + 2 * SAMPLE_RATE)
    times = (np.arange(left, right) - center) / SAMPLE_RATE
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 3.85), sharex=True, sharey=True)
    for ax, signal, label in zip(axes, (raw, filtered), ("Raw MLII", "0.5–40 Hz filtered")):
        ax.plot(times, signal[left:right], color=N_COLOR, linewidth=1)
        ax.set_ylabel("mV")
        ax.set_title(label, loc="left")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time relative to annotated beat (s)")
    fig.tight_layout()
    finish_figure(fig, output_dir / "preprocessing.png")


def plot_beats(
    beats: Beats, examples: list[tuple[int, str, str]], path: Path
) -> None:
    fig, axes = plt.subplots(1, len(examples), figsize=(8.0, 2.75), squeeze=False, sharey=True)
    for ax, (index, title, color) in zip(axes[0], examples):
        ax.plot(WAVEFORM_TIMES, beats.windows[index], color=color, linewidth=1.5)
        ax.axvline(0, color="#5d6975", linestyle="--", linewidth=1)
        ax.grid(alpha=0.2)
        ax.set_title(f"{title}\nRecord {beats.records[index]}, sample {beats.samples[index]}")
        ax.set_xlabel("Time from annotation (ms)")
    axes[0, 0].set_ylabel("Filtered MLII (mV)")
    fig.tight_layout()
    finish_figure(fig, path)


def plot_comparison(metrics: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.1))
    bars = np.arange(len(MODELS))
    for offset, metric, color in (
        (-0.23, "precision", "#17679c"),
        (0, "recall", "#6655a4"),
        (0.23, "f1", "#a2335f"),
    ):
        label = "F1" if metric == "f1" else metric.capitalize()
        ax.bar(
            bars + offset,
            [metrics[name][metric] for name in MODELS],
            width=0.22,
            label=label,
            color=color,
        )
    ax.set_xticks(bars, list(MODELS.values()))
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("PVC score")
    ax.legend(loc="upper center", ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    finish_figure(fig, output_dir / "comparison.png")


def plot_confusion(matrix: list, output_dir: Path) -> None:
    values = np.asarray(matrix)
    normalized = values / values.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    cells = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row, column in np.ndindex(values.shape):
        rate = normalized[row, column]
        ax.text(
            column,
            row,
            f"{values[row, column]:,}\n{rate:.1%}",
            ha="center",
            va="center",
            color="white" if rate > 0.55 else "#152238",
        )
    ax.set_xticks([0, 1], ["Normal", "PVC"])
    ax.set_yticks([0, 1], ["Normal", "PVC"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Reference label")
    fig.colorbar(cells, ax=ax, label="Share of reference class")
    fig.tight_layout()
    finish_figure(fig, output_dir / "confusion_matrix.png")


def make_figures(
    beats: Beats,
    test: np.ndarray,
    predictions: np.ndarray,
    metrics: dict,
    data_dir: Path,
    sos: np.ndarray,
    output_dir: Path,
) -> dict:
    labels = beats.labels[test]
    normal = test[np.flatnonzero(labels == 0)[0]]
    pvc = test[np.flatnonzero(labels == 1)[0]]
    plot_preprocessing(beats, pvc, data_dir, sos, output_dir)
    plot_beats(
        beats,
        [(normal, "Reference normal heartbeat", N_COLOR), (pvc, "Reference PVC", V_COLOR)],
        output_dir / "representative_beats.png",
    )
    plot_comparison(metrics, output_dir)
    plot_confusion(metrics["combined"]["confusion_matrix"], output_dir)

    examples = []
    metadata = {}
    for actual, predicted, kind, title, color in (
        (0, 1, "false_positive", "Normal predicted as PVC", N_COLOR),
        (1, 0, "false_negative", "PVC predicted as normal", V_COLOR),
    ):
        matches = test[(labels == actual) & (predictions == predicted)]
        if matches.size:
            index = matches[0]
            examples.append((index, title, color))
            metadata[kind] = {
                "record": str(beats.records[index]),
                "sample": int(beats.samples[index]),
                "previous_rr_s": float(beats.rr_features[index, 0]),
                "next_rr_s": float(beats.rr_features[index, 1]),
            }
    if examples:
        plot_beats(beats, examples, output_dir / "errors.png")
    return metadata


def write_results(output_dir: Path, report: dict) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = ["model", "feature_count", "precision", "recall", "f1", "accuracy", "tn", "fp", "fn", "tp"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, metrics in report["models"].items():
            (tn, fp), (fn, tp) = metrics["confusion_matrix"]
            row = {field: metrics[field] for field in fields if field in metrics}
            writer.writerow({"model": name, **row, "tn": tn, "fp": fp, "fn": fn, "tp": tp})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="MIT-BIH directory with RECORDS")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--jobs", type=int, default=2, help="Random forest worker threads")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least one")

    output_dir = args.output_dir
    sos = butter(FILTER_ORDER, FILTER_BAND_HZ, btype="bandpass", fs=SAMPLE_RATE, output="sos")
    beats = load_beats(args.data_dir, sos)
    train, test = split_beats(beats)
    print(
        f"Loaded {beats.labels.size:,} beats from {len(beats.record_counts)} records. "
        f"Test partition: {test.size:,} beats from "
        f"{len(set(beats.subject_groups[test]))} subjects.",
        flush=True,
    )
    metrics, predictions = train_models(beats, train, test, args.jobs)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = make_figures(beats, test, predictions, metrics, args.data_dir, sos, output_dir)
    report = {
        "dataset": "MIT-BIH Arrhythmia Database 1.0.0",
        "configuration": {
            "lead": LEAD,
            "sample_rate_hz": SAMPLE_RATE,
            "filter_order": FILTER_ORDER,
            "filter_band_hz": FILTER_BAND_HZ,
            "window_samples_relative_to_annotation": [-PRE_SAMPLES, POST_SAMPLES],
            "window_convention": "start inclusive, stop exclusive",
            "test_fraction_of_subjects": TEST_FRACTION,
            "seed": SEED,
            "trees": TREES,
            "minimum_samples_per_leaf": MIN_LEAF_SIZE,
            "class_weight": "balanced",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "wfdb": wfdb.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "train": partition_details(beats, train),
        "test": partition_details(beats, test),
        "target_labels": list(TARGET_SYMBOLS),
        "excluded_records": beats.excluded_records,
        "record_counts": beats.record_counts,
        "models": metrics,
        "error_examples": examples,
    }
    write_results(output_dir, report)
    for name, scores in metrics.items():
        print(f"{MODELS[name]}: PVC F1 = {scores['f1']:.4f}")
    print(f"Saved metrics and figures under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
