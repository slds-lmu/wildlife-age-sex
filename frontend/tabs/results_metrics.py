import logging
from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)


def _ensure_labels(labels: Iterable[str] | None) -> list[str] | None:
    if not labels:
        return None
    if isinstance(labels, list):
        return labels
    return list(labels)


def plot_confusion_matrix(
    cm_dict: dict[str, int],
    labels: Iterable[str] | None = None,
    title: str = "Confusion Matrix",
):
    label_list = _ensure_labels(labels)
    if not label_list:
        st.warning("Label ordering missing for confusion matrix; attempting inference.")
        inferred = set()
        for key in cm_dict.keys():
            if "_" not in key:
                continue
            true_label, pred_label = key.split("_", 1)
            inferred.update([true_label, pred_label])
        label_list = sorted(inferred)

    if not label_list:
        st.error("Unable to infer label ordering for confusion matrix.")
        return

    n = len(label_list)
    cm = np.zeros((n, n), dtype=int)

    for i, true_label in enumerate(label_list):
        for j, pred_label in enumerate(label_list):
            cm[i, j] = int(cm_dict.get(f"{true_label}_{pred_label}", 0))

    extra_keys = set(cm_dict.keys()) - {f"{t}_{p}" for t in label_list for p in label_list}
    if extra_keys:
        logging.warning(
            "Confusion matrix contains unexpected label pairs that were ignored: %s",
            sorted(extra_keys),
        )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    plt.title(title, pad=20)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_stratified_results(results_dict, metric="accuracy"):
    strata = []
    metrics = []
    for stratum, metrics_dict in results_dict.items():
        if stratum != "overall":
            strata.append(stratum)
            metrics.append(metrics_dict[metric])

    fig = px.bar(
        x=strata,
        y=metrics,
        title=f"Stratified {metric.capitalize()}",
        labels={"x": "Category", "y": metric.capitalize()},
    )
    fig.update_layout(width=800, height=400)
    return fig


def _as_scalar(value, default=0.0):
    if isinstance(value, dict):
        if "mean" in value:
            return float(value["mean"])
        if "value" in value:
            return float(value["value"])
    if value is None:
        return default
    return float(value)


def display_metrics(metrics_dict, title="Metrics"):
    st.subheader(title)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{_as_scalar(metrics_dict.get('accuracy')):.2%}")
    with col2:
        st.metric("Precision", f"{_as_scalar(metrics_dict.get('precision')):.2%}")
    with col3:
        st.metric("Recall", f"{_as_scalar(metrics_dict.get('recall')):.2%}")
    with col4:
        st.metric("F1 Score", f"{_as_scalar(metrics_dict.get('f1-score')):.2%}")


def display_uncertainty_metrics(metrics_dict, title="Uncertainty Metrics"):
    def _scalar(value):
        if isinstance(value, dict) and "mean" in value:
            return float(value["mean"])
        if value is None:
            return None
        return float(value)

    st.subheader(title)
    available_metrics = []

    if metrics_dict.get("uncertainty_threshold") is not None:
        available_metrics.append(
            ("Uncertainty Threshold", f"{_scalar(metrics_dict['uncertainty_threshold']):.2f}")
        )
    if metrics_dict.get("n_certain_images") is not None:
        available_metrics.append(
            ("Certain Images", f"{_scalar(metrics_dict['n_certain_images']):,.0f}")
        )
    if metrics_dict.get("n_uncertain_images") is not None:
        available_metrics.append(
            ("Uncertain Images", f"{_scalar(metrics_dict['n_uncertain_images']):,.0f}")
        )
    if metrics_dict.get("avg_confidence") is not None:
        available_metrics.append(
            ("Avg Prediction Confidence", f"{_scalar(metrics_dict['avg_confidence']):.3f}")
        )

    if not available_metrics:
        st.write("No uncertainty metrics available for this result.")
        return

    cols = st.columns(len(available_metrics))
    for i, (label, value) in enumerate(available_metrics):
        with cols[i]:
            st.metric(label, value)
    st.write("\n")


def calculate_averaged_metrics(results):
    if not results:
        return None

    metric_keys = ["accuracy", "precision", "recall", "f1-score"]
    additional_numeric_keys = [
        "n_test_observations",
        "n_uncertain_images",
        "n_certain_images",
        "avg_confidence",
    ]

    single_value_keys = ["uncertainty_threshold"]

    sections = defaultdict(
        lambda: {
            "labels_known": set(),
            "y_true": [],
            "y_pred": [],
            "numeric_lists": defaultdict(list),
            "boolean_values": defaultdict(list),
            "class_counts": defaultdict(int),
            "confidence_components": [],
            "single_value_lists": defaultdict(list),
        }
    )

    def parse_confusion_key(key, known_labels):
        for label in sorted(known_labels, key=len, reverse=True):
            prefix = f"{label}_"
            if key.startswith(prefix):
                return label, key[len(prefix) :]
        if "_" in key:
            true_label, pred_label = key.split("_", 1)
            return true_label, pred_label
        return None

    def expand_confusion_matrix(cm_dict, known_labels):
        y_true, y_pred = [], []
        for key, count in cm_dict.items():
            if count <= 0:
                continue
            parsed = parse_confusion_key(key, known_labels)
            if not parsed:
                logging.warning(f"Skipping malformed confusion-matrix key: {key}")
                continue
            true_label, pred_label = parsed
            y_true.extend([true_label] * count)
            y_pred.extend([pred_label] * count)
            known_labels.update([true_label, pred_label])
        return y_true, y_pred

    def compute_metrics(y_true, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "accuracy": float(accuracy),
        }

    for _, result in results:
        for section_name, section_data in result.items():
            if "confusion_matrix" not in section_data:
                continue

            section_store = sections[section_name]
            section_store["labels_known"].update(section_data.get("class_distribution", {}).keys())

            y_true_run, y_pred_run = expand_confusion_matrix(
                section_data["confusion_matrix"], section_store["labels_known"]
            )
            if not y_true_run:
                continue

            section_store["y_true"].extend(y_true_run)
            section_store["y_pred"].extend(y_pred_run)

            for key in additional_numeric_keys:
                if key in section_data:
                    section_store["numeric_lists"][key].append(section_data[key])
                    if key == "avg_confidence":
                        section_store["confidence_components"].append(
                            (float(section_data[key]), float(len(y_true_run)))
                        )

            for label, count in section_data.get("class_distribution", {}).items():
                section_store["class_counts"][label] += int(count)

            for key in single_value_keys:
                if key in section_data:
                    section_store["single_value_lists"][key].append(section_data[key])

            n_test = section_data.get("n_test_observations")
            n_uncertain = section_data.get("n_uncertain_images")
            if n_test is not None and n_uncertain is not None:
                section_store["numeric_lists"]["n_total_images"].append(n_test + n_uncertain)

            if "excluded_uncertain_images" in section_data:
                section_store["boolean_values"]["excluded_uncertain_images"].append(
                    bool(section_data["excluded_uncertain_images"])
                )

    averaged_metrics = {}
    for section_name, section_store in sections.items():
        if not section_store["y_true"]:
            continue

        averaged_metrics[section_name] = {}
        combined_metrics = compute_metrics(section_store["y_true"], section_store["y_pred"])
        for key in metric_keys:
            averaged_metrics[section_name][key] = float(combined_metrics[key])

        for key, values in section_store["numeric_lists"].items():
            if not values:
                continue
            if key == "avg_confidence":
                numerator = 0.0
                denom = 0.0
                for conf_value, weight in section_store["confidence_components"]:
                    if weight <= 0:
                        continue
                    numerator += conf_value * weight
                    denom += weight
                if denom > 0:
                    averaged_metrics[section_name][key] = numerator / denom
                continue
            averaged_metrics[section_name][key] = float(np.sum(values))

        for key, values in section_store["boolean_values"].items():
            if not values:
                continue
            unique_values = set(values)
            if len(unique_values) == 1:
                averaged_metrics[section_name][key] = values[0]
            else:
                averaged_metrics[section_name][key] = None

        if section_store["class_counts"]:
            averaged_metrics[section_name]["aggregated_class_distribution"] = dict(
                section_store["class_counts"]
            )

        for key, values in section_store["single_value_lists"].items():
            if not values:
                continue
            unique_values = {v for v in values if v is not None}
            if len(unique_values) == 1:
                averaged_metrics[section_name][key] = float(unique_values.pop())
            else:
                averaged_metrics[section_name][key] = None

        if section_store["y_true"]:
            combined_cm = defaultdict(int)
            for idx, true_label in enumerate(section_store["y_true"]):
                pred_label = section_store["y_pred"][idx]
                combined_cm[f"{true_label}_{pred_label}"] += 1
            averaged_metrics[section_name]["aggregated_confusion_matrix"] = dict(combined_cm)

    return averaged_metrics


def summarize_overall_runs(results):
    summary = {
        "n_runs": len(results),
        "per_run_n_test_observations": [],
        "per_run_n_uncertain_images": [],
        "total_n_test_observations": 0,
        "total_n_uncertain_images": 0,
        "total_images_reviewed": 0,
        "weighted_avg_confidence": None,
        "excluded_flags": set(),
    }

    weighted_conf_numerator = 0.0

    for _, run in results:
        overall = run.get("overall")
        if not overall:
            continue

        n_test_observations = int(overall.get("n_test_observations") or 0)
        uncertain = int(overall.get("n_uncertain_images") or 0)

        summary["per_run_n_test_observations"].append(n_test_observations)
        summary["per_run_n_uncertain_images"].append(uncertain)

        summary["total_n_test_observations"] += n_test_observations
        summary["total_n_uncertain_images"] += uncertain
        summary["total_images_reviewed"] += n_test_observations + uncertain

        avg_conf = overall.get("avg_confidence")
        if avg_conf is not None and n_test_observations:
            weighted_conf_numerator += float(avg_conf) * n_test_observations

        if "excluded_uncertain_images" in overall:
            summary["excluded_flags"].add(bool(overall["excluded_uncertain_images"]))

    summary["weighted_avg_confidence"] = (
        weighted_conf_numerator / summary["total_n_test_observations"]
        if summary["total_n_test_observations"]
        else None
    )

    def _consistent_value(values):
        unique_values = {v for v in values if v is not None}
        if len(unique_values) == 1:
            return unique_values.pop()
        return None

    summary["per_run_n_test_consistent"] = _consistent_value(
        summary["per_run_n_test_observations"]
    )
    summary["per_run_n_uncertain_consistent"] = _consistent_value(
        summary["per_run_n_uncertain_images"]
    )
    summary["excluded_uncertain_consistent"] = len(summary["excluded_flags"]) == 1
    summary["excluded_uncertain_value"] = (
        next(iter(summary["excluded_flags"])) if summary["excluded_flags"] else None
    )

    return summary
