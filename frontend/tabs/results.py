import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)


def plot_confusion_matrix(cm_dict, labels=None):
    """Create a confusion matrix plot using scikit-learn's ConfusionMatrixDisplay.

    Args:
        cm_dict: Dictionary with keys in format "true_label_predicted_label"
        labels: Optional list of labels in correct order. If None, extracted from cm_dict.
    """
    # Extract unique labels from dictionary keys
    if labels is None:
        labels = sorted(set(label for key in cm_dict.keys() for label in key.split("_")))

    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    # Fill confusion matrix using label indices
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for key, value in cm_dict.items():
        try:
            true_label, pred_label = key.split("_")
            i, j = label_to_idx[true_label], label_to_idx[pred_label]
            cm[i, j] = value
        except Exception as e:
            st.write(f"Error processing key {key}: {e}")
            continue

    # Create the confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",  # Show integer values
        colorbar=True,
    )

    # Customize the plot
    plt.title("Confusion Matrix", pad=20)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    return fig


def plot_stratified_results(results_dict, metric="accuracy"):
    """Create an interactive bar plot of stratified results."""
    # Extract metrics for each stratum
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


def display_metrics(metrics_dict, title="Metrics", show_std=False):
    """Display metrics in a row of columns.

    Args:
        metrics_dict: Dictionary containing metric values
        title: Title for the metrics section
        show_std: If True, display as mean ± std format. If False, display as percentage.
    """
    st.subheader(title)
    col1, col2, col3, col4 = st.columns(4)

    if show_std:
        # Display averaged metrics with standard deviation on multiple lines
        with col1:
            acc_data = metrics_dict.get("accuracy", {})
            if isinstance(acc_data, dict) and "mean" in acc_data:
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f8f9fa;">
                    <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Accuracy</div>
                    <div style="font-size: 18px; font-weight: bold; color: #262730;">{acc_data["mean"]:.2%}</div>
                    <div style="font-size: 12px; color: #666;">(±{acc_data["std"]:.2%})</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.metric("Accuracy", f"{metrics_dict.get('accuracy', 0):.2%}")

        with col2:
            prec_data = metrics_dict.get("precision", {})
            if isinstance(prec_data, dict) and "mean" in prec_data:
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f8f9fa;">
                    <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Precision</div>
                    <div style="font-size: 18px; font-weight: bold; color: #262730;">{prec_data["mean"]:.2%}</div>
                    <div style="font-size: 12px; color: #666;">(±{prec_data["std"]:.2%})</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.metric("Precision", f"{metrics_dict.get('precision', 0):.2%}")

        with col3:
            rec_data = metrics_dict.get("recall", {})
            if isinstance(rec_data, dict) and "mean" in rec_data:
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f8f9fa;">
                    <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Recall</div>
                    <div style="font-size: 18px; font-weight: bold; color: #262730;">{rec_data["mean"]:.2%}</div>
                    <div style="font-size: 12px; color: #666;">(±{rec_data["std"]:.2%})</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.metric("Recall", f"{metrics_dict.get('recall', 0):.2%}")

        with col4:
            f1_data = metrics_dict.get("f1-score", {})
            if isinstance(f1_data, dict) and "mean" in f1_data:
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f8f9fa;">
                    <div style="font-size: 14px; color: #666; margin-bottom: 5px;">F1 Score</div>
                    <div style="font-size: 18px; font-weight: bold; color: #262730;">{f1_data["mean"]:.2%}</div>
                    <div style="font-size: 12px; color: #666;">(±{f1_data["std"]:.2%})</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.metric("F1 Score", f"{metrics_dict.get('f1-score', 0):.2%}")
        st.write("\n")  # Add an empty line for spacing in the app
    else:
        # Display individual run metrics as percentages
        with col1:
            st.metric("Accuracy", f"{metrics_dict.get('accuracy', 0):.2%}")
        with col2:
            st.metric("Precision", f"{metrics_dict.get('precision', 0):.2%}")
        with col3:
            st.metric("Recall", f"{metrics_dict.get('recall', 0):.2%}")
        with col4:
            st.metric("F1 Score", f"{metrics_dict.get('f1-score', 0):.2%}")


def display_uncertainty_metrics(metrics_dict, title="Uncertainty Metrics", show_std=False):
    """Display uncertainty-related metrics.

    Args:
        metrics_dict: Dictionary containing uncertainty metric values
        title: Title for the uncertainty metrics section
        show_std: If True, display as mean ± std format. If False, display as single values.
    """
    st.subheader(title)

    # Check which metrics are available and create appropriate columns
    available_metrics = []

    if "uncertainty_threshold" in metrics_dict:
        available_metrics.append(
            ("Uncertainty Threshold", f"{metrics_dict['uncertainty_threshold']:.2f}")
        )

    if "n_certain_images" in metrics_dict:
        if show_std and isinstance(metrics_dict["n_certain_images"], dict):
            certain_data = metrics_dict["n_certain_images"]
            available_metrics.append(
                ("Certain Images", f"{certain_data['mean']:.0f}", f"{certain_data['std']:.0f}")
            )
        else:
            available_metrics.append(("Certain Images", metrics_dict["n_certain_images"]))

    if "n_uncertain_images" in metrics_dict:
        if show_std and isinstance(metrics_dict["n_uncertain_images"], dict):
            uncertain_data = metrics_dict["n_uncertain_images"]
            available_metrics.append(
                (
                    "Uncertain Images",
                    f"{uncertain_data['mean']:.0f}",
                    f"{uncertain_data['std']:.0f}",
                )
            )
        else:
            available_metrics.append(("Uncertain Images", metrics_dict["n_uncertain_images"]))

    if "avg_confidence" in metrics_dict:
        if show_std and isinstance(metrics_dict["avg_confidence"], dict):
            conf_data = metrics_dict["avg_confidence"]
            available_metrics.append(
                (
                    "Avg Prediction Confidence",
                    f"{conf_data['mean']:.3f}",
                    f"{conf_data['std']:.3f}",
                )
            )
        else:
            available_metrics.append(
                ("Avg Prediction Confidence", f"{metrics_dict['avg_confidence']:.3f}")
            )
    if not available_metrics:
        st.write("No uncertainty metrics available for this result.")
        return

    # Create columns based on available metrics
    cols = st.columns(len(available_metrics))
    for i, metric_data in enumerate(available_metrics):
        with cols[i]:
            if len(metric_data) == 3:  # Has std deviation as separate value
                label, value, std = metric_data
                # Use HTML to display on multiple lines
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f8f9fa;">
                    <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
                    <div style="font-size: 18px; font-weight: bold; color: #262730;">{value}</div>
                    <div style="font-size: 12px; color: #666;">(±{std})</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:  # Single value (std already included in value string)
                label, value = metric_data
                st.metric(label, value)
    st.write("\n")  # Add an empty line for spacing in the app


def calculate_averaged_metrics(results):
    """Calculate averaged metrics and standard deviations across all evaluation runs.

    Args:
        results: List of (timestamp, result_dict) tuples from load_model_results

    Returns:
        Dictionary with averaged metrics and standard deviations
    """
    if not results:
        return None

    metric_keys = ["accuracy", "precision", "recall", "f1-score"]
    additional_numeric_keys = [
        "n_test_observations",
        "n_uncertain_images",
        "n_certain_images",
        "avg_confidence",
    ]

    # Store aggregated labels/predictions and per-run metric values per section.
    sections = defaultdict(
        lambda: {
            "labels_known": set(),
            "y_true": [],
            "y_pred": [],
            "per_run_metrics": defaultdict(list),
            "numeric_values": defaultdict(list),
            "boolean_values": defaultdict(list),
        }
    )

    def parse_confusion_key(key, known_labels):
        """Parse a confusion-matrix key back into true/pred labels."""
        for label in sorted(known_labels, key=len, reverse=True):
            prefix = f"{label}_"
            if key.startswith(prefix):
                return label, key[len(prefix) :]
        if "_" in key:
            true_label, pred_label = key.split("_", 1)
            return true_label, pred_label
        return None

    def expand_confusion_matrix(cm_dict, known_labels):
        """Expand confusion-matrix counts into repeated label arrays for sklearn metrics."""
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
        """Recompute weighted-averaged metrics using sklearn with zero_division safeguards."""
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

            run_metrics = compute_metrics(y_true_run, y_pred_run)
            for key in metric_keys:
                section_store["per_run_metrics"][key].append(run_metrics[key])

            for key in additional_numeric_keys:
                if key in section_data:
                    section_store["numeric_values"][key].append(section_data[key])

            if "excluded_uncertain_images" in section_data:
                section_store["boolean_values"]["excluded_uncertain_images"].append(
                    bool(section_data["excluded_uncertain_images"])
                )

    averaged_metrics = {}
    for section_name, section_store in sections.items():
        if not section_store["y_true"]:
            continue

        averaged_metrics[section_name] = {}
        combined_metrics = None
        if section_name == "overall":
            # We only aggregate via the combined confusion matrix for the overall summary.
            combined_metrics = compute_metrics(section_store["y_true"], section_store["y_pred"])

        for key in metric_keys:
            per_run_values = section_store["per_run_metrics"][key]
            if not per_run_values:
                continue
            mean = (
                combined_metrics[key]
                if combined_metrics is not None
                else float(np.mean(per_run_values))
            )
            std = float(np.std(per_run_values, ddof=1)) if len(per_run_values) > 1 else 0.0
            averaged_metrics[section_name][key] = {
                "mean": mean,
                "std": std,
                "count": len(per_run_values),
            }

        for key, values in section_store["numeric_values"].items():
            if not values:
                continue
            avg = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            averaged_metrics[section_name][key] = {"mean": avg, "std": std, "count": len(values)}

        for key, values in section_store["boolean_values"].items():
            if not values:
                continue
            avg = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            averaged_metrics[section_name][key] = {"mean": avg, "std": std, "count": len(values)}

    return averaged_metrics


def get_model_average_accuracy(model_name):
    """Calculate average overall accuracy for a model across all evaluation runs."""
    try:
        results = load_model_results(model_name)
        if not results:
            return -99

        # Extract accuracy values from all evaluation runs
        accuracies = []
        for _, result in results:
            if "overall" in result and "accuracy" in result["overall"]:
                accuracies.append(result["overall"]["accuracy"])

        # Return average accuracy or 0.0 if no valid results
        return sum(accuracies) / len(accuracies) if accuracies else -99
    except Exception as e:
        logging.error(f"Error calculating average accuracy for {model_name}: {e}")
        return -99


def get_models():
    """Get list of available models with evaluation results and their average accuracies."""
    model_dir = Path("models")
    models_with_accuracy = []

    # Recursively search for model directories with evaluation results
    for model_path in model_dir.rglob("*"):
        if model_path.is_dir() and any(
            file.name.endswith("__eval_results.json") for file in model_path.iterdir()
        ):
            # Get relative path from models directory
            relative_path = model_path.relative_to(model_dir)
            model_name = str(relative_path)
            avg_accuracy = get_model_average_accuracy(model_name)
            models_with_accuracy.append((model_name, avg_accuracy))

    # Sort by average accuracy (descending), putting invalid results at the end
    models_with_accuracy.sort(key=lambda x: (x[1] == -99, -x[1] if x[1] != -99 else 0))
    return models_with_accuracy


def load_model_results(model_name):
    """Load evaluation results for a specific model."""
    model_dir = Path("models") / model_name
    results_files = sorted(model_dir.glob("*__eval_results.json"), key=lambda x: x.name)
    print(results_files)
    results = []
    for results_file in results_files:
        with open(results_file) as f:
            result = json.load(f)
        # Convert timestamp to human readable format
        timestamp = results_file.name.replace("__eval_results.json", "")
        formatted_date = datetime.strptime(timestamp, "%Y%m%dT%H%M%S").strftime(
            "%B %d, %Y at %I:%M:%S"
        )
        results.append((formatted_date, result))
    return results


def load_training_specs(model_name):
    """Load training specifications for a specific model."""
    model_dir = Path("models") / model_name
    specs_file = model_dir / "tuning_specs.json"
    with open(specs_file) as f:
        return json.load(f)


def render_results(model):
    """Render evaluation results for a specific model."""
    try:
        results = load_model_results(model)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        logging.error(f"Error loading results: {e}")
        return

    if not results:
        st.warning("No evaluation results found for this model.")
        return

    # Always show averaged results first
    render_averaged_results(results, model)

    # Then show individual runs as collapsed sections
    st.subheader("Individual Evaluation Runs")
    render_individual_results(results)


def render_averaged_results(results, model):
    """Render averaged results across all evaluation runs."""
    # Calculate averaged metrics
    averaged_metrics = calculate_averaged_metrics(results)

    if not averaged_metrics:
        st.warning("Unable to calculate averaged metrics.")
        return

    # Display summary information
    st.write("#### Summary")
    n_runs = len(results)
    st.write(f"Results averaged across **{n_runs}** evaluation runs")

    # Display averaged overall metrics
    if "overall" in averaged_metrics:
        overall_data = averaged_metrics["overall"]

        # Class distribution info (use averaged values)
        st.write("#### Class Distribution")
        if "n_test_observations" in overall_data:
            n_obs_data = overall_data["n_test_observations"]
            if isinstance(n_obs_data, dict) and "mean" in n_obs_data:
                st.write(
                    f"Test set contained {n_obs_data['mean']:.0f} ± {n_obs_data['std']:.0f} observations."
                )
            else:
                st.write(f"Test set contained {n_obs_data} observations.")
        else:
            # Fallback to first run if averaged value not available
            first_result = results[0][1]["overall"]
            st.write(f"Test set contained {first_result['n_test_observations']} observations.")

        # Excluded uncertain images (use first run as representative since it's boolean)
        first_result = results[0][1]["overall"]
        st.write(f"Excluded uncertain images: {first_result['excluded_uncertain_images']}")
        st.info(
            "💡 Traning class distribution varies. Use 'Individual Evaluation Runs' view to see the class distribution for each run."
        )

        # Display averaged uncertainty metrics
        display_uncertainty_metrics(overall_data, "Uncertainty Analysis", show_std=True)

        # Display averaged performance metrics
        display_metrics(overall_data, "Overall Performance", show_std=True)
        # Note about confusion matrix
        st.info(
            "💡 **Note:** Confusion matrices are not summarized. Use 'Individual Evaluation Runs' view to see confusion matrices for each run."
        )

        # Display stratified results (averaged)
        st.subheader("Stratified Results")
        for stratum, metrics in averaged_metrics.items():
            if stratum != "overall":
                st.write(f"### {stratum}")
                display_metrics(metrics, show_std=True)
                display_uncertainty_metrics(metrics, "Stratum Uncertainty Analysis", show_std=True)


def render_individual_results(results):
    """Render individual evaluation runs."""
    for eval_run, result in results:
        with st.expander(f"**{eval_run}**"):
            st.write("#### Class Distribution")
            st.write(
                f"Test set contained {result['overall']['n_test_observations']} observations."
            )
            st.write(
                f"Excluded uncertain images: {result['overall']['excluded_uncertain_images']}"
            )
            st.write(f"Number of uncertain images: {result['overall']['n_uncertain_images']}")
            st.write(
                f"Avg prediction confidence of included images: {result['overall']['avg_confidence']:.3f}"
            )

            class_dist = result["overall"]["class_distribution"]
            fig = px.treemap(
                names=list(class_dist.keys()),
                parents=["Test Data"] * len(class_dist),
                values=list(class_dist.values()),
                title="Test Data Class Distribution",
            )
            fig.update_layout(width=800, height=400)
            st.plotly_chart(fig, key=f"{eval_run.replace(' ', '_')}__class_distribution")

            # Display overall metrics
            display_metrics(result["overall"], "Overall Performance")

            # Display uncertainty metrics
            display_uncertainty_metrics(result["overall"], "Uncertainty Analysis")

            # Display overall confusion matrix
            st.subheader("Overall Confusion Matrix")
            st.pyplot(plot_confusion_matrix(result["overall"]["confusion_matrix"]))

            # Display stratified results
            st.subheader("Stratified Results")
            for stratum, metrics in result.items():
                if stratum != "overall":
                    st.write(f"### {stratum}")
                    display_metrics(metrics)
                    # Display uncertainty metrics for stratified results
                    display_uncertainty_metrics(metrics, "Stratum Uncertainty Analysis")
                    st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]))


def render_training_specs(model):
    """Render training specifications for a specific model."""
    try:
        specs = load_training_specs(model)

        # Display training parameters
        st.subheader("Training Parameters")
        # Display class distribution
        st.write("#### Class Distribution")
        st.write(f"Training set contained {specs['n_train_observations']} observations.")
        class_dist = specs["class_distribution"]
        fig = px.treemap(
            names=list(class_dist.keys()),
            parents=["Training Data"] * len(class_dist),
            values=list(class_dist.values()),
            title="Training Data Class Distribution",
        )
        fig.update_layout(width=800, height=400)
        st.plotly_chart(fig, key=f"{model.replace(' ', '_')}__training_class_distribution")
        params = specs["training_params"]

        # Transfer learning phase
        st.write("#### Transfer Learning Phase")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Epochs", params["transfer_epochs"])
            st.metric("Patience", params["transfer_patience"])
        with col2:
            st.metric("Optimizer", params["transfer_optimizer"]["name"])
            st.metric("Learning Rate", f"{params['transfer_optimizer']['learning_rate']:.4f}")

        # Fine-tuning phase
        st.write("#### Fine-tuning Phase")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Epochs", params["finetune_epochs"])
            st.metric("Patience", params["finetune_patience"])
            st.metric("Fine-tuned Layers", params["finetune_layers"])
        with col2:
            st.metric("Optimizer", params["finetune_optimizer"]["name"])
            st.metric("Learning Rate", f"{params['finetune_optimizer']['learning_rate']:.4f}")

        # Other parameters
        st.write("#### Other Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Loss Function", params["loss_function"])
            st.metric("Batch Size", params["batch_size"])
        with col2:
            st.metric("Early Stop Metric", params["earlystop_metric"])

        # Model architecture
        st.subheader("Model Architecture")
        st.code(specs["model_summary"], language="text")

    except Exception as e:
        st.error(f"Error loading training specifications: {e}")
        logging.error(f"Error loading training specifications: {e}")


def render_results_page():
    """Model results visualization."""
    st.title("Model Results")

    # Model selection
    models = get_models()
    if not models:
        st.warning("No models with evaluation results found.")
        return

    # Create formatted options with average accuracy
    model_options = []
    model_names = []
    for name, accuracy in models:
        if accuracy == -99:
            # Handle case where no valid results found
            model_options.append(f"{name} (No valid results)")
        else:
            model_options.append(f"{name} (Avg Accuracy: {accuracy:.2%})")
        model_names.append(name)

    selected_option = st.selectbox("Select model", options=model_options)
    selected_model = model_names[model_options.index(selected_option)]

    if st.button("View Results"):
        st.divider()
        st.write("### Evaluation Runs")
        render_results(selected_model)

        # add collapsible section for training specs
        st.subheader("Training Specifications")
        with st.expander("Training Specs"):
            render_training_specs(selected_model)
