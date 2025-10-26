import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime


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
                ("Avg Confidence", f"{conf_data['mean']:.3f}", f"{conf_data['std']:.3f}")
            )
        else:
            available_metrics.append(("Avg Confidence", f"{metrics_dict['avg_confidence']:.3f}"))
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

    # Collect all metrics across runs
    all_metrics = {
        "overall": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1-score": [],
            "n_test_observations": [],
            "excluded_uncertain_images": [],
            "n_uncertain_images": [],
            "avg_confidence": [],
        }
    }

    # Collect stratified metrics
    stratified_keys = set()
    for _, result in results:
        for key in result.keys():
            if key != "overall":
                stratified_keys.add(key)

    # Initialize stratified metrics collection
    for key in stratified_keys:
        all_metrics[key] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1-score": [],
            "n_test_observations": [],
            "excluded_uncertain_images": [],
            "n_uncertain_images": [],
            "avg_confidence": [],
        }

    # Collect metrics from each run
    for _, result in results:
        # Overall metrics
        if "overall" in result:
            for metric in all_metrics["overall"].keys():
                if metric in result["overall"]:
                    all_metrics["overall"][metric].append(result["overall"][metric])

        # Stratified metrics
        for key in stratified_keys:
            if key in result:
                for metric in all_metrics[key].keys():
                    if metric in result[key]:
                        all_metrics[key][metric].append(result[key][metric])

    # Calculate averages and standard deviations
    averaged_metrics = {}
    for section, metrics in all_metrics.items():
        averaged_metrics[section] = {}
        for metric, values in metrics.items():
            if values:  # Only calculate if we have values
                avg = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0.0
                averaged_metrics[section][metric] = {"mean": avg, "std": std, "count": len(values)}

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

        # Class distribution info (use first run as representative)
        first_result = results[0][1]["overall"]
        st.write("#### Class Distribution")
        st.write(f"Test set contained {first_result['n_test_observations']} observations.")
        st.write(f"Excluded uncertain images: {first_result['excluded_uncertain_images']}")
        st.info(
            "💡 Traning class distribution varies. Use 'Individual Runs' view to see the class distribution for each run."
        )

        # Display averaged uncertainty metrics
        display_uncertainty_metrics(overall_data, "Uncertainty Analysis", show_std=True)

        # Display averaged performance metrics
        display_metrics(overall_data, "Overall Performance", show_std=True)
        # Note about confusion matrix
        st.info(
            "💡 **Note:** Confusion matrices are not averaged. Use 'Individual Runs' view to see confusion matrices for each evaluation run."
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
                f"Avg confidence of included images: {result['overall']['avg_confidence']:.3f}"
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
