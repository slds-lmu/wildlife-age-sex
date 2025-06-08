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


def display_metrics(metrics_dict, title="Metrics"):
    """Display metrics in a row of columns."""
    st.subheader(title)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics_dict['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics_dict['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics_dict['recall']:.2%}")
    with col4:
        st.metric("F1 Score", f"{metrics_dict['f1-score']:.2%}")


def get_models():
    """Get list of available models with evaluation results."""
    model_dir = Path("models")
    return [
        model.name
        for model in model_dir.iterdir()
        if model.is_dir()
        and any(file.name.endswith("__eval_results.json") for file in model.iterdir())
    ]


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

    for eval_run, result in results:
        with st.expander(f"**{eval_run}**"):
            st.write("#### Class Distribution")
            st.write(
                f"Test set contained {result['overall']['n_test_observations']} observations."
            )
            class_dist = result["overall"]["class_distribution"]
            fig = px.treemap(
                names=list(class_dist.keys()),
                parents=["Test Data"] * len(class_dist),
                values=list(class_dist.values()),
                title="Test Data Class Distribution",
            )
            fig.update_layout(width=800, height=400)
            st.plotly_chart(fig)
            # Display overall metrics
            display_metrics(result["overall"], "Overall Performance")

            # Display overall confusion matrix
            st.subheader("Overall Confusion Matrix")
            st.pyplot(plot_confusion_matrix(result["overall"]["confusion_matrix"]))

            # Display stratified results
            st.subheader("Stratified Results")
            st.write("Select a metric to display")
            metric = st.selectbox(
                label=f"{eval_run.replace(' ', '_')}__metric",
                options=["accuracy", "precision", "recall", "f1-score"],
                label_visibility="hidden",
            )
            st.plotly_chart(
                plot_stratified_results(result, metric),
                key=f"{eval_run.replace(' ', '_')}__stratified_results",
            )

            # Display detailed stratified metrics
            st.subheader("Detailed Stratified Metrics")
            for stratum, metrics in result.items():
                if stratum != "overall":
                    st.write(f"### {stratum}")
                    display_metrics(metrics)
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
        st.plotly_chart(fig)
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

    selected_model = st.selectbox("Select model", options=models)

    if st.button("View Results"):
        st.divider()
        st.write("### Evaluation Runs")
        render_results(selected_model)

        # add collapsible section for training specs
        with st.expander("Training Specs"):
            render_training_specs(selected_model)
