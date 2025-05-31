import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import logging
from pathlib import Path


def plot_confusion_matrix(cm_dict, labels=None):
    """Create an interactive confusion matrix plot from dictionary format."""
    # Convert dictionary to matrix
    n = int(np.sqrt(len(cm_dict)))
    cm = np.zeros((n, n), dtype=int)
    for key, value in cm_dict.items():
        i, j = map(int, key.split("_"))
        cm[i, j] = value

    # Create labels if not provided
    if labels is None:
        labels = [f"Class {i}" for i in range(n)]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        width=600,
        height=500,
    )
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
    results_file = next(model_dir.glob("*__eval_results.json"))
    with open(results_file) as f:
        return json.load(f)


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

        # Display overall metrics
        display_metrics(results["overall"], "Overall Performance")

        # Display overall confusion matrix
        st.subheader("Overall Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(results["overall"]["confusion_matrix"]))

        # Display stratified results
        st.subheader("Stratified Results")
        metric = st.selectbox(
            "Select metric to display", ["accuracy", "precision", "recall", "f1-score"]
        )
        st.plotly_chart(plot_stratified_results(results, metric))

        # Display detailed stratified metrics
        st.subheader("Detailed Stratified Metrics")
        for stratum, metrics in results.items():
            if stratum != "overall":
                st.write(f"### {stratum}")
                display_metrics(metrics)
                st.plotly_chart(plot_confusion_matrix(metrics["confusion_matrix"]))

    except Exception as e:
        st.error(f"Error loading results: {e}")
        logging.error(f"Error loading results: {e}")


def render_training_specs(model):
    """Render training specifications for a specific model."""
    try:
        specs = load_training_specs(model)

        # Display training parameters
        st.subheader("Training Parameters")
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
        render_results(selected_model)

        # add collapsible section for training specs
        with st.expander("Training Specs"):
            render_training_specs(selected_model)
