import json
import logging
from datetime import datetime
from pathlib import Path

import plotly.express as px
import streamlit as st

from .results_metrics import (
    calculate_averaged_metrics,
    display_metrics,
    display_uncertainty_metrics,
    plot_confusion_matrix,
    summarize_overall_runs,
)


def _labels_from_section(section_data, fallback_labels=None):
    """Return label order based on class distributions."""
    if not section_data:
        return fallback_labels
    distribution = section_data.get("aggregated_class_distribution") or section_data.get(
        "class_distribution"
    )
    if distribution:
        return list(distribution.keys())
    return fallback_labels


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
    summary_stats = summarize_overall_runs(results)
    st.write(f"Results averaged across **{summary_stats['n_runs']}** evaluation runs")

    # Display averaged overall metrics
    if "overall" in averaged_metrics:
        overall_data = averaged_metrics["overall"]

        st.write("#### Dataset Overview")

        total_test = summary_stats["total_n_test_observations"]
        total_uncertain = summary_stats["total_n_uncertain_images"]
        total_images = summary_stats["total_images_reviewed"]

        overview_lines = [
            f"- **{summary_stats['n_runs']}** evaluation runs with identical specs.",
            (
                f"- Ran **{total_images:,}** test predictions "
                f"({total_test:,} used for scoring + {total_uncertain:,} uncertain excluded)."
            ),
        ]

        per_run = summary_stats["per_run_n_test_consistent"]
        if per_run is not None:
            overview_lines.append(f"- **{per_run:,}** test observations per run.")

        per_run_unc = summary_stats["per_run_n_uncertain_consistent"]
        if per_run_unc is not None and total_uncertain:
            overview_lines.append(f"- **{per_run_unc:,}** uncertain images flagged per run.")

        if summary_stats["weighted_avg_confidence"] is not None:
            overview_lines.append(
                f"- Average prediction confidence across certain predictions: "
                f"**{summary_stats['weighted_avg_confidence']:.3f}**."
            )
        if overall_data.get("uncertainty_threshold") is not None:
            overview_lines.append(
                "- Uncertainty threshold applied before scoring: "
                f"**{float(overall_data['uncertainty_threshold']):.2f}**."
            )

        if summary_stats["excluded_uncertain_consistent"]:
            if summary_stats["excluded_uncertain_value"]:
                overview_lines.append("- Uncertain images were excluded before scoring.")
            else:
                overview_lines.append("- Uncertain images remained in the scored dataset.")
        else:
            overview_lines.append("- Handling of uncertain images varied between runs.")

        st.markdown("\n".join(overview_lines))

        with st.expander("Class Distribution & Confusion Matrix"):
            aggregated_distribution = overall_data.get("aggregated_class_distribution")
            if aggregated_distribution:
                total_count = sum(aggregated_distribution.values())
                names = ["Aggregated Test Data"]
                parents = [""]
                values = [total_count]
                for label, count in sorted(aggregated_distribution.items()):
                    names.append(label)
                    parents.append("Aggregated Test Data")
                    values.append(count)
                fig = px.treemap(
                    names=names,
                    parents=parents,
                    values=values,
                    title="Aggregated Test Data Class Distribution",
                    branchvalues="total",
                )
                fig.update_layout(width=800, height=400)
                st.plotly_chart(fig, key="aggregated_test_class_distribution")

            aggregated_cm = overall_data.get("aggregated_confusion_matrix")
            if aggregated_cm:
                aggregated_labels = _labels_from_section(overall_data)
                st.pyplot(
                    plot_confusion_matrix(
                        aggregated_cm,
                        labels=aggregated_labels,
                        title="Aggregated Confusion Matrix",
                    )
                )

        # Display aggregated uncertainty metrics
        display_uncertainty_metrics(overall_data, "Uncertainty Analysis")

        # Display aggregated performance metrics
        display_metrics(overall_data, "Overall Performance")

        # Display stratified results (averaged)
        st.subheader("Stratified Results")
        for stratum, metrics in averaged_metrics.items():
            if stratum != "overall":
                st.write(f"### {stratum}")
                display_metrics(metrics)
                display_uncertainty_metrics(metrics, "Stratum Uncertainty Analysis")


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
                "Avg prediction confidence of included images: "
                f"{result['overall']['avg_confidence']:.3f}"
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
            overall_labels = _labels_from_section(result["overall"])
            st.pyplot(
                plot_confusion_matrix(
                    result["overall"]["confusion_matrix"],
                    labels=overall_labels,
                    title="Overall Confusion Matrix",
                )
            )

            # Display stratified results
            st.subheader("Stratified Results")
            for stratum, metrics in result.items():
                if stratum != "overall":
                    st.write(f"### {stratum}")
                    display_metrics(metrics)
                    # Display uncertainty metrics for stratified results
                    display_uncertainty_metrics(metrics, "Stratum Uncertainty Analysis")
                    stratum_labels = _labels_from_section(metrics, fallback_labels=overall_labels)
                    st.pyplot(
                        plot_confusion_matrix(
                            metrics["confusion_matrix"],
                            labels=stratum_labels,
                            title=f"{stratum} Confusion Matrix",
                        )
                    )


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
