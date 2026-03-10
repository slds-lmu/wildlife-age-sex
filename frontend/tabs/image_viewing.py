import streamlit as st
import os
from pathlib import Path
import logging
import json
import plotly.express as px
import glob
import pandas as pd
from wildlifeml.io import load
from datetime import datetime


def get_experiment_average_accuracy(experiment_name):
    """Calculate average overall accuracy for an experiment across all evaluation runs."""
    try:
        experiment_dir = Path("models") / experiment_name
        eval_files = glob.glob(str(experiment_dir / "*__eval_results.json"))

        if not eval_files:
            return -99

        # Extract accuracy values from all evaluation runs
        accuracies = []
        for eval_file in eval_files:
            with open(eval_file, "r") as f:
                results = json.load(f)
                if "overall" in results and "accuracy" in results["overall"]:
                    accuracies.append(results["overall"]["accuracy"])

        # Return average accuracy or -99 if no valid results
        return sum(accuracies) / len(accuracies) if accuracies else -99
    except Exception as e:
        logging.error(f"Error calculating average accuracy for {experiment_name}: {e}")
        return -99


def get_experiments():
    """
    Scan the models/ directory for experiments containing .pt files.
    Returns list of tuples (experiment_name, avg_accuracy) sorted by accuracy.
    """
    models_dir = Path("models")
    experiments_with_accuracy = []

    # Check if models directory exists
    if not models_dir.exists():
        st.error("Models directory not found!")
        return []

    # Walk through all subdirectories in models/
    for root, dirs, files in os.walk(models_dir):
        # Check if any .pt files exist in current directory
        if any(file.endswith(".pt") for file in files):
            # Get relative path from models/ directory
            relative_path = Path(root).relative_to(models_dir)
            experiment_name = str(relative_path)
            avg_accuracy = get_experiment_average_accuracy(experiment_name)
            experiments_with_accuracy.append((experiment_name, avg_accuracy))

    # Sort by average accuracy (descending), putting invalid results at the end
    experiments_with_accuracy.sort(key=lambda x: (x[1] == -99, -x[1] if x[1] != -99 else 0))
    return experiments_with_accuracy


def render_results_summary(experiment_name, additional_metrics=None):
    """Render summary cards for all evaluation results in the experiment directory.

    Args:
        experiment_name: Name of the experiment
        additional_metrics: List of additional metric keys to display in summary cards
    """

    # Get the experiment directory path
    experiment_dir = Path("models") / experiment_name

    # Find all evaluation result files
    eval_files = glob.glob(str(experiment_dir / "*__eval_results.json"))

    if not eval_files:
        st.info("No evaluation results found for this experiment.")
        return

    st.subheader("Evaluation Results Summary")

    # Sort files by timestamp (newest first)
    eval_files.sort(reverse=True)

    # Create columns for the summary cards
    cols = st.columns(min(3, len(eval_files)))

    for i, eval_file in enumerate(eval_files):
        col_idx = i % 3
        with cols[col_idx]:
            try:
                # Load evaluation results
                with open(eval_file, "r") as f:
                    results = json.load(f)

                # Extract timestamp from filename
                filename = Path(eval_file).name
                # Extract and format timestamp as YYMMDDTHHMMSS
                from datetime import datetime

                raw_timestamp = filename.split("__")[0]
                try:
                    dt = datetime.strptime(raw_timestamp, "%Y%m%dT%H%M%S")
                    timestamp = dt.strftime("%B %d, %Y at %I:%M:%S")
                except Exception:
                    timestamp = raw_timestamp  # fallback if format is unexpected

                # Get overall metrics
                overall = results.get("overall", {})

                # Create summary card
                with st.container():
                    st.write(f"**Timestamp:** {timestamp}")
                    st.write(
                        "Accuracy",
                        f"{overall.get('accuracy', 'N/A'):.3f}"
                        if overall.get("accuracy") is not None
                        else "N/A",
                    )
                    st.write(
                        "F1-Score",
                        f"{overall.get('f1-score', 'N/A'):.3f}"
                        if overall.get("f1-score") is not None
                        else "N/A",
                    )
                    st.write(
                        "Precision",
                        f"{overall.get('precision', 'N/A'):.3f}"
                        if overall.get("precision") is not None
                        else "N/A",
                    )
                    st.write(
                        "Recall",
                        f"{overall.get('recall', 'N/A'):.3f}"
                        if overall.get("recall") is not None
                        else "N/A",
                    )
                    st.write(f"**Test Samples:** {overall.get('n_test_observations', 'N/A')}")
                    st.write(
                        f"**Excluded Uncertain Images:** {overall.get('excluded_uncertain_images', 'N/A')}"
                    )
                    st.write(
                        f"**Number of Uncertain Images:** {overall.get('n_uncertain_images', 'N/A')}"
                    )
                    st.write(
                        f"**Avg Prediction Confidence of Included Images:** {overall.get('avg_confidence', 'N/A'):.3f}"
                    )
            except Exception as e:
                st.error(f"Error loading evaluation results: {e}")
                logging.error(f"Error loading evaluation results: {e}")


def render_image_slideshow(
    experiment_name,
    file_pattern,
    slideshow_title,
    session_state_key,
    additional_metadata=None,
    confidence_column=None,
):
    """Render images in a simple infinite slideshow.

    Args:
        experiment_name: Name of the experiment
        file_pattern: Glob pattern to find image files (e.g., "*__eval_errors.parquet")
        slideshow_title: Title for the slideshow section
        session_state_key: Key for session state to track current index
        additional_metadata: List of additional metadata columns to display
        confidence_column: Column name for confidence scores (if applicable)
    """

    # Get the experiment directory path
    experiment_dir = Path("models") / experiment_name

    # Find all image files
    image_files = glob.glob(str(experiment_dir / file_pattern))

    if not image_files:
        st.info(f"No {slideshow_title.lower()} files found for this experiment.")
        return

    st.subheader(f"{slideshow_title} - Slideshow")

    try:
        # Load all image data and combine
        all_image_data = []
        for image_file in image_files:
            data = load(image_file)
            all_image_data.append(data)

        if not all_image_data:
            st.info(f"No {slideshow_title.lower()} found in any files.")
            return

        # Combine all data and deduplicate by filename (not full path)
        combined_data = pd.concat(all_image_data, ignore_index=True)
        # Extract filename from image_path for deduplication
        combined_data["filename"] = combined_data["image_path"].apply(
            lambda x: os.path.basename(x) if pd.notna(x) else ""
        )
        combined_data = combined_data.drop_duplicates(subset=["filename"], keep="first")
        # Remove the temporary filename column
        combined_data = combined_data.drop(columns=["filename"])

        if len(combined_data) == 0:
            st.info(f"No unique {slideshow_title.lower()} found after deduplication.")
            return

        # Filter out rows with missing images to avoid errors during viewing
        original_count = len(combined_data)
        combined_data = combined_data[combined_data["image"].notna()].copy()
        filtered_count = len(combined_data)

        if filtered_count == 0:
            st.info(f"No {slideshow_title.lower()} with valid images found.")
            return

        target_column = get_target_column(experiment_dir)

        # Initialize current index in session state
        if session_state_key not in st.session_state:
            st.session_state[session_state_key] = 0

        # Ensure index is within bounds after filtering
        if st.session_state[session_state_key] >= len(combined_data):
            st.session_state[session_state_key] = 0

        # Progress indicator
        st.write(f"**Image {st.session_state[session_state_key] + 1} of {len(combined_data)}**")
        progress = (st.session_state[session_state_key] + 1) / len(combined_data)
        st.progress(progress)

        # Display current image (all images are guaranteed to exist after filtering)
        current_row = combined_data.iloc[st.session_state[session_state_key]]

        # Display image
        st.image(current_row["image"], caption=f"Image ID: {current_row['image_id']}", width=600)
        render_slide_controls(combined_data, session_state_key)

        # Display metadata in columns
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**True {target_column.title()}:** {current_row.get(target_column)}")
            st.write(
                f"**Predicted {target_column.title()}:** {current_row.get('predicted_label')}"
            )
            if confidence_column and confidence_column in current_row:
                st.write(f"**Confidence:** {current_row.get(confidence_column, 'N/A'):.3f}")

        with col2:
            if "DateTime" in current_row:
                date_str = (
                    current_row["DateTime"].strftime("%Y-%m-%d")
                    if pd.notna(current_row["DateTime"])
                    else "N/A"
                )
                st.write(f"**Date:** {date_str}")
            if "is_summer" in current_row:
                season = "Summer" if current_row["is_summer"] else "Winter"
                st.write(f"**Season:** {season}")

            # Display additional metadata if provided
            if additional_metadata:
                for metadata_col in additional_metadata:
                    if metadata_col in current_row:
                        st.write(
                            f"**{metadata_col.replace('_', ' ').title()}:** {current_row[metadata_col]}"
                        )

    except Exception as e:
        st.error(f"Error loading image data: {e}")
        logging.error(f"Error loading image data: {e}")


def get_target_column(experiment_dir):
    # Take first eval_results file, all should be the same target
    eval_file = Path(glob.glob(str(experiment_dir / "*__eval_results.json"))[0])
    if eval_file.exists():
        with open(eval_file) as f:
            eval_results = json.load(f)
        target_column = eval_results["overall"].get("target_column")
    else:
        st.error("Could not find eval_results file to determine target column.")
        return
    return target_column


def render_slide_controls(combined_data, session_state_key):
    # Navigation controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("⏮️ First"):
            st.session_state[session_state_key] = 0
            st.rerun()

    with col2:
        if st.button("⬅️ Previous"):
            st.session_state[session_state_key] = max(0, st.session_state[session_state_key] - 1)
            st.rerun()

    with col3:
        if st.button("Next ➡️"):
            st.session_state[session_state_key] = min(
                len(combined_data) - 1, st.session_state[session_state_key] + 1
            )
            st.rerun()

    with col4:
        if st.button("⏭️ Last"):
            st.session_state[session_state_key] = len(combined_data) - 1
            st.rerun()


def load_training_specs(model_name):
    """Load training specifications for a specific model."""
    model_dir = Path("models") / model_name
    specs_file = model_dir / "tuning_specs.json"
    with open(specs_file) as f:
        return json.load(f)


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
