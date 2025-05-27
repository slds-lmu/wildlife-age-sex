import argparse
import logging
from datetime import datetime
from typing import Literal
import numpy as np
import tensorflow as tf


def get_session_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working-dir",
        type=str,
        help="Path to the working directory containing data and results directories.",
    )
    parser.add_argument(
        "--log-level",
        type=Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--use-log-file",
        type=bool,
        default=True,
        help="""
        Whether to use a log file. If True, a log file will be created in the working directory.
        If False, logs will be printed to the console.
        """,
    )
    args = parser.parse_args()

    return {
        "level": getattr(logging, args.log_level.upper()),
        "filename": args.working_dir
        / "logs"
        / f"wildlifeml_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"
        if args.use_log_file
        else None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "working_dir": args.working_dir,
    }


def convert_to_numeric_indices(
    targets: np.ndarray, classes: list[str] | None = None
) -> np.ndarray:
    if classes is None:
        classes = targets.unique()

    # Check for additional classes not in the provided list
    unique_targets = set(targets.unique())
    if not unique_targets.issubset(set(classes)):
        extra_classes = unique_targets - set(classes)
        raise ValueError(f"Found additional classes not in provided list: {extra_classes}")

    category_to_idx = {cat: idx for idx, cat in enumerate(classes)}  # Zero-based indices
    # Convert string labels to numeric indices
    numeric_labels = targets.map(category_to_idx).values
    # One-hot encode the numeric labels
    labels = tf.keras.utils.to_categorical(numeric_labels)
    return labels


def get_model_summary(model: tf.keras.Model) -> str:
    """Get a formatted model summary suitable for JSON storage.

    Args:
        model: Keras model to summarize

    Returns:
        String containing model summary with newlines preserved and special characters removed
    """
    # Capture model summary output
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return "\n".join(summary_list)
