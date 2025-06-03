import numpy as np
import tensorflow as tf


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
    return category_to_idx, labels


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
