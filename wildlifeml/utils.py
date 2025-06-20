import numpy as np
import torch
import torch.nn as nn


def convert_to_numeric_indices(
    targets: np.ndarray, classes: list[str] | None = None
) -> tuple[dict, np.ndarray]:
    """
    Converts string class labels to numeric indices and one-hot encodes them using PyTorch.

    Args:
        targets: Array of string class labels.
        classes: Optional list of class names. If None, inferred from targets.

    Returns:
        category_to_idx: Mapping from class name to index.
        labels: One-hot encoded labels as a NumPy array.
    """
    if classes is None:
        classes = np.unique(targets).tolist()

    unique_targets = set(np.unique(targets))
    if not unique_targets.issubset(set(classes)):
        extra_classes = unique_targets - set(classes)
        raise ValueError(f"Found additional classes not in provided list: {extra_classes}")

    category_to_idx = {cat: idx for idx, cat in enumerate(classes)}
    # Convert string labels to numeric indices
    numeric_labels = np.vectorize(category_to_idx.get)(targets)
    # One-hot encode using PyTorch
    labels = torch.nn.functional.one_hot(
        torch.tensor(numeric_labels), num_classes=len(classes)
    ).numpy()
    return category_to_idx, labels


def get_model_summary(model: nn.Module) -> str:
    """
    Get a formatted model summary for a PyTorch model.

    Args:
        model: PyTorch nn.Module to summarize.

    Returns:
        String containing model summary.
    """
    # Use torchinfo if available for a detailed summary, else fallback to str(model)
    try:
        from torchinfo import summary

        # You may need to specify input_size for your model
        return str(summary(model))
    except ImportError:
        # Fallback: just return the model architecture as a string
        return str(model)
