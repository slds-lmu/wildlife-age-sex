from torch.nn import Module
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import logging
from torch.utils.data import DataLoader, TensorDataset
# Age: 4 (yearling, juvenile, adult, unknown)
# Sex: 3 (male, female, unknown)

# Confusion matrix
# Precision, recall, f1-score, acc

# For each of the time periods (winter, summer):
## summer - anlters, white spots for the yearlings

# Concept bottlenecks
# Anlters, white spots, long faces (adult females)
# If not features, then do standard classification
# Feed image AND features into final prediction task
# Orthogonalize concepts from images after extraction


def evaluate_model(
    model: Module,
    test_data: pd.DataFrame,
    target_column: str,
    classes: list[str],
    stratify_by: str | None = None,
) -> dict:
    """Evaluate a model on a test dataset. Optionally stratify
    evaluation by the provided metadatacolumn.

    Evaluation metrics:
    - Precision, recall, f1-score, accuracy
    - Confusion matrix
    - ROC curve
    - PR curve
    - Confusion matrix

    Args:
        model: The model to evaluate.
        test_data: The test dataset.
        target_column: The column to evaluate the model on.
        stratify_by: The column to stratify the evaluation by.
    Returns:
        A dictionary containing the evaluation results.
    """
    logging.info(f"Starting model evaluation with {len(test_data)} test samples")
    logging.info(f"Target column: {target_column}, Stratify by: {stratify_by}")

    if stratify_by and stratify_by not in test_data.columns:
        raise ValueError(f"Stratify by column {stratify_by} not found in test data.")

    num_missing_images = sum([img is None for img in test_data["image"]])
    if num_missing_images > 0:
        test_data = test_data[test_data["image"].notna()]
        logging.warning(
            f"Found {num_missing_images} missing images. Continuing with {len(test_data)} images."
        )
    results = {}

    # Calculate overall metrics

    # Convert images to tensor
    inputs = torch.stack([torch.from_numpy(img).float() for img in test_data["image"].values])
    inputs = inputs.permute(0, 3, 1, 2)  # Convert from (N, H, W, C) to (N, C, H, W)

    category_to_idx = {cat: idx for idx, cat in enumerate(classes)}
    idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}

    labels = test_data[target_column].map(category_to_idx).values
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    test_dataset = TensorDataset(inputs, labels_tensor)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )  # adjust batch_size as needed

    logging.info("Generating predictions...")
    predictions = _get_predictions(model, test_loader)
    # Use a boolean mask as a Series aligned to test_data's index to select error rows
    error_mask = [l != p for l, p in zip(labels, predictions)]
    errors = test_data[error_mask].copy()
    errors["predicted_label"] = [idx_to_category[p] for p in predictions[error_mask]]

    logging.info("Calculating overall metrics...")
    results["overall"] = _get_metrics(labels, predictions, idx_to_category, target_column)
    logging.info(f"Overall accuracy: {results['overall']['accuracy']:.3f}")

    # Calculate stratified metrics if requested
    if stratify_by is not None:
        logging.info(f"Calculating stratified metrics by {stratify_by}...")
        for val in test_data[stratify_by].unique():
            subset_labels = labels[test_data[stratify_by] == val]
            subset_predictions = predictions[test_data[stratify_by] == val]
            logging.debug(f"Stratum {val}: {len(subset_labels)} samples")
            stratum_key = f"{stratify_by}: {val}"
            results[stratum_key] = _get_metrics(
                subset_labels, subset_predictions, idx_to_category, target_column
            )
            logging.info(f"{val} accuracy: {results[stratum_key]['accuracy']:.3f}")

    return results, errors


def _get_predictions(model: Module, data_loader: DataLoader) -> np.ndarray:
    """Helper function to get model predictions for a dataset using a DataLoader."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds.append(probs.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def _get_metrics(
    labels: np.ndarray, predictions: np.ndarray, idx_to_category: dict, target_column: str
) -> dict:
    """Helper function to calculate all evaluation metrics."""
    # Convert confusion matrix to dict with original labels as keys
    cm = confusion_matrix(labels, predictions)
    logging.debug(f"Confusion matrix shape: {cm.shape}")

    # Convert confusion matrix to dict with original labels
    cm_dict = {
        f"{idx_to_category[i]}_{idx_to_category[j]}": int(
            cm[i, j]
        )  # Convert to int for JSON serialization
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
    }

    class_distribution = pd.Series(labels).map(idx_to_category).value_counts().to_dict()

    metrics = {
        "target_column": target_column,
        "n_test_observations": len(labels),
        "class_distribution": class_distribution,
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1-score": f1_score(labels, predictions, average="weighted", zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
        "confusion_matrix": cm_dict,
    }
    logging.debug(f"Calculated metrics: {metrics}")
    return metrics
