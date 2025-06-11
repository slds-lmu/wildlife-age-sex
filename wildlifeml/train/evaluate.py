from tensorflow.keras.models import Sequential
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
    model: Sequential,
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
    # Get raw numeric indices
    category_to_idx = {cat: idx for idx, cat in enumerate(classes)}
    labels = test_data[target_column].map(category_to_idx).values
    print(np.isnan(labels).sum())

    logging.info("Generating predictions...")
    predictions = _get_predictions(model, test_data["image"].values)

    logging.info("Calculating overall metrics...")
    results["overall"] = _get_metrics(labels, predictions, category_to_idx)
    logging.info(f"Overall accuracy: {results['overall']['accuracy']:.3f}")

    # Calculate stratified metrics if requested
    if stratify_by is not None:
        logging.info(f"Calculating stratified metrics by {stratify_by}...")
        for val in test_data[stratify_by].unique():
            subset_labels = labels[test_data[stratify_by] == val]
            subset_predictions = predictions[test_data[stratify_by] == val]
            logging.debug(f"Stratum {val}: {len(subset_labels)} samples")
            stratum_key = f"{stratify_by}: {val}"
            results[stratum_key] = _get_metrics(subset_labels, subset_predictions, category_to_idx)
            logging.info(f"{val} accuracy: {results[stratum_key]['accuracy']:.3f}")

    return results


def _get_predictions(model: Sequential, input_data: np.ndarray) -> np.ndarray:
    """Helper function to get model predictions for a dataset."""
    predictions = model.predict(np.stack(input_data).astype(np.float32))
    return np.argmax(predictions, axis=1)  # Convert probabilities to class labels


def _get_metrics(labels: np.ndarray, predictions: np.ndarray, category_to_idx: dict) -> dict:
    """Helper function to calculate all evaluation metrics."""
    # Convert confusion matrix to dict with original labels as keys
    cm = confusion_matrix(labels, predictions)
    logging.debug(f"Confusion matrix shape: {cm.shape}")

    # Create reverse mapping from indices to original labels
    idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}

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
