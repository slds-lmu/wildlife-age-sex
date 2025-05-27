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
    stratify_by: str = "age",
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
    if stratify_by and stratify_by not in test_data.columns:
        raise ValueError(f"Stratify by column {stratify_by} not found in test data.")

    num_missing_images = len(test_data["image"].isna())
    if num_missing_images > 0:
        test_data = test_data[test_data["image"].notna()]
        logging.warning(
            f"Found {num_missing_images} missing images. Continuing with {len(test_data)} images."
        )
    results = {}

    # Calculate overall metrics
    # Get raw numeric indices
    categories = test_data[target_column].unique()
    category_to_idx = {cat: idx for idx, cat in enumerate(sorted(categories))}
    labels = test_data[target_column].map(category_to_idx).values

    predictions = _get_predictions(model, test_data["image"].values)
    results["overall"] = _get_metrics(labels, predictions)

    # Calculate stratified metrics if requested
    if stratify_by is not None:
        for val in test_data[stratify_by].unique():
            subset_labels = labels[test_data[stratify_by] == val]
            subset_predictions = predictions[test_data[stratify_by] == val]
            results[str(val)] = _get_metrics(subset_labels, subset_predictions)

    return results


def _get_predictions(model: Sequential, input_data: np.ndarray) -> np.ndarray:
    """Helper function to get model predictions for a dataset."""
    predictions = model.predict(np.stack(input_data).astype(np.float32))
    return np.argmax(predictions, axis=1)  # Convert probabilities to class labels


def _get_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Helper function to calculate all evaluation metrics."""
    # Convert confusion matrix to dict with row/col indices as keys
    cm = confusion_matrix(labels, predictions)
    cm_dict = {
        f"{i}_{j}": int(cm[i, j])  # Convert to int for JSON serialization
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
    }
    return {
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1-score": f1_score(labels, predictions, average="weighted", zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
        "confusion_matrix": cm_dict,
    }
