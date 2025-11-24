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
    uncertainty_threshold: float = 0.5,
    exclude_uncertain: bool = False,
) -> dict:
    """Evaluate a model on a test dataset. Optionally stratify
    evaluation by the provided metadatacolumn.

    Evaluation metrics:
    - Precision, recall, f1-score, accuracy
    - Confusion matrix

    Args:
        model: The model to evaluate.
        test_data: The test dataset.
        target_column: The column to evaluate the model on.
        stratify_by: The column to stratify the evaluation by.
        uncertainty_threshold: Minimum confidence threshold (0-1). Predictions below this
                              threshold are considered uncertain and should be manually labeled.
    Returns:
        A dictionary containing the evaluation results.
    """
    logging.info(f"Starting model evaluation with {len(test_data)} test samples")
    logging.info(f"Target column: {target_column}, Stratify by: {stratify_by}")
    logging.info(f"Uncertainty threshold: {uncertainty_threshold}")

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
    predictions, confidence_scores, uncertain_indices = _get_predictions(
        model, test_loader, uncertainty_threshold
    )
    print(predictions)
    print(confidence_scores)
    print(uncertain_indices)

    # Create uncertain images dataframe for manual labeling
    uncertain_images = (
        test_data.iloc[uncertain_indices].copy() if len(uncertain_indices) > 0 else pd.DataFrame()
    )
    if len(uncertain_images) > 0:
        uncertain_images["predicted_label"] = [
            idx_to_category[p] for p in predictions[uncertain_indices]
        ]
        uncertain_images["confidence_score"] = confidence_scores[uncertain_indices]
        logging.info(
            f"Found {len(uncertain_images)} uncertain images (confidence < {uncertainty_threshold})"
        )

    # Handle uncertain predictions based on exclude_uncertain parameter
    if exclude_uncertain and len(uncertain_indices) > 0:
        # Create mask for certain predictions (exclude uncertain ones)
        certain_mask = np.ones(len(predictions), dtype=bool)
        certain_mask[uncertain_indices] = False

        # Filter data for evaluation
        labels_filtered = labels[certain_mask]
        predictions_filtered = predictions[certain_mask]
        confidence_scores_filtered = confidence_scores[certain_mask]
        test_data_filtered = test_data.iloc[certain_mask].copy()

        logging.info(f"Excluding {len(uncertain_indices)} uncertain predictions from evaluation")
        logging.info(f"Evaluating on {len(labels_filtered)} certain predictions")
    else:
        # Use all predictions for evaluation
        labels_filtered = labels
        predictions_filtered = predictions
        confidence_scores_filtered = confidence_scores
        test_data_filtered = test_data.copy()
        logging.info(
            f"Evaluating on all {len(labels_filtered)} predictions (including uncertain ones)"
        )

    # Use a boolean mask as a Series aligned to test_data's index to select error rows
    error_mask = [l != p for l, p in zip(labels_filtered, predictions_filtered)]
    errors = test_data_filtered[error_mask].copy()
    errors["predicted_label"] = [idx_to_category[p] for p in predictions_filtered[error_mask]]
    errors["confidence_score"] = confidence_scores_filtered[error_mask]

    logging.info("Calculating overall metrics...")
    results["overall"] = _get_metrics(
        labels_filtered, predictions_filtered, idx_to_category, target_column
    )
    results["overall"]["uncertainty_threshold"] = uncertainty_threshold
    results["overall"]["excluded_uncertain_images"] = exclude_uncertain
    results["overall"]["n_uncertain_images"] = len(uncertain_indices)
    results["overall"]["n_certain_images"] = len(labels_filtered)
    results["overall"]["avg_confidence"] = float(np.nanmean(confidence_scores_filtered))
    logging.info(f"Overall accuracy: {results['overall']['accuracy']:.3f}")
    logging.info(f"Average confidence: {results['overall']['avg_confidence']:.3f}")

    # Calculate stratified metrics if requested
    if stratify_by is not None:
        logging.info(f"Calculating stratified metrics by {stratify_by}...")
        for val in test_data_filtered[stratify_by].unique():
            subset_mask = test_data_filtered[stratify_by] == val
            subset_labels = labels_filtered[subset_mask]
            subset_predictions = predictions_filtered[subset_mask]
            subset_confidences = confidence_scores_filtered[subset_mask]
            logging.debug(f"Stratum {val}: {len(subset_labels)} samples")
            stratum_key = f"{stratify_by}: {val}"
            results[stratum_key] = _get_metrics(
                subset_labels, subset_predictions, idx_to_category, target_column
            )
            results[stratum_key]["avg_confidence"] = float(np.nanmean(subset_confidences))
            logging.info(f"{val} accuracy: {results[stratum_key]['accuracy']:.3f}")

    return results, errors, uncertain_images


def _get_predictions(
    model: Module, data_loader: DataLoader, uncertainty_threshold: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to get model predictions for a dataset using a DataLoader.

    Args:
        model: The model to evaluate
        data_loader: DataLoader containing the test data
        uncertainty_threshold: Minimum confidence threshold (0-1). Predictions below this
                              threshold are considered uncertain and should be manually labeled.

    Returns:
        tuple: (predictions, confidence_scores, uncertain_indices)
            - predictions: Array of predicted class indices
            - confidence_scores: Array of confidence scores for each prediction (normalized to sum to 1)
            - uncertain_indices: Array of indices where confidence < threshold
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    preds = []
    confidences = []
    uncertain_indices = []
    current_idx = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)

            # Check for problematic outputs before softmax
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logging.warning(f"Found NaN/Inf in model outputs: {outputs}")

            # Check for extreme values that might cause numerical issues
            if torch.abs(outputs).max() > 100:
                logging.warning(
                    f"Found large logit values: max={outputs.max():.2f}, min={outputs.min():.2f}"
                )

            # Apply softmax to get normalized probabilities (sum to 1)
            # Use log_softmax + exp for better numerical stability
            log_probs = torch.log_softmax(outputs, dim=1)
            probs = torch.exp(log_probs)

            # Check for NaN in probabilities after softmax
            if torch.isnan(probs).any():
                logging.warning(f"Found NaN in softmax probabilities: {probs}")
                # Replace NaN with uniform distribution
                probs = torch.where(
                    torch.isnan(probs), torch.ones_like(probs) / probs.shape[1], probs
                )

            # Get predictions and confidence scores
            batch_preds = probs.argmax(dim=1).cpu()
            batch_confidences = probs.max(dim=1)[0].cpu()  # Max probability for each sample

            # Identify uncertain predictions
            uncertain_mask = batch_confidences < uncertainty_threshold
            batch_uncertain_indices = torch.where(uncertain_mask)[0] + current_idx

            preds.append(batch_preds)
            confidences.append(batch_confidences)
            uncertain_indices.append(batch_uncertain_indices)

            current_idx += len(batch_preds)

    # Concatenate all batches
    predictions = torch.cat(preds).numpy()
    confidence_scores = torch.cat(confidences).numpy()
    uncertain_indices = torch.cat(uncertain_indices).numpy() if uncertain_indices else np.array([])

    return predictions, confidence_scores, uncertain_indices


def _get_metrics(
    labels: np.ndarray, predictions: np.ndarray, idx_to_category: dict, target_column: str
) -> dict:
    """Helper function to calculate all evaluation metrics."""
    ordered_indices = sorted(idx_to_category.keys())
    label_names = [idx_to_category[idx] for idx in ordered_indices]

    cm_table = {
        true_label: {pred_label: 0 for pred_label in label_names} for true_label in label_names
    }
    for true_idx, pred_idx in zip(labels, predictions):
        true_label = idx_to_category[int(true_idx)]
        pred_label = idx_to_category[int(pred_idx)]
        cm_table[true_label][pred_label] += 1

    cm_dict = {
        f"{true_label}_{pred_label}": count
        for true_label, pred_counts in cm_table.items()
        for pred_label, count in pred_counts.items()
    }

    class_distribution_series = pd.Series(labels).map(idx_to_category).value_counts()
    class_distribution = {label: int(count) for label, count in class_distribution_series.items()}

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
