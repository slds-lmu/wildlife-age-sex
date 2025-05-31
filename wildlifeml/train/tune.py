from typing import Literal
import logging
import numpy as np
from ..utils import convert_to_numeric_indices, get_model_summary

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, Optimizer
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, Loss
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure all GPUs to use memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Set memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        logging.error(f"GPU configuration error: {e}")


def tune_model(
    model,
    train_data,
    target_column: str,
    classes: list[str],
    batch_size: int,
    loss_function: Literal[
        "binary_crossentropy", "categorical_crossentropy"
    ] = "binary_crossentropy",
    transfer_epochs: int = 10,
    finetune_epochs: int = 10,
    transfer_optimizer: Literal["adam", "sgd"] = "adam",
    transfer_learning_rate: float = 0.01,
    finetune_optimizer: Literal["adam", "sgd"] = "adam",
    finetune_learning_rate: float = 0.001,
    finetune_layers: int = 1,
    earlystop_metric: str = "val_loss",
    transfer_patience: int = 10,
    finetune_patience: int = 10,
    num_workers: int = 0,
    eval_metrics: list | None = None,
    **kwargs,
):
    # Postprocess inputs
    # Collect tuning specifications
    tuning_specs = {
        "training_params": {
            "loss_function": loss_function,
            "transfer_epochs": transfer_epochs,
            "finetune_epochs": finetune_epochs,
            "transfer_optimizer": {
                "name": transfer_optimizer,
                "learning_rate": transfer_learning_rate,
            },
            "finetune_optimizer": {
                "name": finetune_optimizer,
                "learning_rate": finetune_learning_rate,
            },
            "finetune_layers": finetune_layers,
            "batch_size": batch_size,
            "earlystop_metric": earlystop_metric,
            "transfer_patience": transfer_patience,
            "finetune_patience": finetune_patience,
        }
    }

    logging.info("Postprocessing inputs...")
    # Convert images to numpy array
    num_missing_images = sum([img is None for img in train_data["image"]])
    if num_missing_images > 0:
        train_data = train_data[train_data["image"].notna()]
        logging.warning(
            f"Found {num_missing_images} missing images. Continuing with {len(train_data)} images."
        )
    inputs = np.stack(train_data["image"].values).astype(np.float32)  # Convert to float32

    # Get labels and convert to numeric indices
    assert (
        target_column in train_data.columns
    ), f"Target column {target_column} not found in train_data"
    # Get unique categories and create mapping
    labels = convert_to_numeric_indices(train_data[target_column], classes)

    loss_function = (
        BinaryCrossentropy()
        if loss_function == "binary_crossentropy"
        else SparseCategoricalCrossentropy()
    )
    transfer_optimizer = (
        Adam(learning_rate=transfer_learning_rate)
        if transfer_optimizer == "adam"
        else SGD(learning_rate=transfer_learning_rate)
    )
    finetune_optimizer = (
        Adam(learning_rate=finetune_learning_rate)
        if finetune_optimizer == "adam"
        else SGD(learning_rate=finetune_learning_rate)
    )

    # Define callbacks for transfer/finetuning
    transfer_callbacks = [
        EarlyStopping(
            monitor=earlystop_metric,
            patience=2 * transfer_patience,
        ),
        ReduceLROnPlateau(
            monitor=earlystop_metric,
            patience=transfer_patience,
            factor=0.1,
            verbose=1,
        ),
    ]
    finetune_callbacks = [
        EarlyStopping(
            monitor=earlystop_metric,
            patience=2 * finetune_patience,
        ),
        ReduceLROnPlateau(
            monitor=earlystop_metric,
            patience=finetune_patience,
            factor=0.1,
            verbose=1,
        ),
    ]

    if transfer_epochs > 0:
        model = do_transfer_learning(
            model,
            inputs,
            labels,
            transfer_optimizer,
            loss_function,
            transfer_epochs,
            transfer_callbacks,
            eval_metrics,
            batch_size,
            num_workers,
        )

    if finetune_epochs > 0 and finetune_layers > 0:
        model = do_finetuning(
            model,
            inputs,
            labels,
            finetune_optimizer,
            loss_function,
            finetune_epochs,
            finetune_callbacks,
            eval_metrics,
            batch_size,
            finetune_layers,
            num_workers,
        )
    logging.info("Generating model summary...")
    tuning_specs["model_summary"] = get_model_summary(model)

    return (model, tuning_specs)


def do_transfer_learning(
    model: Sequential,
    inputs: np.ndarray,
    labels: np.ndarray,
    transfer_optimizer: Optimizer,
    loss_function: Loss,
    transfer_epochs: int = 10,
    transfer_callbacks: list | None = None,
    eval_metrics: list | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
):
    logging.info("Compiling model...")
    model.compile(
        optimizer=transfer_optimizer,
        loss=loss_function,
        metrics=eval_metrics,
    )

    logging.info("Starting transfer learning...")
    model.fit(
        x=inputs,
        y=labels,
        batch_size=batch_size,
        epochs=transfer_epochs,
        callbacks=transfer_callbacks,
        validation_split=0.15,
        workers=num_workers,
        use_multiprocessing=num_workers > 0,
        shuffle=False,
    )
    return model


def do_finetuning(
    model: Sequential,
    inputs: np.ndarray,
    labels: np.ndarray,
    finetune_optimizer: Optimizer,
    loss_function: Loss,
    finetune_epochs: int = 10,
    finetune_callbacks: list | None = None,
    eval_metrics: list | None = None,
    batch_size: int = 32,
    finetune_layers: int = 1,
    num_workers: int = 0,
):
    # Freeze all layers except the last finetune_layers
    for layer in model.layers[:-finetune_layers]:
        layer.trainable = False
    # Unfreeze the last finetune_layers
    for layer in model.layers[-finetune_layers:]:
        layer.trainable = True

    logging.info("Compiling model...")
    model.compile(
        optimizer=finetune_optimizer,
        loss=loss_function,
        metrics=eval_metrics,
    )

    logging.info("Starting fine tuning...")
    model.fit(
        x=inputs,
        y=labels,
        batch_size=batch_size,
        epochs=finetune_epochs,
        callbacks=finetune_callbacks,
        validation_split=0.15,
        workers=num_workers,
        use_multiprocessing=num_workers > 0,
        shuffle=False,
    )
    return model
