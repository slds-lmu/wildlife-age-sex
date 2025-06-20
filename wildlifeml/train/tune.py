import logging
from typing import Literal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils import convert_to_numeric_indices, get_model_summary


def tune_model(
    model: nn.Module,
    train_data,
    target_column: str,
    classes: list[str],
    batch_size: int,
    loss_function: Literal[
        "binary_crossentropy", "categorical_crossentropy"
    ] = "categorical_crossentropy",
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
):
    """
    Train a model using transfer learning and finetuning.

    Args:
    -----
    model: nn.Module
        The model to train.
    train_data: pd.DataFrame
        The training data.
    target_column: str
        The column in train_data that contains the target variable.
    classes: list[str]
        The classes in the target variable.
    batch_size: int
        The batch size to use for training.
    loss_function: Literal["binary_crossentropy", "categorical_crossentropy"]
        The loss function to use.
    transfer_epochs: int
        The number of epochs to use for transfer learning.
    finetune_epochs: int
        The number of epochs to use for finetuning.
    transfer_optimizer: Literal["adam", "sgd"]
        The optimizer to use for transfer learning.
    transfer_learning_rate: float
        The learning rate to use for transfer learning.
    finetune_optimizer: Literal["adam", "sgd"]
        The optimizer to use for finetuning.
    finetune_learning_rate: float
        The learning rate to use for finetuning.
    finetune_layers: int
        The number of layers of the base model to finetune. All classifier layers are also trainable in this step.
    earlystop_metric: str
        The metric to use for early stopping.
    transfer_patience: int
        The number of epochs to wait before early stopping transfer learning.
    finetune_patience: int
        The number of epochs to wait before early stopping finetuning.
    num_workers: int
        The number of workers to use for training.
    device: str
        The device to use for training.
    **kwargs: dict
        Additional keyword arguments to pass to the model.

    Returns:
    --------
    tuple[nn.Module, dict]
        The trained model and the tuning specifications.
    """
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
        },
    }

    logging.info("Postprocessing inputs...")
    # Convert images to tensor
    num_missing_images = sum([img is None for img in train_data["image"]])
    if num_missing_images > 0:
        train_data = train_data[train_data["image"].notna()]
        logging.warning(
            f"Found {num_missing_images} missing images. Continuing with {len(train_data)} images."
        )
    tuning_specs["n_train_observations"] = len(train_data)

    # Stack images and convert to tensor
    inputs = torch.stack([torch.from_numpy(img).float() for img in train_data["image"].values])
    inputs = inputs.permute(0, 3, 1, 2)  # Convert from (N, H, W, C) to (N, C, H, W)

    # Get labels and convert to numeric indices
    assert (
        target_column in train_data.columns
    ), f"Target column {target_column} not found in train_data"
    class_mappings, labels = convert_to_numeric_indices(train_data[target_column], classes)
    tuning_specs["class_mappings"] = class_mappings
    tuning_specs["class_distribution"] = train_data[target_column].value_counts().to_dict()

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.float)

    # Create dataset and dataloader
    dataset = TensorDataset(inputs, labels)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Setup loss function
    criterion = nn.BCELoss() if loss_function == "binary_crossentropy" else nn.CrossEntropyLoss()

    # Move model to device
    model = model.to(device)

    if transfer_epochs > 0:
        model = do_transfer_learning(
            model,
            train_loader,
            val_loader,
            transfer_optimizer,
            transfer_learning_rate,
            criterion,
            transfer_epochs,
            transfer_patience,
            device,
        )

    if finetune_epochs > 0 and finetune_layers > 0:
        model = do_finetuning(
            model,
            train_loader,
            val_loader,
            finetune_optimizer,
            finetune_learning_rate,
            criterion,
            finetune_epochs,
            finetune_patience,
            finetune_layers,
            device,
        )

    logging.info("Generating model summary...")
    tuning_specs["model_summary"] = get_model_summary(model)

    return model, tuning_specs


def do_transfer_learning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_name: str,
    learning_rate: float,
    criterion: nn.Module,
    epochs: int,
    patience: int,
    device: str,
):
    # Setup optimizer
    optimizer = (
        Adam(model.classifier.parameters(), lr=learning_rate)
        if optimizer_name == "adam"
        else SGD(model.classifier.parameters(), lr=learning_rate)
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=0.1)

    best_val_loss = float("inf")
    patience_counter = 0

    logging.info("Starting transfer learning...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            train_inputs, train_labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                val_inputs, val_labels = inputs.to(device), labels.to(device)
                outputs = model(val_inputs)
                val_loss += criterion(outputs, val_labels).item()

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

        logging.info(
            f"""
            Transfer Learning Epoch {epoch}:
            Train Loss = {train_loss / len(train_loader):.4f},
            Val Loss = {val_loss / len(val_loader):.4f}
            """
        )

    return model


def do_finetuning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_name: str,
    learning_rate: float,
    criterion: nn.Module,
    epochs: int,
    patience: int,
    finetune_layers: int,
    device: str,
):
    # Unfreeze the last finetune_layers
    for param in list(model.base.parameters())[-finetune_layers:]:
        param.requires_grad = True

    # Setup optimizer (only for unfrozen parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = (
        Adam(trainable_params, lr=learning_rate)
        if optimizer_name == "adam"
        else SGD(trainable_params, lr=learning_rate)
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=0.1)

    best_val_loss = float("inf")
    patience_counter = 0

    logging.info("Starting fine tuning...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            train_inputs, train_labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                val_inputs, val_labels = inputs.to(device), labels.to(device)
                outputs = model(val_inputs)
                val_loss += criterion(outputs, val_labels).item()

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

        logging.info(
            f"""
            Finetuning Epoch {epoch}:
            Train Loss = {train_loss / len(train_loader):.4f},
            Val Loss = {val_loss / len(val_loader):.4f}
            """
        )

    return model
