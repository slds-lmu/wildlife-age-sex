import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
import torchvision.models as models


# Dictionary of available PyTorch models
TORCH_AVAILABLE_MODELS = {
    "resnet50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
    "inception_v3": models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1),
    "vgg19": models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1),
    "densenet161": models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1),
    "densenet201": models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1),
}


def load(filepath: str | Path, image_mode: Literal["RGB", "L"] = "RGB"):
    """
    Load data from a parquet file.

    Args:
    -----
    filepath: Path
        The path to the file to load.
    image_mode: Literal['RGB', 'L']
        The mode to convert the image to. 'RGB' for color, 'L' for grayscale.
        Only used if the file is a parquet file and contains an 'image_path' column.

    Returns:
    --------
    pd.DataFrame
        The loaded data.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if filepath.suffix == ".parquet":
        logging.info(f"Loading parquet file: {filepath}.")
        data = pd.read_parquet(filepath)
        if "image_path" in data.columns:
            data["image"] = data["image_path"].apply(lambda x: try_load_image(x, image_mode))
        return data
    else:
        raise ValueError(f"File {filepath} is not a parquet file.")


def try_load_image(image_path: str | Path, mode: Literal["RGB", "L"]) -> np.ndarray:
    """Load an image from a path.

    Args:
    -----
    image_path: str | Path
        The path to the image.
    mode: Literal['RGB', 'L']
        The mode to convert the image to. 'RGB' for color, 'L' for grayscale.

    Returns:
    --------
    np.ndarray or None if file not found
    """
    try:
        img = Image.open(image_path)
        img.load()
        return np.array(img.convert(mode))
    except FileNotFoundError:
        logging.warning(f"File not found: {image_path}")
        return None


class WildlifeModel(nn.Module):
    """
    A model class that creates standardized models for image classification.

    By default, the model is frozen and the classifier is trainable.
    The base is all layers except the classifier.
    The classifier is always nn.Linear with num_classes outputs.
    """

    def __init__(self, model: nn.Module, num_classes: int | None = None):
        super().__init__()
        if num_classes is not None:
            self.model = reshape_classifier(model, num_classes)
        else:
            self.model = model
        # Freeze all layers in base, leave classifier trainable
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        self.classifier = self.model.classifier
        self.base = self.model.base

    def forward(self, x):
        return self.classifier(self.base(x))

    def save(self, filepath: str | Path):
        torch.save(self.model.state_dict(), filepath)


def reshape_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """

    Args:
    -----
    model: nn.Module
        The model to refactor.
    num_classes: int
        The number of output classes for the model.

    Returns:
    --------
    nn.Module
        The refactored model
    """
    # If model has .fc (e.g., ResNet)
    if hasattr(model, "fc"):
        in_ftrs = model.fc.in_features
        # Make all layers (including what was .fc) part of model.base
        base_layers = [m for n, m in model.named_children() if n != "fc"]
        base_layers.append(nn.Flatten(1))
        model.base = nn.Sequential(*base_layers)
        model.classifier = nn.Linear(in_ftrs, num_classes)
        return model

    # If model has .classifier (e.g., VGG, DenseNet)
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            modules = list(clf)
            if modules and isinstance(modules[-1], nn.Linear):
                last = modules[-1]
                in_ftrs = last.in_features
                modules[-1] = nn.Linear(in_ftrs, num_classes)
                model.classifier = nn.Sequential(*modules)
        elif isinstance(clf, nn.Linear):
            in_ftrs = clf.in_features
            model.classifier = nn.Linear(in_ftrs, num_classes)
        # Move all layers except .classifier to base
        base_layers = [m for n, m in model.named_children() if n != "classifier"]
        model.base = nn.Sequential(*base_layers)
        return model
    raise RuntimeError("Model has no recognized final classifier layer")


def load_model(
    backbone_model: str | Path,
    num_classes: int = 2,
    weights_path: str | Path | None = None,
    **kwargs,
) -> nn.Module:
    """
    Load a backbone model from a string identifier or path. Refactor the model to have a 'base' and 'classifier' section.
    The classifier is always nn.Linear with num_classes outputs. The base is all layers except the classifier.
    The base is frozen and the classifier is trainable.

    Args:
    -----
    model: str | Path
        Either a model name from TORCH_AVAILABLE_MODELS or path to a model file
    num_classes: int
        Number of output classes for the model

    Returns:
    --------
    torch.nn.Module
        The loaded model
    """
    if backbone_model in TORCH_AVAILABLE_MODELS.keys():
        logging.info(f"Loading model from pretrained model factory: {backbone_model}.")
        model = TORCH_AVAILABLE_MODELS[backbone_model]
    else:
        try:
            logging.info(f"Loading model from path: {backbone_model}.")
            model = torch.load(backbone_model)
        except FileNotFoundError as e:
            raise ValueError(
                f"""
                Model not found. Please specify a valid model name from {TORCH_AVAILABLE_MODELS.keys()}
                or provide a path to a valid PyTorch model file (.pt or .pth).
                """
            ) from e

    model = WildlifeModel(model, num_classes)
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model
