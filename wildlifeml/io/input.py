import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential

KERAS_AVAILABLE_MODELS = {
    "resnet50v2": {
        "model": tf.keras.applications.ResNet50V2,
        "preproc_func": tf.keras.applications.resnet_v2.preprocess_input,
    },
    "inception_resnet_v2": {
        "model": tf.keras.applications.InceptionResNetV2,
        "preproc_func": tf.keras.applications.inception_resnet_v2.preprocess_input,
    },
    "vgg19": {
        "model": tf.keras.applications.VGG19,
        "preproc_func": tf.keras.applications.vgg19.preprocess_input,
    },
    "xception": {
        "model": tf.keras.applications.Xception,
        "preproc_func": tf.keras.applications.xception.preprocess_input,
    },
    "densenet121": {
        "model": tf.keras.applications.DenseNet121,
        "preproc_func": tf.keras.applications.densenet.preprocess_input,
    },
    "densenet201": {
        "model": tf.keras.applications.DenseNet201,
        "preproc_func": tf.keras.applications.densenet.preprocess_input,
    },
}


def load(filepath: str | Path, image_mode: Literal["RGB", "L"] = "RGB"):
    """
    Load data from a parquet file or model from a keras file.

    NOTE: If the file is a parquet file, the 'image_path' column is used to load the image.
    The 'image_path' column is created from the 'image_id' column by the save function when
    saving a DataFrame with an 'image' column.

    Args:
    -----
    filepath: Path
        The path to the file to load.
    image_mode: Literal['RGB', 'L']
        The mode to convert the image to. 'RGB' for color, 'L' for grayscale.
        Only used if the file is a parquet file and contains an 'image_path' column.

    Returns:
    --------
    pd.DataFrame or tf.keras.Model
        The loaded data or model.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if filepath.suffix == ".parquet":
        logging.info(f"Loading parquet file: {filepath}.")
        data = pd.read_parquet(filepath)
        if "image_path" in data.columns:
            data["image"] = data["image_path"].apply(lambda x: try_load_image(x, image_mode))
        return data
    elif filepath.suffix == ".keras":
        logging.info(f"Loading keras model: {filepath}.")
        return tf.keras.models.load_model(filepath)
    else:
        raise ValueError(f"File {filepath} is not a parquet file or keras file.")


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


class ModelFactory:
    """Factory for creating Keras model objects."""

    @staticmethod
    def load(
        model_id: Literal[
            "resnet50",
            "inception_resnet_v2",
            "vgg19",
            "xception",
            "densenet121",
            "densenet201",
        ],
        num_classes: int,
        weights: str = "imagenet",
        include_top: bool = False,
        pooling: str = "avg",
    ) -> Sequential:
        """
        Return an initialized model instance from an identifier.

        Note: The model still needs to be compiled.
        """
        model_entry = KERAS_AVAILABLE_MODELS[model_id]
        model_cls = model_entry["model"]

        logging.info("Configuring pretrained model...")
        model = Sequential()
        model.add(Lambda(model_entry["preproc_func"]))
        model.add(model_cls(weights=weights, include_top=include_top, pooling=pooling))
        model.add(Dense(num_classes, activation="softmax"))

        # Freeze all layers from backbone model
        for layer in model.get_layer(model_id).layers:
            layer.trainable = False

        return model


def load_backbone_model(backbone_model: str | Path, num_classes: int = 2, **kwargs) -> Sequential:
    """
    Load a backbone model from a string identifier or path. By default, all but the final
    layer are frozen.

    Args:
    -----
    model: str | Path
        Either a model name from KERAS_AVAILABLE_MODELS or path to a model file
    num_classes: int
        Number of output classes for the model

    Returns:
    --------
    tf.keras.Model
        The loaded model
    """
    if backbone_model in KERAS_AVAILABLE_MODELS.keys():
        logging.info(f"Loading model from pretrained model factory: {backbone_model}.")
        return ModelFactory.load(
            backbone_model,
            num_classes=num_classes,
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )
    else:
        try:
            logging.info(f"Loading model from path: {backbone_model}.")
            backbone_model = tf.keras.models.load_model(backbone_model)
            # Check if the model has the right output layer
            if not backbone_model.layers[-1].output_shape[1] == num_classes:
                # add dense layer with softmax
                logging.info(
                    f"Adding dense layer with {num_classes} classes to the backbonemodel."
                )
                backbone_model.add(Dense(num_classes, activation="softmax"))
            # Freeze all but the last layer
            for layer in backbone_model.layers[:-1]:
                layer.trainable = False
            return backbone_model
        except FileNotFoundError as e:
            raise ValueError(
                f"""
                Model not found. Please specify a valid model name from {KERAS_AVAILABLE_MODELS.keys()}
                or provide a path to a valid model file [#TODO: add valid extensions].
                """
            ) from e
