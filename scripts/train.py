import argparse
import logging
from pathlib import Path
import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings

import tomli

from wildlifeml.io import load_model, load, save
from wildlifeml.train import tune_model, split_data
from wildlifeml.utils import pathify_args


def main(
    working_dir: Path,
    preprocessed_data_filepath: Path,
    train_filepath: Path,
    test_filepath: Path,
    model_dir: Path,
    target_column: str,
    classes: list[str],
    training_args: dict,
    **kwargs,
):
    # Load data
    preprocessed_data = load(filepath=working_dir / preprocessed_data_filepath)

    # Split data, save to disk
    logging.info(f"Splitting data...")
    for train_data, test_data in split_data(
        preprocessed_data, stratify_by=training_args.get("stratify_by", None)
    ):
        save(train_data, filepath=working_dir / train_filepath)
        save(test_data, filepath=working_dir / test_filepath)

    # Train model
    model = load_model(**training_args, num_classes=len(classes))
    tuned_model, tuning_specs = tune_model(
        model, train_data, target_column, classes, **training_args
    )

    # Save model
    save(tuning_specs, filepath=working_dir / model_dir / "tuning_specs.json")
    save(tuned_model, filepath=working_dir / model_dir / "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/demo__config.toml")
    args = parser.parse_args()

    with open(Path(args.config), "rb") as f:
        args = tomli.load(f)
    logging.basicConfig(**args.get("logging", {}))

    # Preprocess arguments to convert string paths to Path objects
    args = pathify_args(args)

    main(
        **args["globals"],
        **args["io"]["data"],
        **args["io"]["model"],
        training_args=args["train"],
    )
