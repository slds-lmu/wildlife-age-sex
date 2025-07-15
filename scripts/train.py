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


def main(
    working_dir: str,
    preprocessed_data_filepath: str,
    train_filepath: str,
    test_filepath: str,
    model_dir: str,
    target_column: str,
    classes: list[str],
    training_args: dict,
    **kwargs,
):
    # Load data
    preprocessed_data = load(filepath=Path(working_dir) / Path(preprocessed_data_filepath))

    # Split data, save to disk
    logging.info(f"Splitting data...")
    for train_data, test_data in split_data(
        preprocessed_data, stratify_by=training_args.get("stratify_by", None)
    ):
        save(train_data, filepath=Path(working_dir) / Path(train_filepath))
        save(test_data, filepath=Path(working_dir) / Path(test_filepath))

    # Train model
    model = load_model(**training_args, num_classes=len(classes))
    tuned_model, tuning_specs = tune_model(
        model, train_data, target_column, classes, **training_args
    )

    # Save model
    save(tuning_specs, filepath=Path(working_dir) / Path(model_dir) / "tuning_specs.json")
    save(tuned_model, filepath=Path(working_dir) / Path(model_dir) / "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test__config.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        args = tomli.load(f)
    logging.basicConfig(**args.get("logging", {}))

    main(
        **args["globals"],
        **args["io"]["data"],
        **args["io"]["model"],
        training_args=args["train"],
    )
