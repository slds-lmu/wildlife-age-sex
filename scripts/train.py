import argparse
import logging
from pathlib import Path

import tomli

from wildlifeml.io import load, load_backbone_model, save
from wildlifeml.train import tune_model


def main(
    working_dir: str,
    train_filepath: str,
    model_dir: str,
    target_column: str,
    classes: list[str],
    training_args: dict,
    **kwargs,
):
    # Load data
    train_data = load(filepath=Path(working_dir) / Path(train_filepath))

    # Train model
    model = load_backbone_model(**training_args, num_classes=len(classes), mode="keras")
    tuned_model, tuning_specs = tune_model(
        model, train_data, target_column, classes, **training_args
    )

    # Save model
    save(tuning_specs, filepath=Path(working_dir) / Path(model_dir) / "tuning_specs.json")
    save(tuned_model, filepath=Path(working_dir) / Path(model_dir) / "model.keras")


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
