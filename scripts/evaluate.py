import argparse
import logging
from datetime import datetime
from pathlib import Path
import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings

import tomli

from wildlifeml.io import load, save, load_model
from wildlifeml.train import evaluate_model
from wildlifeml.utils import pathify_args


def main(
    working_dir: Path,
    test_filepath: Path,
    model_dir: Path,
    backbone_model: str,
    target_column: str,
    classes: list[str],
    stratify_by: str | None = None,
    confidence_threshold: float = 0.5,
    exclude_uncertain: bool = False,
    **kwargs,
):
    # Load data
    test_data = load(filepath=working_dir / test_filepath)

    # Load model
    model = load_model(
        backbone_model, len(classes), weights_path=working_dir / model_dir / "model.pt"
    )

    # Evaluate model
    results, errors, uncertain_images = evaluate_model(
        model,
        test_data,
        target_column,
        classes,
        stratify_by,
        confidence_threshold,
        exclude_uncertain,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    save(
        results,
        filepath=working_dir / model_dir / f"{timestamp}__eval_results.json",
    )
    save(
        errors,
        filepath=working_dir / model_dir / f"{timestamp}__eval_errors.parquet",
    )
    save(
        uncertain_images,
        filepath=working_dir / model_dir / f"{timestamp}__eval_uncertain_images.parquet",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/demo__config.toml")
    args = parser.parse_args()

    with open(Path(args.config), "rb") as f:
        args = tomli.load(f)

    logging.basicConfig(**args.get("logging", {}))

    # Preprocess arguments to convert string paths to Path objects
    processed_args = pathify_args(args)

    main(
        **processed_args["globals"],
        **processed_args["io"]["model"],
        **processed_args["io"]["data"],
        backbone_model=processed_args["train"]["backbone_model"],
        **processed_args["evaluate"],
    )
