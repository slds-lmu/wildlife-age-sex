import argparse
import logging
from datetime import datetime
from pathlib import Path

import tomli

from wildlifeml.io import load, save
from wildlifeml.train import evaluate_model


def main(
    working_dir: str,
    test_filepath: str,
    model_dir: str,
    target_column: str,
    stratify_by: str | None = None,
    **kwargs,
):
    # Load data
    test_data = load(filepath=Path(working_dir) / Path(test_filepath))

    # Load model
    model = load(filepath=Path(working_dir) / Path(model_dir) / "model.keras")

    # Evaluate model
    results = evaluate_model(model, test_data, target_column, stratify_by)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    save(
        results,
        filepath=Path(working_dir) / Path(model_dir) / f"{timestamp}__eval_results.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test__config.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        args = tomli.load(f)

    logging.basicConfig(**args.get("logging", {}))

    main(
        **args["globals"],
        **args["io"]["model"],
        **args["io"]["data"],
        **args["evaluate"],
    )
