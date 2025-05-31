import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings

import argparse
import logging
from pathlib import Path

import tomli

from wildlifeml.io import load, save
from wildlifeml.preprocess import preprocess_data, split_data


def main(
    working_dir: str,
    raw_data_filepath: str,
    train_filepath: str,
    test_filepath: str,
    preprocess_kwargs: dict,
    **kwargs,
):
    # Load data
    logging.info(f"Loading data from {raw_data_filepath}...")
    data = load(filepath=Path(working_dir) / Path(raw_data_filepath))
    logging.info(f"Loaded {len(data)} rows of data.")

    # Preprocess data
    logging.info(f"Preprocessing data...")
    preprocessed_data = preprocess_data(data, **preprocess_kwargs)

    # Split data, save to disk
    logging.info(f"Splitting data...")
    for train, test in split_data(
        preprocessed_data, stratify_by=preprocess_kwargs.get("stratify_by", None)
    ):
        save(train, filepath=Path(working_dir) / Path(train_filepath))
        save(test, filepath=Path(working_dir) / Path(test_filepath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test__config.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        args = tomli.load(f)
    logging.basicConfig(**args.get("logging", {}))

    main(
        args["globals"]["working_dir"],
        **args["io"]["data"],
        preprocess_kwargs=args.get("preprocess", {}),
    )
