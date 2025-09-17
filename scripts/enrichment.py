#!/usr/bin/env python3
"""
Data enrichment script to convert annotations to training format and add metadata.

This script:
1. Reads annotations.json files from multiple directories
2. Loads and joins metadata file on original_image_id (annotations) and image_id (metadata) (optional)
3. Converts to the required Parquet format for training

Usage:
    poe run-enrichment  # Uses demo data by default
    poe run-enrichment --annotation-dirs data/dir1 data/dir2 --metadata-file data/metadata.csv
    poe run-enrichment --annotation-dirs data/dir1 data/dir2 --no-metadata  # Skip metadata joining
"""

import json
import pandas as pd
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_annotations(annotation_dirs: list) -> list:
    """Load annotations from multiple directories containing annotations.json files."""
    all_annotations = []

    for annotation_dir in annotation_dirs:
        annotations_path = Path(annotation_dir) / "annotations.json"

        if not annotations_path.exists():
            logger.warning(f"Annotations file not found: {annotations_path}")
            continue

        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        # Handle both list format (from web interface) and dict format
        if isinstance(annotations, list):
            all_annotations.extend(annotations)
        else:
            # Convert dict format to list
            all_annotations.extend(annotations.values())

        logger.info(
            f"Loaded {len(annotations) if isinstance(annotations, list) else len(annotations.values())} annotations from {annotations_path}"
        )

    logger.info(f"Total annotations loaded: {len(all_annotations)}")
    return all_annotations


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load additional metadata file (CSV or Parquet)."""
    metadata_file = Path(metadata_path)

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Support both CSV and Parquet metadata files
    if metadata_file.suffix.lower() == ".csv":
        metadata = pd.read_csv(metadata_file)
    elif metadata_file.suffix.lower() == ".parquet":
        metadata = pd.read_parquet(metadata_file)
    else:
        raise ValueError(f"Unsupported metadata file format: {metadata_file.suffix}")

    logger.info(
        f"Loaded metadata with {len(metadata)} rows and columns: {metadata.columns.tolist()}"
    )
    if "image_path" in metadata.columns:
        logger.warning("found competing image_path column in metadata, dropping it")
        metadata.drop(columns=["image_path"], inplace=True)
    if "image_id" not in metadata.columns:
        raise ValueError("Column 'image_id' not found in metadata")
    else:
        metadata.rename(columns={"image_id": "metadata_image_id"}, inplace=True)
    return metadata


def convert_annotations_to_dataframe(annotations: list) -> pd.DataFrame:
    """
    Convert annotations to DataFrame format for training.
    Uses the schema exactly as found in the annotation dicts.
    Checks that required columns are present.
    """
    # Convert list of annotation dicts directly to DataFrame
    df = pd.DataFrame(annotations)
    logger.info(f"Created dataset with {len(df)} rows from annotations")

    # Define required columns (these may change depending on your label config)
    required_cols = ["image_id", "original_image_id", "image_path", "bbox", "confidence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in annotations: {missing_cols}")

    return df


def add_metadata(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional metadata to the dataset by joining on original_image_id (data) and image_id (metadata).

    Args:
        data: Main dataset DataFrame with original_image_id column
        metadata: Additional metadata DataFrame with image_id column

    Returns:
        pd.DataFrame with additional metadata columns
    """
    logger.info("Adding metadata by joining on original_image_id (data) and image_id (metadata)")

    # Check if required columns exist
    if "original_image_id" not in data.columns:
        raise ValueError("Column 'original_image_id' not found in main dataset")

    # Perform left join to preserve all main data
    enriched_data = data.merge(
        metadata, left_on="original_image_id", right_on="metadata_image_id", how="left"
    )
    enriched_data.drop(columns=["metadata_image_id"], inplace=True)

    # Log join results
    original_count = len(data)
    enriched_count = len(enriched_data)
    matched_count = enriched_data[
        enriched_data.iloc[:, -len(metadata.columns) :].notna().any(axis=1)
    ].shape[0]

    logger.info(f"Join results: {original_count} original rows, {enriched_count} enriched rows")
    logger.info(f"Successfully matched {matched_count} rows with metadata")

    # Log new columns added (exclude the join column from metadata)
    new_columns = [col for col in metadata.columns]
    logger.info(f"Added metadata columns: {new_columns}")

    return enriched_data


def save_parquet(data: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to Parquet format."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(data)} rows to {output_path}")


def main():
    """Main function to run data enrichment process."""
    parser = argparse.ArgumentParser(
        description="Convert annotations to training format and add metadata"
    )
    parser.add_argument(
        "--annotation-dirs",
        type=str,
        nargs="+",
        default=["data/demo"],
        help="List of directories containing annotations.json files (default: data/demo)",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="data/demo/sample_metadata.csv",
        help="Metadata file (CSV or Parquet) to join with annotations (default: data/demo/sample_metadata.csv)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/demo/enriched_data.parquet",
        help="Output path for the enriched parquet file (default: data/demo/enriched_data.parquet)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip metadata joining and only convert annotations to training format",
    )

    args = parser.parse_args()

    # Load annotations from multiple directories
    logger.info("Loading annotations...")
    annotations = load_annotations(args.annotation_dirs)

    if not annotations:
        raise ValueError("No annotations found in the specified directories")

    # Convert to DataFrame
    logger.info("Converting annotations to training format...")
    data = convert_annotations_to_dataframe(annotations)

    # Add metadata (unless --no-metadata flag is used)
    if not args.no_metadata:
        logger.info(f"Loading metadata from {args.metadata_file}...")
        metadata = load_metadata(args.metadata_file)
        data = add_metadata(data, metadata)
    else:
        logger.info("Skipping metadata joining (--no-metadata flag used)")

    # Save to parquet
    logger.info(f"Saving enriched data to {args.output_path}...")
    save_parquet(data, args.output_path)

    logger.info("Data enrichment completed successfully!")
    logger.info(f"Output file: {args.output_path}")
    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Columns: {data.columns.tolist()}")


if __name__ == "__main__":
    main()
