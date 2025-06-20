from pathlib import Path

import pandas as pd
import torch
from PIL import Image
import json


def save(content: dict | str | pd.DataFrame, filepath: str | Path):
    """
    Save content to a file.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    # Make dir if not exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content, pd.DataFrame):
        if "image" in content.columns:
            content = postprocess_image_data(content, filepath)
        content.to_parquet(filepath)
    elif isinstance(content, torch.nn.Module):
        torch.save(content.state_dict(), filepath)
    elif isinstance(content, dict):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(content, f, indent=4)
    else:
        raise TypeError("Content must be a dataframe, model, or dictionary")


def postprocess_image_data(data: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """
    Postprocess the image data in the DataFrame by saving the images to the same directory
    as the parquet file. Requires the 'image' and 'image_id' columns.

    Args:
    -----
    data: pd.DataFrame
        The DataFrame to postprocess.
    filepath: Path
        The path to the parquet file.

    Returns:
    --------
    pd.DataFrame
        The postprocessed DataFrame with 'image_path' column added and 'image' column dropped.
    """
    # Check if required columns are present
    if "image" not in data.columns or "image_id" not in data.columns:
        raise ValueError("DataFrame must contain 'image' and 'image_id' columns.")

    # Check if image_id is unique
    if not data.image_id.is_unique:
        raise ValueError("'image_id' must be unique.")

    # Create image directory with same name as parquet file
    image_dir = filepath.parent / filepath.stem
    image_dir.mkdir(exist_ok=True)

    data.loc[:, "image_path"] = data.image_id.apply(lambda id: str(image_dir / (id + ".jpg")))

    for __, row in data.iterrows():
        if row["image"] is not None:
            im = Image.fromarray(row["image"])
            im.save(row["image_path"])

    data = data.drop(columns=["image"], axis=1)
    return data


def format_dict_for_text(dict_content: dict) -> str:
    """
    Format a dictionary for text output.
    """
    return "\n".join([f"{k}: {v}" for k, v in dict_content.items()])
