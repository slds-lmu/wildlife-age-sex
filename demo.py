"""Demo script for the wildlifeml package.

This script transforms the data in `test_data` into a format that can be used for training and evaluation.
Run this script and the use the command `poe run-pipeline` to train and evaluate the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import wildlifeml

df = pd.read_json("/home/chemeryc/wildlife-age-sex/data/demo/annotations.json")
df["metadata"] = [np.random.randint(0, 2) for i in df.image_id]
print(df.head())

test_df = pd.read_parquet("/home/chemeryc/wildlife-age-sex/data/demo/labeled_bbox_data.parquet")
print(test_df.head())


wildlifeml.save(df, Path("data") / "demo" / "labeled_bbox_data.parquet")
