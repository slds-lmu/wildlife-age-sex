"""Demo script for the wildlifeml package.

This script transforms the data in `test_data` into a format that can be used for training and evaluation.
Run this script and the use the command `poe run-pipeline` to train and evaluate the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import wildlifeml

df = pd.read_json("test_data/data__red_deer_cropped/md_unlabeled.json").T
df.loc[:, "image_path"] = df.apply(
    lambda row: f"test_data/data__red_deer_cropped/full_images/{row['file'].split('/')[-1]}",
    axis=1,
)
# use indices for unique image_id, should not be a path!!
df.loc[:, "image_id"] = [
    i.split("/")[-1].split(".")[0] + str(idx) for idx, i in enumerate(df.index)
]
df["age"] = df["file"].apply(lambda x: x.split("/")[0].split("_")[0])
df["sex"] = df["file"].apply(lambda x: x.split("/")[0].split("_")[1])
df = df.drop(columns=["file", "category"]).reset_index(drop=True)

df["dummy_metadata"] = [np.random.randint(0, 5) for i in range(len(df))]
wildlifeml.save(df, Path("test_data/labeled_bbox_data.parquet"))
