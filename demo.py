"""Demo script for the wildlifeml package.

This script transforms the data in `test_data` into a format that can be used for training and evaluation.
Run this script and the use the command `poe run-pipeline` to train and evaluate the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import wildlifeml

# df = pd.read_json("test_data/data__red_deer_cropped/md_unlabeled.json").T
# df.loc[:, "image_path"] = df.apply(
#     lambda row: f"test_data/data__red_deer_cropped/full_images/{row['file'].split('/')[-1]}",
#     axis=1,
# )
# # use indices for unique image_id, should not be a path!!
# df.loc[:, "image_id"] = [
#     i.split("/")[-1].split(".")[0] + str(idx) for idx, i in enumerate(df.index)
# ]
# df["age"] = df["file"].apply(lambda x: x.split("/")[0].split("_")[0])
# df["sex"] = df["file"].apply(lambda x: x.split("/")[0].split("_")[1])
# df = df.drop(columns=["file", "category"]).reset_index(drop=True)

# df["dummy_metadata"] = [np.random.randint(0, 5) for i in range(len(df))]
# wildlifeml.save(df, Path("test_data/labeled_bbox_data.parquet"))


tgt_img_base_path = "/home/chemeryc/wildlife-age-sex/data/raw/"

# Performing the detection on the single image

df = pd.DataFrame()
for class_dir in [
    "adult_female",
    "adult_male",
    "unknown_unknown",
    "adult_unknown",
    "yearling_female",
    "yearling_male",
    "juvenile_unknown",
]:
    class_df = pd.read_json(tgt_img_base_path + class_dir + "/md_unlabeled.json").T
    # Keep original index as bbox_id
    class_df = class_df.rename_axis("bbox_id").reset_index()
    class_df = class_df.drop(columns=["category"])
    df = pd.concat([df, class_df])

metadata_df = pd.read_csv(tgt_img_base_path + "reddeer_ageclasses_image_info.csv", sep=";")
# Convert Date column from YYYY-MM-DD string to datetime and extract month
metadata_df["Date"] = pd.to_datetime(metadata_df["Date"], format="%Y-%m-%d")
metadata_df["is_summer"] = metadata_df["Date"].dt.month.between(5, 9)


metadata_df.loc[:, "image_id"] = metadata_df.apply(
    lambda x: x["Station"] + "_" + x["Session"] + "_" + x["Trigger"] + x["Trigger_Sub"], axis=1
)
df = df.merge(metadata_df, on="image_id", how="left")
df.loc[:, "image_id"] = df.loc[:, "bbox_id"]
df = df.drop(columns=["bbox_id"])

wildlifeml.save(
    df, Path("/home/chemeryc/wildlife-age-sex/data/preprocessed") / "labeled_bbox_data.parquet"
)
