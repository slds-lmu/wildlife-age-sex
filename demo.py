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


tgt_img_base_path = "/Users/clarechemery/Desktop/RD_Classes_F/"

# Performing the detection on the single image

df = pd.DataFrame()
for class_dir in ["F1", "F2"]:
    class_df = pd.read_json(tgt_img_base_path + class_dir + "/md_unlabeled.json").T
    # Keep original index as bbox_id
    class_df = class_df.reset_index(drop=True)
    class_df["image_id"] = class_df.image_path.apply(
        lambda x: ".".join(x.split("/")[-1].split(".")[:-1])
    )
    class_df["bbox_id"] = class_df.bbox_id.astype(str).str[:3] + class_df.image_id
    class_df = class_df.drop(columns=["category"])
    df = pd.concat([df, class_df])
print(df.head())

metadata_df_1 = pd.read_csv(tgt_img_base_path + "ImageData_F1.csv", sep=",")
metadata_df_2 = pd.read_csv(tgt_img_base_path + "ImageData_F2.csv", sep=",")
metadata_df = pd.concat([metadata_df_1, metadata_df_2], axis=0)
# Convert Date column from YYYY-MM-DD string to datetime and extract month
metadata_df["Date"] = pd.to_datetime(metadata_df["DateTime"], format="ISO8601")
metadata_df["is_summer"] = metadata_df["Date"].dt.month.between(5, 9)


metadata_df.loc[:, "image_id"] = metadata_df.apply(
    lambda x: x["RelativePath"].replace("\\", ".") + "." + x["File"].replace(".JPG", ""), axis=1
)
print(metadata_df.head())
df = df.merge(metadata_df, on="image_id", how="left")
df.loc[:, "image_id"] = df.loc[:, "bbox_id"]
df = df.drop(columns=["bbox_id"])

print(df.image_path[:5])

wildlifeml.save(df, Path(tgt_img_base_path) / "labeled_bbox_data.parquet")
