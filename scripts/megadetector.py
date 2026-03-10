from PytorchWildlife.models import detection

import torch
from pathlib import Path
import json
import argparse
import re

# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initializing the MegaDetectorV6 model for image detection
# Valid versions are MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e or MDV6-rtdetr-c
DETECTION_MODEL = detection.MegaDetectorV6(
    device=DEVICE, pretrained=True, version="MDV6-yolov10-e"
)


def clean_image_filename(filename: str) -> str:
    """Remove common image extensions from filename (case insensitive)."""
    return re.sub(r"\.(jpg|jpeg|png|JPG|JPEG|PNG)$", "", filename, flags=re.IGNORECASE)


def main(image_dir: Path):
    results = DETECTION_MODEL.batch_image_detection(str(image_dir))

    processed_results = {}
    for result in results:
        bboxes = result["normalized_coords"]
        confidences = result["detections"].confidence
        categories = result["detections"].class_id

        if not len(bboxes) == len(confidences) == len(categories):
            print(
                f"""Error: {result["img_id"]} has {len(bboxes)} bboxes,
                {len(confidences)} confidences, and {len(categories)} categories"""
            )
            continue
        elif any([bboxes is None, confidences is None, categories is None]):
            print(f"Error: {result['img_id']} does not contain any bounding boxes.")
            # Optional: send for manual check
            continue

        for idx, (bbox, confidence, category) in enumerate(
            zip(bboxes, confidences, categories, strict=True)
        ):
            # Extract filename and clean it of any image extensions
            filename = result["img_id"].split("/")[-1]
            clean_filename = clean_image_filename(filename)

            processed_results[f"{idx}__{clean_filename}"] = {
                # Unique bbox id, used as image_id after cropping
                "bbox_id": f"{idx}__{clean_filename}",
                # Unique file name, x.Station + "_" + x.Session + "_" + x.Trigger + x.Trigger_Sub
                "image_id": clean_filename,
                "image_path": result["img_id"],
                "category": int(category),  # 0: animal
                "bbox": [round(float(i), 4) for i in bbox],
                "confidence": round(float(confidence), 3),
            }
    # Save results to a text file in the class directory
    output_path = image_dir / "md_unlabeled.json"
    with open(output_path, "w") as f:
        json.dump(processed_results, f, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--image-dir", type=str, required=True, default="data/demo/raw")
    args = args.parse_args()
    main(Path(args.image_dir))
