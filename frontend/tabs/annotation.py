import streamlit as st
import os
import pandas as pd
import json
from PIL import Image, ImageDraw
from pathlib import Path

ANNOTATION_CONFIG = {
    "class_name": st.column_config.TextColumn(
        "Class Name", help="The name of the class to annotate"
    ),
    "class_labels": st.column_config.TextColumn(
        "Class Labels", help="The labels of the class to annotate, comma separated"
    ),
}


def render_annotation_page():
    """Render evaluation results for a specific model."""
    st.title("WIP: Annotation")

    if "image_dir" not in st.session_state or "annotation_config" not in st.session_state:
        render_config_picker()

    else:
        annotation_config = st.session_state.annotation_config
        image_dir = st.session_state.image_dir
        annotations_path = st.session_state.annotations_path
        # Display an info section with image directory and class definitions
        render_config_info(image_dir, annotation_config)
        render_coding_interface(image_dir, annotations_path, annotation_config)
        st.divider()
        st.button("Exit", key="exit_annotation", on_click=exit_annotation)
        return


def render_config_picker():
    st.write("## Select Image Directory")
    st.write(
        "If the image directory already contains an annotation config, it will be loaded. Otherwise, you can create a new one."
    )
    image_dir = st.selectbox("Select image directory", get_image_dirs())
    if st.button("Continue", key="select_image_dir"):
        st.session_state.image_dir = image_dir

    if "image_dir" in st.session_state:
        st.write(f"Image directory: {st.session_state.image_dir}")
        # Get annotation config from file or create default
        annotation_config = get_annotation_config(image_dir)
        if annotation_config:
            # Store config in session_state before rerun
            st.session_state.annotation_config = annotation_config
            st.session_state.image_dir = image_dir
            annotations_path = Path("data") / image_dir / "annotations.csv"
            st.session_state.annotations_path = annotations_path
            st.rerun()

        config_df = pd.DataFrame([{"class_name": "ex_class", "class_labels": "label_1, label_2"}])
        st.write("## Edit Class Definitions")
        st.info("Class names must be unique. Class labels must be a comma separated list.")
        config_df = st.data_editor(
            config_df[["class_name", "class_labels"]],
            use_container_width=True,
            num_rows="dynamic",
        )
        if st.button("Continue to annotation", key="config_submit"):
            if config_df.class_name.duplicated().any():
                st.error("Class names must be unique.")
                return

            # Process class_labels from comma-separated strings to lists
            config_df["class_labels"] = config_df["class_labels"].apply(
                lambda x: [label.strip() for label in x.split(",")]
            )

            # Convert (back) to JSON format and save
            annotation_config = {"classes": config_df.to_dict("records")}
            with open(Path("data") / image_dir / "annotation_config.json", "w") as f:
                json.dump(annotation_config, f, indent=2)

            # Store config in session_state before rerun
            st.session_state.annotation_config = annotation_config
            st.session_state.image_dir = image_dir
            annotations_path = Path("data") / image_dir / "annotations.csv"
            st.session_state.annotations_path = annotations_path
            st.rerun()


def get_image_dirs():
    """Get all image directories in the data directory that do not have a csv file in them."""
    return [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]


def get_annotation_config(image_dir: str):
    """Get the class definitions for a specific image directory."""
    annotation_config_path = Path("data") / image_dir / "annotation_config.json"
    if not os.path.exists(annotation_config_path):
        return
    with open(annotation_config_path, "r") as f:
        return json.load(f)


def render_config_info(image_dir: str, annotation_config: dict):
    with st.expander("Annotation Info", expanded=True):
        st.markdown(f"**Image Directory:** `{image_dir}`")
        st.markdown("**Class Definitions:**")
        for cls in annotation_config["classes"]:
            class_name = cls["class_name"]
            class_labels = ", ".join(cls["class_labels"]) if cls["class_labels"] else "None"
            st.markdown(f"- **{class_name}**: {class_labels}")


def render_coding_interface(image_dir: str, annotations_path: str, annotation_config: dict):
    bboxes = load_bbox_results(image_dir)
    unlabeled_bboxes = get_bboxes_to_code(annotations_path, bboxes)

    if not unlabeled_bboxes:
        st.error("No bounding boxes to annotate.")
        return

    bbox = unlabeled_bboxes[0]
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image")
            render_bbox_image(bbox)

        with col2:
            st.subheader("Annotation")
            bbox_annotations = render_bbox_annotation_options(bbox, annotation_config)
            # Next button to save and move to next bbox
            if st.button("Save & Next", key=f"save_next_{bbox['bbox_id']}"):
                # Append annotation to CSV
                df = pd.DataFrame([bbox_annotations])
                if annotations_path.exists():
                    df.to_csv(annotations_path, mode="a", header=False, index=False)
                else:
                    df.to_csv(annotations_path, mode="w", header=True, index=False)
                st.rerun()


def load_bbox_results(image_dir: str):
    """
    Load Megadetector results for a specific image directory.
    Returns a list of dicts with image file, bbox, and other info.
    """
    # Path to the JSON file generated by Megadetector
    json_path = Path("data") / image_dir / "md_unlabeled.json"
    if not os.path.exists(json_path):
        st.error(f"No bounding box JSON found in {image_dir}")
        return []

    with open(json_path) as f:
        bbox_data = json.load(f)

    # Each key is a unique detection, value is a dict with bbox info
    results = []
    for key, value in bbox_data.items():
        result = {
            "image_path": Path("data") / image_dir / value.get("file"),
            "bbox": value.get("bbox"),
            "category": value.get("category"),
            "confidence": value.get("confidence"),
            "bbox_id": key,
        }
        results.append(result)

    return results


def get_bboxes_to_code(annotations_path: str, bboxes: list):
    try:
        existing_annotations = pd.read_csv(annotations_path).bbox_id.tolist()
    except FileNotFoundError:
        existing_annotations = []

    if not bboxes:
        st.warning("No bounding boxes found.")
        return None

    elif len(existing_annotations) == len(bboxes):
        st.success("All bounding boxes annotated!")
        return None
    else:
        return [
            bbox
            for bbox in bboxes
            if bbox["bbox_id"] not in existing_annotations and os.path.exists(bbox["image_path"])
        ]


def render_bbox_image(bbox: dict):
    if os.path.exists(bbox["image_path"]):
        image = Image.open(bbox["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        if bbox["bbox"]:
            width, height = image.size
            try:
                x_min, y_min, bbox_width, bbox_height = bbox["bbox"]
                assert (
                    x_min >= 0
                    and y_min >= 0
                    and x_min + bbox_width <= 1
                    and y_min + bbox_height <= 1
                )
                x_coords = (int(x_min * width), int((x_min + bbox_width) * width))
                y_coords = (int(y_min * height), int((y_min + bbox_height) * height))
            except AssertionError:
                x_min, y_min, x_max, y_max = bbox["bbox"]
                x_coords = (int(x_min * width), int(x_max * width))
                y_coords = (int(y_min * height), int(y_max * height))
            # use coordinates to draw rectangle
            box = [x_coords[0], y_coords[0], x_coords[1], y_coords[1]]
            draw.rectangle(box, outline="red", width=3)
        st.image(image, caption=bbox["image_path"], use_container_width=True)
    else:
        st.error(f"Image not found: {bbox['image_path']}")


def render_bbox_annotation_options(bbox: dict, annotation_config: dict):
    bbox_annotations = {
        "bbox_id": bbox["bbox_id"],
        "image_path": bbox["image_path"],
    }
    for cls in annotation_config["classes"]:
        label = cls["class_name"]
        options = cls["class_labels"]

        selected = st.selectbox(
            label=label,
            options=options,
            key=f"{bbox['bbox_id']}_{label}",
        )
        bbox_annotations[label] = selected
    return bbox_annotations


def exit_annotation():
    st.session_state.pop("annotation_config")
    st.session_state.pop("image_dir")
    st.session_state.pop("annotations_path")
