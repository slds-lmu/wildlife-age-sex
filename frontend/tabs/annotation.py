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
    st.title("Annotation")

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
            annotations_path = Path("data") / image_dir / "annotations.json"
            st.session_state.annotations_path = annotations_path
            st.rerun()

        config_df = pd.DataFrame([{"class_name": "ex_class", "class_labels": "label_1, label_2"}])
        st.write("## Edit Class Definitions")
        st.write("Class names must be unique. Class labels must be a comma separated list.")
        st.info(
            "The first class in each list will be used as the default class. Tip: put the most common class first for extra speed!"
        )
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
            annotations_path = Path("data") / image_dir / "annotations.json"
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
        # get_bboxes_to_code already shows appropriate messages (success or warning)
        return

    # Get total number of images in directory and completed annotations
    total_images = len(bboxes)
    completed_annotations = get_completed_annotation_count(annotations_path)

    # Initialize current index in session state if not present
    if "current_bbox_index" not in st.session_state:
        st.session_state.current_bbox_index = 0

    # Ensure index is within bounds
    if st.session_state.current_bbox_index >= len(unlabeled_bboxes):
        st.session_state.current_bbox_index = 0

    bbox = unlabeled_bboxes[st.session_state.current_bbox_index]

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image")
            render_bbox_image(bbox)

        with col2:
            st.subheader("Annotation")
            bbox_annotations = render_bbox_annotation_options(bbox, annotation_config)

            # Navigation buttons
            col_prev, col_save, col_skip = st.columns(3)

            with col_prev:
                if st.button(
                    "← Previous",
                    key=f"prev_{bbox['bbox_id']}",
                    disabled=st.session_state.current_bbox_index == 0,
                ):
                    st.session_state.current_bbox_index = max(
                        0, st.session_state.current_bbox_index - 1
                    )
                    st.rerun()

            with col_save:
                if st.button("Save/Next", key=f"save_next_{bbox['bbox_id']}"):
                    # Append annotation to JSON
                    save_annotation_to_json(annotations_path, bbox_annotations)
                    # Reset index to 0 since the current bbox will be filtered out
                    st.session_state.current_bbox_index = 0
                    st.rerun()

            with col_skip:
                if st.button("Skip →", key=f"skip_{bbox['bbox_id']}"):
                    # Move to next bbox without saving
                    st.session_state.current_bbox_index += 1
                    st.rerun()

            # Progress indicator - show total images vs completed annotations
            st.progress(completed_annotations / total_images if total_images > 0 else 0)
            st.caption(f"Already Annotated: {completed_annotations} of {total_images} images")


def load_bbox_results(image_dir: str):
    """
    Load Megadetector results for a specific image directory.
    Searches for all md_unlabeled.json files at any level within the directory and combines them.
    Returns a list of dicts with image file, bbox, and other info.
    """
    # Search for all md_unlabeled.json files at any level within the image directory
    image_dir_path = Path("data") / image_dir
    md_files = []

    # Recursively search for md_unlabeled.json files
    for file_path in image_dir_path.rglob("md_unlabeled.json"):
        if file_path.is_file():
            md_files.append(file_path)

    if not md_files:
        st.error(f"No md_unlabeled.json files found in {image_dir}")
        return []

    # Combine all md_unlabeled.json files into one dataset
    combined_bbox_data = {}
    for md_file in md_files:
        try:
            with open(md_file) as f:
                bbox_data = json.load(f)

            # Add source file info to each entry for tracking
            for key, value in bbox_data.items():
                value["source_file"] = str(md_file.parent.name + "/" + md_file.name)
                combined_bbox_data[key] = value

        except (json.JSONDecodeError, FileNotFoundError) as e:
            st.warning(f"Could not load {md_file}: {e}")
            continue

    if not combined_bbox_data:
        st.error(f"No valid MegaDetector data found in {image_dir}")
        return []

    # Show info about combined files
    st.info(
        f"Found and combined {len(md_files)} md_unlabeled.json files: {[f.parent.name + '/' + f.name for f in md_files]}"
    )

    # Each key is a unique detection, value is a dict with bbox info
    results = []
    for key, value in combined_bbox_data.items():
        result = {
            "bbox_id": key,
            "image_path": value.get("image_path"),
            "bbox": value.get("bbox"),
            "image_id": value.get("image_id"),
            "category": value.get("category"),
            "confidence": value.get("confidence")
            or value.get("conf"),  # Handle different field names
            "source_file": value.get("source_file", "unknown"),
        }
        results.append(result)

    return results


def save_annotation_to_json(annotations_path: str, bbox_annotations: dict):
    """Save annotation to JSON file."""
    # Load existing annotations or create new list
    try:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
    except FileNotFoundError:
        annotations = []

    # Add new annotation with source file tracking
    annotations.append(bbox_annotations)

    # Save back to file with proper formatting
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)


def get_bboxes_to_code(annotations_path: str, bboxes: list):
    try:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        existing_annotations = [ann["image_id"] for ann in annotations]
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
            draw.rectangle(box, outline="yellow", width=4)
        st.image(image, caption=bbox["image_path"], use_container_width=True)
    else:
        st.error(f"Image not found: {bbox['image_path']}")


def render_bbox_annotation_options(bbox: dict, annotation_config: dict):
    bbox_annotations = {
        "image_id": bbox["bbox_id"],
        "original_image_id": bbox["image_id"],
        "image_path": bbox["image_path"],
        "bbox": bbox["bbox"],
        "category": bbox["category"],
        "confidence": bbox["confidence"],
        "source_file": bbox.get("source_file", "unknown"),
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
    st.session_state.pop("current_bbox_index", None)


def get_completed_annotation_count(annotations_path: str):
    """Get the number of completed annotations."""
    try:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        return len(annotations)
    except FileNotFoundError:
        return 0
