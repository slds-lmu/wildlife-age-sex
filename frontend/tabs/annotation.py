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
    if (
        st.session_state.get("class_df") is None
        or st.session_state.get("image_dir") is None
        or st.session_state.get("confirmed_overwrite") is None
    ):
        render_config_picker()

    if (
        st.session_state.get("class_df") is not None
        and st.session_state.get("image_dir") is not None
        and st.session_state.get("confirmed_overwrite") is not None
    ):
        class_df = st.session_state.class_df
        image_dir = st.session_state.image_dir
        annotations_path = st.session_state.annotations_path
        # Display an info section with image directory and class definitions
        render_config_info(image_dir, class_df)
        render_coding_interface(image_dir, annotations_path, class_df)
        st.divider()
        st.warning(
            """⚠️ WARNING: Exiting the annotation interface will finalize the annotation file.
            You cannot return to this folder of images without losing all your annotations."""
        )
        st.button("Exit", key="exit_annotation", on_click=exit_annotation)
        return


def render_config_picker():
    image_dir = st.selectbox("Select image directory", get_image_dirs())

    class_df = st.data_editor(
        pd.DataFrame({"class_name": ["ex_class"], "class_labels": ["label_1, label_2"]}),
        use_container_width=True,
        num_rows="dynamic",
    )
    class_df["class_labels"] = class_df["class_labels"].str.split(",")
    class_df["class_labels"] = class_df["class_labels"].apply(
        lambda x: [label.strip() for label in x]
    )

    if st.button("Continue to annotation", key="config_submit"):
        # Store config in session_state before rerun
        st.session_state.class_df = class_df
        st.session_state.image_dir = image_dir
        st.session_state.annotations_path = (
            annotations_path := Path("data") / image_dir / "annotations.csv"
        )
        if os.path.exists(annotations_path):
            confirm_overwrite(annotations_path)
        else:
            st.session_state.confirmed_overwrite = True
            st.rerun()


def get_image_dirs():
    """Get all image directories in the data directory that do not have a csv file in them."""
    return [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]


def confirm_overwrite(annotations_path: str):
    st.warning(
        "You are about to start annotating a folder of images that already has annotations. This will overwrite the existing annotations."
    )
    if st.button(
        "Continue with overwrite",
        key="confirm_overwrite",
        on_click=annotations_path.unlink(),
    ):
        st.session_state.confirmed_overwrite = True


def render_config_info(image_dir: str, class_df: pd.DataFrame):
    with st.expander("Annotation Info", expanded=True):
        st.markdown(f"**Image Directory:** `{image_dir}`")
        st.markdown("**Class Definitions:**")
        for __, row in class_df.iterrows():
            class_name = row["class_name"]
            class_labels = ", ".join(row["class_labels"])
            st.markdown(f"- **{class_name}**: {class_labels}")


def render_coding_interface(image_dir: str, annotations_path: str, class_df: pd.DataFrame):
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
            bbox_annotations = render_bbox_annotation_options(bbox, class_df)
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
        # bbox format: [x_start, y_start, x_end, y_end] normalized (0-1)
        if bbox["bbox"]:
            w, h = image.size
            # Unpack coordinates assuming format is [x_min, y_min, width, height]
            x_min, y_min, width, height = bbox["bbox"]
            # Convert coordinates to pixel values
            box = [x_min * w, y_min * h, (x_min + width) * w, (y_min + height) * h]
            draw.rectangle(box, outline="red", width=3)
        st.image(image, caption=bbox["image_path"], use_container_width=True)
    else:
        st.error(f"Image not found: {bbox['image_path']}")


def render_bbox_annotation_options(bbox: dict, class_df: pd.DataFrame):
    bbox_annotations = {
        "bbox_id": bbox["bbox_id"],
        "image_path": bbox["image_path"],
    }
    for __, row in class_df.iterrows():
        label = row["class_name"]
        options = row["class_labels"]
        selected = st.selectbox(
            label=label,
            options=options,
            key=f"{bbox['bbox_id']}_{label}",
        )
        bbox_annotations[label] = selected
    return bbox_annotations


def exit_annotation():
    st.session_state.pop("class_df")
    st.session_state.pop("image_dir")
    st.session_state.pop("annotations_path")
    st.session_state.pop("confirmed_overwrite")
