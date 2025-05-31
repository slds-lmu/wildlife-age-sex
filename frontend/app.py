"""Please note the frontend cannot be run inside the GPU server. It must have access to a local port."""

import streamlit as st
import logging
import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import page components
from pages.annotation import render_annotation_page
from pages.results import render_results_page


def main():
    st.set_page_config(page_title="Wildlife Age-Sex Classification", page_icon="🦌", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Annotation", "Model Results"])

    if page == "Annotation":
        render_annotation_page()
    elif page == "Model Results":
        render_results_page()


if __name__ == "__main__":
    main()
