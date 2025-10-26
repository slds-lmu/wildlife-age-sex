"""Please note the frontend cannot be run inside the GPU server. It must have access to a local port."""

import streamlit as st
from tabs.annotation import render_annotation_page
from tabs.results import render_results_page
from tabs.error_viewing import render_error_viewing_page
from tabs.uncertainty_viewing import render_uncertainty_viewing_page


def home():
    # Page header
    st.title("🦌 WildlifeML")
    st.caption("A Lightweight and Accessible Machine Learning Pipeline for Ecologists")

    st.markdown("---")

    st.header("Welcome to the WildlifeML App!")
    st.markdown(
        """
**WildlifeML** is a lightweight interface for ecological image classification tasks.
It's adaptable for many applications and based on an open framework described in our paper:

> *Beyond Off-the-Shelf Models: A Lightweight and Accessible Machine Learning Pipeline for Ecologists Working with Image Data*

[View the paper on Arxiv](https://arxiv.org/demo-link)
        """
    )

    st.markdown("---")
    st.header("📋 Interface Sections")
    st.markdown("Explore the main components:")

    # Use columns to visually organize app features
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏷️ Annotation Interface")
        st.markdown(
            """
- **Manually label images** with your own classes
- **Navigate detected objects** efficiently
            """
        )

        st.subheader("❌ Error Viewing")
        st.markdown(
            """
- **Examine misclassified images**
- **See prediction errors with confidence scores**
- **Find patterns in classification mistakes**
            """
        )
    with col2:
        st.subheader("📊 Model Results")
        st.markdown(
            """
- **View interactive performance metrics**
- **Compare accuracy, precision, recall, F1**
- **Review training specs and class distributions**
            """
        )

        st.subheader("❓ Uncertainty Viewing")
        st.markdown(
            """
- **Review images with uncertain predictions**
- **Analyze confidence and threshold values**
            """
        )

    st.markdown("---")
    st.header("🚀 Getting Started")
    st.markdown(
        """
- **Use the sidebar menu** to navigate between annotation, results, error analysis, and uncertainties.
- Adapt this framework for your own data by modifying config files and the pipeline in the linked repository as needed.
        """
    )
    st.info(
        "For more information, visit our [GitHub repository](https://github.com/slds-lmu/wildlife-age-sex).",
        icon="🌐",
    )


st.set_page_config(page_title="Wildlife Age-Sex Classification", page_icon="🦌", layout="centered")

page_names_to_funcs = {
    "Home": home,
    "Annotation Interface": render_annotation_page,
    "Model Results": render_results_page,
    "Error Viewing": render_error_viewing_page,
    "Uncertainty Viewing": render_uncertainty_viewing_page,
}

page_name = st.sidebar.selectbox("Navigate to", page_names_to_funcs.keys())
page_names_to_funcs[page_name]()
