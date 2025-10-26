"""Please note the frontend cannot be run inside the GPU server. It must have access to a local port."""

import streamlit as st
from tabs.annotation import render_annotation_page
from tabs.results import render_results_page
from tabs.error_viewing import render_error_viewing_page
from tabs.uncertainty_viewing import render_uncertainty_viewing_page


def home():
    # Info page
    st.title("🦌 WildlifeML: Machine Learning Framework")

    st.markdown("""
    ## Welcome to WildlifeML

    This is a flexible machine learning framework for image classification tasks. The system
    can be adapted for various image classification applications.
    """)

    st.markdown("""
    ### Key Features:

    • **Configurable Models**: Choose from ResNet, VGG, DenseNet architectures
    • **Flexible Data Pipeline**: Works with different image datasets and formats
    • **Customizable Evaluation**: Adjust metrics and analysis methods
    • **Extensible Design**: Easy to modify for new classification tasks
    """)

    st.markdown("""
    ### 📋 Interface Sections:
    """)

    st.write("### 🏷️ Annotation Interface")
    st.write("• Manually label images with custom classification categories")
    st.write("• Navigate through detected objects efficiently")
    st.write("• Create custom annotation classes")

    st.write("### 📊 Model Results")
    st.write("• View comprehensive performance metrics for trained models")
    st.write("• Compare model accuracy, precision, recall, and F1 scores")
    st.write("• Review training specifications and class distributions")

    st.write("### ❌ Error Viewing")
    st.write("• Examine misclassified images to understand model weaknesses")
    st.write("• Browse through prediction errors with confidence scores")
    st.write("• Identify patterns in classification mistakes")

    st.write("### ❓ Uncertainty Viewing")
    st.write("• Review images where the model is uncertain about predictions")
    st.write("• Analyze confidence scores and uncertainty thresholds")

    st.markdown("""
    ### Getting Started:

    Navigate through the sidebar to explore the current wildlife classification example.
    The framework can be adapted for other image classification tasks by modifying
    the configuration files and data pipeline.
    """)

    st.write(
        "For more information, visit our [GitHub repository](https://github.com/slds-lmu/wildlife-age-sex)."
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
