"""Please note the frontend cannot be run inside the GPU server. It must have access to a local port."""

import streamlit as st
from tabs.annotation import render_annotation_page
from tabs.results import render_results_page
from tabs.error_viewing import render_error_viewing_page


def home():
    # Info page
    st.title("Welcome to the Wildlife Age-Sex Classification App")
    st.write(
        """This project helps classify wildlife based on age and sex using machine learning models.
        This portal is designed to help annotate images of wildlife and compare the performance of
        different trained models.
        """
    )
    st.write("Navigate through the sidebar to access different features of the app.")
    st.write(
        "For more information, visit our [GitHub repository](https://github.com/slds-lmu/wildlife-age-sex)."
    )


st.set_page_config(page_title="Wildlife Age-Sex Classification", page_icon="🦌", layout="centered")

page_names_to_funcs = {
    "Home": home,
    "Annotation Interface": render_annotation_page,
    "Model Results": render_results_page,
    "Error Viewing": render_error_viewing_page,
}

page_name = st.sidebar.selectbox("Navigate to", page_names_to_funcs.keys())
page_names_to_funcs[page_name]()
