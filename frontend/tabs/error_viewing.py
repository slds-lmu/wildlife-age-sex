import streamlit as st
from .image_viewing import (
    get_experiments,
    render_results_summary,
    render_image_slideshow,
    render_training_specs,
)


def render_error_viewing_page():
    st.title("Error Viewing")
    st.write("Here you can view the mislabeled images from each experiment.")

    st.write("## Select Experiment")
    experiment_name = st.selectbox("Select experiment", get_experiments())
    if st.button("Continue", key="select_experiment"):
        st.session_state.experiment_name = experiment_name

    if "experiment_name" in st.session_state:
        st.write(f"Experiment: {st.session_state.experiment_name}")

        # Render results summary (no additional metrics needed for errors)
        render_results_summary(st.session_state.experiment_name)

        # Render error images slideshow
        render_image_slideshow(
            experiment_name=st.session_state.experiment_name,
            file_pattern="*__eval_errors.parquet",
            slideshow_title="Model Errors",
            session_state_key="error_index",
            confidence_column="confidence_score",
        )

        with st.expander("Training Specs"):
            render_training_specs(st.session_state.experiment_name)
