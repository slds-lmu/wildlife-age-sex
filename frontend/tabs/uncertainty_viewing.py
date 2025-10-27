import streamlit as st
from .image_viewing import (
    get_experiments,
    render_results_summary,
    render_image_slideshow,
    render_training_specs,
)


def render_uncertainty_viewing_page():
    st.title("Uncertainty Viewing")
    st.write(
        "Here you can view the images from each experiment where the model is not confident in its predictions."
    )

    st.write("## Select Experiment")
    experiments = get_experiments()

    # Create formatted options with average accuracy
    experiment_options = []
    experiment_names = []
    for name, accuracy in experiments:
        if accuracy == -99:
            # Handle case where no valid results found
            experiment_options.append(f"{name} (No valid results)")
        else:
            experiment_options.append(f"{name} (Avg Accuracy: {accuracy:.2%})")
        experiment_names.append(name)

    selected_option = st.selectbox("Select experiment", options=experiment_options)
    experiment_name = experiment_names[experiment_options.index(selected_option)]

    if st.button("Continue", key="select_experiment"):
        st.session_state.experiment_name = experiment_name

    if "experiment_name" in st.session_state:
        st.write(f"Experiment: {st.session_state.experiment_name}")

        # Render results summary with uncertainty-specific metrics - collapsed by default
        with st.expander("Evaluation Results Summary", expanded=False):
            render_results_summary(
                st.session_state.experiment_name,
                additional_metrics=["n_uncertain_images", "avg_confidence"],
            )

        # Render uncertain images slideshow
        render_image_slideshow(
            experiment_name=st.session_state.experiment_name,
            file_pattern="*__eval_uncertain_images.parquet",
            slideshow_title="Uncertain Images",
            session_state_key="uncertain_index",
            confidence_column="confidence_score",
        )

        with st.expander("Training Specs"):
            render_training_specs(st.session_state.experiment_name)
