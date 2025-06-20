import streamlit as st
from aura_framework import AURA

st.set_page_config(page_title="AURA Artifact Evaluator", layout="wide")
st.title("AURA: Unified Artifact Research Assessment")

with st.sidebar:
    st.header("Artifact & Conference")
    guideline_path = st.text_input("Conference Guidelines Path", value="../../data/conference_guideline_texts/processed/13_icse_2025.md")
    artifact_json_path = st.text_input("Artifact JSON Path", value="C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\ml-image-classifier_analysis.json")
    conference_name = st.text_input("Conference Name", value="ICSE 2025")
    dims = st.multiselect(
        "Select Evaluation Dimensions",
        ['documentation', 'usability', 'accessibility', 'functionality'],
        default=['documentation', 'usability', 'accessibility', 'functionality']
    )
    if st.button("Evaluate"):
        with st.spinner("Running AURA evaluation..."):
            aura = AURA(guideline_path, artifact_json_path, conference_name)
            results = aura.evaluate(dimensions=dims)
        st.success("Evaluation complete!")
        for dim in dims:
            st.subheader(f"{dim.capitalize()} Evaluation")
            st.code(results[dim])

st.markdown("---")
st.info("AURA is modular and supports evaluation of any research artifact according to conference guidelines. Upload your artifact and guidelines to begin.")
