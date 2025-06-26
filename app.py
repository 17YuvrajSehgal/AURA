import os
import time
import glob
import re

import streamlit as st

from scripts.algorithm_4.aura_framework import AURAFramework

st.set_page_config(
    page_title="AURA Artifact Evaluator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 AURA: Unified Artifact Research Assessment</h1>
    <p>Automated Repository Analysis & Evaluation Pipeline</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'repo_analysis_complete' not in st.session_state:
    st.session_state.repo_analysis_complete = False
if 'artifact_json_path' not in st.session_state:
    st.session_state.artifact_json_path = None
if 'repo_name' not in st.session_state:
    st.session_state.repo_name = None

# Sidebar configuration
with st.sidebar:
    st.header("🎯 Workflow Steps")
    st.markdown("**1. Configuration** ✅")
    st.markdown("**2. Evaluation** ⏳")
    st.markdown("---")
    st.header("⚙️ Configuration")

    # --- Conference dropdown and auto-path logic ---
    processed_dir = os.path.join("data", "conference_guideline_texts", "processed")
    md_files = glob.glob(os.path.join(processed_dir, "*.md"))
    conference_options = []
    conference_map = {}
    for f in md_files:
        fname = os.path.basename(f)
        # Try to extract a readable name, e.g., '13_icse_2025.md' -> 'ICSE 2025'
        match = re.match(r"\d+_([a-zA-Z]+)_?(\d{4})?\.md", fname)
        if match:
            conf = match.group(1).upper()
            year = match.group(2) if match.group(2) else ''
            label = f"{conf} {year}".strip()
        else:
            # fallback: use filename without extension
            label = fname.replace(".md", "").replace("_", " ").title()
        conference_options.append(label)
        conference_map[label] = f
    conference_options = sorted(conference_options)

    # Select conference
    selected_conference = st.selectbox("Select Conference", conference_options, index=conference_options.index("ICSE 2025") if "ICSE 2025" in conference_options else 0)
    guideline_path = os.path.abspath(conference_map[selected_conference])

    st.text_input("Conference Name", value=selected_conference, disabled=True)
    st.text_input("Conference Guidelines Path", value=guideline_path, disabled=True)

    # --- End conference dropdown logic ---

    # Evaluation options
    st.subheader("Evaluation Dimensions")
    dims = st.multiselect(
        "Select Dimensions",
        ['documentation', 'usability', 'accessibility', 'functionality', 'experimental', 'reproducibility'],
        default=['documentation', 'usability', 'accessibility', 'functionality', 'experimental', 'reproducibility']
    )

    # Keyword evaluation
    include_keyword_eval = st.checkbox("Include Keyword-Based Evaluation", value=True)
    criteria_csv_path = None
    if include_keyword_eval:
        criteria_csv_path = st.text_input(
            "Criteria CSV Path",
            value="C:\\workplace\\AURA\\algo_outputs\\algorithm_1_output\\algorithm_1_artifact_evaluation_criteria.csv"
        )

    evaluation_mode = st.selectbox(
        "Evaluation Mode",
        ["Full Evaluation", "LLM Only", "Keyword Only", "Comparison Mode", "Grounded Evaluation"]
    )

# Main content area: Only evaluation
st.header("🎯 AURA Artifact Evaluation")

# Let user select or input artifact JSON path
artifact_json_path = st.text_input(
    "Artifact JSON Path",
    value=st.session_state.artifact_json_path if 'artifact_json_path' in st.session_state and st.session_state.artifact_json_path else "",
    help="Provide the path to the artifact analysis JSON file."
)
if artifact_json_path:
    st.session_state.artifact_json_path = artifact_json_path

if artifact_json_path and os.path.exists(artifact_json_path):
    st.markdown('<div class="status-box">', unsafe_allow_html=True)
    st.info(f"Using analysis file: {artifact_json_path}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Start Evaluation", type="primary"):
        with st.spinner("Running AURA evaluation..."):
            try:
                # Initialize AURAFramework
                framework = AURAFramework(artifact_json_path)
                result = framework.evaluate_artifact()
                st.success("🎉 Evaluation complete!")
                # Show overall results
                st.subheader("Overall Results")
                st.metric("Total Weighted Score", f"{result.total_weighted_score:.3f}")
                st.metric("Acceptance Threshold", "0.750")
                if result.acceptance_prediction:
                    st.success("✅ Prediction: ACCEPTED")
                else:
                    st.error("❌ Prediction: REJECTED")
                st.info(result.overall_justification)
                if result.recommendations:
                    st.subheader("Recommendations")
                    for i, rec in enumerate(result.recommendations, 1):
                        st.write(f"{i}. {rec}")
                # Show per-dimension results
                st.subheader("Dimension Evaluations")
                for criterion in result.criteria_scores:
                    with st.expander(f"{criterion.dimension.capitalize()} Evaluation", expanded=False):
                        st.metric("Score", f"{criterion.llm_evaluated_score:.3f}")
                        st.metric("Weight", f"{criterion.normalized_weight:.3f}")
                        st.write(f"**Justification:** {criterion.justification}")
                        if criterion.evidence:
                            st.write("**Evidence:**")
                            for ev in criterion.evidence[:5]:
                                st.write(f"- {ev}")
                            if len(criterion.evidence) > 5:
                                st.write(f"... and {len(criterion.evidence) - 5} more")
                        st.write(f"**LLM Justification:** {criterion.llm_justification}")
                        if criterion.llm_evidence:
                            st.write("**LLM Additional Evidence:**")
                            for ev in criterion.llm_evidence[:5]:
                                st.write(f"- {ev}")
                            if len(criterion.llm_evidence) > 5:
                                st.write(f"... and {len(criterion.llm_evidence) - 5} more")
                # Reset button
                st.markdown("---")
                if st.button("🔄 Start New Evaluation"):
                    st.session_state.artifact_json_path = ""
                    st.rerun()
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
                st.exception(e)
else:
    st.warning("Please provide a valid artifact JSON path to start evaluation.")

# Footer with information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>🚀 AURA Framework Features</h4>
    <p>
        <strong>🤖 LLM-based Evaluation:</strong> Qualitative assessment with detailed reasoning and suggestions<br>
        <strong>🔍 Keyword-based Evaluation:</strong> Quantitative scoring based on guideline-derived criteria<br>
        <strong>🔗 Grounded Evaluation:</strong> LLM assessments grounded with keyword evidence to prevent hallucination<br>
        <strong>📊 Comparison Mode:</strong> Compare both evaluation approaches<br>
        <strong>🧩 Modular Design:</strong> Select specific dimensions to evaluate<br>
        <strong>📄 Artifact-Specific CSV:</strong> Results saved with repository name for easy identification<br>
        <strong>⏱️ Performance Tracking:</strong> Detailed timing data for analysis and evaluation phases<br>
        <strong>📈 Performance Analytics:</strong> Timing data stored in CSV for performance analysis
    </p>
</div>
""", unsafe_allow_html=True)
