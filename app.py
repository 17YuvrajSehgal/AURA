import os
import time

import streamlit as st

from scripts.algorithm_4.aura_framework import AURA
from scripts.algorithm_4.repo_analyzer import analyze_github_repository, get_analysis_summary

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

    # Step indicator
    steps = ["1. Repository Input", "2. Analysis", "3. Evaluation"]
    for i, step in enumerate(steps, 1):
        if i == st.session_state.current_step:
            st.markdown(f"**{step}** 🔄")
        elif i < st.session_state.current_step:
            st.markdown(f"✅ {step}")
        else:
            st.markdown(f"⏳ {step}")

    st.markdown("---")

    st.header("⚙️ Configuration")

    # Conference settings
    conference_name = st.text_input("Conference Name", value="ICSE 2025")
    guideline_path = st.text_input(
        "Conference Guidelines Path",
        value="C:\\workplace\\AURA\\data\\conference_guideline_texts\\processed\\13_icse_2025.md"
    )

    # Evaluation options
    st.subheader("Evaluation Dimensions")
    dims = st.multiselect(
        "Select Dimensions",
        ['documentation', 'usability', 'accessibility', 'functionality'],
        default=['documentation', 'usability', 'accessibility', 'functionality']
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

# Main content area
if st.session_state.current_step == 1:
    st.header("📥 Step 1: Repository Input")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("GitHub Repository URL")
        repo_url = st.text_input(
            "Enter GitHub repository URL",
            placeholder="https://github.com/username/repository",
            help="Enter the full GitHub URL of the repository you want to analyze"
        )

        # Example repositories
        st.subheader("📋 Example Repositories")
        example_repos = [
            "https://github.com/sneh2001patel/ml-image-classifier",
            "https://github.com/nntzuekai/Respector",
            "https://github.com/SageSELab/MotorEase",
            "https://github.com/sola-st/PyTy"
        ]

        for repo in example_repos:
            if st.button(f"Use: {repo.split('/')[-1]}", key=f"example_{repo}"):
                repo_url = repo
                st.session_state.repo_url = repo
                st.rerun()

    with col2:
        st.subheader("ℹ️ Information")
        st.info("""
        **What happens next:**
        1. Repository will be cloned locally
        2. Files will be analyzed and categorized
        3. JSON analysis file will be generated
        4. AURA evaluation will be performed
        
        **Supported repositories:**
        - Public GitHub repositories
        - Repositories with documentation files
        - Code repositories with README files
        """)

    if st.button("🚀 Start Analysis", type="primary", disabled=not repo_url):
        if repo_url:
            st.session_state.repo_url = repo_url
            st.session_state.current_step = 2
            st.rerun()

elif st.session_state.current_step == 2:
    st.header("🔍 Step 2: Repository Analysis")

    if not st.session_state.repo_analysis_complete:
        with st.spinner("Analyzing repository..."):
            try:
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Clone and analyze repository
                status_text.text("📥 Cloning repository...")
                progress_bar.progress(25)

                repo_url = st.session_state.repo_url

                # Analyze repository
                status_text.text("🔍 Analyzing repository structure...")
                progress_bar.progress(50)

                repo_name, artifact_json_path, result = analyze_github_repository(repo_url)
                st.session_state.repo_name = repo_name
                st.session_state.artifact_json_path = artifact_json_path

                status_text.text("💾 Analysis results saved!")
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")

                st.session_state.repo_analysis_complete = True

                # Show analysis summary
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("Repository analysis completed successfully!")
                st.markdown('</div>', unsafe_allow_html=True)

                # Display analysis summary
                summary = get_analysis_summary(result)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Repository", repo_name)
                with col2:
                    st.metric("Files Analyzed", summary["total_files"])
                with col3:
                    st.metric("Documentation Files", summary["documentation_files"])
                with col4:
                    st.metric("Code Files", summary["code_files"])

                # Show file structure preview
                with st.expander("📁 Repository Structure Preview", expanded=False):
                    tree_lines = result.get("tree_structure", [])
                    st.code("\n".join(tree_lines[:20]))  # Show first 20 lines
                    if len(tree_lines) > 20:
                        st.info(f"... and {len(tree_lines) - 20} more files/directories")

                time.sleep(1)  # Brief pause to show completion

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)
                st.session_state.current_step = 1

    # Show analysis results and proceed to evaluation
    if st.session_state.repo_analysis_complete:
        st.markdown('<div class="status-box">', unsafe_allow_html=True)
        st.info(f"✅ Repository '{st.session_state.repo_name}' analyzed successfully!")
        st.info(f"📄 Analysis file: {st.session_state.artifact_json_path}")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🎯 Proceed to Evaluation", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

elif st.session_state.current_step == 3:
    st.header("🎯 Step 3: AURA Evaluation")

    if st.session_state.artifact_json_path and os.path.exists(st.session_state.artifact_json_path):
        st.markdown('<div class="status-box">', unsafe_allow_html=True)
        st.info(f"Evaluating repository: {st.session_state.repo_name}")
        st.info(f"Using analysis file: {st.session_state.artifact_json_path}")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Start Evaluation", type="primary"):
            with st.spinner("Running AURA evaluation..."):
                try:
                    # Initialize AURA framework
                    aura = AURA(
                        guideline_path,
                        st.session_state.artifact_json_path,
                        conference_name,
                        criteria_csv_path
                    )

                    # Run evaluation based on selected mode
                    if evaluation_mode == "Full Evaluation":
                        results = aura.evaluate(dimensions=dims, include_keyword_eval=include_keyword_eval)
                    elif evaluation_mode == "LLM Only":
                        results = aura.evaluate(dimensions=dims, include_keyword_eval=False)
                    elif evaluation_mode == "Keyword Only":
                        if aura.keyword_agent:
                            results = {"keyword_baseline": aura.keyword_agent.evaluate(verbose=False)}
                        else:
                            st.error("Keyword agent not available. Please check the criteria CSV path.")
                            st.stop()
                    elif evaluation_mode == "Comparison Mode":
                        results = aura.compare_evaluations(verbose=False)
                    elif evaluation_mode == "Grounded Evaluation":
                        if not aura.keyword_agent:
                            st.error(
                                "Keyword agent not available for grounded evaluation. Please check the criteria CSV path.")
                            st.stop()
                        results = {}
                        for dim in dims:
                            grounded_result = aura.get_grounded_evaluation(dim, verbose=False)
                            results[dim] = grounded_result

                    st.success("🎉 Evaluation complete!")

                    # Display results based on mode
                    if evaluation_mode == "Comparison Mode":
                        st.subheader("📊 Evaluation Comparison")
                        st.text(results["comparison_notes"])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🤖 LLM Evaluations (Grounded)")
                            for dim in dims:
                                if dim in results["llm_evaluations"]:
                                    st.text_area(f"{dim.capitalize()}", results["llm_evaluations"][dim], height=200)

                        with col2:
                            st.subheader("🔍 Keyword Evaluation")
                            if "keyword_baseline" in results["llm_evaluations"]:
                                keyword_result = results["llm_evaluations"]["keyword_baseline"]
                                if "error" not in keyword_result:
                                    st.text_area("Keyword Summary", keyword_result["summary"], height=400)
                                else:
                                    st.error(f"Keyword evaluation error: {keyword_result['error']}")

                    elif evaluation_mode == "Grounded Evaluation":
                        st.subheader("🔗 Grounded Evaluations (LLM + Keyword Evidence)")

                        for dim, grounded_result in results.items():
                            if "error" in grounded_result:
                                st.error(f"{dim.capitalize()}: {grounded_result['error']}")
                                continue

                            with st.expander(f"{dim.capitalize()} Evaluation", expanded=True):
                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    st.subheader("🤖 LLM Evaluation (Grounded)")
                                    st.text_area("Evaluation", grounded_result["llm_evaluation"], height=300,
                                                 key=f"llm_{dim}")

                                with col2:
                                    st.subheader("🔍 Keyword Evidence")
                                    if grounded_result["keyword_evidence"]:
                                        evidence = grounded_result["keyword_evidence"]
                                        st.metric("Raw Score", evidence["raw_score"])
                                        st.metric("Weighted Score", f"{evidence['weighted_score']:.2f}")

                                        if evidence["keywords_found"]:
                                            st.write("**Keywords Found:**")
                                            for kw in evidence["keywords_found"][:5]:
                                                st.write(f"• {kw}")
                                            if len(evidence["keywords_found"]) > 5:
                                                st.write(f"... and {len(evidence['keywords_found']) - 5} more")
                                        else:
                                            st.write("No keywords found")
                                    else:
                                        st.write("No keyword evidence available")

                                st.info(grounded_result["grounding_info"])

                    else:
                        # Display individual evaluations
                        for key, value in results.items():
                            if key == "keyword_baseline":
                                st.subheader("🔍 Keyword-Based Evaluation")
                                if isinstance(value, dict) and "error" not in value:
                                    st.text_area("Summary", value["summary"], height=300)

                                    # Show detailed breakdown
                                    if "dimensions" in value:
                                        st.subheader("📊 Dimension Breakdown")
                                        for dim in value["dimensions"]:
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric(f"{dim['dimension'].capitalize()}",
                                                          f"{dim['weighted_score']:.2f}")
                                            with col2:
                                                st.metric("Raw Score", dim['raw_score'])
                                            with col3:
                                                st.metric("Weight", f"{dim['weight']:.3f}")

                                    # Show overall score
                                    st.metric("🎯 Overall Score", f"{value['overall_score']:.2f}")
                                else:
                                    st.error(f"Keyword evaluation error: {value}")
                            else:
                                st.subheader(f"📝 {key.capitalize()} Evaluation")
                                if evaluation_mode == "Full Evaluation":
                                    st.info(
                                        "💡 This LLM evaluation was grounded with keyword evidence to prevent hallucination")
                                st.code(value)

                    # Reset button to start over
                    st.markdown("---")
                    if st.button("🔄 Start New Analysis"):
                        st.session_state.current_step = 1
                        st.session_state.repo_analysis_complete = False
                        st.session_state.artifact_json_path = None
                        st.session_state.repo_name = None
                        st.rerun()

                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    st.exception(e)
    else:
        st.error("Analysis file not found. Please go back to step 2.")
        if st.button("⬅️ Go Back to Analysis"):
            st.session_state.current_step = 2
            st.rerun()

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
        <strong>🧩 Modular Design:</strong> Select specific dimensions to evaluate
    </p>
</div>
""", unsafe_allow_html=True)
