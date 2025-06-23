import streamlit as st

from aura_framework import AURA

st.set_page_config(page_title="AURA Artifact Evaluator", layout="wide")
st.title("AURA: Unified Artifact Research Assessment")

with st.sidebar:
    st.header("Artifact & Conference")
    guideline_path = st.text_input("Conference Guidelines Path", value="C:\\workplace\\AURA\\data\\conference_guideline_texts\\processed\\13_icse_2025.md")
    artifact_json_path = st.text_input("Artifact JSON Path", value="C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\PyTy_analysis.json")
    conference_name = st.text_input("Conference Name", value="ICSE 2025")
    
    st.header("Evaluation Options")
    dims = st.multiselect(
        "Select Evaluation Dimensions",
        ['documentation', 'usability', 'accessibility', 'functionality'],
        default=['documentation', 'usability', 'accessibility', 'functionality']
    )
    
    # Keyword evaluation options
    st.subheader("Keyword-Based Evaluation")
    include_keyword_eval = st.checkbox("Include Keyword-Based Evaluation", value=True)
    criteria_csv_path = None
    if include_keyword_eval:
        criteria_csv_path = st.text_input(
            "Criteria CSV Path", 
            value="../../algo_outputs/algorithm_1_output/algorithm_1_artifact_evaluation_criteria.csv"
        )
    
    evaluation_mode = st.selectbox(
        "Evaluation Mode",
        ["Full Evaluation", "LLM Only", "Keyword Only", "Comparison Mode", "Grounded Evaluation"]
    )
    
    if st.button("Evaluate"):
        with st.spinner("Running AURA evaluation..."):
            try:
                aura = AURA(guideline_path, artifact_json_path, conference_name, criteria_csv_path)
                
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
                        st.error("Keyword agent not available for grounded evaluation. Please check the criteria CSV path.")
                        st.stop()
                    results = {}
                    for dim in dims:
                        grounded_result = aura.get_grounded_evaluation(dim, verbose=False)
                        results[dim] = grounded_result
                
                st.success("Evaluation complete!")
                
                # Display results based on mode
                if evaluation_mode == "Comparison Mode":
                    st.subheader("Evaluation Comparison")
                    st.text(results["comparison_notes"])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("LLM Evaluations (Grounded)")
                        for dim in dims:
                            if dim in results["llm_evaluations"]:
                                st.text_area(f"{dim.capitalize()}", results["llm_evaluations"][dim], height=200)
                    
                    with col2:
                        st.subheader("Keyword Evaluation")
                        if "keyword_baseline" in results["llm_evaluations"]:
                            keyword_result = results["llm_evaluations"]["keyword_baseline"]
                            if "error" not in keyword_result:
                                st.text_area("Keyword Summary", keyword_result["summary"], height=400)
                            else:
                                st.error(f"Keyword evaluation error: {keyword_result['error']}")
                
                elif evaluation_mode == "Grounded Evaluation":
                    st.subheader("Grounded Evaluations (LLM + Keyword Evidence)")
                    
                    for dim, grounded_result in results.items():
                        if "error" in grounded_result:
                            st.error(f"{dim.capitalize()}: {grounded_result['error']}")
                            continue
                        
                        with st.expander(f"{dim.capitalize()} Evaluation", expanded=True):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("LLM Evaluation (Grounded)")
                                st.text_area("Evaluation", grounded_result["llm_evaluation"], height=300, key=f"llm_{dim}")
                            
                            with col2:
                                st.subheader("Keyword Evidence")
                                if grounded_result["keyword_evidence"]:
                                    evidence = grounded_result["keyword_evidence"]
                                    st.metric("Raw Score", evidence["raw_score"])
                                    st.metric("Weighted Score", f"{evidence['weighted_score']:.2f}")
                                    
                                    if evidence["keywords_found"]:
                                        st.write("**Keywords Found:**")
                                        for kw in evidence["keywords_found"][:5]:  # Show first 5
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
                            st.subheader("Keyword-Based Evaluation")
                            if isinstance(value, dict) and "error" not in value:
                                st.text_area("Summary", value["summary"], height=300)
                                
                                # Show detailed breakdown
                                if "dimensions" in value:
                                    st.subheader("Dimension Breakdown")
                                    for dim in value["dimensions"]:
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric(f"{dim['dimension'].capitalize()}", f"{dim['weighted_score']:.2f}")
                                        with col2:
                                            st.metric("Raw Score", dim['raw_score'])
                                        with col3:
                                            st.metric("Weight", f"{dim['weight']:.3f}")
                                
                                # Show overall score
                                st.metric("Overall Score", f"{value['overall_score']:.2f}")
                            else:
                                st.error(f"Keyword evaluation error: {value}")
                        else:
                            st.subheader(f"{key.capitalize()} Evaluation")
                            if evaluation_mode == "Full Evaluation":
                                st.info("💡 This LLM evaluation was grounded with keyword evidence to prevent hallucination")
                            st.code(value)
                
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
                st.exception(e)

# Main content area
st.markdown("---")
st.info("""
**AURA Framework Features:**
- **LLM-based Evaluation**: Qualitative assessment with detailed reasoning and suggestions
- **Keyword-based Evaluation**: Quantitative scoring based on guideline-derived criteria
- **Grounded Evaluation**: LLM assessments grounded with keyword evidence to prevent hallucination
- **Comparison Mode**: Compare both evaluation approaches
- **Modular Design**: Select specific dimensions to evaluate

**Evaluation Modes:**
- **Full Evaluation**: Both LLM and keyword-based assessments
- **LLM Only**: Qualitative evaluation only (without grounding)
- **Keyword Only**: Quantitative baseline evaluation
- **Comparison Mode**: Side-by-side comparison of both methods
- **Grounded Evaluation**: LLM evaluations with keyword evidence integration
""")
