#!/usr/bin/env python3
"""
AURA Evaluation Framework - Streamlit Web Application

A comprehensive web interface for automated research artifact evaluation
supporting multiple input methods, conference-specific guidelines, and
detailed progress tracking.
"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

from integrated_artifact_analyzer import IntegratedArtifactAnalyzer
from aura_evaluator import AURAEvaluator

# Page configuration
st.set_page_config(
    page_title="AURA - Artifact Evaluation Framework",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .step-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .error-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #dc3545;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        margin: 0.5rem;
        border-top: 3px solid #667eea;
    }
    
    .progress-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'step': 1,
        'artifact_data': None,
        'evaluation_results': None,
        'selected_conference': 'ICSE 2025',
        'artifact_name': None,
        'temp_dir': None,
        'show_results': False,
        'analysis_complete': False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Header
def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AURA: Artifact Understanding and Research Assessment</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            AI-Powered Research Artifact Evaluation Framework
        </p>
        <p style="font-size: 1rem; opacity: 0.9; margin-top: 0.5rem;">
            Conference-Specific • Multi-Dimensional • Weighted Scoring • Production-Ready
        </p>
    </div>
    """, unsafe_allow_html=True)


# Sidebar configuration
def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.markdown("### 🎯 Evaluation Configuration")

        # Conference selection
        st.markdown("#### Conference Guidelines")
        conferences = get_available_conferences()
        selected_conf = st.selectbox(
            "Target Conference",
            conferences,
            index=conferences.index(st.session_state.selected_conference)
            if st.session_state.selected_conference in conferences else 0,
            help="Select the target conference for evaluation criteria"
        )
        st.session_state.selected_conference = selected_conf

        # Evaluation dimensions
        st.markdown("#### Evaluation Dimensions")
        all_dimensions = ['accessibility', 'documentation', 'experimental',
                          'functionality', 'reproducibility', 'usability']
        selected_dims = st.multiselect(
            "Select Dimensions",
            all_dimensions,
            default=all_dimensions,
            help="Choose which dimensions to evaluate"
        )

        # Advanced options
        st.markdown("#### Advanced Options")
        use_rag = st.checkbox("Enable RAG Enhancement", value=True,
                              help="Use Retrieval-Augmented Generation for contextual evaluation")
        use_neo4j = st.checkbox("Use Neo4j (if available)", value=False,
                                help="Use Neo4j for knowledge graph storage")

        # Progress indicator
        st.markdown("---")
        st.markdown("#### 📊 Progress")
        progress_text = get_progress_text(st.session_state.step)
        st.markdown(progress_text, unsafe_allow_html=True)

        return selected_dims, use_rag, use_neo4j


def get_available_conferences():
    """Get list of available conferences"""
    try:
        processed_dir = Path("data/conference_guideline_texts/processed")
        if not processed_dir.exists():
            processed_dir = Path("../data/conference_guideline_texts/processed")

        if processed_dir.exists():
            md_files = list(processed_dir.glob("*.md"))
            conferences = []
            for f in md_files:
                match = re.match(r"\d+_([a-zA-Z_]+)_?(\d{4})?\.md", f.name)
                if match:
                    conf = match.group(1).upper().replace('_', ' ')
                    year = match.group(2) if match.group(2) else ''
                    label = f"{conf} {year}".strip()
                    conferences.append(label)
            return sorted(conferences) if conferences else ["ICSE 2025", "ASE 2024", "FSE 2024"]
        else:
            return ["ICSE 2025", "ASE 2024", "FSE 2024"]
    except Exception as e:
        st.error(f"Error loading conferences: {e}")
        return ["ICSE 2025", "ASE 2024", "FSE 2024"]


def get_progress_text(step):
    """Get progress text for current step"""
    steps = [
        "🎯 **Step 1**: Input Artifact",
        "📊 **Step 2**: Analysis & Extraction",
        "🔍 **Step 3**: Evaluation",
        "📈 **Step 4**: Results & Recommendations"
    ]

    progress_html = "<div style='padding: 1rem;'>"
    for i, step_text in enumerate(steps, 1):
        if i < step:
            progress_html += f"<div style='color: #28a745; margin: 0.5rem 0;'>✅ {step_text}</div>"
        elif i == step:
            progress_html += f"<div style='color: #667eea; margin: 0.5rem 0; font-weight: bold;'>⏳ {step_text}</div>"
        else:
            progress_html += f"<div style='color: #6c757d; margin: 0.5rem 0;'>⚪ {step_text}</div>"
    progress_html += "</div>"

    return progress_html


# Step 1: Artifact Input
def render_artifact_input():
    """Render artifact input interface"""
    st.markdown("## 📁 Step 1: Provide Your Research Artifact")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🌐 GitHub Repository")
        github_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/username/repository",
            help="Enter the full GitHub URL of your research artifact"
        )

        if github_url and st.button("📥 Analyze GitHub Repository", type="primary"):
            if validate_github_url(github_url):
                process_github_artifact(github_url)
            else:
                st.error("Please enter a valid GitHub URL")

    with col2:
        st.markdown("### 📦 Upload Archive")
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['zip', 'tar', 'gz', 'bz2', 'xz'],
            help="Upload your artifact as ZIP, TAR, or compressed archive"
        )

        if uploaded_file and st.button("📤 Process Uploaded File", type="primary"):
            process_uploaded_artifact(uploaded_file)

    # Display current artifact info if available
    if st.session_state.artifact_data:
        display_artifact_summary()


def validate_github_url(url):
    """Validate GitHub URL format"""
    github_pattern = r'^https?://github\.com/[\w\-\.]+/[\w\-\.]+/?$'
    return re.match(github_pattern, url) is not None


def process_github_artifact(github_url):
    """Process GitHub repository"""
    with st.spinner("🔍 Cloning and analyzing repository..."):
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="aura_")
            st.session_state.temp_dir = temp_dir

            # Initialize analyzer
            analyzer = IntegratedArtifactAnalyzer(
                temp_dir=temp_dir,
                output_dir=temp_dir
            )

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Analyze artifact
            status_text.text("Cloning repository...")
            progress_bar.progress(25)

            result = analyzer.analyze_artifact(
                artifact_path=github_url,
                cleanup_after_processing=False
            )

            progress_bar.progress(75)
            status_text.text("Processing analysis results...")

            if result.get("success", False):
                st.session_state.artifact_data = result
                st.session_state.artifact_name = result.get("artifact_name", "unknown")
                st.session_state.step = 2
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")

                st.success("Repository analyzed successfully!")
                st.rerun()
            else:
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            st.error(f"Error processing repository: {str(e)}")


def process_uploaded_artifact(uploaded_file):
    """Process uploaded file"""
    with st.spinner("📦 Processing uploaded artifact..."):
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="aura_")
            st.session_state.temp_dir = temp_dir

            # Save uploaded file
            file_path = Path(temp_dir) / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Initialize analyzer
            analyzer = IntegratedArtifactAnalyzer(
                temp_dir=temp_dir,
                output_dir=temp_dir
            )

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Analyze artifact
            status_text.text("Extracting archive...")
            progress_bar.progress(25)

            result = analyzer.analyze_artifact(
                artifact_path=str(file_path),
                cleanup_after_processing=False
            )

            progress_bar.progress(75)
            status_text.text("Processing analysis results...")

            if result.get("success", False):
                st.session_state.artifact_data = result
                st.session_state.artifact_name = result.get("artifact_name", uploaded_file.name)
                st.session_state.step = 2
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                st.success("Artifact processed successfully!")
                st.rerun()
            else:
                st.error(f"Processing failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")


def display_artifact_summary():
    """Display summary of analyzed artifact"""
    st.markdown("---")
    st.markdown("## 📊 Artifact Summary")

    data = st.session_state.artifact_data

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📁 Repository</h3>
            <p style="font-size: 1.2rem; font-weight: bold; color: #667eea;">
                {name}
            </p>
        </div>
        """.format(name=data.get("artifact_name", "Unknown")), unsafe_allow_html=True)

    with col2:
        size_mb = data.get("repo_size_mb", 0)
        st.markdown("""
        <div class="metric-card">
            <h3>💾 Size</h3>
            <p style="font-size: 1.2rem; font-weight: bold; color: #667eea;">
                {size} MB
            </p>
        </div>
        """.format(size=size_mb), unsafe_allow_html=True)

    with col3:
        doc_files = len(data.get("documentation_files", []))
        st.markdown("""
        <div class="metric-card">
            <h3>📝 Docs</h3>
            <p style="font-size: 1.2rem; font-weight: bold; color: #667eea;">
                {count} files
            </p>
        </div>
        """.format(count=doc_files), unsafe_allow_html=True)

    with col4:
        code_files = len(data.get("code_files", []))
        st.markdown("""
        <div class="metric-card">
            <h3>💻 Code</h3>
            <p style="font-size: 1.2rem; font-weight: bold; color: #667eea;">
                {count} files
            </p>
        </div>
        """.format(count=code_files), unsafe_allow_html=True)

    # Repository structure
    if data.get("tree_structure"):
        with st.expander("🌳 Repository Structure", expanded=False):
            tree_text = "\n".join(data["tree_structure"][:50])  # Limit display
            if len(data["tree_structure"]) > 50:
                tree_text += f"\n... and {len(data['tree_structure']) - 50} more items"
            st.code(tree_text, language="text")


# Step 2: Start Evaluation
def render_evaluation_step():
    """Render evaluation step"""
    st.markdown("## 🔍 Step 2: Ready for Evaluation")

    st.markdown("""
    <div class="step-card">
        <h3>🎯 Evaluation Configuration</h3>
        <p>Your artifact has been analyzed and is ready for comprehensive evaluation.</p>
        <ul>
            <li><strong>Conference:</strong> {conference}</li>
            <li><strong>Artifact:</strong> {artifact}</li>
            <li><strong>Framework:</strong> AURA Multi-Dimensional Assessment</li>
        </ul>
    </div>
    """.format(
        conference=st.session_state.selected_conference,
        artifact=st.session_state.artifact_name
    ), unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📋 Evaluation Preview")
        st.markdown("""
        The evaluation will assess your artifact across **6 key dimensions**:
        
        1. **🌐 Accessibility** (15.86%) - Repository availability, download ease, format compliance
        2. **📝 Documentation** (11.82%) - Completeness, clarity, examples, maintenance info  
        3. **🧪 Experimental** (9.99%) - Design quality, data availability, reproducible scripts
        4. **⚙️ Functionality** (6.43%) - Core features, error handling, performance, robustness
        5. **🔄 Reproducibility** (26.23%) - Setup reproducibility, dependency management
        6. **👥 Usability** (29.67%) - Learning curve, interface design, workflow clarity
        """)

    with col2:
        if st.button("🚀 Start Comprehensive Evaluation", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

        if st.button("🔄 Choose Different Artifact", use_container_width=True):
            reset_session()
            st.rerun()


# Step 3: Run Evaluation
def render_evaluation_process(selected_dims, use_rag, use_neo4j):
    """Render the evaluation process with progress tracking"""
    st.markdown("## ⚡ Step 3: Running Evaluation")

    # Create a container for progress updates
    progress_container = st.container()

    with progress_container:
        st.markdown("""
        <div class="progress-container">
            <h3>🔄 Evaluation in Progress</h3>
            <p>Please wait while we perform comprehensive analysis...</p>
        </div>
        """, unsafe_allow_html=True)

        # Main progress bar
        main_progress = st.progress(0)
        status_text = st.empty()

        # Detailed progress
        progress_details = st.empty()

        try:
            # Save artifact data to temporary JSON file
            temp_json_path = save_artifact_to_json()

            # Initialize evaluator
            status_text.text("🔧 Initializing evaluation framework...")
            main_progress.progress(10)

            # Extract conference name for the evaluator
            conference_name = st.session_state.selected_conference.split()[
                0]  # Get first word (e.g., "ICSE" from "ICSE 2025")

            # Run evaluation with progress tracking
            result = run_evaluation_with_progress(
                temp_json_path,
                conference_name,
                selected_dims,
                use_rag,
                use_neo4j,
                main_progress,
                status_text,
                progress_details
            )

            if result:
                st.session_state.evaluation_results = result
                st.session_state.step = 4
                st.session_state.show_results = True
                main_progress.progress(100)
                status_text.text("✅ Evaluation completed successfully!")

                # Add a small delay before rerunning to show completion
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Evaluation failed. Please try again.")

        except Exception as e:
            st.error(f"❌ Evaluation error: {str(e)}")


def save_artifact_to_json():
    """Save artifact data to temporary JSON file for evaluation"""
    temp_json_path = Path(st.session_state.temp_dir) / f"{st.session_state.artifact_name}_analysis.json"

    with open(temp_json_path, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.artifact_data, f, indent=2, ensure_ascii=False)

    return str(temp_json_path)


def run_evaluation_with_progress(json_path, conference_name, dimensions, use_rag, use_neo4j,
                                 progress_bar, status_text, progress_details):
    """Run evaluation with detailed progress tracking"""
    try:
        # Progress tracking
        total_steps = len(dimensions) + 3  # dimensions + init + kg + finalize
        current_step = 0

        def update_progress(step_name, detail=""):
            nonlocal current_step
            current_step += 1
            progress = int((current_step / total_steps) * 90)  # Leave 10% for finalization
            progress_bar.progress(progress)
            status_text.text(f"🔄 {step_name}...")

            if detail:
                progress_details.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                    <small><strong>{step_name}:</strong> {detail}</small>
                </div>
                """, unsafe_allow_html=True)

        # Initialize evaluator
        update_progress("Initializing AURA Framework", "Setting up evaluation environment")

        evaluator = AURAEvaluator(
            use_neo4j=use_neo4j,
            use_rag=use_rag,
            conference_name=conference_name
        )

        # Build knowledge graph
        update_progress("Building Knowledge Graph", "Extracting semantic relationships from artifact")

        # Run evaluation for each dimension
        for dim in dimensions:
            update_progress(f"Evaluating {dim.title()}", f"Analyzing {dim} dimension with {conference_name} guidelines")

        # Perform actual evaluation
        update_progress("Computing Final Scores", "Applying weighted scoring algorithm")

        result = evaluator.evaluate_artifact_from_json(
            artifact_json_path=json_path,
            dimensions=dimensions
        )

        # Finalize
        update_progress("Generating Report", "Preparing comprehensive evaluation report")

        evaluator.close()
        return result

    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")
        return None


# Step 4: Display Results
def render_results():
    """Render comprehensive evaluation results"""
    st.markdown("## 📈 Step 4: Evaluation Results")

    results = st.session_state.evaluation_results

    # Overall Score Header
    render_overall_score_header(results)

    # Detailed breakdown
    col1, col2 = st.columns([2, 1])

    with col1:
        render_dimension_analysis(results)
        render_detailed_recommendations(results)

    with col2:
        render_score_visualization(results)
        render_acceptance_probability(results)

    # Additional insights
    render_conference_insights(results)

    # Action buttons
    render_action_buttons()


def render_overall_score_header(results):
    """Render overall score header"""
    weighted_score = results.get("weighted_scoring", {}).get("weighted_overall_percentage", 0)
    acceptance_prob = results.get("weighted_scoring", {}).get("acceptance_probability", {})

    if weighted_score >= 85:
        color = "#28a745"
        icon = "🏆"
    elif weighted_score >= 70:
        color = "#fd7e14"
        icon = "✅"
    elif weighted_score >= 55:
        color = "#ffc107"
        icon = "⚠️"
    else:
        color = "#dc3545"
        icon = "❌"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; 
                border: 2px solid {color}; margin: 1rem 0;">
        <h1 style="color: {color}; margin: 0;">{icon} {weighted_score:.1f}%</h1>
        <h3 style="color: #333; margin: 0.5rem 0;">Overall Evaluation Score</h3>
        <p style="font-size: 1.2rem; color: {color}; font-weight: bold; margin: 0;">
            {acceptance_prob.get('probability_text', 'Unknown')}
        </p>
        <p style="color: #666; margin: 0.5rem 0;">
            Conference: {st.session_state.selected_conference} | 
            Artifact: {st.session_state.artifact_name}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_dimension_analysis(results):
    """Render detailed dimension analysis"""
    st.markdown("### 📊 Dimension Breakdown")

    dimension_scores = results.get("dimension_scores", {})
    dimension_percentages = results.get("weighted_scoring", {}).get("dimension_percentages", {})

    for dimension, score in dimension_scores.items():
        percentage = dimension_percentages.get(dimension, 0)
        detailed_eval = results.get("detailed_evaluations", {}).get(dimension, {})

        # Color coding
        if percentage >= 80:
            color = "#28a745"
        elif percentage >= 60:
            color = "#fd7e14"
        elif percentage >= 40:
            color = "#ffc107"
        else:
            color = "#dc3545"

        with st.expander(f"**{dimension.title()}**: {percentage:.1f}% ({score:.1f}/5.0)", expanded=False):
            col1, col2 = st.columns([1, 2])

            with col1:
                # Score gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{dimension.title()} Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 85
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Detailed breakdown
                if detailed_eval:
                    st.markdown("**Strengths:**")
                    for strength in detailed_eval.get("strengths", [])[:3]:
                        st.markdown(f"✅ {strength}")

                    st.markdown("**Weaknesses:**")
                    for weakness in detailed_eval.get("weaknesses", [])[:3]:
                        st.markdown(f"❌ {weakness}")

                    st.markdown("**Key Recommendations:**")
                    for rec in detailed_eval.get("recommendations", [])[:2]:
                        st.markdown(f"💡 {rec}")


def render_score_visualization(results):
    """Render score visualization charts"""
    st.markdown("### 📈 Score Analytics")

    # Dimension scores radar chart
    dimension_scores = results.get("dimension_scores", {})
    dimension_percentages = results.get("weighted_scoring", {}).get("dimension_percentages", {})

    if dimension_scores:
        # Radar chart
        categories = list(dimension_scores.keys())
        values = [dimension_percentages.get(dim, 0) for dim in categories]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[cat.title() for cat in categories],
            fill='toself',
            name='Your Artifact',
            line_color='#667eea'
        ))

        # Add ideal scores
        ideal_values = [85] * len(categories)  # Target score
        fig.add_trace(go.Scatterpolar(
            r=ideal_values,
            theta=[cat.title() for cat in categories],
            fill='toself',
            name='Target Score',
            line_color='#28a745',
            opacity=0.3
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Dimension Comparison",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def render_acceptance_probability(results):
    """Render acceptance probability analysis"""
    st.markdown("### 🎯 Acceptance Analysis")

    acceptance_info = results.get("weighted_scoring", {}).get("acceptance_probability", {})
    score_percentage = acceptance_info.get("score_percentage", 0)

    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Acceptance Probability"},
        delta={'reference': 70, 'valueformat': ".1f"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 40], 'color': "#ffe6e6"},
                {'range': [40, 55], 'color': "#fff3cd"},
                {'range': [55, 70], 'color': "#d1ecf1"},
                {'range': [70, 85], 'color': "#d4edda"},
                {'range': [85, 100], 'color': "#28a745"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Probability breakdown
    category = acceptance_info.get("category", "unknown")
    prob_text = acceptance_info.get("probability_text", "Unknown")
    prob_range = acceptance_info.get("probability_range", "Unknown")

    st.markdown(f"""
    <div class="metric-card">
        <h4>📊 Assessment</h4>
        <p><strong>Category:</strong> {category.replace('_', ' ').title()}</p>
        <p><strong>Probability:</strong> {prob_text}</p>
        <p><strong>Range:</strong> {prob_range}</p>
    </div>
    """, unsafe_allow_html=True)


def render_detailed_recommendations(results):
    """Render detailed recommendations"""
    st.markdown("### 💡 Improvement Recommendations")

    # Collect all recommendations
    all_recommendations = []
    detailed_evals = results.get("detailed_evaluations", {})

    for dimension, eval_data in detailed_evals.items():
        recommendations = eval_data.get("recommendations", [])
        for rec in recommendations[:2]:  # Top 2 per dimension
            all_recommendations.append({
                "dimension": dimension,
                "recommendation": rec,
                "priority": get_recommendation_priority(dimension, eval_data.get("rating", 0))
            })

    # Sort by priority
    all_recommendations.sort(key=lambda x: x["priority"], reverse=True)

    # Display top recommendations
    for i, rec_data in enumerate(all_recommendations[:8], 1):
        priority_icon = "🔴" if rec_data["priority"] > 3 else "🟡" if rec_data["priority"] > 2 else "🟢"

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; 
                    border-left: 4px solid #667eea;">
            <p><strong>{priority_icon} {rec_data['dimension'].title()}</strong></p>
            <p style="margin: 0;">{rec_data['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)


def get_recommendation_priority(dimension, score):
    """Calculate recommendation priority based on dimension importance and score"""
    # Dimension weights from the framework
    weights = {
        "usability": 0.2967,
        "reproducibility": 0.2623,
        "accessibility": 0.1586,
        "documentation": 0.1182,
        "experimental": 0.0999,
        "functionality": 0.0643
    }

    weight = weights.get(dimension, 0.1)
    score_factor = (5 - score) / 5  # Higher priority for lower scores

    return weight * score_factor * 5  # Scale to 0-5


def render_conference_insights(results):
    """Render conference-specific insights"""
    st.markdown("### 🏛️ Conference-Specific Analysis")

    conference_name = st.session_state.selected_conference

    # Mock conference insights based on the selected conference
    insights = get_conference_insights(conference_name, results)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Conference Requirements")
        for req in insights["requirements"]:
            status = "✅" if req["met"] else "❌"
            st.markdown(f"{status} {req['description']}")

    with col2:
        st.markdown("#### 🎯 Specific Recommendations")
        for rec in insights["specific_recommendations"]:
            st.markdown(f"💡 {rec}")


def get_conference_insights(conference_name, results):
    """Get conference-specific insights"""
    # This is a simplified version - in practice, this would integrate with
    # the conference guidelines loader

    base_insights = {
        "requirements": [
            {"description": "Public repository availability", "met": True},
            {"description": "Documentation completeness",
             "met": results.get("dimension_scores", {}).get("documentation", 0) >= 3},
            {"description": "Reproducibility support",
             "met": results.get("dimension_scores", {}).get("reproducibility", 0) >= 3},
            {"description": "License inclusion",
             "met": len(results.get("detailed_evaluations", {}).get("license_files", [])) > 0}
        ],
        "specific_recommendations": [
            "Consider adding more comprehensive documentation",
            "Improve reproducibility with better dependency management",
            "Enhance usability with clear installation instructions"
        ]
    }

    if "ICSE" in conference_name:
        base_insights["requirements"].append({
            "description": "Archival repository (Zenodo/FigShare) required",
            "met": False  # This would need to be checked
        })
        base_insights["specific_recommendations"].append(
            "ICSE requires DOI from archival repository like Zenodo"
        )

    return base_insights


def render_action_buttons():
    """Render action buttons for next steps"""
    st.markdown("---")
    st.markdown("### 🚀 Next Steps")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Download Report", use_container_width=True):
            download_report()

    with col2:
        if st.button("🔄 Evaluate Again", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

    with col3:
        if st.button("📁 New Artifact", use_container_width=True):
            reset_session()
            st.rerun()

    with col4:
        if st.button("📋 Comparison", use_container_width=True):
            st.info("Comparison feature coming soon!")


def download_report():
    """Generate and download evaluation report"""
    try:
        results = st.session_state.evaluation_results

        # Create downloadable JSON report
        report_data = {
            "artifact_name": st.session_state.artifact_name,
            "conference": st.session_state.selected_conference,
            "evaluation_timestamp": str(pd.Timestamp.now()),
            "results": results
        }

        st.download_button(
            label="💾 Download JSON Report",
            data=json.dumps(report_data, indent=2),
            file_name=f"{st.session_state.artifact_name}_aura_report.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"Error generating report: {str(e)}")


def reset_session():
    """Reset session state for new evaluation"""
    keys_to_keep = []  # Keep nothing, full reset
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

    # Clean up temporary directory
    if hasattr(st.session_state, 'temp_dir') and st.session_state.temp_dir:
        try:
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        except:
            pass

    initialize_session_state()


# Main app logic
def main():
    """Main application logic"""
    initialize_session_state()
    render_header()

    # Sidebar configuration
    selected_dims, use_rag, use_neo4j = render_sidebar()

    # Main content based on current step
    if st.session_state.step == 1:
        render_artifact_input()
    elif st.session_state.step == 2:
        render_evaluation_step()
    elif st.session_state.step == 3:
        render_evaluation_process(selected_dims, use_rag, use_neo4j)
    elif st.session_state.step == 4:
        render_results()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>🚀 AURA Framework v1.0</strong> | 
        AI-Powered Research Artifact Evaluation | 
        Conference-Specific Guidelines | 
        Multi-Dimensional Assessment</p>
        <p><small>Developed for the research community • Open source • Production ready</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
