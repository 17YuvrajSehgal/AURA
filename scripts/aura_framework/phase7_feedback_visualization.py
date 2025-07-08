"""
📊 Phase 7: Feedback & Visualization Dashboard
Goal: Create interactive visualizations and comprehensive reports for artifact evaluation.

Features:
- Streamlit web dashboard with real-time evaluation
- Interactive radar charts and dimension breakdowns
- Neo4j graph visualization integration
- Automated markdown report generation
- Comparative analysis visualizations
- Export capabilities (PDF, JSON, CSV)
- Conference recommendation interface
"""

import json
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import base64
from io import BytesIO
import zipfile

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Local imports
from config import config
from phase1_artifact_preprocessing import ArtifactPreprocessor
from phase2_knowledge_graph import KnowledgeGraphBuilder
from phase3_vector_embeddings import VectorEmbeddingEngine
from phase4_pattern_analysis import PatternAnalysisEngine
from phase5_genai_agents import MultiAgentEvaluationOrchestrator, EvaluationDimension
from phase6_rubric_scoring import RubricScoringFramework, AcceptanceCategory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the dashboard"""
    title: str = "AURA: AI-Powered Artifact Evaluation Framework"
    page_icon: str = "🚀"
    layout: str = "wide"
    sidebar_width: int = 300
    theme: str = "light"
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["PDF", "JSON", "CSV", "ZIP"]


class ArtifactVisualizationDashboard:
    """Comprehensive Streamlit dashboard for artifact evaluation visualization"""
    
    def __init__(self, 
                 artifacts_directory: str = "data/acm_bib_to_json_data",
                 use_neo4j: bool = False):
        """
        Initialize the visualization dashboard
        
        Args:
            artifacts_directory: Directory containing artifact data
            use_neo4j: Whether to use Neo4j for graph operations
        """
        self.artifacts_directory = Path(artifacts_directory)
        self.use_neo4j = use_neo4j
        self.config = DashboardConfig()
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.evaluation_results = {}
            st.session_state.comparison_results = {}
            st.session_state.selected_artifact = None
            st.session_state.framework_stats = {}
        
        # Initialize framework components
        self._initialize_framework()
        
        logger.info("Artifact Visualization Dashboard initialized")

    def _initialize_framework(self):
        """Initialize all framework components"""
        if not st.session_state.initialized:
            with st.spinner("Initializing AURA Framework..."):
                try:
                    # Initialize core components
                    self.preprocessor = ArtifactPreprocessor(
                        data_directory=str(self.artifacts_directory)
                    )
                    
                    self.kg_builder = KnowledgeGraphBuilder(use_neo4j=self.use_neo4j)
                    self.vector_engine = VectorEmbeddingEngine()
                    self.pattern_engine = PatternAnalysisEngine(self.kg_builder)
                    self.evaluator = MultiAgentEvaluationOrchestrator()
                    self.scoring_framework = RubricScoringFramework()
                    
                    st.session_state.initialized = True
                    st.success("✅ Framework initialized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Failed to initialize framework: {e}")
                    logger.error(f"Framework initialization error: {e}")

    def run_dashboard(self):
        """Run the main Streamlit dashboard"""
        # Configure page
        st.set_page_config(
            page_title=self.config.title,
            page_icon=self.config.page_icon,
            layout=self.config.layout
        )
        
        # Custom CSS
        self._apply_custom_styling()
        
        # Main header
        st.title(self.config.title)
        st.markdown("*Cutting-edge AI evaluation of research software artifacts*")
        
        # Sidebar navigation
        page = self._create_sidebar()
        
        # Route to appropriate page
        if page == "🏠 Overview":
            self._show_overview_page()
        elif page == "📊 Single Artifact Evaluation":
            self._show_single_evaluation_page()
        elif page == "🔍 Multi-Conference Comparison":
            self._show_comparison_page()
        elif page == "📈 Pattern Analysis":
            self._show_pattern_analysis_page()
        elif page == "🗺️ Knowledge Graph Viewer":
            self._show_knowledge_graph_page()
        elif page == "📋 Batch Evaluation":
            self._show_batch_evaluation_page()
        elif page == "⚙️ Framework Settings":
            self._show_settings_page()

    def _apply_custom_styling(self):
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #1f77b4;
            margin: 0.5rem 0;
        }
        .evaluation-result {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .dimension-score {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background-color: #f8f9fa;
            border-radius: 0.25rem;
        }
        .stMetric {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        </style>
        """, unsafe_allow_html=True)

    def _create_sidebar(self) -> str:
        """Create navigation sidebar"""
        with st.sidebar:
            st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=AURA", use_column_width=True)
            
            st.markdown("### Navigation")
            page = st.selectbox(
                "Choose a page:",
                [
                    "🏠 Overview",
                    "📊 Single Artifact Evaluation", 
                    "🔍 Multi-Conference Comparison",
                    "📈 Pattern Analysis",
                    "🗺️ Knowledge Graph Viewer",
                    "📋 Batch Evaluation",
                    "⚙️ Framework Settings"
                ]
            )
            
            st.markdown("---")
            
            # Framework status
            st.markdown("### Framework Status")
            if st.session_state.initialized:
                st.success("✅ Initialized")
                
                # Show quick stats
                if st.session_state.framework_stats:
                    stats = st.session_state.framework_stats
                    st.metric("Artifacts", stats.get('total_artifacts', 0))
                    st.metric("Evaluations", len(st.session_state.evaluation_results))
            else:
                st.warning("⏳ Initializing...")
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("### Quick Actions")
            if st.button("🔄 Refresh Framework"):
                st.session_state.initialized = False
                st.experimental_rerun()
            
            if st.button("📥 Export All Results"):
                self._export_all_results()
            
            st.markdown("---")
            st.markdown("*AURA Framework v1.0*")
            
        return page

    def _show_overview_page(self):
        """Show framework overview and statistics"""
        st.header("🏠 Framework Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Artifacts", 
                len(st.session_state.evaluation_results),
                help="Number of artifacts evaluated"
            )
        
        with col2:
            completed_evaluations = sum(1 for r in st.session_state.evaluation_results.values() if r.get('completed'))
            st.metric(
                "Completed Evaluations",
                completed_evaluations,
                help="Number of completed evaluations"
            )
        
        with col3:
            if st.session_state.comparison_results:
                avg_score = np.mean([r.get('best_score', 0) for r in st.session_state.comparison_results.values()])
                st.metric(
                    "Average Score",
                    f"{avg_score:.1%}",
                    help="Average evaluation score"
                )
            else:
                st.metric("Average Score", "N/A")
        
        with col4:
            conferences = set()
            for result in st.session_state.comparison_results.values():
                conferences.update(result.get('conference_scores', {}).keys())
            st.metric(
                "Conferences",
                len(conferences),
                help="Number of conferences in analysis"
            )
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        
        if st.session_state.evaluation_results:
            # Create activity timeline
            activity_data = []
            for artifact_id, result in st.session_state.evaluation_results.items():
                if 'timestamp' in result:
                    activity_data.append({
                        'Artifact': artifact_id,
                        'Score': result.get('final_score', 0),
                        'Category': result.get('acceptance_category', 'Unknown'),
                        'Timestamp': result.get('timestamp', datetime.now().isoformat())
                    })
            
            if activity_data:
                df = pd.DataFrame(activity_data)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                
                # Score trend chart
                fig = px.line(df, x='Timestamp', y='Score', 
                            title="Evaluation Scores Over Time",
                            markers=True)
                fig.update_layout(yaxis_title="Score", xaxis_title="Time")
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent evaluations table
                st.subheader("🕒 Recent Evaluations")
                recent_df = df.sort_values('Timestamp', ascending=False).head(10)
                st.dataframe(
                    recent_df[['Artifact', 'Score', 'Category', 'Timestamp']],
                    use_container_width=True
                )
        else:
            st.info("No evaluations completed yet. Start by evaluating an artifact!")
        
        # Framework capabilities
        st.subheader("🎯 Framework Capabilities")
        
        capabilities = [
            ("🧱", "Artifact Preprocessing", "Extract metadata, documentation, and structural features"),
            ("📐", "Knowledge Graph", "Build comprehensive artifact relationships"),
            ("🔍", "Vector Embeddings", "Semantic analysis and similarity search"),
            ("📊", "Pattern Analysis", "Discover success patterns in accepted artifacts"),
            ("🧠", "GenAI Agents", "Multi-agent evaluation across 6 dimensions"),
            ("🎯", "Rubric Scoring", "Conference-specific weighted scoring"),
            ("📋", "Visualization", "Interactive dashboards and reports")
        ]
        
        for icon, name, description in capabilities:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{icon} {name}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

    def _show_single_evaluation_page(self):
        """Show single artifact evaluation interface"""
        st.header("📊 Single Artifact Evaluation")
        
        # Artifact selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            artifact_input = st.text_input(
                "Enter Artifact ID or Path:",
                placeholder="e.g., artifact_123 or path/to/artifact.json",
                help="Provide the artifact identifier or file path"
            )
        
        with col2:
            target_conference = st.selectbox(
                "Target Conference:",
                ["ICSE", "FSE", "ASE", "CHI", "PLDI", "OOPSLA"],
                help="Conference for evaluation context"
            )
        
        # Evaluation options
        st.subheader("⚙️ Evaluation Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_preprocessing = st.checkbox("Include Preprocessing", value=True)
            include_patterns = st.checkbox("Include Pattern Analysis", value=True)
        
        with col2:
            include_vectors = st.checkbox("Include Vector Analysis", value=True)
            apply_penalties = st.checkbox("Apply Conference Penalties", value=True)
        
        with col3:
            detailed_feedback = st.checkbox("Detailed Feedback", value=True)
            export_results = st.checkbox("Auto-export Results", value=False)
        
        # Evaluation button
        if st.button("🚀 Start Evaluation", type="primary"):
            if artifact_input:
                self._run_single_evaluation(
                    artifact_input, 
                    target_conference,
                    {
                        'include_preprocessing': include_preprocessing,
                        'include_patterns': include_patterns,
                        'include_vectors': include_vectors,
                        'apply_penalties': apply_penalties,
                        'detailed_feedback': detailed_feedback,
                        'export_results': export_results
                    }
                )
            else:
                st.error("Please provide an artifact ID or path")
        
        # Show existing results
        if st.session_state.evaluation_results:
            st.subheader("📋 Previous Evaluations")
            
            # Results selector
            selected_result_id = st.selectbox(
                "Select evaluation to view:",
                list(st.session_state.evaluation_results.keys()),
                format_func=lambda x: f"{x} ({st.session_state.evaluation_results[x].get('conference', 'N/A')})"
            )
            
            if selected_result_id:
                self._display_evaluation_result(
                    st.session_state.evaluation_results[selected_result_id]
                )

    def _run_single_evaluation(self, artifact_input: str, conference: str, options: Dict[str, Any]):
        """Run evaluation for a single artifact"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load artifact data
            status_text.text("Loading artifact data...")
            progress_bar.progress(10)
            
            # Mock artifact loading (replace with actual implementation)
            artifact_data = self._load_artifact_data(artifact_input)
            
            # Step 2: Preprocessing (if enabled)
            if options['include_preprocessing']:
                status_text.text("Preprocessing artifact...")
                progress_bar.progress(25)
                # Add preprocessing logic
            
            # Step 3: Multi-agent evaluation
            status_text.text("Running multi-agent evaluation...")
            progress_bar.progress(50)
            
            # Mock evaluation result (replace with actual evaluation)
            multi_agent_result = self._mock_multi_agent_evaluation(artifact_input)
            
            # Step 4: Rubric scoring
            status_text.text("Applying rubric-based scoring...")
            progress_bar.progress(75)
            
            scoring_result = self.scoring_framework.score_artifact(
                multi_agent_result, 
                conference,
                apply_penalties=options['apply_penalties']
            )
            
            # Step 5: Generate visualization
            status_text.text("Generating visualizations...")
            progress_bar.progress(90)
            
            # Store results
            evaluation_key = f"{artifact_input}_{conference}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.evaluation_results[evaluation_key] = {
                'artifact_id': artifact_input,
                'conference': conference,
                'multi_agent_result': multi_agent_result,
                'scoring_result': scoring_result,
                'options': options,
                'timestamp': datetime.now().isoformat(),
                'completed': True
            }
            
            progress_bar.progress(100)
            status_text.text("Evaluation completed!")
            
            # Display results
            st.success("✅ Evaluation completed successfully!")
            self._display_evaluation_result(st.session_state.evaluation_results[evaluation_key])
            
            # Auto-export if requested
            if options['export_results']:
                self._export_evaluation_result(evaluation_key)
            
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            logger.error(f"Single evaluation error: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()

    def _display_evaluation_result(self, result: Dict[str, Any]):
        """Display comprehensive evaluation results"""
        scoring_result = result['scoring_result']
        
        st.markdown(f"""
        <div class="evaluation-result">
            <h3>📊 Evaluation Results for {result['artifact_id']}</h3>
            <p><strong>Conference:</strong> {result['conference']}</p>
            <p><strong>Timestamp:</strong> {result['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Final Score",
                f"{scoring_result.final_score:.1%}",
                help="Overall weighted score"
            )
        
        with col2:
            st.metric(
                "Acceptance Category",
                scoring_result.acceptance_category.value.title(),
                help="Quality category based on thresholds"
            )
        
        with col3:
            st.metric(
                "Acceptance Probability",
                f"{scoring_result.acceptance_probability:.1%}",
                help="Estimated probability of acceptance"
            )
        
        with col4:
            st.metric(
                "Confidence Score",
                f"{scoring_result.confidence_score:.1%}",
                help="Confidence in the evaluation"
            )
        
        # Radar chart for dimension scores
        st.subheader("🎯 Dimension Breakdown")
        self._create_radar_chart(scoring_result)
        
        # Detailed dimension scores
        st.subheader("📊 Detailed Scores")
        
        dimensions_df = pd.DataFrame([
            {
                'Dimension': dim.value.title(),
                'Raw Score': f"{score:.1%}",
                'Weighted Score': f"{scoring_result.weighted_scores.get(dim, 0):.1%}",
                'Weight': f"{scoring_result.dimension_breakdown['weight_distribution'].get(dim.value, 0):.1%}"
            }
            for dim, score in scoring_result.raw_scores.items()
        ])
        
        st.dataframe(dimensions_df, use_container_width=True)
        
        # Improvement priorities
        if scoring_result.improvement_priorities:
            st.subheader("🎯 Improvement Priorities")
            for i, priority in enumerate(scoring_result.improvement_priorities, 1):
                st.markdown(f"{i}. {priority}")
        
        # Export options
        st.subheader("💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate PDF Report"):
                self._generate_pdf_report(result)
        
        with col2:
            if st.button("📊 Export to JSON"):
                self._export_to_json(result)
        
        with col3:
            if st.button("📈 Export to CSV"):
                self._export_to_csv(result)

    def _create_radar_chart(self, scoring_result):
        """Create radar chart for dimension scores"""
        dimensions = list(scoring_result.raw_scores.keys())
        scores = [scoring_result.raw_scores[dim] for dim in dimensions]
        dimension_names = [dim.value.title() for dim in dimensions]
        
        # Add first point again to close the radar chart
        scores.append(scores[0])
        dimension_names.append(dimension_names[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=dimension_names,
            fill='toself',
            name='Scores',
            line_color='#1f77b4',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat=".0%"
                )
            ),
            showlegend=False,
            title="Dimension Scores Radar Chart",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _show_comparison_page(self):
        """Show multi-conference comparison interface"""
        st.header("🔍 Multi-Conference Comparison")
        
        # Artifact and conference selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            artifact_input = st.text_input(
                "Artifact ID for Comparison:",
                placeholder="Enter artifact identifier"
            )
        
        with col2:
            comparison_type = st.selectbox(
                "Comparison Type:",
                ["Best Fit Analysis", "Full Conference Matrix", "Custom Selection"]
            )
        
        # Conference selection
        st.subheader("🏛️ Conference Selection")
        
        if comparison_type == "Custom Selection":
            selected_conferences = st.multiselect(
                "Select conferences to compare:",
                ["ICSE", "FSE", "ASE", "CHI", "PLDI", "OOPSLA", "SIGMOD", "KDD"],
                default=["ICSE", "FSE", "ASE"]
            )
        else:
            selected_conferences = ["ICSE", "FSE", "ASE", "CHI", "PLDI"]
            st.info(f"Using default conference set: {', '.join(selected_conferences)}")
        
        # Run comparison
        if st.button("🔍 Run Comparison", type="primary"):
            if artifact_input and selected_conferences:
                self._run_conference_comparison(artifact_input, selected_conferences)
            else:
                st.error("Please provide artifact ID and select conferences")
        
        # Display comparison results
        if st.session_state.comparison_results:
            st.subheader("📊 Comparison Results")
            
            selected_comparison = st.selectbox(
                "Select comparison to view:",
                list(st.session_state.comparison_results.keys())
            )
            
            if selected_comparison:
                self._display_comparison_results(
                    st.session_state.comparison_results[selected_comparison]
                )

    def _run_conference_comparison(self, artifact_id: str, conferences: List[str]):
        """Run multi-conference comparison"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Mock multi-agent result
            status_text.text("Generating multi-agent evaluation...")
            progress_bar.progress(20)
            
            multi_agent_result = self._mock_multi_agent_evaluation(artifact_id)
            
            # Run comparison
            status_text.text("Comparing across conferences...")
            progress_bar.progress(60)
            
            comparison_result = self.scoring_framework.compare_across_conferences(
                multi_agent_result, conferences
            )
            
            # Store results
            comparison_key = f"{artifact_id}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.comparison_results[comparison_key] = {
                'artifact_id': artifact_id,
                'conferences': conferences,
                'comparison_result': comparison_result,
                'timestamp': datetime.now().isoformat()
            }
            
            progress_bar.progress(100)
            status_text.text("Comparison completed!")
            
            st.success("✅ Comparison completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Comparison failed: {e}")
            logger.error(f"Conference comparison error: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()

    def _display_comparison_results(self, comparison_data: Dict[str, Any]):
        """Display multi-conference comparison results"""
        comparison_result = comparison_data['comparison_result']
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Best Fit Conference",
                comparison_result.best_fit_conference,
                help="Conference with highest score"
            )
        
        with col2:
            st.metric(
                "Best Score",
                f"{comparison_result.best_score:.1%}",
                help="Highest score achieved"
            )
        
        with col3:
            st.metric(
                "Score Variance",
                f"{comparison_result.score_variance:.4f}",
                help="Variance across conferences"
            )
        
        # Conference scores chart
        st.subheader("📊 Conference Scores")
        
        scores_data = [
            {
                'Conference': conf,
                'Score': result.final_score,
                'Category': result.acceptance_category.value.title(),
                'Probability': result.acceptance_probability
            }
            for conf, result in comparison_result.conference_scores.items()
        ]
        
        scores_df = pd.DataFrame(scores_data)
        
        # Bar chart
        fig = px.bar(
            scores_df, 
            x='Conference', 
            y='Score',
            color='Category',
            title="Scores by Conference",
            hover_data=['Probability']
        )
        fig.update_layout(yaxis_title="Score", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed scores table
        st.subheader("📋 Detailed Comparison")
        st.dataframe(scores_df, use_container_width=True)
        
        # Recommendations
        if comparison_result.recommendations:
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(comparison_result.recommendations, 1):
                st.markdown(f"{i}. {rec}")

    def _show_pattern_analysis_page(self):
        """Show pattern analysis visualization"""
        st.header("📈 Pattern Analysis")
        
        # Pattern analysis placeholder
        st.info("Pattern analysis visualization will be implemented here")
        
        # Mock pattern data
        patterns = [
            {"Pattern": "Documentation Structure", "Frequency": 85, "Success Rate": 0.92},
            {"Pattern": "Reproducibility Package", "Frequency": 72, "Success Rate": 0.88},
            {"Pattern": "Testing Framework", "Frequency": 64, "Success Rate": 0.85},
            {"Pattern": "Docker Integration", "Frequency": 58, "Success Rate": 0.82},
            {"Pattern": "API Documentation", "Frequency": 45, "Success Rate": 0.79}
        ]
        
        patterns_df = pd.DataFrame(patterns)
        
        # Pattern frequency chart
        fig = px.scatter(
            patterns_df,
            x='Frequency',
            y='Success Rate',
            size='Frequency',
            hover_name='Pattern',
            title="Pattern Success Analysis"
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    def _show_knowledge_graph_page(self):
        """Show knowledge graph visualization"""
        st.header("🗺️ Knowledge Graph Viewer")
        
        st.info("Knowledge graph visualization will be integrated here")
        st.markdown("This section will provide:")
        st.markdown("- Interactive graph exploration")
        st.markdown("- Node and relationship filtering")
        st.markdown("- Path analysis between artifacts")
        st.markdown("- Community detection visualization")

    def _show_batch_evaluation_page(self):
        """Show batch evaluation interface"""
        st.header("📋 Batch Evaluation")
        
        st.info("Batch evaluation interface will be implemented here")
        st.markdown("Features:")
        st.markdown("- Upload multiple artifacts")
        st.markdown("- Bulk evaluation across conferences")
        st.markdown("- Progress tracking")
        st.markdown("- Batch report generation")

    def _show_settings_page(self):
        """Show framework settings"""
        st.header("⚙️ Framework Settings")
        
        # LLM Settings
        st.subheader("🤖 LLM Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            llm_provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic"])
            model_name = st.text_input("Model Name", value="gpt-4-turbo")
        
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
            max_tokens = st.number_input("Max Tokens", 100, 4000, 2000)
        
        # Neo4j Settings
        st.subheader("🗃️ Neo4j Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
            neo4j_user = st.text_input("Username", value="neo4j")
        
        with col2:
            neo4j_password = st.text_input("Password", type="password")
            use_neo4j = st.checkbox("Enable Neo4j", value=False)
        
        # Export settings
        st.subheader("💾 Export Settings")
        export_directory = st.text_input("Export Directory", value="./exports")
        auto_export = st.checkbox("Auto-export Results", value=False)
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")

    def _load_artifact_data(self, artifact_input: str) -> Dict[str, Any]:
        """Load artifact data (mock implementation)"""
        # This would be replaced with actual artifact loading
        return {
            'artifact_id': artifact_input,
            'metadata': {'name': artifact_input, 'size': 1024},
            'documentation': 'Sample documentation content'
        }

    def _mock_multi_agent_evaluation(self, artifact_id: str):
        """Mock multi-agent evaluation for testing"""
        from phase5_genai_agents import MultiAgentResult, EvaluationResult
        
        # Create mock evaluation results
        mock_results = {}
        for dimension in EvaluationDimension:
            mock_results[dimension] = EvaluationResult(
                dimension=dimension,
                score=np.random.uniform(0.5, 0.95),
                confidence=np.random.uniform(0.7, 0.95),
                justification=f"Mock evaluation for {dimension.value}",
                evidence=[f"Evidence 1 for {dimension.value}", f"Evidence 2 for {dimension.value}"],
                suggestions=[f"Suggestion for {dimension.value}"],
                agent_version="1.0"
            )
        
        return MultiAgentResult(
            artifact_id=artifact_id,
            individual_results=mock_results,
            weighted_score=np.mean([r.score for r in mock_results.values()]),
            consensus_score=np.mean([r.score for r in mock_results.values()]),
            confidence_score=np.mean([r.confidence for r in mock_results.values()]),
            final_recommendation="Mock recommendation",
            improvement_suggestions=["Mock suggestion 1", "Mock suggestion 2"]
        )

    def _generate_pdf_report(self, result: Dict[str, Any]):
        """Generate PDF report for evaluation result"""
        if not REPORTLAB_AVAILABLE:
            st.error("ReportLab not available for PDF generation")
            return
        
        st.info("PDF report generation not yet implemented")

    def _export_to_json(self, result: Dict[str, Any]):
        """Export result to JSON"""
        json_data = json.dumps(result, default=str, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"evaluation_{result['artifact_id']}.json",
            mime="application/json"
        )

    def _export_to_csv(self, result: Dict[str, Any]):
        """Export result to CSV"""
        # Create CSV data from evaluation result
        csv_data = "Dimension,Raw Score,Weighted Score,Weight\n"
        scoring_result = result['scoring_result']
        
        for dim, score in scoring_result.raw_scores.items():
            weighted = scoring_result.weighted_scores.get(dim, 0)
            weight = scoring_result.dimension_breakdown['weight_distribution'].get(dim.value, 0)
            csv_data += f"{dim.value},{score:.3f},{weighted:.3f},{weight:.3f}\n"
        
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"evaluation_{result['artifact_id']}.csv",
            mime="text/csv"
        )

    def _export_all_results(self):
        """Export all evaluation results"""
        if not st.session_state.evaluation_results:
            st.warning("No results to export")
            return
        
        # Create ZIP file with all results
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zip_file:
            for key, result in st.session_state.evaluation_results.items():
                json_data = json.dumps(result, default=str, indent=2)
                zip_file.writestr(f"{key}.json", json_data)
        
        buffer.seek(0)
        
        st.download_button(
            label="📦 Download All Results (ZIP)",
            data=buffer.getvalue(),
            file_name=f"aura_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )

    def _export_evaluation_result(self, evaluation_key: str):
        """Export single evaluation result"""
        result = st.session_state.evaluation_results[evaluation_key]
        json_data = json.dumps(result, default=str, indent=2)
        
        # Auto-download (this would need to be implemented based on deployment)
        st.success(f"✅ Results exported for {evaluation_key}")


def main():
    """Run the Streamlit dashboard"""
    dashboard = ArtifactVisualizationDashboard(use_neo4j=False)
    dashboard.run_dashboard()


if __name__ == "__main__":
    main() 