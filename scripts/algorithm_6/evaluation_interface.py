"""
Artifact Evaluation Interface

This module provides a comprehensive interface for the artifact evaluation framework,
bringing together knowledge graph analysis, ML prediction models, conference-specific
insights, and RAG-powered explanations into a unified system.

Key Features:
- Unified artifact evaluation pipeline
- Interactive query interface for pattern research
- Batch evaluation capabilities
- Comprehensive reporting and visualization
- RESTful API for integration
- Command-line interface for researchers
"""

import logging
import sys
import argparse
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from datetime import datetime
import time

# Web interface components
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not available - web interface disabled")

# API components
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available - API interface disabled")

# Core framework components
from unified_kg_builder import UnifiedKnowledgeGraphBuilder
from pattern_analysis_engine import PatternAnalysisEngine
from vector_embeddings_analyzer import VectorEmbeddingsAnalyzer
from scoring_framework import ArtifactScoringFramework, PredictionResult
from conference_models import ConferenceSpecificModels, ConferenceRecommendation
from rag_insights_generator import RAGInsightsGenerator, GeneratedInsight
from config import config, NODE_TYPES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationRequest:
    """Request for artifact evaluation"""
    artifact_path: Optional[str] = None
    artifact_id: Optional[str] = None
    target_conference: Optional[str] = None
    evaluation_type: str = "comprehensive"  # comprehensive, quick, conference_specific
    include_insights: bool = True
    include_comparisons: bool = True

@dataclass
class EvaluationResult:
    """Comprehensive evaluation result"""
    artifact_id: str
    timestamp: str
    
    # Core predictions
    acceptance_prediction: Optional[PredictionResult] = None
    conference_recommendation: Optional[ConferenceRecommendation] = None
    
    # Analysis results
    pattern_analysis: Dict[str, Any] = None
    semantic_analysis: Dict[str, Any] = None
    quality_metrics: Dict[str, Any] = None
    
    # Insights and explanations
    generated_insights: List[GeneratedInsight] = None
    summary_report: str = ""
    
    # Metadata
    evaluation_time_seconds: float = 0.0
    components_used: List[str] = None
    confidence_scores: Dict[str, float] = None

class ArtifactEvaluationFramework:
    """Main framework interface for artifact evaluation"""
    
    def __init__(self, data_directory: Optional[str] = None, 
                 models_directory: Optional[str] = None):
        """
        Initialize the evaluation framework
        
        Args:
            data_directory: Directory containing artifact data
            models_directory: Directory containing pre-trained models
        """
        self.data_directory = data_directory
        self.models_directory = models_directory
        
        # Core components
        self.kg_builder: Optional[UnifiedKnowledgeGraphBuilder] = None
        self.pattern_analyzer: Optional[PatternAnalysisEngine] = None
        self.vector_analyzer: Optional[VectorEmbeddingsAnalyzer] = None
        self.scoring_framework: Optional[ArtifactScoringFramework] = None
        self.conference_models: Optional[ConferenceSpecificModels] = None
        self.insights_generator: Optional[RAGInsightsGenerator] = None
        
        # Initialization status
        self.is_initialized = False
        self.components_loaded = []
        
        # Cache for performance
        self.evaluation_cache: Dict[str, EvaluationResult] = {}
        self.pattern_cache: Dict[str, Any] = {}
        
    def initialize_framework(self, artifacts_directory: str, 
                           max_artifacts: Optional[int] = None,
                           use_neo4j: bool = True) -> Dict[str, Any]:
        """
        Initialize the complete evaluation framework
        
        Args:
            artifacts_directory: Directory containing artifact JSON files
            max_artifacts: Maximum number of artifacts to process
            use_neo4j: Whether to use Neo4j for knowledge graph storage
            
        Returns:
            Initialization results and statistics
        """
        logger.info("Initializing Artifact Evaluation Framework")
        start_time = time.time()
        
        initialization_results = {
            'start_time': datetime.now().isoformat(),
            'components_initialized': [],
            'failed_components': [],
            'statistics': {},
            'performance_metrics': {}
        }
        
        try:
            # 1. Initialize Knowledge Graph Builder
            logger.info("Step 1: Initializing Knowledge Graph Builder")
            self.kg_builder = UnifiedKnowledgeGraphBuilder(use_neo4j=use_neo4j)
            kg_results = self.kg_builder.build_unified_graph(
                artifacts_directory, max_artifacts=max_artifacts
            )
            initialization_results['components_initialized'].append('knowledge_graph')
            initialization_results['statistics']['knowledge_graph'] = kg_results
            
            # 2. Initialize Pattern Analysis Engine
            logger.info("Step 2: Initializing Pattern Analysis Engine")
            self.pattern_analyzer = PatternAnalysisEngine(self.kg_builder)
            pattern_results = self.pattern_analyzer.analyze_documentation_patterns()
            initialization_results['components_initialized'].append('pattern_analysis')
            initialization_results['statistics']['pattern_analysis'] = {
                'communities_found': len(pattern_results.get('community_analysis', {})),
                'motifs_discovered': len(pattern_results.get('motif_discovery', {}))
            }
            
            # 3. Initialize Vector Embeddings Analyzer
            logger.info("Step 3: Initializing Vector Embeddings Analyzer")
            self.vector_analyzer = VectorEmbeddingsAnalyzer(self.kg_builder)
            embedding_results = self.vector_analyzer.extract_embeddings()
            clustering_results = self.vector_analyzer.perform_semantic_clustering()
            initialization_results['components_initialized'].append('vector_analysis')
            initialization_results['statistics']['vector_analysis'] = embedding_results
            
            # 4. Initialize Scoring Framework
            logger.info("Step 4: Initializing Scoring Framework")
            self.scoring_framework = ArtifactScoringFramework(
                self.kg_builder, self.pattern_analyzer, self.vector_analyzer
            )
            
            # Train models if we have enough data
            # Get actual artifact IDs from the knowledge graph
            artifact_list = []
            if hasattr(self.kg_builder, 'nx_graph'):
                G = self.kg_builder.nx_graph
                for node in G.nodes():
                    node_data = G.nodes[node]
                    if node_data.get('node_type') == NODE_TYPES['ARTIFACT']:
                        artifact_list.append(node)
            
            if len(artifact_list) >= 10:
                try:
                    training_results = self.scoring_framework.train_acceptance_models(artifact_list)
                    initialization_results['statistics']['scoring_models'] = training_results
                except Exception as e:
                    logger.warning(f"Failed to train scoring models: {e}")
                    initialization_results['failed_components'].append('scoring_models')
            
            initialization_results['components_initialized'].append('scoring_framework')
            
            # 5. Initialize Conference Models
            logger.info("Step 5: Initializing Conference Models")
            self.conference_models = ConferenceSpecificModels(self.scoring_framework)
            
            # Build conference profiles
            if hasattr(self.scoring_framework, 'training_data') and self.scoring_framework.training_data is not None:
                artifacts_data = self.scoring_framework.training_data.to_dict('records')
                conference_profiles = self.conference_models.build_conference_profiles(artifacts_data)
                initialization_results['statistics']['conference_models'] = {
                    'profiles_built': len(conference_profiles)
                }
            
            initialization_results['components_initialized'].append('conference_models')
            
            # 6. Initialize RAG Insights Generator
            logger.info("Step 6: Initializing RAG Insights Generator")
            self.insights_generator = RAGInsightsGenerator(
                self.kg_builder, self.pattern_analyzer, self.vector_analyzer,
                self.scoring_framework, self.conference_models
            )
            initialization_results['components_initialized'].append('insights_generator')
            
            # Mark as initialized
            self.is_initialized = True
            self.components_loaded = initialization_results['components_initialized']
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            initialization_results['performance_metrics'] = {
                'total_initialization_time': total_time,
                'artifacts_processed': kg_results.get('processed_artifacts', 0),
                'processing_rate': kg_results.get('processed_artifacts', 0) / total_time if total_time > 0 else 0
            }
            
            logger.info(f"Framework initialization completed in {total_time:.2f} seconds")
            return initialization_results
            
        except Exception as e:
            logger.error(f"Framework initialization failed: {e}")
            initialization_results['failed_components'].append('framework')
            raise
    
    def evaluate_artifact(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Evaluate a single artifact comprehensively
        
        Args:
            request: Evaluation request with artifact information
            
        Returns:
            Comprehensive evaluation result
        """
        if not self.is_initialized:
            raise ValueError("Framework not initialized. Call initialize_framework() first.")
        
        start_time = time.time()
        logger.info(f"Evaluating artifact: {request.artifact_id}")
        
        # Check cache first
        cache_key = f"{request.artifact_id}_{request.evaluation_type}_{request.target_conference}"
        if cache_key in self.evaluation_cache:
            logger.info("Returning cached evaluation result")
            return self.evaluation_cache[cache_key]
        
        result = EvaluationResult(
            artifact_id=request.artifact_id,
            timestamp=datetime.now().isoformat(),
            components_used=[],
            confidence_scores={}
        )
        
        try:
            # 1. Acceptance Prediction
            if 'scoring_framework' in self.components_loaded:
                logger.info("Generating acceptance prediction")
                result.acceptance_prediction = self.scoring_framework.predict_acceptance(
                    request.artifact_id, request.target_conference
                )
                result.components_used.append('acceptance_prediction')
                result.confidence_scores['acceptance'] = result.acceptance_prediction.confidence_score
            
            # 2. Conference Recommendation
            if 'conference_models' in self.components_loaded and request.target_conference is None:
                logger.info("Generating conference recommendation")
                artifact_features = self.scoring_framework._extract_single_artifact_features(request.artifact_id)
                if artifact_features:
                    features_dict = self.scoring_framework._features_to_dict(artifact_features)
                    result.conference_recommendation = self.conference_models.predict_best_conference(features_dict)
                    result.components_used.append('conference_recommendation')
                    result.confidence_scores['conference'] = result.conference_recommendation.confidence_score
            
            # 3. Pattern Analysis (if comprehensive evaluation)
            if request.evaluation_type == "comprehensive" and 'pattern_analysis' in self.components_loaded:
                logger.info("Performing pattern analysis")
                # Get artifact-specific patterns
                result.pattern_analysis = self._get_artifact_patterns(request.artifact_id)
                result.components_used.append('pattern_analysis')
            
            # 4. Semantic Analysis
            if 'vector_analysis' in self.components_loaded:
                logger.info("Performing semantic analysis")
                result.semantic_analysis = self._get_semantic_analysis(request.artifact_id)
                result.components_used.append('semantic_analysis')
            
            # 5. Quality Metrics
            result.quality_metrics = self._extract_quality_metrics(request.artifact_id)
            result.components_used.append('quality_metrics')
            
            # 6. Generate Insights (if requested)
            if request.include_insights and 'insights_generator' in self.components_loaded:
                logger.info("Generating insights")
                result.generated_insights = []
                
                # Artifact analysis insight
                artifact_insight = self.insights_generator.generate_artifact_analysis(request.artifact_id)
                result.generated_insights.append(artifact_insight)
                
                # Improvement recommendations
                improvement_insight = self.insights_generator.generate_improvement_recommendations(
                    request.artifact_id, request.target_conference
                )
                result.generated_insights.append(improvement_insight)
                
                result.components_used.append('insights_generation')
                
                # Calculate average insight confidence
                if result.generated_insights:
                    avg_confidence = sum(i.confidence_score for i in result.generated_insights) / len(result.generated_insights)
                    result.confidence_scores['insights'] = avg_confidence
            
            # 7. Generate Summary Report
            result.summary_report = self._generate_summary_report(result)
            
            # Calculate total evaluation time
            result.evaluation_time_seconds = time.time() - start_time
            
            # Cache the result
            self.evaluation_cache[cache_key] = result
            
            logger.info(f"Artifact evaluation completed in {result.evaluation_time_seconds:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Artifact evaluation failed: {e}")
            result.summary_report = f"Evaluation failed: {str(e)}"
            result.evaluation_time_seconds = time.time() - start_time
            return result
    
    def batch_evaluate_artifacts(self, artifact_ids: List[str],
                                target_conference: Optional[str] = None,
                                evaluation_type: str = "quick") -> List[EvaluationResult]:
        """
        Evaluate multiple artifacts in batch
        
        Args:
            artifact_ids: List of artifact IDs to evaluate
            target_conference: Target conference for all artifacts
            evaluation_type: Type of evaluation to perform
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Batch evaluating {len(artifact_ids)} artifacts")
        
        results = []
        for i, artifact_id in enumerate(artifact_ids):
            try:
                request = EvaluationRequest(
                    artifact_id=artifact_id,
                    target_conference=target_conference,
                    evaluation_type=evaluation_type,
                    include_insights=(evaluation_type == "comprehensive"),
                    include_comparisons=(evaluation_type == "comprehensive")
                )
                
                result = self.evaluate_artifact(request)
                results.append(result)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(artifact_ids)} evaluations")
                    
            except Exception as e:
                logger.error(f"Failed to evaluate {artifact_id}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    artifact_id=artifact_id,
                    timestamp=datetime.now().isoformat(),
                    summary_report=f"Evaluation failed: {str(e)}",
                    components_used=[],
                    confidence_scores={}
                )
                results.append(error_result)
        
        return results
    
    def query_patterns(self, query: str, pattern_type: Optional[str] = None,
                      conference: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the knowledge base for specific patterns
        
        Args:
            query: Natural language query about patterns
            pattern_type: Specific type of pattern to search for
            conference: Specific conference to filter by
            
        Returns:
            Query results with matching patterns and explanations
        """
        if not self.is_initialized:
            raise ValueError("Framework not initialized")
        
        logger.info(f"Querying patterns: {query}")
        
        # Use RAG insights generator for natural language queries
        if 'insights_generator' in self.components_loaded:
            # Create a mock pattern data structure for the query
            pattern_data = {
                'query': query,
                'pattern_type': pattern_type,
                'conference': conference,
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate explanation
            if pattern_type:
                insight = self.insights_generator.generate_pattern_explanation(pattern_type, pattern_data)
                return {
                    'query': query,
                    'results': [insight],
                    'explanation': insight.main_insight,
                    'recommendations': insight.recommendations
                }
        
        # Fallback to direct pattern search
        return self._direct_pattern_search(query, pattern_type, conference)
    
    def compare_conferences(self, conferences: List[str]) -> Dict[str, Any]:
        """
        Compare multiple conferences using the framework
        
        Args:
            conferences: List of conference names to compare
            
        Returns:
            Comprehensive comparison results
        """
        if not self.is_initialized:
            raise ValueError("Framework not initialized")
        
        logger.info(f"Comparing conferences: {conferences}")
        
        comparison_results = {
            'conferences': conferences,
            'timestamp': datetime.now().isoformat(),
            'comparison_insight': None,
            'individual_profiles': {},
            'similarities': {},
            'recommendations': {}
        }
        
        # Get individual conference profiles
        if 'conference_models' in self.components_loaded:
            for conf in conferences:
                if conf in self.conference_models.conference_profiles:
                    profile = self.conference_models.conference_profiles[conf]
                    comparison_results['individual_profiles'][conf] = {
                        'category': profile.category,
                        'preferred_sections': profile.preferred_sections,
                        'preferred_tools': profile.preferred_tools,
                        'documentation_style': profile.documentation_style,
                        'avg_doc_length': profile.avg_documentation_length,
                        'reproducibility_emphasis': profile.reproducibility_emphasis
                    }
        
        # Generate comparative insight
        if 'insights_generator' in self.components_loaded:
            comparison_insight = self.insights_generator.generate_conference_comparison(conferences)
            comparison_results['comparison_insight'] = comparison_insight
        
        # Calculate similarities
        comparison_results['similarities'] = self._calculate_conference_similarities(conferences)
        
        return comparison_results
    
    def generate_comprehensive_report(self, artifact_ids: List[str],
                                    output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report
        
        Args:
            artifact_ids: List of artifacts to include in the report
            output_file: Optional file to save the report
            
        Returns:
            Comprehensive report data
        """
        logger.info(f"Generating comprehensive report for {len(artifact_ids)} artifacts")
        
        # Evaluate all artifacts
        results = self.batch_evaluate_artifacts(artifact_ids, evaluation_type="comprehensive")
        
        # Aggregate statistics
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_artifacts': len(artifact_ids),
                'framework_version': '1.0.0',
                'components_used': self.components_loaded
            },
            'summary_statistics': self._calculate_summary_statistics(results),
            'pattern_analysis': self._aggregate_pattern_analysis(results),
            'conference_analysis': self._aggregate_conference_analysis(results),
            'quality_analysis': self._aggregate_quality_analysis(results),
            'recommendations': self._generate_aggregate_recommendations(results),
            'detailed_results': results
        }
        
        # Save to file if requested
        if output_file:
            self._save_report(report, output_file)
        
        return report
    
    def _get_artifact_patterns(self, artifact_id: str) -> Dict[str, Any]:
        """Get patterns specific to an artifact"""
        patterns = {
            'structural_patterns': [],
            'content_patterns': [],
            'tool_patterns': [],
            'quality_patterns': []
        }
        
        # Extract patterns from knowledge graph
        if hasattr(self.kg_builder, 'nx_graph'):
            G = self.kg_builder.nx_graph
            
            # Find artifact node and analyze its patterns
            for node in G.nodes():
                node_data = G.nodes[node]
                if (node_data.get('node_type') == 'Artifact' and 
                    node_data.get('artifact_id') == artifact_id):
                    
                    # Analyze connected components for patterns
                    for neighbor in G.neighbors(node):
                        neighbor_data = G.nodes[neighbor]
                        node_type = neighbor_data.get('node_type')
                        
                        if node_type == 'Tool':
                            patterns['tool_patterns'].append(neighbor_data.get('name', 'unknown'))
                        elif node_type == 'Section':
                            patterns['structural_patterns'].append(neighbor_data.get('section_type', 'unknown'))
        
        return patterns
    
    def _get_semantic_analysis(self, artifact_id: str) -> Dict[str, Any]:
        """Get semantic analysis for an artifact"""
        analysis = {
            'embedding_available': False,
            'similar_artifacts': [],
            'semantic_cluster': -1,
            'coherence_score': 0.0
        }
        
        if 'vector_analysis' in self.components_loaded:
            try:
                # Find similar artifacts
                similar_results = self.vector_analyzer.find_similar_documents(artifact_id)
                analysis['similar_artifacts'] = similar_results.similar_documents[:5]
                analysis['embedding_available'] = True
            except:
                pass
        
        return analysis
    
    def _extract_quality_metrics(self, artifact_id: str) -> Dict[str, Any]:
        """Extract quality metrics for an artifact"""
        metrics = {
            'completeness_score': 0.0,
            'structure_score': 0.0,
            'reproducibility_score': 0.0,
            'clarity_score': 0.0,
            'overall_score': 0.0
        }
        
        # Extract from knowledge graph
        if hasattr(self.kg_builder, 'nx_graph'):
            G = self.kg_builder.nx_graph
            
            for node in G.nodes():
                node_data = G.nodes[node]
                if (node_data.get('node_type') == 'Documentation' and 
                    artifact_id in node_data.get('artifact_id', '')):
                    
                    # Calculate scores based on available features
                    metrics['completeness_score'] = node_data.get('quality_score', 0) / 10.0
                    metrics['structure_score'] = min(node_data.get('has_headers', 0) + 
                                                   node_data.get('has_code_blocks', 0), 1.0)
                    metrics['clarity_score'] = node_data.get('readability_score', 0.5)
                    break
        
        # Calculate overall score
        scores = [metrics['completeness_score'], metrics['structure_score'], 
                 metrics['reproducibility_score'], metrics['clarity_score']]
        metrics['overall_score'] = sum(scores) / len(scores) if scores else 0.0
        
        return metrics
    
    def _generate_summary_report(self, result: EvaluationResult) -> str:
        """Generate a human-readable summary report"""
        report_parts = []
        
        report_parts.append(f"# Artifact Evaluation Report: {result.artifact_id}")
        report_parts.append(f"Generated: {result.timestamp}")
        report_parts.append("")
        
        # Acceptance prediction
        if result.acceptance_prediction:
            pred = result.acceptance_prediction
            report_parts.append("## Acceptance Prediction")
            report_parts.append(f"- Probability: {pred.acceptance_probability:.2%}")
            report_parts.append(f"- Classification: {pred.predicted_class}")
            report_parts.append(f"- Confidence: {pred.confidence_score:.2%}")
            report_parts.append("")
        
        # Conference recommendation
        if result.conference_recommendation:
            conf_rec = result.conference_recommendation
            report_parts.append("## Conference Recommendation")
            report_parts.append(f"- Recommended: {conf_rec.recommended_conference}")
            report_parts.append(f"- Match Score: {conf_rec.match_score:.2%}")
            report_parts.append(f"- Confidence: {conf_rec.confidence_score:.2%}")
            report_parts.append("")
        
        # Quality metrics
        if result.quality_metrics:
            report_parts.append("## Quality Assessment")
            metrics = result.quality_metrics
            report_parts.append(f"- Overall Score: {metrics.get('overall_score', 0):.2%}")
            report_parts.append(f"- Completeness: {metrics.get('completeness_score', 0):.2%}")
            report_parts.append(f"- Structure: {metrics.get('structure_score', 0):.2%}")
            report_parts.append(f"- Clarity: {metrics.get('clarity_score', 0):.2%}")
            report_parts.append("")
        
        # Key insights
        if result.generated_insights:
            report_parts.append("## Key Insights")
            for insight in result.generated_insights[:2]:  # Top 2 insights
                report_parts.append(f"### {insight.insight_type.replace('_', ' ').title()}")
                # Truncate long insights
                main_insight = insight.main_insight[:300] + "..." if len(insight.main_insight) > 300 else insight.main_insight
                report_parts.append(main_insight)
                report_parts.append("")
        
        # Performance info
        report_parts.append("## Evaluation Metadata")
        report_parts.append(f"- Evaluation Time: {result.evaluation_time_seconds:.2f} seconds")
        report_parts.append(f"- Components Used: {', '.join(result.components_used)}")
        
        return '\n'.join(report_parts)
    
    def _direct_pattern_search(self, query: str, pattern_type: Optional[str], 
                             conference: Optional[str]) -> Dict[str, Any]:
        """Direct pattern search without RAG"""
        # Simplified pattern search implementation
        return {
            'query': query,
            'results': [],
            'explanation': "Direct pattern search not yet implemented",
            'recommendations': ["Use RAG-powered search for better results"]
        }
    
    def _calculate_conference_similarities(self, conferences: List[str]) -> Dict[str, float]:
        """Calculate similarities between conferences"""
        similarities = {}
        
        if 'conference_models' in self.components_loaded:
            profiles = self.conference_models.conference_profiles
            
            for i, conf1 in enumerate(conferences):
                for j, conf2 in enumerate(conferences[i+1:], i+1):
                    if conf1 in profiles and conf2 in profiles:
                        # Simple similarity based on shared sections and tools
                        profile1 = profiles[conf1]
                        profile2 = profiles[conf2]
                        
                        shared_sections = set(profile1.preferred_sections) & set(profile2.preferred_sections)
                        shared_tools = set(profile1.preferred_tools) & set(profile2.preferred_tools)
                        
                        similarity = (len(shared_sections) + len(shared_tools)) / 10.0  # Normalize
                        similarities[f"{conf1}_vs_{conf2}"] = min(similarity, 1.0)
        
        return similarities
    
    def _calculate_summary_statistics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results"""
        stats = {
            'total_evaluated': len(results),
            'successful_evaluations': 0,
            'average_evaluation_time': 0.0,
            'acceptance_rate': 0.0,
            'top_conferences': {},
            'average_confidence': 0.0
        }
        
        successful_results = [r for r in results if r.acceptance_prediction is not None]
        stats['successful_evaluations'] = len(successful_results)
        
        if successful_results:
            # Average evaluation time
            stats['average_evaluation_time'] = sum(r.evaluation_time_seconds for r in successful_results) / len(successful_results)
            
            # Acceptance rate
            accepted_count = sum(1 for r in successful_results if r.acceptance_prediction.acceptance_probability > 0.5)
            stats['acceptance_rate'] = accepted_count / len(successful_results)
            
            # Top recommended conferences
            conf_counts = {}
            for r in successful_results:
                if r.conference_recommendation:
                    conf = r.conference_recommendation.recommended_conference
                    conf_counts[conf] = conf_counts.get(conf, 0) + 1
            
            stats['top_conferences'] = dict(sorted(conf_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            
            # Average confidence
            confidences = []
            for r in successful_results:
                if r.confidence_scores:
                    confidences.extend(r.confidence_scores.values())
            
            if confidences:
                stats['average_confidence'] = sum(confidences) / len(confidences)
        
        return stats
    
    def _aggregate_pattern_analysis(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate pattern analysis from multiple results"""
        # Simplified aggregation
        return {
            'common_patterns': [],
            'pattern_frequency': {},
            'pattern_effectiveness': {}
        }
    
    def _aggregate_conference_analysis(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate conference analysis from multiple results"""
        conference_stats = {}
        
        for result in results:
            if result.conference_recommendation:
                conf = result.conference_recommendation.recommended_conference
                if conf not in conference_stats:
                    conference_stats[conf] = {
                        'count': 0,
                        'avg_match_score': 0.0,
                        'avg_confidence': 0.0
                    }
                
                conference_stats[conf]['count'] += 1
                conference_stats[conf]['avg_match_score'] += result.conference_recommendation.match_score
                conference_stats[conf]['avg_confidence'] += result.conference_recommendation.confidence_score
        
        # Calculate averages
        for conf_data in conference_stats.values():
            if conf_data['count'] > 0:
                conf_data['avg_match_score'] /= conf_data['count']
                conf_data['avg_confidence'] /= conf_data['count']
        
        return conference_stats
    
    def _aggregate_quality_analysis(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate quality analysis from multiple results"""
        quality_scores = []
        
        for result in results:
            if result.quality_metrics:
                quality_scores.append(result.quality_metrics.get('overall_score', 0))
        
        if quality_scores:
            return {
                'average_quality': sum(quality_scores) / len(quality_scores),
                'quality_distribution': {
                    'high_quality': sum(1 for q in quality_scores if q > 0.7) / len(quality_scores),
                    'medium_quality': sum(1 for q in quality_scores if 0.4 <= q <= 0.7) / len(quality_scores),
                    'low_quality': sum(1 for q in quality_scores if q < 0.4) / len(quality_scores)
                }
            }
        
        return {'average_quality': 0.0, 'quality_distribution': {}}
    
    def _generate_aggregate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
        """Generate aggregate recommendations from multiple results"""
        all_recommendations = []
        
        for result in results:
            if result.generated_insights:
                for insight in result.generated_insights:
                    all_recommendations.extend(insight.recommendations)
        
        # Count frequency of recommendations
        from collections import Counter
        recommendation_counts = Counter(all_recommendations)
        
        # Return top 10 most common recommendations
        return [rec for rec, count in recommendation_counts.most_common(10)]
    
    def _save_report(self, report: Dict[str, Any], output_file: str):
        """Save report to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_file}")
    
    def save_framework_state(self, output_dir: str):
        """Save the current framework state"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save component states
        if self.scoring_framework:
            self.scoring_framework.save_models(output_path / "scoring_models")
        
        if self.conference_models:
            self.conference_models.save_conference_models(output_path / "conference_models")
        
        # Save framework metadata
        metadata = {
            'initialized': self.is_initialized,
            'components_loaded': self.components_loaded,
            'initialization_time': datetime.now().isoformat(),
            'config': asdict(config) if hasattr(config, '__dict__') else str(config)
        }
        
        with open(output_path / "framework_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Framework state saved to {output_dir}")

# Command Line Interface
def create_cli():
    """Create command line interface"""
    parser = argparse.ArgumentParser(description="Artifact Evaluation Framework")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize the framework')
    init_parser.add_argument('--artifacts-dir', required=True, help='Directory containing artifact data')
    init_parser.add_argument('--max-artifacts', type=int, help='Maximum number of artifacts to process')
    init_parser.add_argument('--use-neo4j', action='store_true', help='Use Neo4j for knowledge graph storage')
    init_parser.add_argument('--output-dir', help='Directory to save framework state')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate artifacts')
    eval_parser.add_argument('--artifact-id', help='Single artifact ID to evaluate')
    eval_parser.add_argument('--artifact-list', help='File containing list of artifact IDs')
    eval_parser.add_argument('--target-conference', help='Target conference for evaluation')
    eval_parser.add_argument('--evaluation-type', choices=['quick', 'comprehensive'], default='comprehensive')
    eval_parser.add_argument('--output-file', help='Output file for results')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query patterns and insights')
    query_parser.add_argument('--query', required=True, help='Natural language query')
    query_parser.add_argument('--pattern-type', help='Specific pattern type to search')
    query_parser.add_argument('--conference', help='Filter by conference')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare conferences')
    compare_parser.add_argument('--conferences', nargs='+', required=True, help='Conferences to compare')
    compare_parser.add_argument('--output-file', help='Output file for comparison results')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive report')
    report_parser.add_argument('--artifact-list', required=True, help='File containing list of artifact IDs')
    report_parser.add_argument('--output-file', required=True, help='Output file for report')
    
    return parser

def main():
    """Main entry point for CLI"""
    parser = create_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize framework
    framework = ArtifactEvaluationFramework()
    
    try:
        if args.command == 'init':
            # Initialize framework
            results = framework.initialize_framework(
                args.artifacts_dir,
                max_artifacts=args.max_artifacts,
                use_neo4j=args.use_neo4j
            )
            print(f"Framework initialized successfully: {results['components_initialized']}")
            
            if args.output_dir:
                framework.save_framework_state(args.output_dir)
        
        elif args.command == 'evaluate':
            # Load framework state or initialize
            if not framework.is_initialized:
                print("Framework not initialized. Please run 'init' command first.")
                return
            
            if args.artifact_id:
                # Single artifact evaluation
                request = EvaluationRequest(
                    artifact_id=args.artifact_id,
                    target_conference=args.target_conference,
                    evaluation_type=args.evaluation_type
                )
                result = framework.evaluate_artifact(request)
                print(result.summary_report)
                
                if args.output_file:
                    with open(args.output_file, 'w') as f:
                        json.dump(asdict(result), f, indent=2, default=str)
            
            elif args.artifact_list:
                # Batch evaluation
                with open(args.artifact_list, 'r') as f:
                    artifact_ids = [line.strip() for line in f if line.strip()]
                
                results = framework.batch_evaluate_artifacts(
                    artifact_ids,
                    target_conference=args.target_conference,
                    evaluation_type=args.evaluation_type
                )
                
                print(f"Evaluated {len(results)} artifacts")
                
                if args.output_file:
                    with open(args.output_file, 'w') as f:
                        json.dump([asdict(r) for r in results], f, indent=2, default=str)
        
        elif args.command == 'query':
            # Query patterns
            if not framework.is_initialized:
                print("Framework not initialized. Please run 'init' command first.")
                return
            
            results = framework.query_patterns(
                args.query,
                pattern_type=args.pattern_type,
                conference=args.conference
            )
            print(json.dumps(results, indent=2, default=str))
        
        elif args.command == 'compare':
            # Compare conferences
            if not framework.is_initialized:
                print("Framework not initialized. Please run 'init' command first.")
                return
            
            results = framework.compare_conferences(args.conferences)
            print(json.dumps(results, indent=2, default=str))
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
        
        elif args.command == 'report':
            # Generate comprehensive report
            if not framework.is_initialized:
                print("Framework not initialized. Please run 'init' command first.")
                return
            
            with open(args.artifact_list, 'r') as f:
                artifact_ids = [line.strip() for line in f if line.strip()]
            
            report = framework.generate_comprehensive_report(artifact_ids, args.output_file)
            print(f"Report generated for {len(artifact_ids)} artifacts")
            print(f"Average quality score: {report['quality_analysis'].get('average_quality', 0):.2%}")
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 