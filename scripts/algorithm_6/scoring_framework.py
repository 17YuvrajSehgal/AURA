"""
Scoring Framework for Artifact Evaluation

This module implements ML and rule-based scoring systems to predict artifact
acceptance probability at research conferences. It combines knowledge graph
patterns, semantic embeddings, and conference-specific criteria.

Key Features:
- Multi-level feature extraction (graph, semantic, structural)
- Conference-specific prediction models
- Ensemble learning for robust predictions
- Explainable AI with feature importance analysis
- Quality scoring and actionable feedback generation
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import shap
# ML and model building
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# NLP and text analysis
from textstat import flesch_reading_ease, automated_readability_index

from config import config, NODE_TYPES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArtifactFeatures:
    """Comprehensive features extracted from an artifact"""
    # Basic metadata
    artifact_id: str
    conference: str
    year: int

    # Documentation features
    has_readme: bool = False
    readme_length: int = 0
    section_count: int = 0
    header_depth_avg: float = 0.0

    # Structure features
    has_installation_section: bool = False
    has_usage_section: bool = False
    has_requirements_section: bool = False
    has_examples_section: bool = False
    has_license_section: bool = False
    has_citation_section: bool = False

    # Tool support features
    has_docker: bool = False
    has_conda: bool = False
    has_pip_requirements: bool = False
    has_build_scripts: bool = False
    has_test_scripts: bool = False

    # Quality indicators
    code_block_count: int = 0
    numbered_list_count: int = 0
    bullet_point_count: int = 0
    image_count: int = 0
    link_count: int = 0

    # Readability metrics
    flesch_score: float = 0.0
    readability_index: float = 0.0

    # Graph-based features
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    pagerank_score: float = 0.0
    clustering_coefficient: float = 0.0

    # Semantic features
    readme_embedding: List[float] = None
    semantic_cluster_id: int = -1
    semantic_coherence_score: float = 0.0

    # Conference-specific features
    conference_alignment_score: float = 0.0
    tool_alignment_score: float = 0.0
    style_alignment_score: float = 0.0

    # Target variable
    acceptance_probability: float = 1.0  # Default for accepted artifacts


@dataclass
class PredictionResult:
    """Result of artifact acceptance prediction"""
    artifact_id: str
    acceptance_probability: float
    confidence_score: float
    predicted_class: str
    feature_importance: Dict[str, float]
    recommendations: List[str]
    conference_specific_feedback: Dict[str, Any]


class ArtifactScoringFramework:
    """ML-powered framework for predicting artifact acceptance"""

    def __init__(self, kg_builder, pattern_analyzer, vector_analyzer):
        self.kg_builder = kg_builder
        self.pattern_analyzer = pattern_analyzer
        self.vector_analyzer = vector_analyzer

        # ML models
        self.models: Dict[str, Any] = {}
        self.conference_models: Dict[str, Any] = {}
        self.feature_scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k='all')

        # Feature extraction components
        self.feature_extractors = {}
        self.trained_features: List[str] = []

        # Analysis results storage
        self.training_data: pd.DataFrame = None
        self.model_performance: Dict[str, Dict[str, float]] = {}

        # SHAP explainer for model interpretability
        self.shap_explainers: Dict[str, Any] = {}

    def train_acceptance_models(self, training_artifacts: List[str],
                                validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train ML models to predict artifact acceptance
        
        Args:
            training_artifacts: List of artifact IDs to use for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results and model performance metrics
        """
        logger.info("Training artifact acceptance prediction models")

        # Extract features from training artifacts
        training_features = self._extract_features_batch(training_artifacts)

        if len(training_features) < 10:
            raise ValueError("Need at least 10 training artifacts")

        # Convert to DataFrame
        features_df = pd.DataFrame([self._features_to_dict(f) for f in training_features])

        # Prepare training data
        X, y, feature_names = self._prepare_training_data(features_df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        # Train multiple models
        models_config = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }

        training_results = {
            'model_performance': {},
            'feature_importance': {},
            'best_model': None,
            'best_score': 0.0
        }

        # Train and evaluate each model
        for model_name, model in models_config.items():
            logger.info(f"Training {model_name}")

            # Train model
            model.fit(X_train_selected, y_train)

            # Evaluate
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]

            # Calculate metrics
            performance = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5)
            performance['cv_mean'] = cv_scores.mean()
            performance['cv_std'] = cv_scores.std()

            training_results['model_performance'][model_name] = performance

            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                selected_feature_names = [feature_names[i] for i in self.feature_selector.get_support(indices=True)]
                importance_dict = dict(zip(selected_feature_names, model.feature_importances_))
                training_results['feature_importance'][model_name] = importance_dict

            # Store model
            self.models[model_name] = model

            # Track best model
            if performance['f1_score'] > training_results['best_score']:
                training_results['best_score'] = performance['f1_score']
                training_results['best_model'] = model_name

            # Setup SHAP explainer for interpretability
            try:
                if model_name == 'random_forest':
                    self.shap_explainers[model_name] = shap.TreeExplainer(model)
                elif model_name == 'logistic_regression':
                    self.shap_explainers[model_name] = shap.LinearExplainer(model, X_train_selected)
            except Exception as e:
                logger.warning(f"Failed to create SHAP explainer for {model_name}: {e}")

        # Create ensemble model
        ensemble_models = [(name, model) for name, model in self.models.items()
                           if name in ['random_forest', 'gradient_boosting', 'logistic_regression']]

        if len(ensemble_models) >= 2:
            ensemble = VotingClassifier(ensemble_models, voting='soft')
            ensemble.fit(X_train_selected, y_train)

            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(X_test_selected)
            y_pred_proba_ensemble = ensemble.predict_proba(X_test_selected)[:, 1]

            ensemble_performance = {
                'accuracy': accuracy_score(y_test, y_pred_ensemble),
                'precision': precision_score(y_test, y_pred_ensemble, average='weighted'),
                'recall': recall_score(y_test, y_pred_ensemble, average='weighted'),
                'f1_score': f1_score(y_test, y_pred_ensemble, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba_ensemble)
            }

            training_results['model_performance']['ensemble'] = ensemble_performance
            self.models['ensemble'] = ensemble

            if ensemble_performance['f1_score'] > training_results['best_score']:
                training_results['best_score'] = ensemble_performance['f1_score']
                training_results['best_model'] = 'ensemble'

        # Store training data and feature names
        self.training_data = features_df
        self.trained_features = feature_names

        # Train conference-specific models
        conference_results = self._train_conference_specific_models(features_df)
        training_results['conference_models'] = conference_results

        logger.info(f"Model training completed. Best model: {training_results['best_model']}")
        return training_results

    def predict_acceptance(self, artifact_id: str,
                           target_conference: Optional[str] = None,
                           use_ensemble: bool = True) -> PredictionResult:
        """
        Predict acceptance probability for an artifact
        
        Args:
            artifact_id: ID of the artifact to evaluate
            target_conference: Target conference for submission
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            Prediction result with probability, confidence, and recommendations
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_acceptance_models() first.")

        # Extract features for the artifact
        features = self._extract_single_artifact_features(artifact_id)

        if not features:
            raise ValueError(f"Could not extract features for artifact {artifact_id}")

        # Convert to model input format
        features_dict = self._features_to_dict(features)
        feature_vector = self._dict_to_vector(features_dict)

        # Scale and select features
        feature_vector_scaled = self.feature_scaler.transform([feature_vector])
        feature_vector_selected = self.feature_selector.transform(feature_vector_scaled)

        # Choose model
        model_name = 'ensemble' if use_ensemble and 'ensemble' in self.models else 'random_forest'
        model = self.models.get(model_name, list(self.models.values())[0])

        # Make prediction
        acceptance_probability = model.predict_proba(feature_vector_selected)[0][1]
        predicted_class = "LIKELY_ACCEPTED" if acceptance_probability > 0.5 else "NEEDS_IMPROVEMENT"

        # Calculate confidence score
        confidence_score = max(acceptance_probability, 1 - acceptance_probability)

        # Get feature importance for this prediction
        feature_importance = self._get_prediction_feature_importance(
            feature_vector_selected, model_name, features_dict
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(features, feature_importance)

        # Conference-specific feedback
        conference_feedback = {}
        if target_conference:
            conference_feedback = self._get_conference_specific_feedback(
                features, target_conference
            )

        return PredictionResult(
            artifact_id=artifact_id,
            acceptance_probability=float(acceptance_probability),
            confidence_score=float(confidence_score),
            predicted_class=predicted_class,
            feature_importance=feature_importance,
            recommendations=recommendations,
            conference_specific_feedback=conference_feedback
        )

    def batch_evaluate_artifacts(self, artifact_ids: List[str],
                                 target_conference: Optional[str] = None) -> List[PredictionResult]:
        """Evaluate multiple artifacts in batch"""
        logger.info(f"Batch evaluating {len(artifact_ids)} artifacts")

        results = []
        for artifact_id in artifact_ids:
            try:
                result = self.predict_acceptance(artifact_id, target_conference)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate artifact {artifact_id}: {e}")

        return results

    def _extract_features_batch(self, artifact_ids: List[str]) -> List[ArtifactFeatures]:
        """Extract features from multiple artifacts"""
        features_list = []

        for artifact_id in artifact_ids:
            try:
                features = self._extract_single_artifact_features(artifact_id)
                if features:
                    features_list.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features for {artifact_id}: {e}")

        return features_list

    def _extract_single_artifact_features(self, artifact_id: str) -> Optional[ArtifactFeatures]:
        """Extract comprehensive features from a single artifact"""
        if not hasattr(self.kg_builder, 'nx_graph'):
            logger.error("Knowledge graph not available")
            return None

        G = self.kg_builder.nx_graph

        # Find artifact node
        artifact_node = None
        for node in G.nodes():
            node_data = G.nodes[node]
            if (node_data.get('node_type') == NODE_TYPES['ARTIFACT'] and
                    node_data.get('artifact_id') == artifact_id):
                artifact_node = node
                break

        if not artifact_node:
            logger.warning(f"Artifact {artifact_id} not found in knowledge graph")
            return None

        artifact_data = G.nodes[artifact_node]

        # Initialize features
        features = ArtifactFeatures(
            artifact_id=artifact_id,
            conference=artifact_data.get('conference', 'unknown'),
            year=artifact_data.get('year', 2024)
        )

        # Extract documentation features
        self._extract_documentation_features(G, artifact_node, features)

        # Extract structure features
        self._extract_structure_features(G, artifact_node, features)

        # Extract tool support features
        self._extract_tool_features(G, artifact_node, features)

        # Extract quality indicators
        self._extract_quality_features(G, artifact_node, features)

        # Extract graph-based features
        self._extract_graph_features(G, artifact_node, features)

        # Extract semantic features
        self._extract_semantic_features(artifact_node, features)

        # Calculate conference alignment
        self._calculate_conference_alignment(features)

        return features

    def _extract_documentation_features(self, G, artifact_node: str, features: ArtifactFeatures):
        """Extract documentation-related features"""
        # Find documentation files
        for neighbor in G.neighbors(artifact_node):
            neighbor_data = G.nodes[neighbor]
            if neighbor_data.get('node_type') == NODE_TYPES['DOCUMENTATION']:
                features.has_readme = True
                features.readme_length = neighbor_data.get('content_length', 0)

                # Extract readability metrics
                file_name = neighbor_data.get('file_name', '').lower()
                if 'readme' in file_name:
                    # Get content for readability analysis
                    content_length = neighbor_data.get('content_length', 0)
                    if content_length > 0:
                        # Use simple heuristics since we don't have full content
                        features.flesch_score = neighbor_data.get('readability_score', 0.5) * 100
                        features.readability_index = min(content_length / 1000, 10)

                # Count sections
                for section_neighbor in G.neighbors(neighbor):
                    section_data = G.nodes[section_neighbor]
                    if section_data.get('node_type') == NODE_TYPES['SECTION']:
                        features.section_count += 1

                        # Calculate average header depth
                        level = section_data.get('level', 1)
                        features.header_depth_avg = (features.header_depth_avg * (
                                    features.section_count - 1) + level) / features.section_count

    def _extract_structure_features(self, G, artifact_node: str, features: ArtifactFeatures):
        """Extract structural features from documentation sections"""
        # Analyze sections for required components
        section_types = []

        for neighbor in G.neighbors(artifact_node):
            neighbor_data = G.nodes[neighbor]
            if neighbor_data.get('node_type') == NODE_TYPES['DOCUMENTATION']:
                # Check sections
                for section_neighbor in G.neighbors(neighbor):
                    section_data = G.nodes[section_neighbor]
                    if section_data.get('node_type') == NODE_TYPES['SECTION']:
                        section_type = section_data.get('section_type', '').lower()
                        section_types.append(section_type)

        # Set structure features based on found sections
        features.has_installation_section = any('install' in s or 'setup' in s for s in section_types)
        features.has_usage_section = any('usage' in s or 'example' in s for s in section_types)
        features.has_requirements_section = any('requirement' in s or 'dependency' in s for s in section_types)
        features.has_examples_section = any('example' in s or 'tutorial' in s for s in section_types)
        features.has_license_section = any('license' in s for s in section_types)
        features.has_citation_section = any('citation' in s or 'cite' in s for s in section_types)

    def _extract_tool_features(self, G, artifact_node: str, features: ArtifactFeatures):
        """Extract tool support features"""
        # Find tools connected to the artifact
        for neighbor in G.neighbors(artifact_node):
            neighbor_data = G.nodes[neighbor]
            if neighbor_data.get('node_type') == NODE_TYPES['TOOL']:
                tool_name = neighbor_data.get('name', '').lower()

                if 'docker' in tool_name:
                    features.has_docker = True
                elif 'conda' in tool_name:
                    features.has_conda = True
                elif 'pip' in tool_name:
                    features.has_pip_requirements = True
                elif any(build_tool in tool_name for build_tool in ['make', 'cmake', 'gradle', 'maven']):
                    features.has_build_scripts = True
                elif 'test' in tool_name:
                    features.has_test_scripts = True

    def _extract_quality_features(self, G, artifact_node: str, features: ArtifactFeatures):
        """Extract quality indicator features"""
        # Extract quality metrics from documentation nodes
        for neighbor in G.neighbors(artifact_node):
            neighbor_data = G.nodes[neighbor]
            if neighbor_data.get('node_type') == NODE_TYPES['DOCUMENTATION']:
                features.code_block_count += neighbor_data.get('has_code_blocks', 0)
                features.numbered_list_count += neighbor_data.get('has_numbered_lists', 0)
                features.bullet_point_count += neighbor_data.get('has_bullet_points', 0)
                features.image_count += neighbor_data.get('has_images', 0)
                features.link_count += neighbor_data.get('has_links', 0)

    def _extract_graph_features(self, G, artifact_node: str, features: ArtifactFeatures):
        """Extract graph-based centrality features"""
        # Get centrality metrics if available
        if hasattr(self.pattern_analyzer, 'centrality_metrics'):
            metrics = self.pattern_analyzer.centrality_metrics.get(artifact_node)
            if metrics:
                features.degree_centrality = metrics.degree_centrality
                features.betweenness_centrality = metrics.betweenness_centrality
                features.pagerank_score = metrics.pagerank
                features.clustering_coefficient = metrics.clustering_coefficient

    def _extract_semantic_features(self, artifact_node: str, features: ArtifactFeatures):
        """Extract semantic embedding features"""
        # Get artifact embedding
        if artifact_node in self.vector_analyzer.artifact_embeddings:
            embedding = self.vector_analyzer.artifact_embeddings[artifact_node]
            features.readme_embedding = embedding.tolist()

            # Calculate semantic coherence (average similarity to other artifacts)
            other_embeddings = [emb for node, emb in self.vector_analyzer.artifact_embeddings.items() if
                                node != artifact_node]
            if other_embeddings:
                similarities = [np.dot(embedding, other) / (np.linalg.norm(embedding) * np.linalg.norm(other))
                                for other in other_embeddings[:10]]  # Sample for efficiency
                features.semantic_coherence_score = float(np.mean(similarities))

        # Get cluster assignment if available
        if hasattr(self.vector_analyzer, 'clustering_results'):
            # Find cluster assignment (simplified)
            features.semantic_cluster_id = 0

    def _calculate_conference_alignment(self, features: ArtifactFeatures):
        """Calculate how well the artifact aligns with conference expectations"""
        conference = features.conference

        if conference in config.conference.conference_categories:
            # Find the category for this conference
            category = None
            for cat_name, cat_info in config.conference.conference_categories.items():
                if conference in cat_info['conferences']:
                    category = cat_name
                    break

            if category:
                cat_info = config.conference.conference_categories[category]

                # Tool alignment
                required_tools = cat_info['required_tools']
                tool_score = 0
                if 'docker' in required_tools and features.has_docker:
                    tool_score += 1
                if 'scripts' in required_tools and (features.has_build_scripts or features.has_test_scripts):
                    tool_score += 1
                features.tool_alignment_score = tool_score / max(len(required_tools), 1)

                # Style alignment (simplified)
                style = cat_info['documentation_style']
                if style == 'technical_detailed':
                    features.style_alignment_score = min(features.section_count / 8, 1.0)
                elif style == 'methodology_focused':
                    features.style_alignment_score = float(features.has_usage_section and features.has_examples_section)
                else:
                    features.style_alignment_score = 0.5

                # Overall conference alignment
                features.conference_alignment_score = (
                                                                  features.tool_alignment_score + features.style_alignment_score) / 2

    def _features_to_dict(self, features: ArtifactFeatures) -> Dict[str, Any]:
        """Convert ArtifactFeatures to dictionary"""
        feature_dict = {}

        # Skip non-numeric fields for ML
        skip_fields = ['artifact_id', 'conference', 'readme_embedding']

        for field_name, field_value in features.__dict__.items():
            if field_name not in skip_fields:
                if isinstance(field_value, bool):
                    feature_dict[field_name] = int(field_value)
                elif isinstance(field_value, (int, float)):
                    feature_dict[field_name] = field_value
                else:
                    feature_dict[field_name] = 0

        return feature_dict

    def _prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for ML models"""
        # Remove non-numeric columns
        exclude_cols = ['artifact_id', 'conference', 'acceptance_probability']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        X = features_df[feature_cols].fillna(0).values
        y = (features_df['acceptance_probability'] > 0.5).astype(int).values  # Binary classification

        return X, y, feature_cols

    def _dict_to_vector(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to vector using trained feature order"""
        vector = []
        for feature_name in self.trained_features:
            vector.append(features_dict.get(feature_name, 0))
        return np.array(vector)

    def _get_prediction_feature_importance(self, feature_vector: np.ndarray,
                                           model_name: str, features_dict: Dict[str, Any]) -> Dict[str, float]:
        """Get feature importance for a specific prediction"""
        importance_dict = {}

        # Use SHAP if available
        if model_name in self.shap_explainers:
            try:
                shap_values = self.shap_explainers[model_name].shap_values(feature_vector)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification

                selected_feature_names = [self.trained_features[i]
                                          for i in self.feature_selector.get_support(indices=True)]

                for i, importance in enumerate(shap_values[0]):
                    if i < len(selected_feature_names):
                        importance_dict[selected_feature_names[i]] = float(importance)

            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        # Fallback to model feature importance
        if not importance_dict and model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                selected_feature_names = [self.trained_features[i]
                                          for i in self.feature_selector.get_support(indices=True)]
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(selected_feature_names):
                        importance_dict[selected_feature_names[i]] = float(importance)

        return importance_dict

    def _generate_recommendations(self, features: ArtifactFeatures,
                                  feature_importance: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations for improvement"""
        recommendations = []

        # High-impact missing features
        high_impact_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        for feature_name, importance in high_impact_features:
            if importance > 0.1:  # Significant importance
                if 'installation' in feature_name and not features.has_installation_section:
                    recommendations.append("Add a detailed installation/setup section with step-by-step instructions")
                elif 'usage' in feature_name and not features.has_usage_section:
                    recommendations.append("Include comprehensive usage examples and tutorials")
                elif 'docker' in feature_name and not features.has_docker:
                    recommendations.append("Consider adding Docker support for reproducibility")
                elif 'requirements' in feature_name and not features.has_requirements_section:
                    recommendations.append("Clearly document all dependencies and requirements")

        # General quality improvements
        if features.section_count < 5:
            recommendations.append("Expand documentation with more detailed sections")

        if features.code_block_count == 0:
            recommendations.append("Add code examples and command-line snippets")

        if not features.has_examples_section:
            recommendations.append("Include practical examples and use cases")

        if features.readme_length < 500:
            recommendations.append("Expand README with more comprehensive documentation")

        return recommendations[:5]  # Limit to top 5 recommendations

    def _get_conference_specific_feedback(self, features: ArtifactFeatures,
                                          target_conference: str) -> Dict[str, Any]:
        """Generate conference-specific feedback"""
        feedback = {
            'target_conference': target_conference,
            'alignment_score': features.conference_alignment_score,
            'specific_recommendations': []
        }

        # Find conference category
        category = None
        for cat_name, cat_info in config.conference.conference_categories.items():
            if target_conference in cat_info['conferences']:
                category = cat_name
                break

        if category:
            cat_info = config.conference.conference_categories[category]

            # Category-specific recommendations
            if category == 'systems':
                if not features.has_docker:
                    feedback['specific_recommendations'].append(
                        "Systems conferences highly value reproducibility - add Docker support"
                    )
                if not features.has_build_scripts:
                    feedback['specific_recommendations'].append(
                        "Include build scripts and performance benchmarks"
                    )

            elif category == 'software_engineering':
                if not features.has_test_scripts:
                    feedback['specific_recommendations'].append(
                        "SE conferences expect test suites and validation scripts"
                    )
                if not features.has_usage_section:
                    feedback['specific_recommendations'].append(
                        "Add detailed methodology and replication instructions"
                    )

            elif category == 'machine_learning':
                if not features.has_conda:
                    feedback['specific_recommendations'].append(
                        "ML conferences prefer Conda environments for dependency management"
                    )
                if not features.has_examples_section:
                    feedback['specific_recommendations'].append(
                        "Include Jupyter notebooks with example experiments"
                    )

        return feedback

    def _train_conference_specific_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train conference-specific models"""
        conference_results = {}

        # Group by conference
        conference_groups = features_df.groupby('conference')

        for conference, group_df in conference_groups:
            if len(group_df) < 10:  # Skip conferences with too few samples
                continue

            try:
                # Prepare conference-specific training data
                X, y, feature_names = self._prepare_training_data(group_df)

                if len(np.unique(y)) < 2:  # Need both classes
                    continue

                # Train simple model for this conference
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)

                # Evaluate with cross-validation
                cv_scores = cross_val_score(model, X, y, cv=min(5, len(group_df)))

                conference_results[conference] = {
                    'model': model,
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std(),
                    'training_samples': len(group_df)
                }

                self.conference_models[conference] = model

            except Exception as e:
                logger.warning(f"Failed to train model for {conference}: {e}")

        return conference_results

    def save_models(self, output_dir: str):
        """Save trained models to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main models
        models_file = output_path / "acceptance_models.pkl"
        joblib.dump({
            'models': self.models,
            'feature_scaler': self.feature_scaler,
            'feature_selector': self.feature_selector,
            'trained_features': self.trained_features
        }, models_file)

        # Save conference models
        conference_models_file = output_path / "conference_models.pkl"
        joblib.dump(self.conference_models, conference_models_file)

        # Save performance metrics
        performance_file = output_path / "model_performance.json"
        with open(performance_file, 'w') as f:
            json.dump(self.model_performance, f, indent=2)

        logger.info(f"Models saved to {output_dir}")

    def load_models(self, models_dir: str):
        """Load trained models from disk"""
        models_path = Path(models_dir)

        # Load main models
        models_file = models_path / "acceptance_models.pkl"
        if models_file.exists():
            model_data = joblib.load(models_file)
            self.models = model_data['models']
            self.feature_scaler = model_data['feature_scaler']
            self.feature_selector = model_data['feature_selector']
            self.trained_features = model_data['trained_features']

        # Load conference models
        conference_models_file = models_path / "conference_models.pkl"
        if conference_models_file.exists():
            self.conference_models = joblib.load(conference_models_file)

        logger.info(f"Models loaded from {models_dir}")

    def generate_evaluation_report(self, artifact_results: List[PredictionResult],
                                   output_file: str):
        """Generate comprehensive evaluation report"""
        report_data = {
            'summary': {
                'total_artifacts': len(artifact_results),
                'likely_accepted': sum(1 for r in artifact_results if r.predicted_class == "LIKELY_ACCEPTED"),
                'needs_improvement': sum(1 for r in artifact_results if r.predicted_class == "NEEDS_IMPROVEMENT"),
                'average_confidence': np.mean([r.confidence_score for r in artifact_results])
            },
            'detailed_results': []
        }

        for result in artifact_results:
            report_data['detailed_results'].append({
                'artifact_id': result.artifact_id,
                'acceptance_probability': result.acceptance_probability,
                'confidence_score': result.confidence_score,
                'predicted_class': result.predicted_class,
                'top_recommendations': result.recommendations[:3],
                'top_feature_importance': dict(list(result.feature_importance.items())[:5])
            })

        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Evaluation report saved to {output_file}")
