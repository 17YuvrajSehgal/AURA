#!/usr/bin/env python3
"""
Pattern Analysis System for Accepted Artifacts

This module analyzes patterns across multiple accepted artifacts to identify
key success factors and evaluation criteria using graph analytics.
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from artifact_evaluation_system import ArtifactEvaluationSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternAnalysisSystem:
    """
    Analyze patterns across accepted artifacts to identify success factors.
    """

    def __init__(
            self,
            neo4j_uri: str = "bolt://localhost:7687",
            neo4j_user: str = "neo4j",
            neo4j_password: str = "12345678",
            openai_api_key: str = None
    ):
        """Initialize a pattern analysis system."""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.openai_api_key = openai_api_key

        # Initialize evaluation system
        self.evaluator = ArtifactEvaluationSystem(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            openai_api_key=openai_api_key
        )

        self.kg_builder = self.evaluator.kg_builder
        self.graph = self.kg_builder.graph

        # Pattern storage
        self.artifacts_data = []
        self.pattern_analysis = {}
        self.success_indicators = {}
        self.evaluation_criteria = {}

    def build_unified_knowledge_graph(self, artifacts_directory: str) -> Dict[str, Any]:
        """
        Build unified knowledge graph from all accepted artifacts.
        
        Args:
            artifacts_directory: Directory containing JSON analysis files
            
        Returns:
            Dictionary with build results and statistics
        """
        artifacts_dir = Path(artifacts_directory)
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Directory not found: {artifacts_directory}")

        # Find all JSON files
        json_files = list(artifacts_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to analyze")

        build_results = {
            "total_files": len(json_files),
            "successful_builds": 0,
            "failed_builds": 0,
            "artifacts_processed": [],
            "total_nodes": 0,
            "total_relationships": 0,
            "patterns_discovered": {},
            "build_errors": []
        }

        # Process each artifact
        for json_file in json_files:
            try:
                logger.info(f"Processing: {json_file.name}")

                # Load and evaluate artifact
                result = self.evaluator.evaluate_artifact_from_json(str(json_file))

                if result["success"]:
                    self.artifacts_data.append(result)
                    build_results["successful_builds"] += 1
                    build_results["artifacts_processed"].append(result["artifact_name"])

                    # Add to node/relationship counts
                    kg_result = result.get("kg_result", {})
                    build_results["total_nodes"] += kg_result.get("nodes_created", 0)

                    logger.info(f"✅ Successfully processed: {result['artifact_name']}")
                else:
                    build_results["failed_builds"] += 1
                    build_results["build_errors"].append({
                        "file": str(json_file),
                        "error": result.get("error", "Unknown error")
                    })
                    logger.warning(f"❌ Failed to process: {json_file.name}")

            except Exception as e:
                build_results["failed_builds"] += 1
                build_results["build_errors"].append({
                    "file": str(json_file),
                    "error": str(e)
                })
                logger.error(f"Error processing {json_file.name}: {e}")

        # Perform pattern analysis
        if self.artifacts_data:
            logger.info("Performing pattern analysis...")
            self.pattern_analysis = self._analyze_patterns()
            build_results["patterns_discovered"] = self.pattern_analysis

            # Extract success indicators
            self.success_indicators = self._extract_success_indicators()

            # Define evaluation criteria based on patterns
            self.evaluation_criteria = self._define_evaluation_criteria()

        logger.info(
            f"Knowledge graph building completed: {build_results['successful_builds']}/{build_results['total_files']} artifacts processed")

        return build_results

    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across all processed artifacts."""
        patterns = {
            "feature_patterns": self._analyze_feature_patterns(),
            "documentation_patterns": self._analyze_documentation_patterns(),
            "structure_patterns": self._analyze_structure_patterns(),
            "technology_patterns": self._analyze_technology_patterns(),
            "complexity_patterns": self._analyze_complexity_patterns(),
            "score_correlations": self._analyze_score_correlations(),
            "graph_patterns": self._analyze_graph_patterns()
        }

        return patterns

    def _analyze_feature_patterns(self) -> Dict[str, Any]:
        """Analyze common features across accepted artifacts."""
        feature_stats = defaultdict(list)

        for artifact in self.artifacts_data:
            features = artifact.get("features", {})
            for feature, value in features.items():
                if isinstance(value, bool):
                    feature_stats[feature].append(value)
                elif isinstance(value, (int, float)):
                    feature_stats[feature].append(value)
                elif isinstance(value, str):
                    feature_stats[f"{feature}_{value}"].append(1)

        # Calculate statistics
        patterns = {}
        for feature, values in feature_stats.items():
            if all(isinstance(v, bool) for v in values):
                # Boolean features
                true_count = sum(values)
                patterns[feature] = {
                    "prevalence": true_count / len(values),
                    "count": true_count,
                    "total": len(values),
                    "is_success_indicator": true_count / len(values) > 0.7
                }
            elif all(isinstance(v, (int, float)) for v in values):
                # Numeric features
                patterns[feature] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        return patterns

    def _analyze_documentation_patterns(self) -> Dict[str, Any]:
        """Analyze documentation patterns."""
        doc_patterns = {
            "readme_analysis": defaultdict(int),
            "section_patterns": defaultdict(int),
            "length_analysis": [],
            "keyword_patterns": defaultdict(int)
        }

        for artifact in self.artifacts_data:
            features = artifact.get("features", {})

            # README analysis
            if features.get("has_readme", False):
                doc_patterns["readme_analysis"]["has_readme"] += 1
                readme_length = features.get("readme_length", 0)
                doc_patterns["length_analysis"].append(readme_length)

                # Categorize README length
                if readme_length > 2000:
                    doc_patterns["readme_analysis"]["comprehensive"] += 1
                elif readme_length > 500:
                    doc_patterns["readme_analysis"]["detailed"] += 1
                else:
                    doc_patterns["readme_analysis"]["basic"] += 1

            # Documentation sections
            doc_sections = features.get("documentation_sections", 0)
            if doc_sections > 5:
                doc_patterns["section_patterns"]["well_structured"] += 1
            elif doc_sections > 2:
                doc_patterns["section_patterns"]["moderate"] += 1
            else:
                doc_patterns["section_patterns"]["minimal"] += 1

        # Calculate statistics
        if doc_patterns["length_analysis"]:
            doc_patterns["length_stats"] = {
                "mean": np.mean(doc_patterns["length_analysis"]),
                "median": np.median(doc_patterns["length_analysis"]),
                "recommended_min": np.percentile(doc_patterns["length_analysis"], 25)
            }

        return dict(doc_patterns)

    def _analyze_structure_patterns(self) -> Dict[str, Any]:
        """Analyze repository structure patterns."""
        structure_patterns = {
            "file_counts": [],
            "code_ratios": [],
            "tree_depths": [],
            "size_categories": defaultdict(int),
            "organization_patterns": defaultdict(int)
        }

        for artifact in self.artifacts_data:
            features = artifact.get("features", {})

            total_files = features.get("total_files", 0)
            code_files = features.get("code_files", 0)
            tree_depth = features.get("tree_depth", 0)
            repo_size = features.get("repo_size_mb", 0)

            structure_patterns["file_counts"].append(total_files)
            structure_patterns["tree_depths"].append(tree_depth)

            # Code ratio
            if total_files > 0:
                code_ratio = code_files / total_files
                structure_patterns["code_ratios"].append(code_ratio)

            # Size categories
            if repo_size < 1:
                structure_patterns["size_categories"]["small"] += 1
            elif repo_size < 10:
                structure_patterns["size_categories"]["medium"] += 1
            elif repo_size < 100:
                structure_patterns["size_categories"]["large"] += 1
            else:
                structure_patterns["size_categories"]["very_large"] += 1

            # Organization patterns
            if tree_depth > 3:
                structure_patterns["organization_patterns"]["deep_hierarchy"] += 1
            elif tree_depth > 1:
                structure_patterns["organization_patterns"]["organized"] += 1
            else:
                structure_patterns["organization_patterns"]["flat"] += 1

        # Calculate statistics
        for metric in ["file_counts", "code_ratios", "tree_depths"]:
            if structure_patterns[metric]:
                structure_patterns[f"{metric}_stats"] = {
                    "mean": np.mean(structure_patterns[metric]),
                    "median": np.median(structure_patterns[metric]),
                    "std": np.std(structure_patterns[metric]),
                    "percentile_75": np.percentile(structure_patterns[metric], 75),
                    "percentile_25": np.percentile(structure_patterns[metric], 25)
                }

        return structure_patterns

    def _analyze_technology_patterns(self) -> Dict[str, Any]:
        """Analyze technology and tool patterns."""
        tech_patterns = {
            "docker_usage": 0,
            "setup_complexity": defaultdict(int),
            "build_systems": defaultdict(int),
            "programming_languages": defaultdict(int)
        }

        for artifact in self.artifacts_data:
            features = artifact.get("features", {})

            # Docker usage
            if features.get("has_docker", False):
                tech_patterns["docker_usage"] += 1

            # Setup complexity
            complexity = features.get("setup_complexity", "unknown")
            tech_patterns["setup_complexity"][complexity] += 1

            # Programming languages (from original analysis)
            # This would need to be extracted from the original JSON data

        return tech_patterns

    def _analyze_complexity_patterns(self) -> Dict[str, Any]:
        """Analyze complexity patterns in successful artifacts."""
        complexity_patterns = {
            "setup_complexity_distribution": defaultdict(int),
            "size_vs_success": [],
            "complexity_scores": []
        }

        for artifact in self.artifacts_data:
            features = artifact.get("features", {})
            scores = artifact.get("evaluation_scores", {})

            # Setup complexity distribution
            complexity = features.get("setup_complexity", "unknown")
            complexity_patterns["setup_complexity_distribution"][complexity] += 1

            # Size vs success correlation
            repo_size = features.get("repo_size_mb", 0)
            overall_score = artifact.get("acceptance_prediction", {}).get("score", 0)
            complexity_patterns["size_vs_success"].append((repo_size, overall_score))

            # Complexity scores
            complexity_score = scores.get("complexity", 0)
            complexity_patterns["complexity_scores"].append(complexity_score)

        return complexity_patterns

    def _analyze_score_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different evaluation scores."""
        scores_data = []

        for artifact in self.artifacts_data:
            scores = artifact.get("evaluation_scores", {})
            prediction = artifact.get("acceptance_prediction", {})

            score_row = {
                "overall_score": prediction.get("score", 0),
                "doc_quality": scores.get("documentation_quality", 0),
                "reproducibility": scores.get("reproducibility", 0),
                "availability": scores.get("availability", 0),
                "code_structure": scores.get("code_structure", 0),
                "complexity": scores.get("complexity", 0)
            }
            scores_data.append(score_row)

        if not scores_data:
            return {}

        # Convert to DataFrame for correlation analysis
        df = pd.DataFrame(scores_data)

        correlations = {}
        criteria = ["doc_quality", "reproducibility", "availability", "code_structure", "complexity"]

        for criterion in criteria:
            if criterion in df.columns and "overall_score" in df.columns:
                corr = df[criterion].corr(df["overall_score"])
                correlations[criterion] = {
                    "correlation_with_overall": corr,
                    "importance_rank": None  # Will be filled after sorting
                }

        # Rank by correlation strength
        sorted_criteria = sorted(correlations.items(), key=lambda x: abs(x[1]["correlation_with_overall"]),
                                 reverse=True)
        for i, (criterion, data) in enumerate(sorted_criteria):
            correlations[criterion]["importance_rank"] = i + 1

        return {
            "correlations": correlations,
            "score_statistics": df.describe().to_dict(),
            "high_performers": df[df["overall_score"] > 0.8].to_dict("records"),
            "criteria_rankings": sorted_criteria
        }

    def _analyze_graph_patterns(self) -> Dict[str, Any]:
        """Analyze graph-level patterns using Neo4j queries."""
        try:
            # Get graph statistics
            graph_stats = self.kg_builder.get_graph_statistics()

            # Find most connected artifacts
            most_connected_query = """
            MATCH (a:Artifact)
            RETURN a.name as artifact_name, 
                   a.evaluation_score as score,
                   COUNT { (a)-[]->() } as outgoing_connections,
                   COUNT { (a)<-[]-() } as incoming_connections,
                   COUNT { (a)-[]-() } as total_connections
            ORDER BY total_connections DESC
            LIMIT 10
            """

            most_connected = self.graph.run(most_connected_query).data()

            # Find common relationship patterns
            relationship_patterns_query = """
            MATCH (a:Artifact)-[r]->(n)
            WHERE a.evaluation_score IS NOT NULL
            RETURN type(r) as relationship_type,
                   labels(n)[0] as target_node_type,
                   COUNT(*) as frequency,
                   AVG(a.evaluation_score) as avg_score_of_artifacts
            ORDER BY frequency DESC
            LIMIT 20
            """

            relationship_patterns = self.graph.run(relationship_patterns_query).data()

            # Find high-scoring artifact characteristics
            high_score_patterns_query = """
            MATCH (a:Artifact)-[r]->(n)
            WHERE a.evaluation_score > 0.7
            RETURN type(r) as relationship_type,
                   labels(n)[0] as target_node_type,
                   COUNT(*) as frequency_in_high_score,
                   AVG(a.evaluation_score) as avg_score
            ORDER BY frequency_in_high_score DESC
            LIMIT 15
            """

            high_score_patterns = self.graph.run(high_score_patterns_query).data()

            return {
                "graph_statistics": graph_stats,
                "most_connected_artifacts": most_connected,
                "common_relationship_patterns": relationship_patterns,
                "high_score_patterns": high_score_patterns
            }

        except Exception as e:
            logger.error(f"Error analyzing graph patterns: {e}")
            return {"error": str(e)}

    def _extract_success_indicators(self) -> Dict[str, Any]:
        """Extract key success indicators from pattern analysis."""
        indicators = {
            "critical_features": [],
            "success_thresholds": {},
            "warning_signals": [],
            "best_practices": []
        }

        # Extract from feature patterns
        feature_patterns = self.pattern_analysis.get("feature_patterns", {})
        for feature, stats in feature_patterns.items():
            if isinstance(stats, dict) and "is_success_indicator" in stats:
                if stats["is_success_indicator"]:
                    indicators["critical_features"].append({
                        "feature": feature,
                        "prevalence": stats["prevalence"],
                        "importance": "high" if stats["prevalence"] > 0.8 else "medium"
                    })

        # Extract thresholds from structure patterns
        structure_patterns = self.pattern_analysis.get("structure_patterns", {})
        for metric in ["file_counts", "tree_depths"]:
            stats_key = f"{metric}_stats"
            if stats_key in structure_patterns:
                stats = structure_patterns[stats_key]
                indicators["success_thresholds"][metric] = {
                    "recommended_min": stats.get("percentile_25", 0),
                    "optimal_range": [stats.get("percentile_25", 0), stats.get("percentile_75", 0)],
                    "mean": stats.get("mean", 0)
                }

        # Extract best practices from documentation patterns
        doc_patterns = self.pattern_analysis.get("documentation_patterns", {})
        if "length_stats" in doc_patterns:
            recommended_min = doc_patterns["length_stats"].get("recommended_min", 500)
            indicators["best_practices"].append({
                "category": "documentation",
                "practice": f"README should be at least {recommended_min:.0f} characters",
                "evidence": "Based on successful artifact analysis"
            })

        # Extract warning signals from complexity patterns
        complexity_patterns = self.pattern_analysis.get("complexity_patterns", {})
        complexity_dist = complexity_patterns.get("setup_complexity_distribution", {})
        high_complexity_ratio = complexity_dist.get("high", 0) / sum(complexity_dist.values()) if complexity_dist else 0

        if high_complexity_ratio < 0.2:
            indicators["warning_signals"].append({
                "signal": "high_setup_complexity",
                "description": "High setup complexity correlates with lower acceptance",
                "threshold": "Avoid if possible"
            })

        return indicators

    def _define_evaluation_criteria(self) -> Dict[str, Any]:
        """Define evaluation criteria based on discovered patterns."""
        criteria = {
            "pattern_based_weights": {},
            "dynamic_thresholds": {},
            "artifact_type_criteria": {},
            "conference_adjustments": {}
        }

        # Calculate pattern-based weights from correlations
        score_correlations = self.pattern_analysis.get("score_correlations", {})
        correlations = score_correlations.get("correlations", {})

        total_weight = 0
        for criterion, data in correlations.items():
            weight = abs(data.get("correlation_with_overall", 0))
            criteria["pattern_based_weights"][criterion] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for criterion in criteria["pattern_based_weights"]:
                criteria["pattern_based_weights"][criterion] /= total_weight

        # Set dynamic thresholds based on score distribution
        score_stats = score_correlations.get("score_statistics", {})
        if "overall_score" in score_stats:
            overall_stats = score_stats["overall_score"]
            criteria["dynamic_thresholds"] = {
                "high_acceptance": overall_stats.get("75%", 0.8),
                "medium_acceptance": overall_stats.get("50%", 0.6),
                "low_acceptance": overall_stats.get("25%", 0.4)
            }

        return criteria

    def predict_artifact_acceptance(self, artifact_json_path: str) -> Dict[str, Any]:
        """
        Predict acceptance for a new artifact using discovered patterns.
        
        Args:
            artifact_json_path: Path to new artifact's JSON analysis
            
        Returns:
            Detailed prediction with pattern-based reasoning
        """
        try:
            # Evaluate the new artifact
            result = self.evaluator.evaluate_artifact_from_json(artifact_json_path)

            if not result["success"]:
                return {"success": False, "error": result.get("error", "Evaluation failed")}

            features = result.get("features", {})
            scores = result.get("evaluation_scores", {})
            prediction = result.get("acceptance_prediction", {})

            # Pattern-based analysis
            pattern_analysis = self._analyze_against_patterns(features, scores)

            # Enhanced prediction using patterns
            enhanced_prediction = self._enhance_prediction_with_patterns(
                features, scores, prediction, pattern_analysis
            )

            return {
                "success": True,
                "artifact_name": result["artifact_name"],
                "standard_prediction": prediction,
                "pattern_based_prediction": enhanced_prediction,
                "pattern_analysis": pattern_analysis,
                "comparison_with_successful_artifacts": self._compare_with_successful_artifacts(features, scores),
                "recommendations": self._generate_pattern_based_recommendations(pattern_analysis)
            }

        except Exception as e:
            logger.error(f"Error predicting artifact acceptance: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_against_patterns(self, features: Dict, scores: Dict) -> Dict[str, Any]:
        """Analyze artifact against discovered patterns."""
        analysis = {
            "feature_alignment": {},
            "score_alignment": {},
            "pattern_matches": [],
            "pattern_violations": [],
            "similarity_score": 0
        }

        # Check feature alignment with success indicators
        success_indicators = self.success_indicators.get("critical_features", [])
        for indicator in success_indicators:
            feature_name = indicator["feature"]
            if feature_name in features:
                feature_value = features[feature_name]
                expected = indicator["prevalence"] > 0.7  # Expected to be True for high prevalence

                analysis["feature_alignment"][feature_name] = {
                    "actual": feature_value,
                    "expected": expected,
                    "matches_pattern": feature_value == expected,
                    "importance": indicator["importance"],
                    "prevalence_in_successful": indicator["prevalence"]
                }

                if feature_value == expected:
                    analysis["pattern_matches"].append(feature_name)
                else:
                    analysis["pattern_violations"].append(feature_name)

        # Check score alignment
        score_correlations = self.pattern_analysis.get("score_correlations", {})
        correlations = score_correlations.get("correlations", {})

        for criterion, score in scores.items():
            if criterion in correlations:
                correlation_data = correlations[criterion]
                analysis["score_alignment"][criterion] = {
                    "score": score,
                    "correlation_with_success": correlation_data.get("correlation_with_overall", 0),
                    "importance_rank": correlation_data.get("importance_rank", 0)
                }

        # Calculate similarity score
        total_matches = len(analysis["pattern_matches"])
        total_features = len(success_indicators)
        analysis["similarity_score"] = total_matches / total_features if total_features > 0 else 0

        return analysis

    def _enhance_prediction_with_patterns(self, features: Dict, scores: Dict, original_prediction: Dict,
                                          pattern_analysis: Dict) -> Dict[str, Any]:
        """Enhance prediction using pattern-based insights."""
        # Start with original prediction
        enhanced_score = original_prediction.get("score", 0)

        # Adjust based on pattern alignment
        similarity_score = pattern_analysis.get("similarity_score", 0)
        pattern_weight = 0.3  # Weight for pattern-based adjustment

        # Boost score if highly similar to successful patterns
        if similarity_score > 0.8:
            enhanced_score += pattern_weight * 0.2
        elif similarity_score > 0.6:
            enhanced_score += pattern_weight * 0.1
        elif similarity_score < 0.3:
            enhanced_score -= pattern_weight * 0.1

        # Adjust based on critical feature violations
        pattern_violations = pattern_analysis.get("pattern_violations", [])
        for violation in pattern_violations:
            # Check if it's a high-importance feature
            feature_alignment = pattern_analysis.get("feature_alignment", {})
            if violation in feature_alignment:
                importance = feature_alignment[violation].get("importance", "medium")
                if importance == "high":
                    enhanced_score -= 0.1
                elif importance == "medium":
                    enhanced_score -= 0.05

        # Ensure score stays within bounds
        enhanced_score = max(0, min(1, enhanced_score))

        # Determine likelihood based on enhanced score
        if enhanced_score >= 0.8:
            likelihood = "high"
            confidence = 0.9
        elif enhanced_score >= 0.6:
            likelihood = "medium"
            confidence = 0.75
        else:
            likelihood = "low"
            confidence = 0.6

        return {
            "score": enhanced_score,
            "likelihood": likelihood,
            "confidence": confidence,
            "pattern_adjustment": enhanced_score - original_prediction.get("score", 0),
            "reasoning": self._generate_prediction_reasoning(pattern_analysis, enhanced_score)
        }

    def _compare_with_successful_artifacts(self, features: Dict, scores: Dict) -> Dict[str, Any]:
        """Compare artifact with successful artifacts in the dataset."""
        # Find top-performing artifacts
        top_artifacts = []
        for artifact in self.artifacts_data:
            artifact_score = artifact.get("acceptance_prediction", {}).get("score", 0)
            if artifact_score > 0.7:  # High-performing threshold
                top_artifacts.append({
                    "name": artifact["artifact_name"],
                    "score": artifact_score,
                    "features": artifact.get("features", {}),
                    "scores": artifact.get("evaluation_scores", {})
                })

        # Sort by score
        top_artifacts.sort(key=lambda x: x["score"], reverse=True)

        # Compare features with top 5 artifacts
        comparison = {
            "most_similar_artifacts": [],
            "feature_gaps": [],
            "score_gaps": {},
            "improvement_potential": 0
        }

        for top_artifact in top_artifacts[:5]:
            similarity = self._calculate_feature_similarity(features, top_artifact["features"])
            comparison["most_similar_artifacts"].append({
                "name": top_artifact["name"],
                "score": top_artifact["score"],
                "similarity": similarity,
                "key_differences": self._identify_key_differences(features, top_artifact["features"])
            })

        # Identify feature gaps
        for top_artifact in top_artifacts[:3]:
            top_features = top_artifact["features"]
            for feature, value in top_features.items():
                if isinstance(value, bool) and value and not features.get(feature, False):
                    comparison["feature_gaps"].append({
                        "feature": feature,
                        "present_in_successful": True,
                        "present_in_current": False,
                        "example_artifact": top_artifact["name"]
                    })

        return comparison

    def _calculate_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two feature sets."""
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0

        matches = 0
        for feature in common_features:
            if features1[feature] == features2[feature]:
                matches += 1

        return matches / len(common_features)

    def _identify_key_differences(self, features1: Dict, features2: Dict) -> List[str]:
        """Identify key differences between feature sets."""
        differences = []

        all_features = set(features1.keys()) | set(features2.keys())
        for feature in all_features:
            val1 = features1.get(feature)
            val2 = features2.get(feature)

            if val1 != val2:
                differences.append(f"{feature}: {val1} vs {val2}")

        return differences[:5]  # Return top 5 differences

    def _generate_pattern_based_recommendations(self, pattern_analysis: Dict) -> List[Dict[str, str]]:
        """Generate recommendations based on pattern analysis."""
        recommendations = []

        # Recommendations for pattern violations
        pattern_violations = pattern_analysis.get("pattern_violations", [])
        feature_alignment = pattern_analysis.get("feature_alignment", {})

        for violation in pattern_violations:
            if violation in feature_alignment:
                data = feature_alignment[violation]
                prevalence = data.get("prevalence_in_successful", 0)

                recommendations.append({
                    "category": "pattern_alignment",
                    "priority": "high" if data.get("importance") == "high" else "medium",
                    "recommendation": f"Consider implementing '{violation}' - present in {prevalence:.1%} of successful artifacts",
                    "evidence": f"Pattern analysis of {len(self.artifacts_data)} accepted artifacts"
                })

        # Recommendations based on similarity score
        similarity_score = pattern_analysis.get("similarity_score", 0)
        if similarity_score < 0.5:
            recommendations.append({
                "category": "overall_pattern",
                "priority": "high",
                "recommendation": f"Low pattern similarity ({similarity_score:.1%}) with successful artifacts - review critical features",
                "evidence": "Pattern analysis comparison"
            })

        return recommendations

    def _generate_prediction_reasoning(self, pattern_analysis: Dict, enhanced_score: float) -> List[str]:
        """Generate human-readable reasoning for the prediction."""
        reasoning = []

        similarity_score = pattern_analysis.get("similarity_score", 0)
        pattern_matches = pattern_analysis.get("pattern_matches", [])
        pattern_violations = pattern_analysis.get("pattern_violations", [])

        # Similarity reasoning
        reasoning.append(f"Pattern similarity with successful artifacts: {similarity_score:.1%}")

        # Positive factors
        if pattern_matches:
            reasoning.append(f"Matches {len(pattern_matches)} success patterns: {', '.join(pattern_matches[:3])}")

        # Negative factors
        if pattern_violations:
            reasoning.append(f"Violates {len(pattern_violations)} common patterns: {', '.join(pattern_violations[:3])}")

        # Score interpretation
        if enhanced_score > 0.8:
            reasoning.append("Strong alignment with acceptance patterns")
        elif enhanced_score > 0.6:
            reasoning.append("Moderate alignment with acceptance patterns")
        else:
            reasoning.append("Weak alignment with acceptance patterns")

        return reasoning

    def export_pattern_analysis_report(self, output_path: str) -> str:
        """Export comprehensive pattern analysis report."""
        try:
            report_data = {
                "analysis_metadata": {
                    "artifacts_analyzed": len(self.artifacts_data),
                    "analysis_date": datetime.now().isoformat(),
                    "total_patterns_identified": len(self.pattern_analysis)
                },
                "pattern_analysis": self.pattern_analysis,
                "success_indicators": self.success_indicators,
                "evaluation_criteria": self.evaluation_criteria,
                "artifacts_summary": [
                    {
                        "name": artifact["artifact_name"],
                        "score": artifact.get("acceptance_prediction", {}).get("score", 0),
                        "key_features": {
                            "has_readme": artifact.get("features", {}).get("has_readme", False),
                            "has_docker": artifact.get("features", {}).get("has_docker", False),
                            "has_zenodo_doi": artifact.get("features", {}).get("has_zenodo_doi", False),
                            "setup_complexity": artifact.get("features", {}).get("setup_complexity", "unknown")
                        }
                    }
                    for artifact in self.artifacts_data
                ]
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)

            logger.info(f"Pattern analysis report exported to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting pattern analysis report: {e}")
            return f"Error: {str(e)}"

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get a summary of discovered patterns."""
        if not self.pattern_analysis:
            return {"error": "No pattern analysis available. Run build_unified_knowledge_graph first."}

        summary = {
            "artifacts_analyzed": len(self.artifacts_data),
            "key_success_indicators": [],
            "critical_patterns": [],
            "evaluation_insights": [],
            "recommendations_for_new_artifacts": []
        }

        # Extract key success indicators
        critical_features = self.success_indicators.get("critical_features", [])
        for feature in critical_features[:5]:  # Top 5
            summary["key_success_indicators"].append({
                "feature": feature["feature"],
                "prevalence": f"{feature['prevalence']:.1%}",
                "importance": feature["importance"]
            })

        # Extract critical patterns
        score_correlations = self.pattern_analysis.get("score_correlations", {})
        correlations = score_correlations.get("correlations", {})

        for criterion, data in correlations.items():
            summary["critical_patterns"].append({
                "criterion": criterion,
                "correlation_strength": f"{data.get('correlation_with_overall', 0):.3f}",
                "importance_rank": data.get("importance_rank", 0)
            })

        # Sort by importance rank
        summary["critical_patterns"].sort(key=lambda x: x["importance_rank"])

        return summary

    def close(self):
        """Close connections and cleanup."""
        if self.evaluator:
            self.evaluator.close()


def main():
    """Example usage of the Pattern Analysis System."""
    import argparse

    parser = argparse.ArgumentParser(description="Pattern Analysis System for Accepted Artifacts")
    parser.add_argument("--artifacts-dir", default="../../algo_outputs/algorithm_2_output",
                        help="Directory containing artifact JSON files")
    parser.add_argument("--output-dir", default="pattern_analysis_results",
                        help="Output directory for results")
    parser.add_argument("--neo4j-password", default="12345678", help="Neo4j password")
    parser.add_argument("--openai-api-key", help="OpenAI API key")
    parser.add_argument("--new-artifact", help="Path to new artifact JSON for prediction")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize pattern analysis system
    pattern_analyzer = PatternAnalysisSystem(
        neo4j_password=args.neo4j_password,
        openai_api_key=args.openai_api_key
    )

    try:
        print("🔍 Building Unified Knowledge Graph from Accepted Artifacts...")
        print("=" * 60)

        # Build unified knowledge graph
        build_results = pattern_analyzer.build_unified_knowledge_graph(args.artifacts_dir)

        print(f"✅ Processed {build_results['successful_builds']}/{build_results['total_files']} artifacts")
        print(f"📊 Total nodes created: {build_results['total_nodes']}")
        print(f"❌ Failed builds: {build_results['failed_builds']}")

        if build_results['successful_builds'] > 0:
            # Export pattern analysis report
            report_path = output_dir / "pattern_analysis_report.json"
            pattern_analyzer.export_pattern_analysis_report(str(report_path))

            # Get pattern summary
            summary = pattern_analyzer.get_pattern_summary()

            print("\n🎯 Key Success Indicators Found:")
            print("-" * 40)
            for indicator in summary.get("key_success_indicators", []):
                print(
                    f"  • {indicator['feature']}: {indicator['prevalence']} prevalence ({indicator['importance']} importance)")

            print("\n📈 Critical Evaluation Patterns:")
            print("-" * 40)
            for pattern in summary.get("critical_patterns", [])[:5]:
                print(
                    f"  • {pattern['criterion']}: correlation {pattern['correlation_strength']} (rank #{pattern['importance_rank']})")

            # Test prediction on new artifact if provided
            if args.new_artifact:
                print(f"\n🔮 Testing Prediction on New Artifact...")
                print("-" * 40)

                prediction_result = pattern_analyzer.predict_artifact_acceptance(args.new_artifact)

                if prediction_result["success"]:
                    artifact_name = prediction_result["artifact_name"]
                    standard_pred = prediction_result["standard_prediction"]
                    pattern_pred = prediction_result["pattern_based_prediction"]

                    print(f"Artifact: {artifact_name}")
                    print(f"Standard Prediction: {standard_pred['likelihood'].upper()} ({standard_pred['score']:.3f})")
                    print(
                        f"Pattern-Based Prediction: {pattern_pred['likelihood'].upper()} ({pattern_pred['score']:.3f})")
                    print(f"Pattern Adjustment: {pattern_pred['pattern_adjustment']:+.3f}")

                    print("\nPattern-Based Reasoning:")
                    for reason in pattern_pred["reasoning"]:
                        print(f"  • {reason}")
                else:
                    print(f"❌ Prediction failed: {prediction_result.get('error')}")

        print(f"\n📁 Results exported to: {output_dir}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    finally:
        pattern_analyzer.close()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
