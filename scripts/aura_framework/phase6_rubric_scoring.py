"""
🎯 Phase 6: Rubric-Weighted Scoring Framework
Goal: Convert multi-agent evaluations into conference-specific weighted scores.

Features:
- Conference-specific rubric weights and criteria
- Multi-dimensional score aggregation with confidence weighting
- Threshold-based categorization (Excellent, Good, Needs Improvement, Poor)
- Acceptance probability prediction based on historical data
- Comparative scoring across multiple conferences
- Explainable scoring with dimension breakdown
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
from collections import defaultdict

# Local imports
from config import config
from phase5_genai_agents import MultiAgentResult, EvaluationDimension, EvaluationResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AcceptanceCategory(Enum):
    """Artifact acceptance categories based on scoring"""
    EXCELLENT = "excellent"           # 85-100%: Very likely to be accepted
    GOOD = "good"                    # 70-84%: Likely to be accepted
    NEEDS_IMPROVEMENT = "needs_improvement"  # 50-69%: May be accepted with improvements
    POOR = "poor"                    # 0-49%: Unlikely to be accepted


@dataclass
class ConferenceRubric:
    """Conference-specific evaluation rubric"""
    conference_name: str
    year: int
    dimension_weights: Dict[EvaluationDimension, float]
    quality_thresholds: Dict[AcceptanceCategory, float]
    specific_criteria: Dict[str, Any]
    historical_acceptance_rate: Optional[float] = None
    emphasis_areas: List[str] = field(default_factory=list)
    penalty_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScoringResult:
    """Result from rubric-based scoring"""
    artifact_id: str
    conference: str
    raw_scores: Dict[EvaluationDimension, float]
    weighted_scores: Dict[EvaluationDimension, float]
    final_score: float
    confidence_score: float
    acceptance_category: AcceptanceCategory
    acceptance_probability: float
    dimension_breakdown: Dict[str, Any]
    improvement_priorities: List[str]
    comparative_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MultiConferenceComparison:
    """Comparison across multiple conferences"""
    artifact_id: str
    conference_scores: Dict[str, ScoringResult]
    best_fit_conference: str
    best_score: float
    score_variance: float
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RubricScoringFramework:
    """Advanced rubric-based scoring framework"""
    
    def __init__(self, 
                 rubrics_directory: str = "data/conference_rubrics",
                 default_weights: Optional[Dict[EvaluationDimension, float]] = None):
        """
        Initialize the Rubric Scoring Framework
        
        Args:
            rubrics_directory: Directory containing conference rubric files
            default_weights: Default dimension weights if conference-specific not available
        """
        self.rubrics_directory = Path(rubrics_directory)
        self.rubrics_directory.mkdir(parents=True, exist_ok=True)
        
        # Default weights if no conference-specific rubric available
        self.default_weights = default_weights or {
            EvaluationDimension.REPRODUCIBILITY: 0.20,
            EvaluationDimension.DOCUMENTATION: 0.25,
            EvaluationDimension.ACCESSIBILITY: 0.15,
            EvaluationDimension.USABILITY: 0.15,
            EvaluationDimension.EXPERIMENTAL: 0.15,
            EvaluationDimension.FUNCTIONALITY: 0.10
        }
        
        # Load conference rubrics
        self.conference_rubrics: Dict[str, ConferenceRubric] = {}
        self._load_conference_rubrics()
        
        # Historical scoring data for prediction
        self.historical_scores: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"Rubric Scoring Framework initialized with {len(self.conference_rubrics)} conference rubrics")

    def _load_conference_rubrics(self):
        """Load conference-specific rubrics from files"""
        # Create default rubrics for major conferences
        self._create_default_rubrics()
        
        # Load custom rubrics from JSON files
        for rubric_file in self.rubrics_directory.glob("*.json"):
            try:
                with open(rubric_file, 'r', encoding='utf-8') as f:
                    rubric_data = json.load(f)
                    rubric = self._parse_rubric_data(rubric_data)
                    self.conference_rubrics[rubric.conference_name] = rubric
                    logger.info(f"Loaded rubric for {rubric.conference_name}")
            except Exception as e:
                logger.error(f"Failed to load rubric from {rubric_file}: {e}")

    def _create_default_rubrics(self):
        """Create default rubrics for major software engineering conferences"""
        
        # ICSE - International Conference on Software Engineering
        icse_rubric = ConferenceRubric(
            conference_name="ICSE",
            year=2025,
            dimension_weights={
                EvaluationDimension.REPRODUCIBILITY: 0.30,  # High emphasis on reproducibility
                EvaluationDimension.DOCUMENTATION: 0.25,
                EvaluationDimension.EXPERIMENTAL: 0.20,
                EvaluationDimension.FUNCTIONALITY: 0.15,
                EvaluationDimension.USABILITY: 0.07,
                EvaluationDimension.ACCESSIBILITY: 0.03
            },
            quality_thresholds={
                AcceptanceCategory.EXCELLENT: 0.85,
                AcceptanceCategory.GOOD: 0.70,
                AcceptanceCategory.NEEDS_IMPROVEMENT: 0.50,
                AcceptanceCategory.POOR: 0.0
            },
            specific_criteria={
                "requires_replication_package": True,
                "needs_statistical_evaluation": True,
                "minimum_documentation_sections": 8
            },
            historical_acceptance_rate=0.22,
            emphasis_areas=["empirical studies", "replication", "open science"],
            penalty_factors={"missing_replication": 0.2, "poor_documentation": 0.15}
        )
        
        # FSE - Foundations of Software Engineering
        fse_rubric = ConferenceRubric(
            conference_name="FSE",
            year=2025,
            dimension_weights={
                EvaluationDimension.FUNCTIONALITY: 0.30,  # High emphasis on functionality
                EvaluationDimension.EXPERIMENTAL: 0.25,
                EvaluationDimension.DOCUMENTATION: 0.20,
                EvaluationDimension.REPRODUCIBILITY: 0.15,
                EvaluationDimension.USABILITY: 0.07,
                EvaluationDimension.ACCESSIBILITY: 0.03
            },
            quality_thresholds={
                AcceptanceCategory.EXCELLENT: 0.82,
                AcceptanceCategory.GOOD: 0.68,
                AcceptanceCategory.NEEDS_IMPROVEMENT: 0.48,
                AcceptanceCategory.POOR: 0.0
            },
            specific_criteria={
                "requires_tool_demo": True,
                "needs_performance_evaluation": True,
                "minimum_documentation_sections": 6
            },
            historical_acceptance_rate=0.24,
            emphasis_areas=["tools", "empirical studies", "innovation"],
            penalty_factors={"missing_tool_demo": 0.25, "poor_performance": 0.15}
        )
        
        # ASE - Automated Software Engineering
        ase_rubric = ConferenceRubric(
            conference_name="ASE",
            year=2024,
            dimension_weights={
                EvaluationDimension.FUNCTIONALITY: 0.35,  # Very high emphasis on functionality
                EvaluationDimension.EXPERIMENTAL: 0.25,
                EvaluationDimension.REPRODUCIBILITY: 0.20,
                EvaluationDimension.DOCUMENTATION: 0.15,
                EvaluationDimension.USABILITY: 0.03,
                EvaluationDimension.ACCESSIBILITY: 0.02
            },
            quality_thresholds={
                AcceptanceCategory.EXCELLENT: 0.80,
                AcceptanceCategory.GOOD: 0.65,
                AcceptanceCategory.NEEDS_IMPROVEMENT: 0.45,
                AcceptanceCategory.POOR: 0.0
            },
            specific_criteria={
                "requires_automation_tool": True,
                "needs_scalability_test": True,
                "minimum_documentation_sections": 5
            },
            historical_acceptance_rate=0.19,
            emphasis_areas=["automation", "tools", "scalability"],
            penalty_factors={"missing_automation": 0.3, "poor_scalability": 0.2}
        )
        
        # CHI - Computer-Human Interaction
        chi_rubric = ConferenceRubric(
            conference_name="CHI",
            year=2024,
            dimension_weights={
                EvaluationDimension.USABILITY: 0.35,  # Very high emphasis on usability
                EvaluationDimension.ACCESSIBILITY: 0.25,  # High emphasis on accessibility
                EvaluationDimension.DOCUMENTATION: 0.20,
                EvaluationDimension.EXPERIMENTAL: 0.15,
                EvaluationDimension.REPRODUCIBILITY: 0.03,
                EvaluationDimension.FUNCTIONALITY: 0.02
            },
            quality_thresholds={
                AcceptanceCategory.EXCELLENT: 0.78,
                AcceptanceCategory.GOOD: 0.62,
                AcceptanceCategory.NEEDS_IMPROVEMENT: 0.42,
                AcceptanceCategory.POOR: 0.0
            },
            specific_criteria={
                "requires_user_study": True,
                "needs_accessibility_evaluation": True,
                "minimum_documentation_sections": 6
            },
            historical_acceptance_rate=0.25,
            emphasis_areas=["user experience", "accessibility", "human factors"],
            penalty_factors={"missing_user_study": 0.4, "poor_accessibility": 0.3}
        )
        
        # Store default rubrics
        self.conference_rubrics.update({
            "ICSE": icse_rubric,
            "FSE": fse_rubric, 
            "ASE": ase_rubric,
            "CHI": chi_rubric
        })

    def _parse_rubric_data(self, rubric_data: Dict[str, Any]) -> ConferenceRubric:
        """Parse rubric data from JSON"""
        # Convert dimension weights
        dimension_weights = {}
        for dim_name, weight in rubric_data.get('dimension_weights', {}).items():
            try:
                dimension = EvaluationDimension(dim_name.lower())
                dimension_weights[dimension] = float(weight)
            except ValueError:
                logger.warning(f"Unknown dimension: {dim_name}")
        
        # Convert quality thresholds
        quality_thresholds = {}
        for category_name, threshold in rubric_data.get('quality_thresholds', {}).items():
            try:
                category = AcceptanceCategory(category_name.lower())
                quality_thresholds[category] = float(threshold)
            except ValueError:
                logger.warning(f"Unknown category: {category_name}")
        
        return ConferenceRubric(
            conference_name=rubric_data.get('conference_name', 'Unknown'),
            year=int(rubric_data.get('year', 2024)),
            dimension_weights=dimension_weights,
            quality_thresholds=quality_thresholds,
            specific_criteria=rubric_data.get('specific_criteria', {}),
            historical_acceptance_rate=rubric_data.get('historical_acceptance_rate'),
            emphasis_areas=rubric_data.get('emphasis_areas', []),
            penalty_factors=rubric_data.get('penalty_factors', {})
        )

    def score_artifact(self, 
                      multi_agent_result: MultiAgentResult, 
                      target_conference: str,
                      apply_penalties: bool = True) -> ScoringResult:
        """
        Score an artifact using conference-specific rubric
        
        Args:
            multi_agent_result: Results from multi-agent evaluation
            target_conference: Target conference for scoring
            apply_penalties: Whether to apply conference-specific penalties
            
        Returns:
            Comprehensive scoring result
        """
        logger.info(f"Scoring artifact {multi_agent_result.artifact_id} for {target_conference}")
        
        # Get conference rubric
        rubric = self.conference_rubrics.get(target_conference)
        if not rubric:
            logger.warning(f"No rubric found for {target_conference}, using default weights")
            rubric = self._create_default_rubric(target_conference)
        
        # Extract raw scores from multi-agent results
        raw_scores = {}
        confidence_weights = {}
        
        for dimension, eval_result in multi_agent_result.individual_results.items():
            raw_scores[dimension] = eval_result.score
            confidence_weights[dimension] = eval_result.confidence
        
        # Apply conference-specific weights
        weighted_scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in rubric.dimension_weights.items():
            if dimension in raw_scores:
                # Apply confidence weighting
                confidence = confidence_weights.get(dimension, 1.0)
                adjusted_weight = weight * confidence
                
                weighted_score = raw_scores[dimension] * adjusted_weight
                weighted_scores[dimension] = weighted_score
                
                total_weighted_score += weighted_score
                total_weight += adjusted_weight
            else:
                logger.warning(f"Missing evaluation for dimension {dimension}")
        
        # Normalize final score
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Apply penalties if enabled
        if apply_penalties:
            final_score = self._apply_penalties(final_score, multi_agent_result, rubric)
        
        # Calculate overall confidence
        confidence_score = np.mean(list(confidence_weights.values()))
        
        # Determine acceptance category
        acceptance_category = self._determine_acceptance_category(final_score, rubric)
        
        # Calculate acceptance probability
        acceptance_probability = self._calculate_acceptance_probability(
            final_score, rubric, confidence_score
        )
        
        # Generate dimension breakdown
        dimension_breakdown = self._generate_dimension_breakdown(
            raw_scores, weighted_scores, rubric
        )
        
        # Identify improvement priorities
        improvement_priorities = self._identify_improvement_priorities(
            raw_scores, rubric, multi_agent_result
        )
        
        return ScoringResult(
            artifact_id=multi_agent_result.artifact_id,
            conference=target_conference,
            raw_scores=raw_scores,
            weighted_scores=weighted_scores,
            final_score=final_score,
            confidence_score=confidence_score,
            acceptance_category=acceptance_category,
            acceptance_probability=acceptance_probability,
            dimension_breakdown=dimension_breakdown,
            improvement_priorities=improvement_priorities
        )

    def compare_across_conferences(self, 
                                 multi_agent_result: MultiAgentResult,
                                 conferences: List[str]) -> MultiConferenceComparison:
        """
        Compare artifact scores across multiple conferences
        
        Args:
            multi_agent_result: Results from multi-agent evaluation
            conferences: List of conferences to compare
            
        Returns:
            Multi-conference comparison results
        """
        logger.info(f"Comparing artifact {multi_agent_result.artifact_id} across {len(conferences)} conferences")
        
        conference_scores = {}
        scores_list = []
        
        # Score for each conference
        for conference in conferences:
            try:
                scoring_result = self.score_artifact(multi_agent_result, conference)
                conference_scores[conference] = scoring_result
                scores_list.append(scoring_result.final_score)
            except Exception as e:
                logger.error(f"Failed to score for {conference}: {e}")
        
        if not scores_list:
            raise ValueError("No successful scores generated")
        
        # Find best fit conference
        best_conference = max(conference_scores.keys(), 
                            key=lambda c: conference_scores[c].final_score)
        best_score = conference_scores[best_conference].final_score
        
        # Calculate score variance
        score_variance = np.var(scores_list) if len(scores_list) > 1 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_conference_recommendations(
            conference_scores, best_conference, score_variance
        )
        
        return MultiConferenceComparison(
            artifact_id=multi_agent_result.artifact_id,
            conference_scores=conference_scores,
            best_fit_conference=best_conference,
            best_score=best_score,
            score_variance=score_variance,
            recommendations=recommendations
        )

    def _create_default_rubric(self, conference_name: str) -> ConferenceRubric:
        """Create a default rubric for unknown conferences"""
        return ConferenceRubric(
            conference_name=conference_name,
            year=2024,
            dimension_weights=self.default_weights,
            quality_thresholds={
                AcceptanceCategory.EXCELLENT: 0.80,
                AcceptanceCategory.GOOD: 0.65,
                AcceptanceCategory.NEEDS_IMPROVEMENT: 0.45,
                AcceptanceCategory.POOR: 0.0
            },
            specific_criteria={},
            historical_acceptance_rate=0.20,
            emphasis_areas=[],
            penalty_factors={}
        )

    def _apply_penalties(self, 
                        score: float, 
                        multi_agent_result: MultiAgentResult, 
                        rubric: ConferenceRubric) -> float:
        """Apply conference-specific penalties"""
        penalized_score = score
        
        for penalty_type, penalty_factor in rubric.penalty_factors.items():
            if self._check_penalty_condition(penalty_type, multi_agent_result, rubric):
                penalized_score *= (1.0 - penalty_factor)
                logger.info(f"Applied {penalty_type} penalty: -{penalty_factor:.1%}")
        
        return max(0.0, penalized_score)  # Ensure non-negative

    def _check_penalty_condition(self, 
                               penalty_type: str, 
                               multi_agent_result: MultiAgentResult, 
                               rubric: ConferenceRubric) -> bool:
        """Check if a penalty condition is met"""
        # Check for common penalty conditions
        if penalty_type == "missing_replication":
            repro_score = multi_agent_result.individual_results.get(
                EvaluationDimension.REPRODUCIBILITY
            )
            return repro_score and repro_score.score < 0.3
        
        elif penalty_type == "poor_documentation":
            doc_score = multi_agent_result.individual_results.get(
                EvaluationDimension.DOCUMENTATION
            )
            return doc_score and doc_score.score < 0.4
        
        elif penalty_type == "missing_tool_demo":
            func_score = multi_agent_result.individual_results.get(
                EvaluationDimension.FUNCTIONALITY
            )
            return func_score and func_score.score < 0.3
        
        elif penalty_type == "poor_accessibility":
            access_score = multi_agent_result.individual_results.get(
                EvaluationDimension.ACCESSIBILITY
            )
            return access_score and access_score.score < 0.2
        
        # Default: no penalty
        return False

    def _determine_acceptance_category(self, 
                                     score: float, 
                                     rubric: ConferenceRubric) -> AcceptanceCategory:
        """Determine acceptance category based on score and thresholds"""
        thresholds = rubric.quality_thresholds
        
        if score >= thresholds.get(AcceptanceCategory.EXCELLENT, 0.85):
            return AcceptanceCategory.EXCELLENT
        elif score >= thresholds.get(AcceptanceCategory.GOOD, 0.70):
            return AcceptanceCategory.GOOD
        elif score >= thresholds.get(AcceptanceCategory.NEEDS_IMPROVEMENT, 0.50):
            return AcceptanceCategory.NEEDS_IMPROVEMENT
        else:
            return AcceptanceCategory.POOR

    def _calculate_acceptance_probability(self, 
                                        score: float, 
                                        rubric: ConferenceRubric, 
                                        confidence: float) -> float:
        """Calculate probability of acceptance based on score and historical data"""
        # Base probability from score
        base_probability = min(1.0, score / 0.85)  # Normalize to excellent threshold
        
        # Adjust for conference acceptance rate
        if rubric.historical_acceptance_rate:
            rate_adjustment = rubric.historical_acceptance_rate / 0.20  # Normalize to 20% baseline
            base_probability *= rate_adjustment
        
        # Adjust for confidence
        confidence_adjustment = 0.5 + (confidence * 0.5)
        adjusted_probability = base_probability * confidence_adjustment
        
        return min(1.0, max(0.0, adjusted_probability))

    def _generate_dimension_breakdown(self, 
                                    raw_scores: Dict[EvaluationDimension, float],
                                    weighted_scores: Dict[EvaluationDimension, float],
                                    rubric: ConferenceRubric) -> Dict[str, Any]:
        """Generate detailed dimension breakdown"""
        breakdown = {
            "dimensions": {},
            "strongest_dimensions": [],
            "weakest_dimensions": [],
            "weight_distribution": {}
        }
        
        # Dimension details
        for dimension in EvaluationDimension:
            if dimension in raw_scores:
                breakdown["dimensions"][dimension.value] = {
                    "raw_score": raw_scores[dimension],
                    "weighted_score": weighted_scores.get(dimension, 0.0),
                    "weight": rubric.dimension_weights.get(dimension, 0.0),
                    "contribution": weighted_scores.get(dimension, 0.0) / sum(weighted_scores.values()) if weighted_scores else 0.0
                }
        
        # Identify strongest and weakest dimensions
        sorted_scores = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
        breakdown["strongest_dimensions"] = [dim.value for dim, _ in sorted_scores[:2]]
        breakdown["weakest_dimensions"] = [dim.value for dim, _ in sorted_scores[-2:]]
        
        # Weight distribution
        breakdown["weight_distribution"] = {
            dim.value: weight for dim, weight in rubric.dimension_weights.items()
        }
        
        return breakdown

    def _identify_improvement_priorities(self, 
                                       raw_scores: Dict[EvaluationDimension, float],
                                       rubric: ConferenceRubric,
                                       multi_agent_result: MultiAgentResult) -> List[str]:
        """Identify improvement priorities based on scores and conference emphasis"""
        priorities = []
        
        # Find dimensions with low scores that have high weights
        for dimension, weight in rubric.dimension_weights.items():
            if dimension in raw_scores:
                score = raw_scores[dimension]
                impact = weight * (1.0 - score)  # Higher impact = high weight + low score
                
                if impact > 0.15:  # Significant improvement opportunity
                    eval_result = multi_agent_result.individual_results.get(dimension)
                    if eval_result and eval_result.suggestions:
                        priorities.extend(eval_result.suggestions[:2])  # Top 2 suggestions
        
        # Add conference-specific priorities
        for emphasis_area in rubric.emphasis_areas:
            if emphasis_area.lower() in ["reproducibility", "replication"]:
                if raw_scores.get(EvaluationDimension.REPRODUCIBILITY, 1.0) < 0.7:
                    priorities.append(f"Improve {emphasis_area} to meet conference standards")
            elif emphasis_area.lower() in ["documentation", "clarity"]:
                if raw_scores.get(EvaluationDimension.DOCUMENTATION, 1.0) < 0.7:
                    priorities.append(f"Enhance {emphasis_area} for better evaluation")
        
        return priorities[:5]  # Return top 5 priorities

    def _generate_conference_recommendations(self, 
                                          conference_scores: Dict[str, ScoringResult],
                                          best_conference: str,
                                          score_variance: float) -> List[str]:
        """Generate recommendations based on multi-conference comparison"""
        recommendations = []
        
        # Best fit recommendation
        best_score = conference_scores[best_conference].final_score
        best_category = conference_scores[best_conference].acceptance_category
        
        recommendations.append(
            f"Best fit: {best_conference} (score: {best_score:.1%}, {best_category.value})"
        )
        
        # Score variance analysis
        if score_variance < 0.01:
            recommendations.append("Consistent performance across conferences - good general quality")
        elif score_variance > 0.05:
            recommendations.append("Variable performance - consider conference-specific improvements")
        
        # Alternative recommendations
        sorted_conferences = sorted(conference_scores.items(), 
                                  key=lambda x: x[1].final_score, 
                                  reverse=True)
        
        if len(sorted_conferences) > 1:
            second_best = sorted_conferences[1]
            if second_best[1].final_score > 0.65:
                recommendations.append(
                    f"Alternative venue: {second_best[0]} (score: {second_best[1].final_score:.1%})"
                )
        
        # Improvement recommendations
        for conf_name, result in sorted_conferences:
            if result.final_score < 0.5 and result.improvement_priorities:
                recommendations.append(
                    f"For {conf_name}: Focus on {result.improvement_priorities[0]}"
                )
        
        return recommendations

    def export_rubric(self, conference_name: str, output_path: str):
        """Export conference rubric to JSON file"""
        if conference_name not in self.conference_rubrics:
            raise ValueError(f"No rubric found for {conference_name}")
        
        rubric = self.conference_rubrics[conference_name]
        
        # Convert to serializable format
        export_data = {
            "conference_name": rubric.conference_name,
            "year": rubric.year,
            "dimension_weights": {dim.value: weight for dim, weight in rubric.dimension_weights.items()},
            "quality_thresholds": {cat.value: thresh for cat, thresh in rubric.quality_thresholds.items()},
            "specific_criteria": rubric.specific_criteria,
            "historical_acceptance_rate": rubric.historical_acceptance_rate,
            "emphasis_areas": rubric.emphasis_areas,
            "penalty_factors": rubric.penalty_factors
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {conference_name} rubric to {output_path}")

    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scoring statistics"""
        stats = {
            "total_rubrics": len(self.conference_rubrics),
            "conferences": list(self.conference_rubrics.keys()),
            "dimension_weight_analysis": {},
            "threshold_analysis": {},
            "historical_data": dict(self.historical_scores)
        }
        
        # Analyze dimension weights across conferences
        all_weights = defaultdict(list)
        for rubric in self.conference_rubrics.values():
            for dim, weight in rubric.dimension_weights.items():
                all_weights[dim.value].append(weight)
        
        for dim, weights in all_weights.items():
            stats["dimension_weight_analysis"][dim] = {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "min": min(weights),
                "max": max(weights)
            }
        
        # Analyze quality thresholds
        all_thresholds = defaultdict(list)
        for rubric in self.conference_rubrics.values():
            for cat, thresh in rubric.quality_thresholds.items():
                all_thresholds[cat.value].append(thresh)
        
        for cat, thresholds in all_thresholds.items():
            stats["threshold_analysis"][cat] = {
                "mean": np.mean(thresholds),
                "std": np.std(thresholds),
                "min": min(thresholds),
                "max": max(thresholds)
            }
        
        return stats


def main():
    """Example usage of the Rubric Scoring Framework"""
    from phase5_genai_agents import MultiAgentResult, EvaluationResult, EvaluationDimension
    
    # Initialize scoring framework
    scoring_framework = RubricScoringFramework()
    
    # Create mock multi-agent result for testing
    mock_results = {
        EvaluationDimension.REPRODUCIBILITY: EvaluationResult(
            dimension=EvaluationDimension.REPRODUCIBILITY,
            score=0.75,
            confidence=0.85,
            justification="Good reproducibility package with clear instructions",
            evidence=["Dockerfile provided", "Requirements.txt present", "Scripts included"],
            suggestions=["Add environment.yml for conda users"],
            agent_version="1.0"
        ),
        EvaluationDimension.DOCUMENTATION: EvaluationResult(
            dimension=EvaluationDimension.DOCUMENTATION,
            score=0.80,
            confidence=0.90,
            justification="Well-structured documentation with examples",
            evidence=["Comprehensive README", "API documentation", "Examples provided"],
            suggestions=["Add troubleshooting section"],
            agent_version="1.0"
        ),
        EvaluationDimension.FUNCTIONALITY: EvaluationResult(
            dimension=EvaluationDimension.FUNCTIONALITY,
            score=0.70,
            confidence=0.75,
            justification="Core functionality works but lacks advanced features",
            evidence=["Basic tests pass", "Main functions work"],
            suggestions=["Add more comprehensive tests", "Improve error handling"],
            agent_version="1.0"
        )
    }
    
    mock_multi_agent_result = MultiAgentResult(
        artifact_id="sample_artifact_123",
        individual_results=mock_results,
        weighted_score=0.75,
        consensus_score=0.80,
        confidence_score=0.83,
        final_recommendation="Good artifact with minor improvements needed",
        improvement_suggestions=["Enhance testing", "Add conda support"]
    )
    
    # Test single conference scoring
    print("\n🎯 Phase 6: Rubric-Weighted Scoring Results")
    print("=" * 60)
    
    icse_result = scoring_framework.score_artifact(mock_multi_agent_result, "ICSE")
    print(f"✅ ICSE Scoring:")
    print(f"   Final Score: {icse_result.final_score:.1%}")
    print(f"   Category: {icse_result.acceptance_category.value}")
    print(f"   Acceptance Probability: {icse_result.acceptance_probability:.1%}")
    print(f"   Top Priorities: {icse_result.improvement_priorities[:2]}")
    
    # Test multi-conference comparison
    conferences = ["ICSE", "FSE", "ASE", "CHI"]
    comparison = scoring_framework.compare_across_conferences(mock_multi_agent_result, conferences)
    
    print(f"\n🏆 Multi-Conference Comparison:")
    print(f"   Best Fit: {comparison.best_fit_conference} ({comparison.best_score:.1%})")
    print(f"   Score Variance: {comparison.score_variance:.4f}")
    print(f"   Recommendations:")
    for rec in comparison.recommendations[:3]:
        print(f"     - {rec}")
    
    # Show conference-specific scores
    print(f"\n📊 Conference-Specific Scores:")
    for conf, result in comparison.conference_scores.items():
        print(f"   {conf}: {result.final_score:.1%} ({result.acceptance_category.value})")
    
    # Get framework statistics
    stats = scoring_framework.get_scoring_statistics()
    print(f"\n📈 Framework Statistics:")
    print(f"   Total Rubrics: {stats['total_rubrics']}")
    print(f"   Conferences: {', '.join(stats['conferences'])}")
    
    # Export example rubric
    scoring_framework.export_rubric("ICSE", "data/conference_rubrics/icse_2025_rubric.json")
    print(f"\n💾 Exported ICSE rubric to JSON file")


if __name__ == "__main__":
    main() 