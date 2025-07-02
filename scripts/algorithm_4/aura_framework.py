import json
import logging
import os
from typing import List, Optional
import time

from pydantic import BaseModel, Field

from scripts.algorithm_4.agents.accessibility_evaluation_agent import AccessibilityEvaluationAgent
from scripts.algorithm_4.agents.documentation_evaluation_agent import DocumentationEvaluationAgent
from scripts.algorithm_4.agents.experimental_evaluation_agent import ExperimentalEvaluationAgent
from scripts.algorithm_4.agents.functionality_evaluation_agent import FunctionalityEvaluationAgent
from scripts.algorithm_4.agents.llm_evaluator import LLMEvaluator
from scripts.algorithm_4.agents.repository_knowledge_graph_agent import RepositoryKnowledgeGraphAgent
from scripts.algorithm_4.agents.reproducibility_evaluation_agent import ReproducibilityEvaluationAgent
from scripts.algorithm_4.agents.usability_evaluation_agent import UsabilityEvaluationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default acceptance threshold - can be overridden by conference profiles
DEFAULT_ACCEPTANCE_THRESHOLD = 0.50


class CriterionScore(BaseModel):
    dimension: str
    raw_score: float
    normalized_weight: float
    llm_evaluated_score: float = Field(default=0.0)
    justification: str = Field(default="")
    evidence: List[str] = Field(default_factory=list)
    llm_justification: str = Field(default="")
    llm_evidence: List[str] = Field(default_factory=list)
    conference_weight: float = Field(default=0.0)  # Conference-specific weight


class ArtifactEvaluationResult(BaseModel):
    criteria_scores: List[CriterionScore]
    total_weighted_score: float
    acceptance_prediction: bool
    overall_justification: str = Field(default="")
    recommendations: List[str] = Field(default_factory=list)
    timing: dict = Field(default_factory=dict)
    conference_info: dict = Field(default_factory=dict)  # Conference-specific information


class AURAFramework:
    def __init__(self, artifact_json_path: str, neo4j_uri: str = "bolt://localhost:7687", 
                 openai_api_key: str = None, use_llm: bool = True, 
                 conference_profile: Optional[dict] = None):
        """
        Initialize the AURA evaluation framework with conference-specific capabilities.
        
        Args:
            artifact_json_path: Path to the artifact JSON file
            neo4j_uri: Neo4j database URI for knowledge graph
            openai_api_key: OpenAI API key for LLM augmentation
            use_llm: Whether to use the LLM evaluator (default True)
            conference_profile: Conference-specific profile for targeted evaluation
        """
        self.artifact_json_path = artifact_json_path
        self.neo4j_uri = neo4j_uri
        self.conference_profile = conference_profile or self._get_default_conference_profile()

        # --- Timing: Analysis ---
        analysis_start = time.time()

        # Initialize knowledge graph agent
        self.kg_agent = RepositoryKnowledgeGraphAgent(
            artifact_json_path=artifact_json_path,
            neo4j_uri=neo4j_uri
        )
        analysis_finish = time.time()
        self.analysis_time = analysis_finish - analysis_start
        self.analysis_start = analysis_start
        self.analysis_finish = analysis_finish

        # Initialize LLM evaluator if enabled
        self.llm_evaluator = None
        if use_llm:
            self.llm_evaluator = LLMEvaluator(
                openai_api_key=openai_api_key,
                model_name="gpt-4-turbo",
                temperature=0.1
            )

        # Initialize evaluation agents with conference-specific profiles
        self.agents = {
            "accessibility": AccessibilityEvaluationAgent(self.kg_agent, self.llm_evaluator, self.conference_profile),
            "documentation": DocumentationEvaluationAgent(self.kg_agent, self.llm_evaluator),
            "experimental": ExperimentalEvaluationAgent(self.kg_agent, self.llm_evaluator),
            "functionality": FunctionalityEvaluationAgent(self.kg_agent, self.llm_evaluator),
            "reproducibility": ReproducibilityEvaluationAgent(self.kg_agent, self.llm_evaluator),
            "usability": UsabilityEvaluationAgent(self.kg_agent, self.llm_evaluator),
        }
        
        # Load scoring criteria with conference-specific weights
        self.criteria_scores = self._load_criteria_scores()
        
        # Calculate conference-specific acceptance threshold
        self.acceptance_threshold = self.conference_profile.get('quality_threshold', DEFAULT_ACCEPTANCE_THRESHOLD)

    def _get_default_conference_profile(self) -> dict:
        """Get default conference profile if none provided."""
        return {
            "category": "general",
            "emphasis_weights": {
                "reproducibility": 0.20,
                "documentation": 0.15,
                "accessibility": 0.15,
                "usability": 0.20,
                "experimental": 0.15,
                "functionality": 0.15
            },
            "quality_threshold": DEFAULT_ACCEPTANCE_THRESHOLD,
            "evaluation_style": "standard",
            "domain_keywords": []
        }

    def _load_criteria_scores(self) -> List[CriterionScore]:
        """Load and adapt criteria scores based on conference profile."""
        try:
            # Get the path relative to the current script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            criteria_file = os.path.join(script_dir,
                                         "../../algo_outputs/algorithm_1_output/aura_integration_data_20250626_044643.json")

            # Try to load existing criteria
            base_criteria = []
            if os.path.exists(criteria_file):
                with open(criteria_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    base_criteria = data.get("structured_criteria", [])

            if not base_criteria:
                # Fallback to default criteria
                base_criteria = self._get_default_criteria_data()

            # Apply conference-specific weights
            criteria_scores = []
            conference_weights = self.conference_profile.get('emphasis_weights', {})
            
            # Normalize conference weights to sum to 1
            total_weight = sum(conference_weights.values())
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in conference_weights.items()}
            else:
                normalized_weights = conference_weights

            for criterion in base_criteria:
                dimension = criterion.get("dimension", "")
                
                # Use conference-specific weight if available, otherwise use original
                conference_weight = normalized_weights.get(dimension, criterion.get("normalized_weight", 0.16))
                
                criteria_scores.append(CriterionScore(
                    dimension=dimension,
                    raw_score=criterion.get("raw_score", 0.0),
                    normalized_weight=criterion.get("normalized_weight", 0.16),
                    conference_weight=conference_weight
                ))
                
            return criteria_scores
            
        except Exception as e:
            logger.error(f"Failed to load criteria scores: {e}")
            # Fallback to default criteria with conference weights
            return self._get_default_criteria()

    def _get_default_criteria_data(self) -> List[dict]:
        """Get default criteria data structure."""
        return [
            {"dimension": "reproducibility", "raw_score": 6.78, "normalized_weight": 0.207},
            {"dimension": "documentation", "raw_score": 5.04, "normalized_weight": 0.154},
            {"dimension": "accessibility", "raw_score": 4.55, "normalized_weight": 0.139},
            {"dimension": "usability", "raw_score": 6.47, "normalized_weight": 0.198},
            {"dimension": "experimental", "raw_score": 4.84, "normalized_weight": 0.148},
            {"dimension": "functionality", "raw_score": 5.09, "normalized_weight": 0.155},
        ]

    def _get_default_criteria(self) -> List[CriterionScore]:
        """Default criteria with conference-specific weights applied."""
        conference_weights = self.conference_profile.get('emphasis_weights', {})
        
        # Normalize weights
        total_weight = sum(conference_weights.values()) if conference_weights else 1.0
        
        default_data = self._get_default_criteria_data()
        criteria_scores = []
        
        for criterion in default_data:
            dimension = criterion["dimension"]
            conference_weight = conference_weights.get(dimension, 0.16) / total_weight if total_weight > 0 else 0.16
            
            criteria_scores.append(CriterionScore(
                dimension=dimension,
                raw_score=criterion["raw_score"],
                normalized_weight=criterion["normalized_weight"],
                conference_weight=conference_weight
            ))
            
        return criteria_scores

    def evaluate_artifact(self, progress_callback=None) -> ArtifactEvaluationResult:
        """
        Evaluate the artifact using conference-specific evaluation.
        """
        eval_start = time.time()
        if progress_callback:
            progress_callback(f"Starting conference-specific evaluation for {self.conference_profile.get('category', 'general')} venue...")
        
        logger.info(f"Starting conference-specific artifact evaluation for {self.conference_profile.get('category', 'general')} venue...")

        # Analyze conference-specific patterns
        conference_patterns = self.kg_agent.check_conference_specific_patterns(
            self.conference_profile.get('category', 'general')
        )
        
        # Calculate repository structure quality
        structure_quality = self.kg_agent.analyze_repository_structure_quality()
        
        # Calculate conference relevance
        conference_relevance = self.kg_agent.get_conference_relevance_score(
            self.conference_profile.get('domain_keywords', [])
        )

        # Evaluate each dimension
        for idx, criterion in enumerate(self.criteria_scores):
            dimension = criterion.dimension
            if dimension in self.agents:
                msg = f"Evaluating {dimension} dimension (conference weight: {criterion.conference_weight:.3f})..."
                if progress_callback:
                    progress_callback(msg)
                logger.info(msg)
                try:
                    agent_result = self.agents[dimension].evaluate()
                    criterion.llm_evaluated_score = agent_result["score"]
                    criterion.justification = agent_result["justification"]
                    criterion.evidence = agent_result.get("evidence", [])
                    
                    # Save LLM justification and evidence if present
                    if "llm_justification" in agent_result:
                        criterion.llm_justification = agent_result["llm_justification"]
                    if "llm_evidence" in agent_result:
                        criterion.llm_evidence = agent_result["llm_evidence"]
                        
                except Exception as e:
                    logger.error(f"Error evaluating {dimension}: {e}")
                    criterion.llm_evaluated_score = 0.0
                    criterion.justification = f"Evaluation failed: {str(e)}"
            else:
                logger.warning(f"No agent found for dimension: {dimension}")
                if progress_callback:
                    progress_callback(f"No agent found for dimension: {dimension}")

        # Calculate total weighted score using conference-specific weights
        total_score = self._calculate_conference_weighted_score()
        if progress_callback:
            progress_callback(f"Conference-weighted score calculated: {total_score:.3f}")

        # Determine acceptance based on conference threshold
        acceptance = total_score >= self.acceptance_threshold

        # Generate conference-aware justification and recommendations
        overall_justification = self._generate_conference_aware_justification(
            total_score, acceptance, structure_quality, conference_relevance
        )
        recommendations = self._generate_conference_specific_recommendations()
        
        if progress_callback:
            progress_callback("Conference-specific artifact evaluation complete.")
        
        eval_finish = time.time()
        eval_time = eval_finish - eval_start

        timing = {
            "analysis_start_time": self.analysis_start,
            "analysis_finish_time": self.analysis_finish,
            "analysis_duration_seconds": round(self.analysis_time, 2),
            "evaluation_start_time": eval_start,
            "evaluation_finish_time": eval_finish,
            "evaluation_duration_seconds": round(eval_time, 2)
        }
        
        conference_info = {
            "category": self.conference_profile.get('category'),
            "quality_threshold": self.acceptance_threshold,
            "evaluation_style": self.conference_profile.get('evaluation_style'),
            "structure_quality": structure_quality,
            "conference_relevance": conference_relevance,
            "conference_patterns": conference_patterns
        }

        return ArtifactEvaluationResult(
            criteria_scores=self.criteria_scores,
            total_weighted_score=total_score,
            acceptance_prediction=acceptance,
            overall_justification=overall_justification,
            recommendations=recommendations,
            timing=timing,
            conference_info=conference_info
        )

    def _calculate_conference_weighted_score(self) -> float:
        """Calculate the total weighted score using conference-specific weights."""
        total = 0.0
        for criterion in self.criteria_scores:
            total += criterion.llm_evaluated_score * criterion.conference_weight
        return total

    def _generate_conference_aware_justification(self, total_score: float, acceptance: bool, 
                                               structure_quality: dict, conference_relevance: float) -> str:
        """Generate conference-aware overall justification."""
        conference_name = self.conference_profile.get('category', 'general').replace('_', ' ').title()
        
        justification = f"Conference-specific evaluation for {conference_name} venue. "
        justification += f"Total conference-weighted score: {total_score:.3f} "
        justification += f"(threshold: {self.acceptance_threshold:.3f}). "
        
        if acceptance:
            justification += f"ACCEPTED: Artifact meets {conference_name} standards. "
        else:
            justification += f"REJECTED: Artifact below {conference_name} threshold. "
        
        # Add structure quality assessment
        justification += f"Repository structure quality: {structure_quality.get('quality_score', 0):.2f}. "
        
        # Add conference relevance
        justification += f"Conference relevance score: {conference_relevance:.2f}. "
        
        # Identify strengths and weaknesses based on conference emphasis
        strengths = []
        weaknesses = []
        emphasis_weights = self.conference_profile.get('emphasis_weights', {})
        
        for criterion in self.criteria_scores:
            emphasis = emphasis_weights.get(criterion.dimension, 0)
            score = criterion.llm_evaluated_score
            
            if emphasis > 0.2 and score >= 0.8:  # High emphasis, high score
                strengths.append(f"{criterion.dimension} (emphasis: {emphasis:.2f})")
            elif emphasis > 0.2 and score < 0.6:  # High emphasis, low score
                weaknesses.append(f"{criterion.dimension} (emphasis: {emphasis:.2f})")

        if strengths:
            justification += f"Strong performance in conference priorities: {', '.join(strengths)}. "
        if weaknesses:
            justification += f"Needs improvement in conference priorities: {', '.join(weaknesses)}. "

        return justification

    def _generate_conference_specific_recommendations(self) -> List[str]:
        """Generate conference-specific recommendations for improvement."""
        recommendations = []
        conference_category = self.conference_profile.get('category', 'general')
        emphasis_weights = self.conference_profile.get('emphasis_weights', {})

        for criterion in self.criteria_scores:
            emphasis = emphasis_weights.get(criterion.dimension, 0)
            score = criterion.llm_evaluated_score
            
            # Focus on high-emphasis, low-scoring dimensions
            if emphasis > 0.2 and score < 0.6:
                if criterion.dimension == "accessibility":
                    if conference_category == "software_engineering":
                        recommendations.append("Improve accessibility by adding GitHub releases and proper software distribution")
                    else:
                        recommendations.append("Enhance accessibility with archival repository (Zenodo/FigShare) and DOI")
                        
                elif criterion.dimension == "reproducibility":
                    if conference_category == "software_engineering":
                        recommendations.append("Add Docker containerization and CI/CD pipeline for reproducibility")
                    elif conference_category == "data_systems":
                        recommendations.append("Provide comprehensive data provenance and experimental setup")
                    else:
                        recommendations.append("Include detailed environment setup and dependency management")
                        
                elif criterion.dimension == "documentation":
                    recommendations.append(f"Improve documentation quality for {conference_category} audience with domain-specific examples")
                    
                elif criterion.dimension == "usability":
                    if conference_category == "hci":
                        recommendations.append("Enhance user interface design and accessibility features")
                    else:
                        recommendations.append("Simplify installation and usage procedures")

        # Add conference-specific recommendations
        if conference_category == "software_engineering":
            recommendations.append("Consider adding automated testing and code quality metrics")
        elif conference_category == "data_systems":
            recommendations.append("Include data quality validation and performance benchmarks")
        elif conference_category == "hci":
            recommendations.append("Provide user study materials and interaction design documentation")

        return recommendations

    def save_results(self, result: ArtifactEvaluationResult, output_path: str):
        """Save evaluation results to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")

    def print_results(self, result: ArtifactEvaluationResult):
        """Print evaluation results in a formatted way."""
        conference_name = result.conference_info.get('category', 'General').replace('_', ' ').title()
        
        print("\n" + "=" * 80)
        print(f"AURA FRAMEWORK - {conference_name.upper()} CONFERENCE ARTIFACT EVALUATION")
        print("=" * 80)
        
        print(f"\nConference Category: {conference_name}")
        print(f"Quality Threshold: {result.conference_info.get('quality_threshold', 'N/A'):.3f}")
        print(f"Repository Structure Quality: {result.conference_info.get('structure_quality', {}).get('quality_score', 0):.3f}")
        print(f"Conference Relevance: {result.conference_info.get('conference_relevance', 0):.3f}")

        for criterion in result.criteria_scores:
            print(f"\n{criterion.dimension.upper()}")
            print(f"  Score: {criterion.llm_evaluated_score:.3f}")
            print(f"  Conference Weight: {criterion.conference_weight:.3f}")
            print(f"  Original Weight: {criterion.normalized_weight:.3f}")
            print(f"  Justification: {criterion.justification}")
            if criterion.evidence:
                print(f"  Evidence: {', '.join(criterion.evidence[:3])}...")

        print(f"\n{'=' * 80}")
        print(f"CONFERENCE-WEIGHTED SCORE: {result.total_weighted_score:.3f}")
        print(f"ACCEPTANCE THRESHOLD: {result.conference_info.get('quality_threshold', 'N/A'):.3f}")

        if result.acceptance_prediction:
            print(f"✅ PREDICTION: ACCEPTED FOR {conference_name.upper()}")
        else:
            print(f"❌ PREDICTION: REJECTED FOR {conference_name.upper()}")

        print(f"\nOverall Justification: {result.overall_justification}")

        if result.recommendations:
            print(f"\n{conference_name}-Specific Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")

        print("=" * 80)

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'kg_agent'):
            self.kg_agent.close()


def main():
    """Main function to run the AURA evaluation framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Conference-Aware AURA Framework for Artifact Evaluation")
    parser.add_argument("artifact_json", help="Path to artifact JSON file")
    parser.add_argument("--output", "-o", help="Output file for results", default="aura_evaluation_results.json")
    parser.add_argument("--neo4j-uri", help="Neo4j database URI", default="bolt://localhost:7687")
    parser.add_argument("--conference-profile", help="Path to conference profile JSON file")

    args = parser.parse_args()

    # Load conference profile if provided
    conference_profile = None
    if args.conference_profile and os.path.exists(args.conference_profile):
        with open(args.conference_profile, 'r') as f:
            conference_profile = json.load(f)

    # Initialize and run evaluation
    framework = AURAFramework(args.artifact_json, args.neo4j_uri, conference_profile=conference_profile)

    try:
        result = framework.evaluate_artifact()
        framework.print_results(result)
        framework.save_results(result, args.output)
    finally:
        framework.close()


if __name__ == "__main__":
    main()
