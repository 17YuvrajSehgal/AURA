import json
import logging
import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path

from .agents.accessibility_evaluation_agent import AccessibilityEvaluationAgent
from .agents.documentation_evaluation_agent import DocumentationEvaluationAgent
from .agents.experimental_evaluation_agent import ExperimentalEvaluationAgent
from .agents.functionality_evaluation_agent import FunctionalityEvaluationAgent
from .agents.reproducibility_evaluation_agent import ReproducibilityEvaluationAgent
from .agents.usability_evaluation_agent import UsabilityEvaluationAgent
from .agents.repository_knowledge_graph_agent import RepositoryKnowledgeGraphAgent
from .agents.llm_evaluator import LLMEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Acceptance threshold based on ICSE 2025 standards
ACCEPTANCE_THRESHOLD = 0.75

class CriterionScore(BaseModel):
    dimension: str
    raw_score: float
    normalized_weight: float
    llm_evaluated_score: float = Field(default=0.0)
    justification: str = Field(default="")
    evidence: List[str] = Field(default_factory=list)
    llm_justification: str = Field(default="")
    llm_evidence: List[str] = Field(default_factory=list)

class ArtifactEvaluationResult(BaseModel):
    criteria_scores: List[CriterionScore]
    total_weighted_score: float
    acceptance_prediction: bool
    overall_justification: str = Field(default="")
    recommendations: List[str] = Field(default_factory=list)

class AURAFramework:
    def __init__(self, artifact_json_path: str, neo4j_uri: str = "bolt://localhost:7687", openai_api_key: str = None):
        """
        Initialize the AURA evaluation framework.
        
        Args:
            artifact_json_path: Path to the artifact JSON file
            neo4j_uri: Neo4j database URI for knowledge graph
            openai_api_key: OpenAI API key for LLM augmentation
        """
        self.artifact_json_path = artifact_json_path
        self.neo4j_uri = neo4j_uri
        
        # Initialize knowledge graph agent
        self.kg_agent = RepositoryKnowledgeGraphAgent(
            artifact_json_path=artifact_json_path,
            neo4j_uri=neo4j_uri
        )
        
        # Initialize LLM evaluator
        self.llm_evaluator = LLMEvaluator(openai_api_key=openai_api_key)
        
        # Initialize evaluation agents with LLM augmentation
        self.agents = {
            "accessibility": AccessibilityEvaluationAgent(self.kg_agent, self.llm_evaluator),
            "documentation": DocumentationEvaluationAgent(self.kg_agent, self.llm_evaluator),
            "experimental": ExperimentalEvaluationAgent(self.kg_agent, self.llm_evaluator),
            "functionality": FunctionalityEvaluationAgent(self.kg_agent, self.llm_evaluator),
            "reproducibility": ReproducibilityEvaluationAgent(self.kg_agent, self.llm_evaluator),
            "usability": UsabilityEvaluationAgent(self.kg_agent, self.llm_evaluator),
        }
        
        # Load scoring criteria
        self.criteria_scores = self._load_criteria_scores()
        
    def _load_criteria_scores(self) -> List[CriterionScore]:
        """Load the pre-defined criteria scores from the integration data."""
        try:
            # Get the path relative to the current script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            criteria_file = os.path.join(script_dir, "../../algo_outputs/algorithm_1_output/aura_integration_data_20250626_013157.json")
            
            with open(criteria_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            criteria_scores = []
            for criterion in data["structured_criteria"]:
                criteria_scores.append(CriterionScore(
                    dimension=criterion["dimension"],
                    raw_score=criterion["raw_score"],
                    normalized_weight=criterion["normalized_weight"]
                ))
            return criteria_scores
        except Exception as e:
            logger.error(f"Failed to load criteria scores: {e}")
            # Fallback to default criteria based on ICSE 2025
            return self._get_default_criteria()
    
    def _get_default_criteria(self) -> List[CriterionScore]:
        """Default criteria based on ICSE 2025 guidelines."""
        return [
            CriterionScore(dimension="reproducibility", raw_score=6.78, normalized_weight=0.207),
            CriterionScore(dimension="documentation", raw_score=5.04, normalized_weight=0.154),
            CriterionScore(dimension="accessibility", raw_score=4.55, normalized_weight=0.139),
            CriterionScore(dimension="usability", raw_score=6.47, normalized_weight=0.198),
            CriterionScore(dimension="experimental", raw_score=4.84, normalized_weight=0.148),
            CriterionScore(dimension="functionality", raw_score=5.09, normalized_weight=0.155),
        ]
    
    def evaluate_artifact(self) -> ArtifactEvaluationResult:
        """
        Evaluate the artifact using all agents and return comprehensive results.
        """
        logger.info("Starting artifact evaluation with AURA framework...")
        
        # Evaluate each dimension
        for criterion in self.criteria_scores:
            dimension = criterion.dimension
            if dimension in self.agents:
                logger.info(f"Evaluating {dimension} dimension...")
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
        
        # Calculate total weighted score
        total_score = self._calculate_total_weighted_score()
        
        # Determine acceptance
        acceptance = total_score >= ACCEPTANCE_THRESHOLD
        
        # Generate overall justification and recommendations
        overall_justification = self._generate_overall_justification()
        recommendations = self._generate_recommendations()
        
        return ArtifactEvaluationResult(
            criteria_scores=self.criteria_scores,
            total_weighted_score=total_score,
            acceptance_prediction=acceptance,
            overall_justification=overall_justification,
            recommendations=recommendations
        )
    
    def _calculate_total_weighted_score(self) -> float:
        """Calculate the total weighted score across all dimensions."""
        total = 0.0
        for criterion in self.criteria_scores:
            total += criterion.llm_evaluated_score * criterion.normalized_weight
        return total
    
    def _generate_overall_justification(self) -> str:
        """Generate overall justification based on evaluation results."""
        strengths = []
        weaknesses = []
        
        for criterion in self.criteria_scores:
            if criterion.llm_evaluated_score >= 0.8:
                strengths.append(criterion.dimension)
            elif criterion.llm_evaluated_score < 0.5:
                weaknesses.append(criterion.dimension)
        
        justification = f"Total weighted score: {self._calculate_total_weighted_score():.3f}. "
        
        if strengths:
            justification += f"Strong performance in: {', '.join(strengths)}. "
        if weaknesses:
            justification += f"Areas needing improvement: {', '.join(weaknesses)}. "
        
        return justification
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations for improvement."""
        recommendations = []
        
        for criterion in self.criteria_scores:
            if criterion.llm_evaluated_score < 0.6:
                if criterion.dimension == "documentation":
                    recommendations.append("Improve README documentation with clear setup and usage instructions")
                elif criterion.dimension == "functionality":
                    recommendations.append("Ensure artifact is executable and includes verification evidence")
                elif criterion.dimension == "reproducibility":
                    recommendations.append("Provide Docker containers or detailed environment setup")
                elif criterion.dimension == "accessibility":
                    recommendations.append("Ensure artifact is publicly accessible with proper licensing")
                elif criterion.dimension == "experimental":
                    recommendations.append("Include comprehensive experimental setup and data")
                elif criterion.dimension == "usability":
                    recommendations.append("Improve user experience and ease of use")
        
        return recommendations
    
    def save_results(self, result: ArtifactEvaluationResult, output_path: str):
        """Save evaluation results to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    def print_results(self, result: ArtifactEvaluationResult):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("AURA FRAMEWORK - ICSE 2025 ARTIFACT EVALUATION")
        print("="*60)
        
        for criterion in result.criteria_scores:
            print(f"\n{criterion.dimension.upper()}")
            print(f"  Score: {criterion.llm_evaluated_score:.3f} (Weight: {criterion.normalized_weight:.3f})")
            print(f"  Justification: {criterion.justification}")
            if criterion.evidence:
                print(f"  Evidence: {', '.join(criterion.evidence[:3])}...")
        
        print(f"\n{'='*60}")
        print(f"TOTAL WEIGHTED SCORE: {result.total_weighted_score:.3f}")
        print(f"ACCEPTANCE THRESHOLD: {ACCEPTANCE_THRESHOLD:.3f}")
        
        if result.acceptance_prediction:
            print("✅ PREDICTION: ACCEPTED")
        else:
            print("❌ PREDICTION: REJECTED")
        
        print(f"\nOverall Justification: {result.overall_justification}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("="*60)
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'kg_agent'):
            self.kg_agent.close()

def main():
    """Main function to run the AURA evaluation framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA Framework for ICSE 2025 Artifact Evaluation")
    parser.add_argument("artifact_json", help="Path to artifact JSON file")
    parser.add_argument("--output", "-o", help="Output file for results", default="aura_evaluation_results.json")
    parser.add_argument("--neo4j-uri", help="Neo4j database URI", default="bolt://localhost:7687")
    
    args = parser.parse_args()
    
    # Initialize and run evaluation
    framework = AURAFramework(args.artifact_json, args.neo4j_uri)
    
    try:
        result = framework.evaluate_artifact()
        framework.print_results(result)
        framework.save_results(result, args.output)
    finally:
        framework.close()

if __name__ == "__main__":
    main()
