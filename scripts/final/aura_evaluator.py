"""
AURA (Artifact Understanding and Research Assessment) Evaluator

This is the main module that orchestrates the complete evaluation pipeline for research artifacts.
It integrates knowledge graph building, RAG retrieval, and LangChain-based evaluation chains.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import config, AuraConfig
from knowledge_graph_builder import KnowledgeGraphBuilder
from rag_retrieval import RAGRetriever
from langchain_chains import ArtifactEvaluationOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AURAEvaluator:
    """
    Main AURA Evaluator that provides a complete pipeline for artifact evaluation
    """
    
    def __init__(self, use_neo4j: bool = True, use_rag: bool = True, 
                 output_dir: str = "evaluation_results", conference_name: Optional[str] = None):
        """
        Initialize the AURA Evaluator
        
        Args:
            use_neo4j: Whether to use Neo4j for knowledge graph storage
            use_rag: Whether to use RAG for contextual evaluation
            output_dir: Directory to save evaluation results
            conference_name: Name of conference for conference-specific evaluation
        """
        self.use_neo4j = use_neo4j
        self.use_rag = use_rag
        self.conference_name = conference_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.kg_builder = None
        self.orchestrator = None
        
        logger.info(f"AURA Evaluator initialized (Neo4j: {use_neo4j}, RAG: {use_rag}, Conference: {conference_name or 'General'})")
    
    def evaluate_artifact_from_json(self, artifact_json_path: str, 
                                   dimensions: Optional[List[str]] = None,
                                   save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate an artifact from its JSON representation
        
        Args:
            artifact_json_path: Path to the artifact JSON file
            dimensions: List of dimensions to evaluate (None for all)
            save_results: Whether to save results to disk
            
        Returns:
            Complete evaluation report
        """
        
        logger.info(f"Starting evaluation of artifact: {artifact_json_path}")
        start_time = time.time()
        
        try:
            # Load artifact data
            artifact_data = self._load_artifact_data(artifact_json_path)
            
            # Build knowledge graph
            logger.info("Building knowledge graph...")
            self.kg_builder = KnowledgeGraphBuilder(use_neo4j=self.use_neo4j)
            kg_stats = self.kg_builder.build_from_artifact_json(artifact_json_path)
            logger.info(f"Knowledge graph built: {kg_stats}")
            
            # Initialize evaluation orchestrator
            logger.info("Initializing evaluation orchestrator...")
            self.orchestrator = ArtifactEvaluationOrchestrator(
                knowledge_graph_builder=self.kg_builder,
                use_rag=self.use_rag,
                conference_name=self.conference_name
            )
            
            # Run evaluation
            logger.info("Running evaluations...")
            evaluation_results = self.orchestrator.evaluate_artifact(
                artifact_data=artifact_data,
                dimensions=dimensions
            )
            
            # Generate comprehensive report
            logger.info("Generating comprehensive report...")
            report = self.orchestrator.generate_comprehensive_report(
                evaluation_results=evaluation_results,
                artifact_data=artifact_data
            )
            
            # Add metadata
            end_time = time.time()
            report["evaluation_metadata"] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(end_time - start_time, 2),
                "evaluator_version": "1.0.0",
                "use_neo4j": self.use_neo4j,
                "use_rag": self.use_rag,
                "conference_name": self.conference_name,
                "knowledge_graph_stats": kg_stats,
                "source_file": artifact_json_path
            }
            
            # Save results if requested
            if save_results:
                self._save_evaluation_results(report, artifact_data)
            
            logger.info(f"Evaluation completed in {report['evaluation_metadata']['duration_seconds']} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
        
        finally:
            self._cleanup()
    
    def evaluate_artifact_from_data(self, artifact_data: Dict[str, Any],
                                   dimensions: Optional[List[str]] = None,
                                   save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate an artifact from artifact data dictionary
        
        Args:
            artifact_data: Dictionary containing artifact information
            dimensions: List of dimensions to evaluate (None for all)
            save_results: Whether to save results to disk
            
        Returns:
            Complete evaluation report
        """
        
        logger.info(f"Starting evaluation of artifact: {artifact_data.get('artifact_name', 'Unknown')}")
        start_time = time.time()
        
        try:
            # Create temporary JSON file for knowledge graph building
            temp_json_path = self.output_dir / "temp_artifact.json"
            with open(temp_json_path, 'w', encoding='utf-8') as f:
                json.dump(artifact_data, f, indent=2)
            
            # Build knowledge graph
            logger.info("Building knowledge graph...")
            self.kg_builder = KnowledgeGraphBuilder(use_neo4j=self.use_neo4j)
            kg_stats = self.kg_builder.build_from_artifact_json(str(temp_json_path))
            logger.info(f"Knowledge graph built: {kg_stats}")
            
            # Initialize evaluation orchestrator
            logger.info("Initializing evaluation orchestrator...")
            self.orchestrator = ArtifactEvaluationOrchestrator(
                knowledge_graph_builder=self.kg_builder,
                use_rag=self.use_rag,
                conference_name=self.conference_name
            )
            
            # Run evaluation
            logger.info("Running evaluations...")
            evaluation_results = self.orchestrator.evaluate_artifact(
                artifact_data=artifact_data,
                dimensions=dimensions
            )
            
            # Generate comprehensive report
            logger.info("Generating comprehensive report...")
            report = self.orchestrator.generate_comprehensive_report(
                evaluation_results=evaluation_results,
                artifact_data=artifact_data
            )
            
            # Add metadata
            end_time = time.time()
            report["evaluation_metadata"] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(end_time - start_time, 2),
                "evaluator_version": "1.0.0",
                "use_neo4j": self.use_neo4j,
                "use_rag": self.use_rag,
                "conference_name": self.conference_name,
                "knowledge_graph_stats": kg_stats,
                "source_type": "data_dictionary"
            }
            
            # Save results if requested
            if save_results:
                self._save_evaluation_results(report, artifact_data)
            
            # Clean up temporary file
            temp_json_path.unlink()
            
            logger.info(f"Evaluation completed in {report['evaluation_metadata']['duration_seconds']} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
        
        finally:
            self._cleanup()
    
    def batch_evaluate_artifacts(self, artifact_paths: List[str],
                                dimensions: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple artifacts in batch
        
        Args:
            artifact_paths: List of paths to artifact JSON files
            dimensions: List of dimensions to evaluate (None for all)
            
        Returns:
            Dictionary mapping artifact names to evaluation reports
        """
        
        logger.info(f"Starting batch evaluation of {len(artifact_paths)} artifacts")
        
        results = {}
        
        for i, artifact_path in enumerate(artifact_paths, 1):
            logger.info(f"Processing artifact {i}/{len(artifact_paths)}: {artifact_path}")
            
            try:
                report = self.evaluate_artifact_from_json(
                    artifact_json_path=artifact_path,
                    dimensions=dimensions,
                    save_results=True
                )
                
                artifact_name = report["artifact_info"]["name"]
                results[artifact_name] = report
                
            except Exception as e:
                logger.error(f"Failed to evaluate {artifact_path}: {e}")
                results[artifact_path] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        # Save batch summary
        self._save_batch_summary(results)
        
        logger.info(f"Batch evaluation completed: {len(results)} artifacts processed")
        return results
    
    def get_dimension_summary(self, evaluation_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a summary of dimension scores from an evaluation report
        
        Args:
            evaluation_report: Complete evaluation report
            
        Returns:
            Summary of dimension scores and statistics
        """
        
        dimension_scores = evaluation_report.get("dimension_scores", {})
        
        if not dimension_scores:
            return {"error": "No dimension scores found in report"}
        
        scores = list(dimension_scores.values())
        
        summary = {
            "dimension_scores": dimension_scores,
            "statistics": {
                "overall_rating": evaluation_report.get("overall_rating", 0.0),
                "average_score": sum(scores) / len(scores) if scores else 0.0,
                "highest_score": max(scores) if scores else 0.0,
                "lowest_score": min(scores) if scores else 0.0,
                "score_range": max(scores) - min(scores) if scores else 0.0,
                "dimensions_evaluated": len(dimension_scores)
            },
            "ranking": {
                "best_dimensions": sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True),
                "worst_dimensions": sorted(dimension_scores.items(), key=lambda x: x[1])
            }
        }
        
        return summary
    
    def _load_artifact_data(self, artifact_json_path: str) -> Dict[str, Any]:
        """Load artifact data from JSON file"""
        try:
            with open(artifact_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load artifact data: {e}")
            raise
    
    def _save_evaluation_results(self, report: Dict[str, Any], artifact_data: Dict[str, Any]):
        """Save evaluation results to disk"""
        try:
            artifact_name = artifact_data.get('artifact_name', 'unknown_artifact')
            
            # Clean artifact name for filename
            safe_name = "".join(c for c in artifact_name if c.isalnum() or c in ('_', '-', '.'))
            
            # Save full report
            report_path = self.output_dir / f"{safe_name}_evaluation_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            # Save summary
            summary = self.get_dimension_summary(report)
            summary_path = self.output_dir / f"{safe_name}_evaluation_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Evaluation results saved: {report_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save evaluation results: {e}")
    
    def _save_batch_summary(self, batch_results: Dict[str, Dict[str, Any]]):
        """Save batch evaluation summary"""
        try:
            summary = {
                "batch_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_artifacts": len(batch_results),
                    "successful_evaluations": len([r for r in batch_results.values() if "error" not in r]),
                    "failed_evaluations": len([r for r in batch_results.values() if "error" in r])
                },
                "artifact_summaries": {}
            }
            
            for artifact_name, result in batch_results.items():
                if "error" not in result:
                    summary["artifact_summaries"][artifact_name] = self.get_dimension_summary(result)
                else:
                    summary["artifact_summaries"][artifact_name] = {"error": result["error"]}
            
            # Calculate overall statistics
            successful_results = [r for r in batch_results.values() if "error" not in r]
            if successful_results:
                overall_ratings = [r.get("overall_rating", 0.0) for r in successful_results]
                summary["batch_statistics"] = {
                    "average_overall_rating": sum(overall_ratings) / len(overall_ratings),
                    "highest_overall_rating": max(overall_ratings),
                    "lowest_overall_rating": min(overall_ratings)
                }
            
            batch_summary_path = self.output_dir / "batch_evaluation_summary.json"
            with open(batch_summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Batch summary saved: {batch_summary_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save batch summary: {e}")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.orchestrator:
            self.orchestrator.close()
        if self.kg_builder:
            self.kg_builder.close()
    
    def close(self):
        """Close the evaluator and clean up resources"""
        self._cleanup()
        logger.info("AURA Evaluator closed")


# Convenience functions for quick evaluation
def quick_evaluate(artifact_json_path: str, 
                  dimensions: Optional[List[str]] = None,
                  use_neo4j: bool = False,
                  use_rag: bool = True,
                  conference_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick evaluation function for single artifacts
    
    Args:
        artifact_json_path: Path to artifact JSON file
        dimensions: Dimensions to evaluate (None for all)
        use_neo4j: Whether to use Neo4j (default: False for simplicity)
        use_rag: Whether to use RAG retrieval
        
    Returns:
        Evaluation report
    """
    
    evaluator = AURAEvaluator(use_neo4j=use_neo4j, use_rag=use_rag, conference_name=conference_name)
    
    try:
        return evaluator.evaluate_artifact_from_json(
            artifact_json_path=artifact_json_path,
            dimensions=dimensions
        )
    finally:
        evaluator.close()


def batch_evaluate(artifact_paths: List[str],
                  dimensions: Optional[List[str]] = None,
                  use_neo4j: bool = False,
                  use_rag: bool = True,
                  conference_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Batch evaluation function for multiple artifacts
    
    Args:
        artifact_paths: List of paths to artifact JSON files
        dimensions: Dimensions to evaluate (None for all)
        use_neo4j: Whether to use Neo4j
        use_rag: Whether to use RAG retrieval
        
    Returns:
        Dictionary of evaluation reports
    """
    
    evaluator = AURAEvaluator(use_neo4j=use_neo4j, use_rag=use_rag, conference_name=conference_name)
    
    try:
        return evaluator.batch_evaluate_artifacts(
            artifact_paths=artifact_paths,
            dimensions=dimensions
        )
    finally:
        evaluator.close()


# Example usage demonstration
def demo_evaluation():
    """Demonstrate the AURA evaluation system"""
    
    logger.info("=== AURA Evaluation System Demo ===")
    
    # Example artifact data (this would normally come from actual artifact processing)
    demo_artifact_data = {
        "artifact_name": "demo_research_artifact",
        "artifact_path": "/path/to/artifact",
        "repo_size_mb": 15.2,
        "extraction_method": "git_clone",
        "success": True,
        "documentation_files": [
            {
                "path": "README.md",
                "content": [
                    "# Demo Research Artifact",
                    "This is a demonstration artifact for testing the AURA evaluation system.",
                    "## Installation",
                    "pip install -r requirements.txt",
                    "## Usage",
                    "python main.py --input data.csv --output results.json"
                ]
            }
        ],
        "code_files": [
            {
                "path": "main.py",
                "content": [
                    "#!/usr/bin/env python",
                    "import argparse",
                    "import pandas as pd",
                    "def main():",
                    "    parser = argparse.ArgumentParser()",
                    "    parser.add_argument('--input', required=True)",
                    "    parser.add_argument('--output', required=True)",
                    "    args = parser.parse_args()",
                    "    # Process data",
                    "    print('Processing complete')"
                ]
            },
            {
                "path": "requirements.txt", 
                "content": ["pandas>=1.0.0", "numpy>=1.18.0"]
            }
        ],
        "data_files": [
            {
                "name": "sample_data.csv",
                "path": "data/sample_data.csv",
                "size_kb": 156,
                "mime_type": "text/csv"
            }
        ],
        "tree_structure": [
            ".",
            "├── README.md",
            "├── main.py", 
            "├── requirements.txt",
            "└── data/",
            "    └── sample_data.csv"
        ]
    }
    
    # Initialize evaluator
    evaluator = AURAEvaluator(use_neo4j=False, use_rag=True)
    
    try:
        # Run evaluation
        logger.info("Running demo evaluation...")
        report = evaluator.evaluate_artifact_from_data(
            artifact_data=demo_artifact_data,
            save_results=True
        )
        
        # Print summary
        logger.info("=== Evaluation Summary ===")
        logger.info(f"Overall Rating: {report['overall_rating']:.2f}/5.0")
        logger.info("Dimension Scores:")
        for dimension, score in report["dimension_scores"].items():
            logger.info(f"  {dimension.title()}: {score:.2f}/5.0")
        
        return report
        
    finally:
        evaluator.close()


if __name__ == "__main__":
    demo_evaluation()
