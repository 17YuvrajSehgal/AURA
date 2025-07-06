#!/usr/bin/env python3
"""
Batch Artifact Evaluation Script

This script processes multiple artifact analysis JSON files and generates
comprehensive evaluation reports and comparisons.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent))

from artifact_evaluation_system import ArtifactEvaluationSystem
from evaluation_config import EvaluationConfig, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchEvaluationProcessor:
    """
    Batch processor for evaluating multiple artifacts and generating reports.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize batch processor.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.evaluator = ArtifactEvaluationSystem(
            neo4j_uri=self.config.neo4j_uri,
            neo4j_user=self.config.neo4j_user,
            neo4j_password=self.config.neo4j_password,
            openai_api_key=self.config.openai_api_key
        )
        
        # Store evaluation results
        self.results = []
        self.failed_evaluations = []
    
    def process_directory(self, input_directory: str, output_directory: str = None) -> Dict[str, Any]:
        """
        Process all JSON files in a directory.
        
        Args:
            input_directory: Directory containing JSON analysis files
            output_directory: Directory to save evaluation results
            
        Returns:
            Dictionary containing batch processing results
        """
        input_path = Path(input_directory)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_directory}")
        
        # Find all JSON files
        json_files = list(input_path.glob("**/*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in: {input_directory}")
            return {
                "success": False,
                "error": "No JSON files found",
                "files_processed": 0
            }
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Create output directory if specified
        if output_directory:
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        successful_evaluations = 0
        for json_file in json_files:
            try:
                logger.info(f"Processing: {json_file.name}")
                result = self.evaluator.evaluate_artifact_from_json(str(json_file))
                
                if result["success"]:
                    self.results.append(result)
                    successful_evaluations += 1
                    
                    # Save individual evaluation result
                    if output_directory:
                        self._save_individual_result(result, output_path)
                else:
                    self.failed_evaluations.append({
                        "file": str(json_file),
                        "error": result.get("error", "Unknown error")
                    })
                    logger.warning(f"Failed to evaluate {json_file.name}: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"Error processing {json_file.name}: {e}")
                self.failed_evaluations.append({
                    "file": str(json_file),
                    "error": str(e)
                })
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary()
        
        # Save results if output directory specified
        if output_directory:
            self._save_batch_results(batch_summary, output_path)
        
        return {
            "success": True,
            "files_processed": len(json_files),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": len(self.failed_evaluations),
            "batch_summary": batch_summary
        }
    
    def process_file_list(self, file_list: List[str], output_directory: str = None) -> Dict[str, Any]:
        """
        Process a list of specific JSON files.
        
        Args:
            file_list: List of JSON file paths
            output_directory: Directory to save evaluation results
            
        Returns:
            Dictionary containing batch processing results
        """
        if output_directory:
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
        
        successful_evaluations = 0
        for json_file in file_list:
            try:
                if not os.path.exists(json_file):
                    logger.warning(f"File not found: {json_file}")
                    continue
                
                logger.info(f"Processing: {os.path.basename(json_file)}")
                result = self.evaluator.evaluate_artifact_from_json(json_file)
                
                if result["success"]:
                    self.results.append(result)
                    successful_evaluations += 1
                    
                    # Save individual evaluation result
                    if output_directory:
                        self._save_individual_result(result, output_path)
                else:
                    self.failed_evaluations.append({
                        "file": json_file,
                        "error": result.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                self.failed_evaluations.append({
                    "file": json_file,
                    "error": str(e)
                })
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary()
        
        # Save results if output directory specified
        if output_directory:
            self._save_batch_results(batch_summary, output_path)
        
        return {
            "success": True,
            "files_processed": len(file_list),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": len(self.failed_evaluations),
            "batch_summary": batch_summary
        }
    
    def _save_individual_result(self, result: Dict[str, Any], output_path: Path):
        """Save individual evaluation result."""
        try:
            artifact_name = result["artifact_name"]
            safe_name = "".join(c for c in artifact_name if c.isalnum() or c in ('-', '_'))
            
            # Save JSON result
            json_path = output_path / f"{safe_name}_evaluation.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Save HTML report
            html_path = output_path / f"{safe_name}_report.html"
            self.evaluator.export_evaluation_report(artifact_name, str(html_path))
            
        except Exception as e:
            logger.error(f"Error saving individual result: {e}")
    
    def _generate_batch_summary(self) -> Dict[str, Any]:
        """Generate comprehensive batch summary."""
        if not self.results:
            return {
                "total_artifacts": 0,
                "successful_evaluations": 0,
                "failed_evaluations": len(self.failed_evaluations)
            }
        
        # Calculate summary statistics
        summary = {
            "total_artifacts": len(self.results),
            "successful_evaluations": len(self.results),
            "failed_evaluations": len(self.failed_evaluations),
            "generated_at": datetime.now().isoformat()
        }
        
        # Overall statistics
        scores = [r["evaluation_scores"] for r in self.results if "evaluation_scores" in r]
        predictions = [r["acceptance_prediction"] for r in self.results if "acceptance_prediction" in r]
        
        if scores:
            # Calculate average scores for each criterion
            criteria_scores = {}
            for criterion in ["documentation_quality", "reproducibility", "availability", "code_structure", "complexity"]:
                criterion_scores = [s.get(criterion, 0) for s in scores]
                criteria_scores[criterion] = {
                    "average": sum(criterion_scores) / len(criterion_scores),
                    "min": min(criterion_scores),
                    "max": max(criterion_scores)
                }
            
            summary["score_statistics"] = criteria_scores
        
        # Acceptance predictions
        if predictions:
            prediction_counts = {}
            for pred in predictions:
                likelihood = pred.get("likelihood", "unknown")
                prediction_counts[likelihood] = prediction_counts.get(likelihood, 0) + 1
            
            summary["acceptance_predictions"] = prediction_counts
        
        # Artifact types
        artifact_types = {}
        for result in self.results:
            features = result.get("features", {})
            artifact_type = self._infer_artifact_type(features)
            artifact_types[artifact_type] = artifact_types.get(artifact_type, 0) + 1
        
        summary["artifact_types"] = artifact_types
        
        # Feature statistics
        feature_stats = self._calculate_feature_statistics()
        summary["feature_statistics"] = feature_stats
        
        # Top artifacts
        top_artifacts = self._get_top_artifacts()
        summary["top_artifacts"] = top_artifacts
        
        # Recommendations analysis
        recommendations_analysis = self._analyze_recommendations()
        summary["recommendations_analysis"] = recommendations_analysis
        
        return summary
    
    def _infer_artifact_type(self, features: Dict[str, Any]) -> str:
        """Infer artifact type from features."""
        # Simple heuristic - can be improved
        if features.get("has_data_files", False) and not features.get("code_files", 0):
            return "dataset"
        elif features.get("code_files", 0) > 0 and features.get("has_examples", False):
            return "tool"
        elif features.get("has_docker", False) or features.get("has_setup_instructions", False):
            return "replication"
        else:
            return "unknown"
    
    def _calculate_feature_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for artifact features."""
        if not self.results:
            return {}
        
        features_list = [r.get("features", {}) for r in self.results]
        
        stats = {}
        
        # Boolean features
        boolean_features = [
            "has_readme", "has_license", "has_docker", "has_zenodo_doi",
            "has_setup_instructions", "has_examples", "has_data_files"
        ]
        
        for feature in boolean_features:
            count = sum(1 for f in features_list if f.get(feature, False))
            stats[feature] = {
                "count": count,
                "percentage": (count / len(features_list)) * 100
            }
        
        # Numeric features
        numeric_features = ["total_files", "code_files", "readme_length", "repo_size_mb"]
        
        for feature in numeric_features:
            values = [f.get(feature, 0) for f in features_list if f.get(feature, 0) > 0]
            if values:
                stats[feature] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return stats
    
    def _get_top_artifacts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top-scoring artifacts."""
        if not self.results:
            return []
        
        # Sort by evaluation score
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get("acceptance_prediction", {}).get("score", 0),
            reverse=True
        )
        
        top_artifacts = []
        for result in sorted_results[:limit]:
            top_artifacts.append({
                "artifact_name": result["artifact_name"],
                "score": result.get("acceptance_prediction", {}).get("score", 0),
                "likelihood": result.get("acceptance_prediction", {}).get("likelihood", "unknown"),
                "key_features": {
                    "has_readme": result.get("features", {}).get("has_readme", False),
                    "has_docker": result.get("features", {}).get("has_docker", False),
                    "has_zenodo_doi": result.get("features", {}).get("has_zenodo_doi", False),
                    "setup_complexity": result.get("features", {}).get("setup_complexity", "unknown")
                }
            })
        
        return top_artifacts
    
    def _analyze_recommendations(self) -> Dict[str, Any]:
        """Analyze common recommendations across artifacts."""
        if not self.results:
            return {}
        
        recommendation_counts = {}
        category_counts = {}
        
        for result in self.results:
            recommendations = result.get("recommendations", [])
            for rec in recommendations:
                category = rec.get("category", "unknown")
                recommendation_text = rec.get("recommendation", "")
                
                # Count by category
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Count similar recommendations (simplified)
                key_words = recommendation_text.lower().split()[:3]  # First 3 words
                rec_key = " ".join(key_words)
                recommendation_counts[rec_key] = recommendation_counts.get(rec_key, 0) + 1
        
        return {
            "most_common_categories": dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)),
            "most_common_recommendations": dict(sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def _save_batch_results(self, batch_summary: Dict[str, Any], output_path: Path):
        """Save batch evaluation results."""
        try:
            # Save JSON summary
            summary_path = output_path / "batch_evaluation_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, indent=2, default=str)
            
            # Save CSV comparison
            self._save_csv_comparison(output_path)
            
            # Generate HTML dashboard
            self._generate_html_dashboard(batch_summary, output_path)
            
            logger.info(f"Batch results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")
    
    def _save_csv_comparison(self, output_path: Path):
        """Save CSV comparison of all artifacts."""
        try:
            if not self.results:
                return
            
            # Prepare data for CSV
            csv_data = []
            for result in self.results:
                features = result.get("features", {})
                scores = result.get("evaluation_scores", {})
                prediction = result.get("acceptance_prediction", {})
                
                row = {
                    "artifact_name": result["artifact_name"],
                    "overall_score": prediction.get("score", 0),
                    "acceptance_likelihood": prediction.get("likelihood", "unknown"),
                    "confidence": prediction.get("confidence", 0),
                    "doc_quality_score": scores.get("documentation_quality", 0),
                    "reproducibility_score": scores.get("reproducibility", 0),
                    "availability_score": scores.get("availability", 0),
                    "code_structure_score": scores.get("code_structure", 0),
                    "complexity_score": scores.get("complexity", 0),
                    "has_readme": features.get("has_readme", False),
                    "has_license": features.get("has_license", False),
                    "has_docker": features.get("has_docker", False),
                    "has_zenodo_doi": features.get("has_zenodo_doi", False),
                    "has_setup_instructions": features.get("has_setup_instructions", False),
                    "has_examples": features.get("has_examples", False),
                    "total_files": features.get("total_files", 0),
                    "code_files": features.get("code_files", 0),
                    "repo_size_mb": features.get("repo_size_mb", 0),
                    "setup_complexity": features.get("setup_complexity", "unknown")
                }
                csv_data.append(row)
            
            # Save to CSV
            df = pd.DataFrame(csv_data)
            csv_path = output_path / "artifact_comparison.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"CSV comparison saved to: {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV comparison: {e}")
    
    def _generate_html_dashboard(self, batch_summary: Dict[str, Any], output_path: Path):
        """Generate HTML dashboard for batch results."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Batch Evaluation Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .stat {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e3f2fd; border-radius: 5px; }}
                    .top-artifact {{ margin: 5px 0; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .high {{ color: #4caf50; }}
                    .medium {{ color: #ff9800; }}
                    .low {{ color: #f44336; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Batch Evaluation Dashboard</h1>
                    <p>Generated on: {batch_summary.get('generated_at', 'Unknown')}</p>
                </div>
                
                <div class="section">
                    <h2>Overview</h2>
                    <div class="stat">
                        <strong>Total Artifacts:</strong> {batch_summary.get('total_artifacts', 0)}
                    </div>
                    <div class="stat">
                        <strong>Successful Evaluations:</strong> {batch_summary.get('successful_evaluations', 0)}
                    </div>
                    <div class="stat">
                        <strong>Failed Evaluations:</strong> {batch_summary.get('failed_evaluations', 0)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Acceptance Predictions</h2>
                    {self._generate_prediction_chart(batch_summary.get('acceptance_predictions', {}))}
                </div>
                
                <div class="section">
                    <h2>Top Artifacts</h2>
                    {self._generate_top_artifacts_html(batch_summary.get('top_artifacts', []))}
                </div>
                
                <div class="section">
                    <h2>Score Statistics</h2>
                    {self._generate_score_statistics_html(batch_summary.get('score_statistics', {}))}
                </div>
                
                <div class="section">
                    <h2>Feature Statistics</h2>
                    {self._generate_feature_statistics_html(batch_summary.get('feature_statistics', {}))}
                </div>
                
                <div class="section">
                    <h2>Common Recommendations</h2>
                    {self._generate_recommendations_html(batch_summary.get('recommendations_analysis', {}))}
                </div>
                
            </body>
            </html>
            """
            
            dashboard_path = output_path / "batch_evaluation_dashboard.html"
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML dashboard saved to: {dashboard_path}")
            
        except Exception as e:
            logger.error(f"Error generating HTML dashboard: {e}")
    
    def _generate_prediction_chart(self, predictions: Dict[str, int]) -> str:
        """Generate HTML for prediction chart."""
        if not predictions:
            return "<p>No prediction data available.</p>"
        
        total = sum(predictions.values())
        html = "<div>"
        for prediction, count in predictions.items():
            percentage = (count / total) * 100
            color = {"high": "#4caf50", "medium": "#ff9800", "low": "#f44336"}.get(prediction, "#757575")
            html += f"""
            <div style="margin: 10px 0;">
                <span style="color: {color}; font-weight: bold;">{prediction.upper()}</span>: {count} ({percentage:.1f}%)
                <div style="background-color: #e0e0e0; height: 20px; border-radius: 10px; margin: 5px 0;">
                    <div style="background-color: {color}; height: 100%; width: {percentage}%; border-radius: 10px;"></div>
                </div>
            </div>
            """
        html += "</div>"
        return html
    
    def _generate_top_artifacts_html(self, top_artifacts: List[Dict[str, Any]]) -> str:
        """Generate HTML for top artifacts."""
        if not top_artifacts:
            return "<p>No artifacts available.</p>"
        
        html = "<div>"
        for artifact in top_artifacts:
            likelihood_class = artifact.get("likelihood", "unknown")
            html += f"""
            <div class="top-artifact">
                <strong>{artifact['artifact_name']}</strong>
                <span class="{likelihood_class}" style="float: right;">
                    {artifact['score']:.3f} ({artifact['likelihood'].upper()})
                </span>
                <br>
                <small>
                    README: {'✓' if artifact.get('key_features', {}).get('has_readme') else '✗'} |
                    Docker: {'✓' if artifact.get('key_features', {}).get('has_docker') else '✗'} |
                    Zenodo: {'✓' if artifact.get('key_features', {}).get('has_zenodo_doi') else '✗'} |
                    Complexity: {artifact.get('key_features', {}).get('setup_complexity', 'unknown')}
                </small>
            </div>
            """
        html += "</div>"
        return html
    
    def _generate_score_statistics_html(self, score_stats: Dict[str, Any]) -> str:
        """Generate HTML for score statistics."""
        if not score_stats:
            return "<p>No score statistics available.</p>"
        
        html = "<table>"
        html += "<tr><th>Criterion</th><th>Average</th><th>Min</th><th>Max</th></tr>"
        
        for criterion, stats in score_stats.items():
            html += f"""
            <tr>
                <td>{criterion.replace('_', ' ').title()}</td>
                <td>{stats['average']:.3f}</td>
                <td>{stats['min']:.3f}</td>
                <td>{stats['max']:.3f}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_feature_statistics_html(self, feature_stats: Dict[str, Any]) -> str:
        """Generate HTML for feature statistics."""
        if not feature_stats:
            return "<p>No feature statistics available.</p>"
        
        html = "<table>"
        html += "<tr><th>Feature</th><th>Count</th><th>Percentage</th></tr>"
        
        for feature, stats in feature_stats.items():
            if isinstance(stats, dict) and "count" in stats:
                html += f"""
                <tr>
                    <td>{feature.replace('_', ' ').title()}</td>
                    <td>{stats['count']}</td>
                    <td>{stats.get('percentage', 0):.1f}%</td>
                </tr>
                """
        
        html += "</table>"
        return html
    
    def _generate_recommendations_html(self, recommendations_analysis: Dict[str, Any]) -> str:
        """Generate HTML for recommendations analysis."""
        if not recommendations_analysis:
            return "<p>No recommendations analysis available.</p>"
        
        html = "<div>"
        
        # Most common categories
        categories = recommendations_analysis.get("most_common_categories", {})
        if categories:
            html += "<h4>Most Common Recommendation Categories</h4><ul>"
            for category, count in list(categories.items())[:5]:
                html += f"<li><strong>{category.title()}</strong>: {count} recommendations</li>"
            html += "</ul>"
        
        # Most common recommendations
        recommendations = recommendations_analysis.get("most_common_recommendations", {})
        if recommendations:
            html += "<h4>Most Common Recommendations</h4><ul>"
            for rec, count in list(recommendations.items())[:5]:
                html += f"<li>{rec.title()}: {count} times</li>"
            html += "</ul>"
        
        html += "</div>"
        return html
    
    def close(self):
        """Close resources."""
        if self.evaluator:
            self.evaluator.close()


def main():
    """Main function for batch evaluation."""
    parser = argparse.ArgumentParser(description="Batch Artifact Evaluation")
    parser.add_argument("--input-dir", help="Directory containing JSON analysis files")
    parser.add_argument("--input-files", nargs="+", help="List of specific JSON files to process")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--openai-api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = EvaluationConfig.from_file(args.config)
    else:
        config = EvaluationConfig()
    
    # Override with command line arguments
    if args.neo4j_uri:
        config.neo4j_uri = args.neo4j_uri
    if args.neo4j_user:
        config.neo4j_user = args.neo4j_user
    if args.neo4j_password:
        config.neo4j_password = args.neo4j_password
    if args.openai_api_key:
        config.openai_api_key = args.openai_api_key
    
    # Create processor
    processor = BatchEvaluationProcessor(config)
    
    try:
        # Process files
        if args.input_dir:
            result = processor.process_directory(args.input_dir, args.output_dir)
        elif args.input_files:
            result = processor.process_file_list(args.input_files, args.output_dir)
        else:
            print("Error: Must specify either --input-dir or --input-files")
            return 1
        
        # Print results
        if result["success"]:
            print(f"✅ Batch evaluation completed successfully!")
            print(f"Files processed: {result['files_processed']}")
            print(f"Successful evaluations: {result['successful_evaluations']}")
            print(f"Failed evaluations: {result['failed_evaluations']}")
            
            if args.output_dir:
                print(f"📁 Results saved to: {args.output_dir}")
        else:
            print(f"❌ Batch evaluation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    except Exception as e:
        print(f"❌ Error during batch evaluation: {e}")
        return 1
    
    finally:
        processor.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 