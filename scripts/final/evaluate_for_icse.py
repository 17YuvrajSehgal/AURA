#!/usr/bin/env python3
"""
ICSE 2025 Artifact Evaluation Script

Production-ready script to evaluate research artifacts for ICSE 2025 acceptance
using conference-specific guidelines and weighted scoring system.

Usage:
    python evaluate_for_icse.py [artifact_json_path]
    
Example:
    python evaluate_for_icse.py evaluation_results/ml-comprehensive-test.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from aura_evaluator import AURAEvaluator

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('icse_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


class ICSeArtifactEvaluator:
    """Production-ready ICSE artifact evaluator"""

    def __init__(self, output_dir: str = "icse_evaluation_results"):
        """
        Initialize ICSE evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.conference_name = "ICSE"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize evaluator with ICSE-specific settings
        self.evaluator = AURAEvaluator(
            use_neo4j=False,  # Use NetworkX for reliability (avoids Neo4j connection issues)
            use_rag=True,     # Enable RAG for contextual evaluation
            conference_name=self.conference_name,
            output_dir=str(self.output_dir)
        )

        logger.info(f"ICSE Artifact Evaluator initialized")
        logger.info(f"Output directory: {self.output_dir}")

    def validate_artifact_json(self, artifact_path: str) -> Dict[str, Any]:
        """
        Validate and load artifact JSON
        
        Args:
            artifact_path: Path to artifact JSON file
            
        Returns:
            Loaded and validated artifact data
            
        Raises:
            ValueError: If artifact data is invalid
            FileNotFoundError: If file doesn't exist
        """
        artifact_file = Path(artifact_path)

        if not artifact_file.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

        try:
            with open(artifact_file, 'r', encoding='utf-8') as f:
                artifact_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        # Validate required fields
        required_fields = ['artifact_name', 'success']
        missing_fields = [field for field in required_fields if field not in artifact_data]

        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        if not artifact_data.get('success', False):
            logger.warning("Artifact extraction was not successful - evaluation may be limited")

        logger.info(f"Validated artifact: {artifact_data.get('artifact_name', 'Unknown')}")
        return artifact_data

    def check_icse_prerequisites(self, artifact_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check ICSE-specific prerequisites
        
        Args:
            artifact_data: Artifact data to check
            
        Returns:
            Dictionary with prerequisite check results
        """
        checks = {
            "has_documentation": bool(artifact_data.get('documentation_files')),
            "has_license": bool(artifact_data.get('license_files')),
            "has_code": bool(artifact_data.get('code_files')),
            "has_docker": bool(artifact_data.get('docker_files')),
            "has_requirements": bool(artifact_data.get('build_files') or artifact_data.get('config_files')),
            "has_tests": False,
            "repository_size_reasonable": artifact_data.get('repo_size_mb', 0) < 1000,  # < 1GB
        }

        # Check for test files
        code_files = artifact_data.get('code_files', [])
        for file_info in code_files:
            if 'test' in file_info.get('path', '').lower():
                checks["has_tests"] = True
                break

        # Calculate overall readiness score
        total_checks = len(checks)
        passed_checks = sum(1 for check in checks.values() if check)
        readiness_score = (passed_checks / total_checks) * 100

        return {
            "checks": checks,
            "readiness_score": readiness_score,
            "passed_checks": passed_checks,
            "total_checks": total_checks
        }

    def evaluate_artifact(self, artifact_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive ICSE evaluation
        
        Args:
            artifact_path: Path to artifact JSON file
            
        Returns:
            Complete evaluation report with ICSE-specific insights
        """
        logger.info("=" * 60)
        logger.info("🎯 ICSE 2025 ARTIFACT EVALUATION")
        logger.info("=" * 60)

        try:
            # Step 1: Validate artifact
            logger.info("Step 1: Validating artifact data...")
            artifact_data = self.validate_artifact_json(artifact_path)

            # Step 2: Check ICSE prerequisites  
            logger.info("Step 2: Checking ICSE prerequisites...")
            prerequisites = self.check_icse_prerequisites(artifact_data)
            self._log_prerequisites(prerequisites)

            # Step 3: Run comprehensive evaluation
            logger.info("Step 3: Running comprehensive evaluation...")
            evaluation_report = self.evaluator.evaluate_artifact_from_data(
                artifact_data=artifact_data,
                save_results=True
            )

            # Step 4: Generate ICSE-specific insights
            logger.info("Step 4: Generating ICSE-specific insights...")
            icse_insights = self._generate_icse_insights(evaluation_report, prerequisites)

            # Step 5: Create final report
            final_report = {
                "artifact_info": evaluation_report["artifact_info"],
                "icse_evaluation": {
                    "conference": self.conference_name,
                    "evaluation_timestamp": evaluation_report["evaluation_metadata"]["timestamp"],
                    "prerequisites": prerequisites,
                    "weighted_scoring": evaluation_report["weighted_scoring"],
                    "dimension_scores": evaluation_report["dimension_scores"],
                    "icse_insights": icse_insights,
                    "recommendation": self._generate_recommendation(evaluation_report, prerequisites)
                },
                "detailed_evaluation": evaluation_report
            }

            # Step 6: Save and display results
            self._save_final_report(final_report)
            self._display_summary(final_report)

            logger.info("✅ ICSE evaluation completed successfully!")
            return final_report

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise

    def _log_prerequisites(self, prerequisites: Dict[str, Any]):
        """Log prerequisite check results"""
        logger.info(
            f"ICSE Prerequisites Check: {prerequisites['passed_checks']}/{prerequisites['total_checks']} passed ({prerequisites['readiness_score']:.1f}%)")

        for check_name, passed in prerequisites["checks"].items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check_name.replace('_', ' ').title()}")

    def _generate_icse_insights(self, evaluation_report: Dict[str, Any], prerequisites: Dict[str, Any]) -> Dict[
        str, Any]:
        """Generate ICSE-specific insights and recommendations"""

        weighted_info = evaluation_report.get("weighted_scoring", {})
        dimension_scores = evaluation_report.get("dimension_scores", {})

        # ICSE-specific analysis
        insights = {
            "archival_readiness": self._assess_archival_readiness(evaluation_report, prerequisites),
            "reproducibility_assessment": self._assess_reproducibility(dimension_scores, prerequisites),
            "documentation_compliance": self._assess_documentation(dimension_scores, prerequisites),
            "artifact_type_classification": self._classify_artifact_type(evaluation_report),
            "estimated_review_time": self._estimate_review_time(weighted_info, prerequisites),
            "key_strengths": self._identify_key_strengths(dimension_scores, prerequisites),
            "improvement_priorities": self._identify_improvement_priorities(dimension_scores, prerequisites)
        }

        return insights

    def _assess_archival_readiness(self, evaluation_report: Dict[str, Any], prerequisites: Dict[str, Any]) -> Dict[
        str, str]:
        """Assess readiness for archival repository (ICSE requirement)"""

        has_license = prerequisites["checks"]["has_license"]
        has_documentation = prerequisites["checks"]["has_documentation"]
        accessibility_score = evaluation_report.get("dimension_scores", {}).get("accessibility", 0)

        if has_license and has_documentation and accessibility_score >= 3.0:
            status = "ready"
            recommendation = "Artifact is ready for archival repository submission (Zenodo/FigShare)"
        elif has_license and has_documentation:
            status = "needs_minor_fixes"
            recommendation = "Minor accessibility improvements needed before archival"
        else:
            status = "needs_major_work"
            recommendation = "Major work needed - missing license or documentation"

        return {
            "status": status,
            "recommendation": recommendation
        }

    def _assess_reproducibility(self, dimension_scores: Dict[str, float], prerequisites: Dict[str, Any]) -> Dict[
        str, Any]:
        """Assess reproducibility for ICSE standards"""

        repro_score = dimension_scores.get("reproducibility", 0)
        has_docker = prerequisites["checks"]["has_docker"]
        has_requirements = prerequisites["checks"]["has_requirements"]

        if repro_score >= 4.0 and has_docker and has_requirements:
            level = "excellent"
            description = "Highly reproducible with containerization"
        elif repro_score >= 3.0 and (has_docker or has_requirements):
            level = "good"
            description = "Good reproducibility with some automation"
        elif repro_score >= 2.0:
            level = "basic"
            description = "Basic reproducibility - manual setup required"
        else:
            level = "poor"
            description = "Significant reproducibility challenges"

        return {
            "level": level,
            "score": repro_score,
            "description": description,
            "has_containerization": has_docker
        }

    def _assess_documentation(self, dimension_scores: Dict[str, float], prerequisites: Dict[str, Any]) -> Dict[
        str, Any]:
        """Assess documentation quality for ICSE standards"""

        doc_score = dimension_scores.get("documentation", 0)
        has_docs = prerequisites["checks"]["has_documentation"]

        if doc_score >= 4.0 and has_docs:
            quality = "excellent"
            description = "Comprehensive documentation meeting ICSE standards"
        elif doc_score >= 3.0 and has_docs:
            quality = "good"
            description = "Good documentation with minor gaps"
        elif has_docs:
            quality = "basic"
            description = "Basic documentation present but needs improvement"
        else:
            quality = "insufficient"
            description = "Insufficient documentation for ICSE requirements"

        return {
            "quality": quality,
            "score": doc_score,
            "description": description
        }

    def _classify_artifact_type(self, evaluation_report: Dict[str, Any]) -> str:
        """Classify artifact type for ICSE submission"""

        artifact_data = evaluation_report.get("detailed_evaluation", {})

        # Check for common patterns
        if "docker" in str(artifact_data).lower():
            return "Containerized Software"
        elif "test" in str(artifact_data).lower():
            return "Tested Software"
        elif "data" in str(artifact_data).lower():
            return "Software with Data"
        else:
            return "Basic Software"

    def _estimate_review_time(self, weighted_info: Dict[str, Any], prerequisites: Dict[str, Any]) -> str:
        """Estimate ICSE review time based on artifact quality"""

        overall_score = weighted_info.get("weighted_overall_percentage", 0)
        readiness_score = prerequisites.get("readiness_score", 0)

        avg_score = (overall_score + readiness_score) / 2

        if avg_score >= 80:
            return "1-2 hours (High quality, minimal reviewer effort)"
        elif avg_score >= 60:
            return "2-4 hours (Good quality, moderate reviewer effort)"
        elif avg_score >= 40:
            return "4-8 hours (Needs work, significant reviewer effort)"
        else:
            return "8+ hours (Major issues, extensive reviewer effort)"

    def _identify_key_strengths(self, dimension_scores: Dict[str, float], prerequisites: Dict[str, Any]) -> list:
        """Identify key strengths for ICSE submission"""

        strengths = []

        if dimension_scores.get("reproducibility", 0) >= 4.0:
            strengths.append("Excellent reproducibility")

        if dimension_scores.get("documentation", 0) >= 4.0:
            strengths.append("High-quality documentation")

        if prerequisites["checks"]["has_docker"]:
            strengths.append("Containerized environment")

        if prerequisites["checks"]["has_tests"]:
            strengths.append("Comprehensive testing")

        if prerequisites["checks"]["has_license"]:
            strengths.append("Proper licensing")

        if dimension_scores.get("usability", 0) >= 4.0:
            strengths.append("User-friendly design")

        return strengths if strengths else ["Basic functionality present"]

    def _identify_improvement_priorities(self, dimension_scores: Dict[str, float],
                                         prerequisites: Dict[str, Any]) -> list:
        """Identify improvement priorities for ICSE acceptance"""

        priorities = []

        # Sort dimensions by score (lowest first) and weight importance
        sorted_dimensions = sorted(dimension_scores.items(), key=lambda x: x[1])

        for dimension, score in sorted_dimensions[:3]:  # Top 3 lowest scores
            if score < 3.0:  # Below acceptable threshold
                if dimension == "reproducibility":
                    priorities.append("🔄 Improve reproducibility: Add detailed setup instructions, Docker support")
                elif dimension == "documentation":
                    priorities.append("📝 Enhance documentation: Add comprehensive README, API docs")
                elif dimension == "accessibility":
                    priorities.append("🌐 Improve accessibility: Ensure public availability, add DOI")
                elif dimension == "functionality":
                    priorities.append("⚙️ Fix functionality issues: Ensure code runs correctly")
                elif dimension == "usability":
                    priorities.append("👥 Enhance usability: Simplify installation, add examples")
                elif dimension == "experimental":
                    priorities.append("🧪 Strengthen evaluation: Add benchmarks, comparisons")

        # Check prerequisites
        if not prerequisites["checks"]["has_license"]:
            priorities.insert(0, "⚖️ Add license file (CRITICAL for ICSE)")

        if not prerequisites["checks"]["has_docker"] and dimension_scores.get("reproducibility", 0) < 3.0:
            priorities.append("🐳 Add Docker containerization")

        return priorities if priorities else ["Minor polishing recommended"]

    def _generate_recommendation(self, evaluation_report: Dict[str, Any], prerequisites: Dict[str, Any]) -> Dict[
        str, Any]:
        """Generate final ICSE acceptance recommendation"""

        weighted_info = evaluation_report.get("weighted_scoring", {})
        overall_score = weighted_info.get("weighted_overall_percentage", 0)
        acceptance_info = weighted_info.get("acceptance_probability", {})
        readiness_score = prerequisites.get("readiness_score", 0)

        # Combined score (70% evaluation, 30% prerequisites)
        combined_score = (overall_score * 0.7) + (readiness_score * 0.3)

        if combined_score >= 75 and acceptance_info.get("category") in ["excellent", "good"]:
            decision = "RECOMMEND ACCEPTANCE"
            confidence = "High"
            reason = "Artifact meets ICSE standards with high quality across dimensions"
        elif combined_score >= 60:
            decision = "CONDITIONAL ACCEPTANCE"
            confidence = "Medium"
            reason = "Artifact is promising but needs minor improvements"
        elif combined_score >= 45:
            decision = "MAJOR REVISION NEEDED"
            confidence = "Low"
            reason = "Artifact has potential but requires significant improvements"
        else:
            decision = "NOT READY FOR ICSE"
            confidence = "Very Low"
            reason = "Artifact needs substantial work before ICSE submission"

        return {
            "decision": decision,
            "confidence": confidence,
            "combined_score": combined_score,
            "reason": reason,
            "next_steps": "See improvement priorities above"
        }

    def _save_final_report(self, final_report: Dict[str, Any]):
        """Save final evaluation report"""

        artifact_name = final_report["artifact_info"]["name"]
        safe_name = "".join(c for c in artifact_name if c.isalnum() or c in ('_', '-', '.'))

        # Save full report
        report_path = self.output_dir / f"{safe_name}_icse_evaluation.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2)

        # Save executive summary
        summary_path = self.output_dir / f"{safe_name}_icse_summary.json"
        summary = {
            "artifact_name": artifact_name,
            "conference": "ICSE 2025",
            "evaluation_timestamp": final_report["icse_evaluation"]["evaluation_timestamp"],
            "recommendation": final_report["icse_evaluation"]["recommendation"],
            "weighted_scoring": final_report["icse_evaluation"]["weighted_scoring"],
            "key_insights": final_report["icse_evaluation"]["icse_insights"]
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📄 Full report saved: {report_path}")
        logger.info(f"📋 Summary saved: {summary_path}")

    def _display_summary(self, final_report: Dict[str, Any]):
        """Display evaluation summary"""

        icse_eval = final_report["icse_evaluation"]
        weighted_scoring = icse_eval["weighted_scoring"]
        recommendation = icse_eval["recommendation"]
        insights = icse_eval["icse_insights"]

        logger.info("\n" + "=" * 60)
        logger.info("📊 ICSE 2025 EVALUATION SUMMARY")
        logger.info("=" * 60)

        # Overall scores
        logger.info(
            f"🎯 Overall Weighted Score: {weighted_scoring['weighted_overall_score']:.2f}/5.0 ({weighted_scoring['weighted_overall_percentage']:.1f}%)")
        logger.info(f"🎲 Acceptance Probability: {weighted_scoring['acceptance_probability']['probability_text']}")
        logger.info(f"⭐ Combined Readiness: {recommendation['combined_score']:.1f}%")

        # Recommendation
        logger.info(f"\n🔮 RECOMMENDATION: {recommendation['decision']}")
        logger.info(f"📈 Confidence: {recommendation['confidence']}")
        logger.info(f"💭 Reason: {recommendation['reason']}")

        # Key insights
        logger.info(f"\n📋 ARTIFACT TYPE: {insights['artifact_type_classification']}")
        logger.info(f"⏱️  Estimated Review Time: {insights['estimated_review_time']}")
        logger.info(f"🏛️  Archival Status: {insights['archival_readiness']['status'].replace('_', ' ').title()}")

        # Strengths
        logger.info(f"\n✅ KEY STRENGTHS:")
        for strength in insights['key_strengths']:
            logger.info(f"   • {strength}")

        # Improvements
        if insights['improvement_priorities']:
            logger.info(f"\n🔧 IMPROVEMENT PRIORITIES:")
            for priority in insights['improvement_priorities']:
                logger.info(f"   • {priority}")

        logger.info("=" * 60)

    def close(self):
        """Clean up resources"""
        if self.evaluator:
            self.evaluator.close()
        logger.info("ICSE evaluator closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate research artifact for ICSE 2025 acceptance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_for_icse.py evaluation_results/ml-comprehensive-test.json
  python evaluate_for_icse.py path/to/artifact.json --output-dir icse_results
        """
    )

    parser.add_argument(
        "artifact_json",
        nargs='?',
        default="evaluation_results/ml-comprehensive-test.json",
        help="Path to artifact JSON file (default: evaluation_results/ml-comprehensive-test.json)"
    )

    parser.add_argument(
        "--output-dir",
        default="icse_evaluation_results",
        help="Output directory for results (default: icse_evaluation_results)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize evaluator
    evaluator = None
    try:
        evaluator = ICSeArtifactEvaluator(output_dir=args.output_dir)

        # Run evaluation
        final_report = evaluator.evaluate_artifact(args.artifact_json)

        # Exit with appropriate code
        recommendation = final_report["icse_evaluation"]["recommendation"]["decision"]
        if "RECOMMEND ACCEPTANCE" in recommendation:
            sys.exit(0)  # Success
        elif "CONDITIONAL" in recommendation:
            sys.exit(1)  # Needs minor work
        else:
            sys.exit(2)  # Needs major work or not ready

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        sys.exit(3)
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(4)
    finally:
        if evaluator:
            evaluator.close()


if __name__ == "__main__":
    main()
