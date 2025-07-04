#!/usr/bin/env python3
"""
ICSE Artifacts Processor

Specialized script for processing ICSE (International Conference on Software Engineering)
artifacts using the Robust Knowledge Graph Pipeline.

This script is designed to:
- Process the 27 ICSE artifacts mentioned by the user
- Handle various artifact formats (zip, tar.gz, directories)
- Generate comprehensive reports and visualizations
- Export results for research analysis
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from algorithm_5 import RobustKGPipeline
    from algorithm_5.config import load_config
except ImportError as e:
    print(f"Error importing algorithm_5: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ICSEArtifactsProcessor:
    """Specialized processor for ICSE artifacts."""
    
    def __init__(
        self,
        artifacts_dir: str,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        output_dir: str = "./icse_analysis_output",
        config_profile: str = "default"
    ):
        """
        Initialize the ICSE artifacts processor.
        
        Args:
            artifacts_dir: Directory containing ICSE artifacts
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            output_dir: Output directory for results
            config_profile: Configuration profile to use
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = load_config(profile=config_profile)
        
        # Override with provided parameters
        self.config.set("neo4j", "uri", value=neo4j_uri)
        self.config.set("neo4j", "user", value=neo4j_user)
        self.config.set("neo4j", "password", value=neo4j_password)
        self.config.set("pipeline", "working_dir", value=str(self.output_dir))
        
        # Processing state
        self.artifacts_found = []
        self.processing_results = []
        self.analysis_report = {}
    
    def discover_artifacts(self) -> List[Dict[str, Any]]:
        """
        Discover all artifacts in the ICSE artifacts directory.
        
        Returns:
            List of discovered artifacts with metadata
        """
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(f"ICSE artifacts directory not found: {self.artifacts_dir}")
        
        logger.info(f"Discovering artifacts in: {self.artifacts_dir}")
        
        artifacts = []
        
        # Supported formats
        archive_extensions = {'.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz'}
        
        for item in self.artifacts_dir.iterdir():
            artifact_info = {
                "path": str(item),
                "name": item.name,
                "type": None,
                "size": 0,
                "format": None
            }
            
            if item.is_dir():
                # Directory (likely git clone)
                artifact_info.update({
                    "type": "directory",
                    "format": "git_repository",
                    "size": sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                })
                artifacts.append(artifact_info)
                
            elif item.is_file():
                artifact_info["size"] = item.stat().st_size
                
                # Check for archive formats
                if any(item.name.lower().endswith(ext) for ext in archive_extensions):
                    artifact_info.update({
                        "type": "archive",
                        "format": self._detect_archive_format(item)
                    })
                    artifacts.append(artifact_info)
                
                # Check for compound extensions (tar.gz, tar.bz2, etc.)
                elif any(item.name.lower().endswith(f".tar{ext}") for ext in ['.gz', '.bz2', '.xz']):
                    artifact_info.update({
                        "type": "archive",
                        "format": self._detect_archive_format(item)
                    })
                    artifacts.append(artifact_info)
        
        self.artifacts_found = artifacts
        
        logger.info(f"Found {len(artifacts)} artifacts:")
        for artifact in artifacts:
            size_mb = artifact["size"] / (1024 * 1024)
            logger.info(f"  - {artifact['name']} ({artifact['format']}, {size_mb:.1f}MB)")
        
        return artifacts
    
    def _detect_archive_format(self, file_path: Path) -> str:
        """Detect the format of an archive file."""
        name_lower = file_path.name.lower()
        
        if name_lower.endswith('.zip'):
            return 'zip'
        elif name_lower.endswith('.tar.gz') or name_lower.endswith('.tgz'):
            return 'tar.gz'
        elif name_lower.endswith('.tar.bz2'):
            return 'tar.bz2'
        elif name_lower.endswith('.tar.xz'):
            return 'tar.xz'
        elif name_lower.endswith('.tar'):
            return 'tar'
        else:
            return 'unknown'
    
    def process_all_artifacts(
        self,
        max_workers: Optional[int] = None,
        enable_advanced_analysis: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Process all discovered ICSE artifacts.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_advanced_analysis: Enable advanced graph analysis
            dry_run: If True, only discover artifacts without processing
            
        Returns:
            Dictionary containing processing results
        """
        # Discover artifacts if not already done
        if not self.artifacts_found:
            self.discover_artifacts()
        
        if dry_run:
            logger.info("Dry run completed. No artifacts were processed.")
            return {
                "dry_run": True,
                "artifacts_discovered": len(self.artifacts_found),
                "artifacts": self.artifacts_found
            }
        
        if not self.artifacts_found:
            logger.warning("No artifacts found to process")
            return {"success": False, "error": "No artifacts found"}
        
        # Update configuration
        if max_workers:
            self.config.set("batch", "max_workers", value=max_workers)
        
        logger.info(f"Starting processing of {len(self.artifacts_found)} ICSE artifacts")
        
        try:
            with RobustKGPipeline(
                neo4j_uri=self.config.get("neo4j", "uri"),
                neo4j_user=self.config.get("neo4j", "user"),
                neo4j_password=self.config.get("neo4j", "password"),
                working_dir=str(self.output_dir),
                enable_advanced_analysis=enable_advanced_analysis,
                clear_existing_graph=True  # Start fresh for ICSE analysis
            ) as pipeline:
                
                # Process artifacts using the pipeline's batch processor
                batch_result = pipeline.process_artifact_directory(
                    artifacts_dir=str(self.artifacts_dir),
                    file_patterns=["*"]  # Process all found artifacts
                )
                
                if batch_result["success"]:
                    self.processing_results = batch_result["artifacts_processed"]
                    
                    # Generate comprehensive analysis
                    self.analysis_report = self._generate_comprehensive_analysis(pipeline)
                    
                    # Export results
                    self._export_results(batch_result)
                    
                    logger.info("✅ ICSE artifacts processing completed successfully!")
                    return batch_result
                else:
                    logger.error("❌ ICSE artifacts processing failed")
                    return batch_result
        
        except Exception as e:
            logger.error(f"Error processing ICSE artifacts: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_comprehensive_analysis(self, pipeline) -> Dict[str, Any]:
        """Generate comprehensive analysis of ICSE artifacts."""
        logger.info("Generating comprehensive analysis...")
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_artifacts": len(self.artifacts_found),
            "processing_summary": {},
            "graph_statistics": {},
            "programming_languages": {},
            "project_characteristics": {},
            "quality_metrics": {},
            "comparative_analysis": {}
        }
        
        try:
            # Get graph statistics
            analysis["graph_statistics"] = pipeline.get_graph_statistics()
            
            # Analyze programming languages across all artifacts
            languages_query = """
            MATCH (a:Artifact)-[:CONTAINS*]->(f:File {type: 'code'})
            RETURN f.extension as extension, count(f) as file_count, 
                   collect(DISTINCT a.name) as artifacts
            ORDER BY file_count DESC
            """
            language_results = pipeline.query_graph(languages_query)
            analysis["programming_languages"] = {
                result["extension"]: {
                    "file_count": result["file_count"],
                    "artifact_count": len(result["artifacts"]),
                    "artifacts": result["artifacts"]
                }
                for result in language_results
            }
            
            # Analyze project characteristics
            characteristics_query = """
            MATCH (a:Artifact)
            RETURN a.name as artifact_name,
                   a.has_readme as has_readme,
                   a.has_license as has_license,
                   a.has_dockerfile as has_dockerfile,
                   a.has_tests as has_tests,
                   a.programming_languages as languages,
                   a.build_systems as build_systems
            """
            char_results = pipeline.query_graph(characteristics_query)
            analysis["project_characteristics"] = {
                result["artifact_name"]: {
                    "has_readme": result.get("has_readme", False),
                    "has_license": result.get("has_license", False),
                    "has_dockerfile": result.get("has_dockerfile", False),
                    "has_tests": result.get("has_tests", False),
                    "languages": result.get("languages", []),
                    "build_systems": result.get("build_systems", [])
                }
                for result in char_results
            }
            
            # Calculate quality metrics
            analysis["quality_metrics"] = self._calculate_quality_metrics(pipeline)
            
            # Comparative analysis
            analysis["comparative_analysis"] = self._perform_comparative_analysis(pipeline)
            
        except Exception as e:
            logger.warning(f"Error generating comprehensive analysis: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _calculate_quality_metrics(self, pipeline) -> Dict[str, Any]:
        """Calculate quality metrics for all artifacts."""
        try:
            # Documentation coverage
            doc_coverage_query = """
            MATCH (a:Artifact)
            OPTIONAL MATCH (a)-[:CONTAINS*]->(doc:File) WHERE doc.type = 'documentation'
            OPTIONAL MATCH (a)-[:CONTAINS*]->(all:File)
            RETURN a.name as artifact,
                   count(DISTINCT doc) as doc_files,
                   count(DISTINCT all) as total_files,
                   CASE WHEN count(DISTINCT all) > 0 
                        THEN toFloat(count(DISTINCT doc)) / count(DISTINCT all) 
                        ELSE 0 END as doc_ratio
            """
            doc_results = pipeline.query_graph(doc_coverage_query)
            
            # Test coverage
            test_coverage_query = """
            MATCH (a:Artifact)
            OPTIONAL MATCH (a)-[:CONTAINS*]->(test:Test)
            OPTIONAL MATCH (a)-[:CONTAINS*]->(code:File {type: 'code'})
            RETURN a.name as artifact,
                   count(DISTINCT test) as test_files,
                   count(DISTINCT code) as code_files,
                   CASE WHEN count(DISTINCT code) > 0 
                        THEN toFloat(count(DISTINCT test)) / count(DISTINCT code) 
                        ELSE 0 END as test_ratio
            """
            test_results = pipeline.query_graph(test_coverage_query)
            
            return {
                "documentation_coverage": {
                    result["artifact"]: {
                        "doc_files": result["doc_files"],
                        "total_files": result["total_files"],
                        "coverage_ratio": result["doc_ratio"]
                    }
                    for result in doc_results
                },
                "test_coverage": {
                    result["artifact"]: {
                        "test_files": result["test_files"],
                        "code_files": result["code_files"],
                        "test_ratio": result["test_ratio"]
                    }
                    for result in test_results
                }
            }
            
        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            return {"error": str(e)}
    
    def _perform_comparative_analysis(self, pipeline) -> Dict[str, Any]:
        """Perform comparative analysis between artifacts."""
        try:
            # Size comparison
            size_query = """
            MATCH (a:Artifact)-[:CONTAINS*]->(f:File)
            RETURN a.name as artifact, 
                   count(f) as total_files,
                   sum(f.size) as total_size
            ORDER BY total_size DESC
            """
            size_results = pipeline.query_graph(size_query)
            
            # Complexity comparison (based on number of functions and classes)
            complexity_query = """
            MATCH (a:Artifact)
            OPTIONAL MATCH (a)-[:CONTAINS*]->()-[:DEFINES]->(fn:Function)
            OPTIONAL MATCH (a)-[:CONTAINS*]->()-[:DEFINES]->(cls:Class)
            RETURN a.name as artifact,
                   count(DISTINCT fn) as functions,
                   count(DISTINCT cls) as classes,
                   (count(DISTINCT fn) + count(DISTINCT cls)) as complexity_score
            ORDER BY complexity_score DESC
            """
            complexity_results = pipeline.query_graph(complexity_query)
            
            return {
                "size_ranking": [
                    {
                        "artifact": result["artifact"],
                        "total_files": result["total_files"],
                        "total_size_mb": result["total_size"] / (1024 * 1024) if result["total_size"] else 0
                    }
                    for result in size_results
                ],
                "complexity_ranking": [
                    {
                        "artifact": result["artifact"],
                        "functions": result["functions"],
                        "classes": result["classes"],
                        "complexity_score": result["complexity_score"]
                    }
                    for result in complexity_results
                ]
            }
            
        except Exception as e:
            logger.warning(f"Error performing comparative analysis: {e}")
            return {"error": str(e)}
    
    def _export_results(self, batch_result: Dict[str, Any]):
        """Export processing results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export batch results
        batch_file = self.output_dir / f"icse_batch_results_{timestamp}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, indent=2, default=str)
        
        # Export comprehensive analysis
        analysis_file = self.output_dir / f"icse_comprehensive_analysis_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_report, f, indent=2, default=str)
        
        # Export summary report
        self._export_summary_report(timestamp)
        
        logger.info(f"Results exported to: {self.output_dir}")
        logger.info(f"  - Batch results: {batch_file}")
        logger.info(f"  - Analysis: {analysis_file}")
    
    def _export_summary_report(self, timestamp: str):
        """Export a human-readable summary report."""
        summary_file = self.output_dir / f"icse_summary_report_{timestamp}.md"
        
        report = []
        report.append("# ICSE Artifacts Analysis Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Artifacts Processed: {len(self.artifacts_found)}")
        
        # Processing summary
        successful = sum(1 for r in self.processing_results if r.get("success", False))
        failed = len(self.processing_results) - successful
        
        report.append(f"\n## Processing Summary")
        report.append(f"- Successful: {successful}")
        report.append(f"- Failed: {failed}")
        report.append(f"- Success Rate: {(successful / len(self.processing_results) * 100):.1f}%")
        
        # Artifacts overview
        report.append(f"\n## Artifacts Overview")
        for artifact in self.artifacts_found:
            size_mb = artifact["size"] / (1024 * 1024)
            report.append(f"- **{artifact['name']}**: {artifact['format']} ({size_mb:.1f}MB)")
        
        # Programming languages
        if "programming_languages" in self.analysis_report:
            report.append(f"\n## Programming Languages")
            for ext, data in self.analysis_report["programming_languages"].items():
                report.append(f"- **{ext}**: {data['file_count']} files in {data['artifact_count']} artifacts")
        
        # Quality metrics summary
        if "quality_metrics" in self.analysis_report:
            report.append(f"\n## Quality Metrics Summary")
            
            # Documentation coverage
            doc_coverage = self.analysis_report["quality_metrics"].get("documentation_coverage", {})
            if doc_coverage:
                avg_doc_coverage = sum(
                    data["coverage_ratio"] for data in doc_coverage.values()
                ) / len(doc_coverage)
                report.append(f"- Average Documentation Coverage: {avg_doc_coverage:.2%}")
            
            # Test coverage
            test_coverage = self.analysis_report["quality_metrics"].get("test_coverage", {})
            if test_coverage:
                avg_test_coverage = sum(
                    data["test_ratio"] for data in test_coverage.values()
                ) / len(test_coverage)
                report.append(f"- Average Test Coverage: {avg_test_coverage:.2%}")
        
        # Top artifacts by complexity
        if "comparative_analysis" in self.analysis_report:
            complexity_ranking = self.analysis_report["comparative_analysis"].get("complexity_ranking", [])
            if complexity_ranking:
                report.append(f"\n## Top 5 Most Complex Artifacts")
                for i, artifact in enumerate(complexity_ranking[:5], 1):
                    report.append(f"{i}. **{artifact['artifact']}**: {artifact['complexity_score']} (Functions: {artifact['functions']}, Classes: {artifact['classes']})")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"  - Summary report: {summary_file}")


def main():
    """Main entry point for ICSE artifacts processing."""
    parser = argparse.ArgumentParser(
        description="Process ICSE artifacts using Robust Knowledge Graph Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover artifacts only (dry run)
  python icse_artifacts_processor.py icse_artifacts/ --dry-run
  
  # Process all artifacts with default settings
  python icse_artifacts_processor.py icse_artifacts/
  
  # Process with custom Neo4j settings and increased workers
  python icse_artifacts_processor.py icse_artifacts/ \\
    --neo4j-uri bolt://localhost:7687 \\
    --neo4j-user neo4j \\
    --neo4j-password mypassword \\
    --max-workers 8 \\
    --output-dir ./my_analysis
        """
    )
    
    parser.add_argument(
        "artifacts_dir",
        help="Directory containing ICSE artifacts (zip files, tar.gz, or git repositories)"
    )
    
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j database URI (default: bolt://localhost:7687)"
    )
    
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    
    parser.add_argument(
        "--neo4j-password",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./icse_analysis_output",
        help="Output directory for results (default: ./icse_analysis_output)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers"
    )
    
    parser.add_argument(
        "--config-profile",
        default="default",
        choices=["default", "development", "production", "testing"],
        help="Configuration profile to use (default: default)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover artifacts only, don't process them"
    )
    
    parser.add_argument(
        "--no-advanced-analysis",
        action="store_true",
        help="Disable advanced graph analysis (faster processing)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize processor
        processor = ICSEArtifactsProcessor(
            artifacts_dir=args.artifacts_dir,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            output_dir=args.output_dir,
            config_profile=args.config_profile
        )
        
        # Process artifacts
        result = processor.process_all_artifacts(
            max_workers=args.max_workers,
            enable_advanced_analysis=not args.no_advanced_analysis,
            dry_run=args.dry_run
        )
        
        if result.get("success", False) or result.get("dry_run", False):
            print("✅ ICSE artifacts processing completed successfully!")
            
            if args.dry_run:
                print(f"📊 Discovered {result['artifacts_discovered']} artifacts")
            else:
                summary = result.get("summary", {})
                print(f"📊 Processed: {summary.get('total_artifacts', 0)} artifacts")
                print(f"✅ Successful: {summary.get('successful_artifacts', 0)}")
                print(f"❌ Failed: {summary.get('failed_artifacts', 0)}")
                print(f"📈 Success Rate: {summary.get('success_rate', 'N/A')}")
                print(f"💾 Results saved to: {args.output_dir}")
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 