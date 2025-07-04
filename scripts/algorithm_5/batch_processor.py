#!/usr/bin/env python3
"""
Batch Processor for Knowledge Graph Pipeline

This module handles batch processing of multiple artifacts with:
- Parallel processing capabilities
- Progress tracking
- Error handling and recovery
- Statistical reporting
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Data class for processing results."""
    artifact_name: str
    success: bool
    processing_time: float
    error: Optional[str] = None
    nodes_created: int = 0
    relationships_created: int = 0
    file_count: int = 0


class BatchProcessor:
    """
    Handles batch processing of multiple artifacts with parallel execution.
    
    Features:
    - Parallel processing with configurable workers
    - Progress tracking and reporting
    - Error handling and recovery
    - Resource management
    - Statistical analysis
    """

    def __init__(
            self,
            extractor,
            kg_builder,
            max_workers: int = 4,
            use_processes: bool = False,
            timeout_per_artifact: int = 300  # 5 minutes
    ):
        """
        Initialize the BatchProcessor.
        
        Args:
            extractor: ArtifactExtractor instance
            kg_builder: EnhancedKGBuilder instance
            max_workers: Maximum number of parallel workers
            use_processes: Use processes instead of threads
            timeout_per_artifact: Timeout per artifact in seconds
        """
        self.extractor = extractor
        self.kg_builder = kg_builder
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.timeout_per_artifact = timeout_per_artifact

        # Progress tracking
        self.progress_lock = threading.Lock()
        self.processed_count = 0
        self.total_count = 0
        self.results = []

        # Statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_artifacts": 0,
            "successful_extractions": 0,
            "successful_kg_builds": 0,
            "failed_artifacts": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "errors": []
        }

    def process_directory(
            self,
            artifacts_dir: str,
            file_patterns: Optional[List[str]] = None,
            enable_advanced_analysis: bool = True,
            progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process all artifacts in a directory.
        
        Args:
            artifacts_dir: Directory containing artifacts
            file_patterns: List of file patterns to match
            enable_advanced_analysis: Enable advanced graph analysis
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing batch processing results
        """
        artifacts_dir = Path(artifacts_dir)
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

        logger.info(f"Starting batch processing of directory: {artifacts_dir}")
        self.stats["start_time"] = datetime.now()

        # Find all artifacts
        artifacts = self._find_artifacts(artifacts_dir, file_patterns or ["*"])
        self.total_count = len(artifacts)
        self.stats["total_artifacts"] = self.total_count

        if not artifacts:
            logger.warning(f"No artifacts found in {artifacts_dir}")
            return {
                "success": True,
                "artifacts_processed": [],
                "stats": self.stats,
                "message": "No artifacts found"
            }

        logger.info(f"Found {len(artifacts)} artifacts to process")

        # Process artifacts
        if self.use_processes:
            processed_results = self._process_with_processes(artifacts, enable_advanced_analysis, progress_callback)
        else:
            processed_results = self._process_with_threads(artifacts, enable_advanced_analysis, progress_callback)

        # Update final statistics
        self.stats["end_time"] = datetime.now()
        self._calculate_final_stats(processed_results)

        result = {
            "success": True,
            "artifacts_processed": processed_results,
            "stats": self.stats,
            "summary": self._generate_summary()
        }

        logger.info(f"Batch processing completed. Processed {len(processed_results)} artifacts")
        return result

    def _find_artifacts(self, artifacts_dir: Path, file_patterns: List[str]) -> List[Dict[str, Any]]:
        """Find all artifacts matching the given patterns."""
        artifacts = []

        for pattern in file_patterns:
            if pattern == "*":
                # Include directories and common archive formats
                for item in artifacts_dir.iterdir():
                    if item.is_dir():
                        artifacts.append({
                            "path": str(item),
                            "name": item.name,
                            "type": "directory"
                        })
                    elif item.suffix.lower() in ['.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz']:
                        artifacts.append({
                            "path": str(item),
                            "name": item.stem,
                            "type": "archive"
                        })
            else:
                # Use glob pattern
                for item in artifacts_dir.glob(pattern):
                    if item.is_file() or item.is_dir():
                        artifacts.append({
                            "path": str(item),
                            "name": item.stem if item.is_file() else item.name,
                            "type": "archive" if item.is_file() else "directory"
                        })

        # Remove duplicates
        seen_paths = set()
        unique_artifacts = []
        for artifact in artifacts:
            if artifact["path"] not in seen_paths:
                unique_artifacts.append(artifact)
                seen_paths.add(artifact["path"])

        return unique_artifacts

    def _process_with_threads(
            self,
            artifacts: List[Dict],
            enable_advanced_analysis: bool,
            progress_callback: Optional[callable]
    ) -> List[Dict]:
        """Process artifacts using thread pool."""
        processed_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_artifact = {
                executor.submit(
                    self._process_single_artifact,
                    artifact,
                    enable_advanced_analysis
                ): artifact
                for artifact in artifacts
            }

            # Collect results as they complete
            for future in as_completed(future_to_artifact, timeout=self.timeout_per_artifact * len(artifacts)):
                artifact = future_to_artifact[future]

                try:
                    result = future.result(timeout=self.timeout_per_artifact)
                    processed_results.append(result)

                    with self.progress_lock:
                        self.processed_count += 1
                        if progress_callback:
                            progress_callback(
                                f"Processed {self.processed_count}/{self.total_count}: {artifact['name']}"
                            )

                    logger.info(f"Completed processing: {artifact['name']}")

                except Exception as e:
                    error_msg = f"Failed to process {artifact['name']}: {str(e)}"
                    logger.error(error_msg)

                    processed_results.append({
                        "artifact_name": artifact['name'],
                        "artifact_path": artifact['path'],
                        "success": False,
                        "error": error_msg,
                        "processing_time": 0
                    })

                    self.stats["errors"].append(error_msg)

        return processed_results

    def _process_with_processes(
            self,
            artifacts: List[Dict],
            enable_advanced_analysis: bool,
            progress_callback: Optional[callable]
    ) -> List[Dict]:
        """Process artifacts using process pool."""
        # Note: Process-based execution would require careful handling of
        # shared resources like Neo4j connections. For now, falling back to threads.
        logger.warning("Process-based execution not fully implemented, using threads")
        return self._process_with_threads(artifacts, enable_advanced_analysis, progress_callback)

    def _process_single_artifact(
            self,
            artifact: Dict[str, Any],
            enable_advanced_analysis: bool
    ) -> Dict[str, Any]:
        """Process a single artifact through the entire pipeline."""
        start_time = time.time()

        result = {
            "artifact_name": artifact["name"],
            "artifact_path": artifact["path"],
            "artifact_type": artifact["type"],
            "success": False,
            "processing_start": datetime.now().isoformat(),
            "extraction_info": None,
            "kg_info": None,
            "analysis_results": None,
            "error": None,
            "processing_time": 0
        }

        try:
            # Step 1: Extract artifact
            logger.debug(f"Extracting artifact: {artifact['name']}")
            extraction_result = self.extractor.extract_artifact(
                artifact_path=artifact["path"],
                artifact_name=artifact["name"]
            )
            result["extraction_info"] = extraction_result

            if not extraction_result["success"]:
                result["error"] = f"Extraction failed: {extraction_result.get('error', 'Unknown error')}"
                return result

            # Step 2: Build knowledge graph
            logger.debug(f"Building knowledge graph for: {artifact['name']}")
            kg_result = self.kg_builder.build_knowledge_graph(
                extracted_path=extraction_result["extracted_path"],
                artifact_name=artifact["name"],
                metadata=extraction_result.get("metadata", {})
            )
            result["kg_info"] = kg_result

            if not kg_result["success"]:
                result["error"] = f"KG build failed: {kg_result.get('error', 'Unknown error')}"
                return result

            # Step 3: Advanced analysis (if enabled)
            if enable_advanced_analysis:
                logger.debug(f"Performing advanced analysis for: {artifact['name']}")
                analysis_result = self.kg_builder.perform_advanced_analysis(artifact["name"])
                result["analysis_results"] = analysis_result

            # Update statistics
            self._update_success_stats(result)

            result["success"] = True
            logger.debug(f"Successfully processed artifact: {artifact['name']}")

        except Exception as e:
            error_msg = f"Error processing {artifact['name']}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            self.stats["errors"].append(error_msg)

        finally:
            result["processing_time"] = time.time() - start_time
            result["processing_end"] = datetime.now().isoformat()

        return result

    def _update_success_stats(self, result: Dict[str, Any]):
        """Update statistics for successful processing."""
        with self.progress_lock:
            if result.get("extraction_info", {}).get("success"):
                self.stats["successful_extractions"] += 1

            if result.get("kg_info", {}).get("success"):
                self.stats["successful_kg_builds"] += 1

    def _calculate_final_stats(self, processed_results: List[Dict]):
        """Calculate final statistics from all results."""
        if not processed_results:
            return

        # Count failures
        self.stats["failed_artifacts"] = sum(
            1 for result in processed_results if not result["success"]
        )

        # Calculate processing times
        processing_times = [
            result["processing_time"] for result in processed_results
            if "processing_time" in result and result["processing_time"] > 0
        ]

        if processing_times:
            self.stats["total_processing_time"] = sum(processing_times)
            self.stats["average_processing_time"] = sum(processing_times) / len(processing_times)

        # Calculate success rates
        total = len(processed_results)
        self.stats["extraction_success_rate"] = (
            self.stats["successful_extractions"] / total if total > 0 else 0
        )
        self.stats["kg_build_success_rate"] = (
            self.stats["successful_kg_builds"] / total if total > 0 else 0
        )
        self.stats["overall_success_rate"] = (
            sum(1 for r in processed_results if r["success"]) / total if total > 0 else 0
        )

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the batch processing results."""
        duration = None
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        summary = {
            "total_artifacts": self.stats["total_artifacts"],
            "successful_artifacts": self.stats["successful_kg_builds"],
            "failed_artifacts": self.stats["failed_artifacts"],
            "success_rate": f"{self.stats.get('overall_success_rate', 0) * 100:.1f}%",
            "total_duration_seconds": duration,
            "average_processing_time": f"{self.stats.get('average_processing_time', 0):.2f}s",
            "errors_count": len(self.stats["errors"])
        }

        return summary

    def process_artifact_list(
            self,
            artifact_paths: List[str],
            enable_advanced_analysis: bool = True,
            progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process a specific list of artifact paths.
        
        Args:
            artifact_paths: List of paths to artifacts
            enable_advanced_analysis: Enable advanced graph analysis
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing batch processing results
        """
        # Convert paths to artifact dictionaries
        artifacts = []
        for path in artifact_paths:
            path_obj = Path(path)
            if path_obj.exists():
                artifacts.append({
                    "path": str(path_obj),
                    "name": path_obj.stem if path_obj.is_file() else path_obj.name,
                    "type": "archive" if path_obj.is_file() else "directory"
                })
            else:
                logger.warning(f"Artifact not found: {path}")

        if not artifacts:
            return {
                "success": False,
                "error": "No valid artifacts found in the provided list"
            }

        self.total_count = len(artifacts)
        self.stats["total_artifacts"] = self.total_count
        self.stats["start_time"] = datetime.now()

        # Process artifacts
        processed_results = self._process_with_threads(artifacts, enable_advanced_analysis, progress_callback)

        # Update final statistics
        self.stats["end_time"] = datetime.now()
        self._calculate_final_stats(processed_results)

        return {
            "success": True,
            "artifacts_processed": processed_results,
            "stats": self.stats,
            "summary": self._generate_summary()
        }

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        with self.progress_lock:
            return {
                "processed_count": self.processed_count,
                "total_count": self.total_count,
                "progress_percentage": (
                    (self.processed_count / self.total_count * 100)
                    if self.total_count > 0 else 0
                ),
                "is_running": self.processed_count < self.total_count,
                "successful_extractions": self.stats["successful_extractions"],
                "successful_kg_builds": self.stats["successful_kg_builds"],
                "errors_count": len(self.stats["errors"])
            }

    def export_results(self, output_path: str, include_detailed_errors: bool = True) -> str:
        """
        Export processing results to JSON file.
        
        Args:
            output_path: Path for the output file
            include_detailed_errors: Include detailed error information
            
        Returns:
            Path to the exported file
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "batch_stats": self.stats,
            "summary": self._generate_summary(),
            "results": self.results
        }

        if not include_detailed_errors:
            # Remove detailed error information
            for result in export_data.get("results", []):
                if "error" in result:
                    result["error"] = "Error occurred (details omitted)"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Results exported to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return ""

    def generate_report(self) -> str:
        """Generate a human-readable processing report."""
        summary = self._generate_summary()

        report = []
        report.append("=" * 60)
        report.append("BATCH PROCESSING REPORT")
        report.append("=" * 60)

        report.append(f"Total Artifacts: {summary['total_artifacts']}")
        report.append(f"Successfully Processed: {summary['successful_artifacts']}")
        report.append(f"Failed: {summary['failed_artifacts']}")
        report.append(f"Success Rate: {summary['success_rate']}")

        if summary.get('total_duration_seconds'):
            report.append(f"Total Duration: {summary['total_duration_seconds']:.1f} seconds")

        report.append(f"Average Processing Time: {summary['average_processing_time']}")

        if self.stats["errors"]:
            report.append(f"\nErrors ({len(self.stats['errors'])}):")
            for i, error in enumerate(self.stats["errors"][:5], 1):  # Show first 5 errors
                report.append(f"  {i}. {error}")

            if len(self.stats["errors"]) > 5:
                report.append(f"  ... and {len(self.stats['errors']) - 5} more errors")

        report.append("=" * 60)

        return "\n".join(report)


def main():
    """Example usage of the BatchProcessor."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch Processor for Knowledge Graph Pipeline")
    parser.add_argument("artifacts_dir", help="Directory containing artifacts")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of workers")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per artifact (seconds)")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--no-advanced-analysis", action="store_true", help="Disable advanced analysis")

    args = parser.parse_args()

    # This would require actual extractor and kg_builder instances
    print("BatchProcessor requires initialized extractor and kg_builder instances")
    print("Use the main RobustKGPipeline instead for complete functionality")


if __name__ == "__main__":
    main()
