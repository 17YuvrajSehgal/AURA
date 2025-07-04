#!/usr/bin/env python3
"""
Robust Knowledge Graph Pipeline - Main Orchestrator

This module provides the main pipeline for processing software artifacts
and creating comprehensive knowledge graphs with enhanced features.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .artifact_extractor import ArtifactExtractor
from .enhanced_kg_builder import EnhancedKGBuilder
from .batch_processor import BatchProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustKGPipeline:
    """
    Main pipeline for robust knowledge graph creation from software artifacts.
    
    Features:
    - Multi-format artifact extraction (zip, tar.gz, folders)
    - Enhanced knowledge graph construction
    - Batch processing capabilities
    - Advanced graph analytics
    - Export and visualization options
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        working_dir: str = "./algo_outputs/algorithm_5_output",
        temp_dir: str = "./temp_extractions",
        enable_advanced_analysis: bool = True,
        clear_existing_graph: bool = False
    ):
        """
        Initialize the Robust Knowledge Graph Pipeline.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            working_dir: Directory for output files
            temp_dir: Temporary directory for extractions
            enable_advanced_analysis: Enable advanced graph analytics
            clear_existing_graph: Whether to clear existing graph data
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.working_dir = Path(working_dir)
        self.temp_dir = Path(temp_dir)
        self.enable_advanced_analysis = enable_advanced_analysis
        self.clear_existing_graph = clear_existing_graph
        
        # Create necessary directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = ArtifactExtractor(temp_dir=str(self.temp_dir))
        self.kg_builder = EnhancedKGBuilder(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            clear_existing=clear_existing_graph
        )
        self.batch_processor = BatchProcessor(
            extractor=self.extractor,
            kg_builder=self.kg_builder
        )
        
        # Pipeline state
        self.processed_artifacts = []
        self.pipeline_stats = {
            "start_time": None,
            "end_time": None,
            "total_artifacts": 0,
            "successful_extractions": 0,
            "successful_kg_builds": 0,
            "errors": []
        }
    
    def process_single_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single artifact (zip, tar.gz, or folder) into knowledge graph.
        
        Args:
            artifact_path: Path to the artifact
            artifact_name: Optional custom name for the artifact
            
        Returns:
            Dictionary containing processing results
        """
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        
        if artifact_name is None:
            artifact_name = artifact_path.stem
        
        logger.info(f"Processing artifact: {artifact_name}")
        
        result = {
            "artifact_name": artifact_name,
            "artifact_path": str(artifact_path),
            "processing_start": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "extraction_info": None,
            "kg_info": None,
            "analysis_results": None
        }
        
        try:
            # Step 1: Extract artifact
            logger.info(f"Extracting artifact: {artifact_name}")
            extraction_result = self.extractor.extract_artifact(
                artifact_path=str(artifact_path),
                artifact_name=artifact_name
            )
            result["extraction_info"] = extraction_result
            
            if not extraction_result["success"]:
                result["error"] = f"Extraction failed: {extraction_result.get('error', 'Unknown error')}"
                return result
            
            # Step 2: Build knowledge graph
            logger.info(f"Building knowledge graph for: {artifact_name}")
            kg_result = self.kg_builder.build_knowledge_graph(
                extracted_path=extraction_result["extracted_path"],
                artifact_name=artifact_name,
                metadata=extraction_result.get("metadata", {})
            )
            result["kg_info"] = kg_result
            
            if not kg_result["success"]:
                result["error"] = f"KG build failed: {kg_result.get('error', 'Unknown error')}"
                return result
            
            # Step 3: Advanced analysis (if enabled)
            if self.enable_advanced_analysis:
                logger.info(f"Performing advanced analysis for: {artifact_name}")
                analysis_result = self.kg_builder.perform_advanced_analysis(artifact_name)
                result["analysis_results"] = analysis_result
            
            result["success"] = True
            result["processing_end"] = datetime.now().isoformat()
            
            # Add to processed artifacts
            self.processed_artifacts.append(result)
            
            logger.info(f"Successfully processed artifact: {artifact_name}")
            
        except Exception as e:
            error_msg = f"Error processing {artifact_name}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            result["processing_end"] = datetime.now().isoformat()
        
        return result
    
    def process_artifact_directory(
        self,
        artifacts_dir: Union[str, Path],
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process all artifacts in a directory.
        
        Args:
            artifacts_dir: Directory containing artifacts
            file_patterns: Optional list of file patterns to match
            
        Returns:
            Dictionary containing batch processing results
        """
        artifacts_dir = Path(artifacts_dir)
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
        
        if file_patterns is None:
            file_patterns = ["*.zip", "*.tar.gz", "*.tgz", "*.tar", "*"]  # Default patterns
        
        logger.info(f"Processing artifacts directory: {artifacts_dir}")
        
        self.pipeline_stats["start_time"] = datetime.now().isoformat()
        
        # Use batch processor
        batch_result = self.batch_processor.process_directory(
            artifacts_dir=str(artifacts_dir),
            file_patterns=file_patterns,
            enable_advanced_analysis=self.enable_advanced_analysis
        )
        
        self.pipeline_stats["end_time"] = datetime.now().isoformat()
        self.pipeline_stats.update(batch_result["stats"])
        
        # Save results
        self._save_pipeline_results(batch_result)
        
        return batch_result
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        return self.kg_builder.get_graph_statistics()
    
    def export_graph_visualization(
        self,
        output_path: Optional[str] = None,
        format: str = "html"
    ) -> str:
        """
        Export graph visualization.
        
        Args:
            output_path: Optional output path
            format: Export format ('html', 'png', 'json')
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.working_dir / f"graph_visualization_{timestamp}.{format}")
        
        return self.kg_builder.export_visualization(output_path, format)
    
    def query_graph(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query on the knowledge graph."""
        return self.kg_builder.execute_query(cypher_query, parameters)
    
    def get_artifact_recommendations(self, artifact_name: str) -> Dict[str, Any]:
        """Get recommendations for improving an artifact based on graph analysis."""
        return self.kg_builder.get_artifact_recommendations(artifact_name)
    
    def cleanup(self):
        """Clean up temporary files and close connections."""
        try:
            # Clean up temporary extraction directory
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            
            # Close KG builder connections
            self.kg_builder.close()
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.working_dir / f"pipeline_results_{timestamp}.json"
        
        # Combine with pipeline stats
        full_results = {
            "pipeline_stats": self.pipeline_stats,
            "batch_results": results,
            "processed_artifacts": self.processed_artifacts,
            "graph_statistics": self.get_graph_statistics()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to: {output_file}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def main():
    """Example usage of the Robust KG Pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Knowledge Graph Pipeline")
    parser.add_argument("input_path", help="Path to artifact or directory of artifacts")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--output-dir", default="./algo_outputs/algorithm_5_output", help="Output directory")
    parser.add_argument("--clear-graph", action="store_true", help="Clear existing graph")
    parser.add_argument("--single-artifact", action="store_true", help="Process as single artifact")
    
    args = parser.parse_args()
    
    # Create pipeline
    with RobustKGPipeline(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        working_dir=args.output_dir,
        clear_existing_graph=args.clear_graph
    ) as pipeline:
        
        if args.single_artifact:
            # Process single artifact
            result = pipeline.process_single_artifact(args.input_path)
            print(f"Processing result: {result['success']}")
            if result['error']:
                print(f"Error: {result['error']}")
        else:
            # Process directory of artifacts
            result = pipeline.process_artifact_directory(args.input_path)
            print(f"Processed {result['stats']['total_artifacts']} artifacts")
            print(f"Successful: {result['stats']['successful_kg_builds']}")
        
        # Export visualization
        viz_path = pipeline.export_graph_visualization()
        print(f"Graph visualization saved to: {viz_path}")


if __name__ == "__main__":
    main() 