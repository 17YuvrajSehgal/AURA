"""
Algorithm 5: Robust Knowledge Graph Pipeline for Artifact Analysis

This module provides a comprehensive pipeline for processing software artifacts
of various formats (zip, tar.gz, git repositories) and creating robust knowledge graphs.

Key Features:
- Multi-format artifact extraction (zip, tar.gz, folders)
- Enhanced knowledge graph construction with Neo4j
- Batch processing with parallel execution
- Advanced graph analytics and pattern mining
- Export and visualization capabilities

Main Components:
- RobustKGPipeline: Main orchestrator class
- ArtifactExtractor: Handles various artifact formats
- EnhancedKGBuilder: Creates comprehensive knowledge graphs
- BatchProcessor: Handles parallel processing of multiple artifacts

Usage Example:
    from scripts.algorithm_5 import RobustKGPipeline
    
    with RobustKGPipeline(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    ) as pipeline:
        # Process single artifact
        result = pipeline.process_single_artifact("path/to/artifact.zip")
        
        # Or process directory of artifacts
        batch_result = pipeline.process_artifact_directory("icse_artifacts/")
        
        # Export visualization
        viz_path = pipeline.export_graph_visualization()
"""

__version__ = "1.0.0"
__author__ = "AURA Framework Team"

# Import main classes
try:
    from .robust_kg_pipeline import RobustKGPipeline
    from .artifact_extractor import RobustArtifactExtractor
    from .enhanced_kg_builder import EnhancedKGBuilder
    from .batch_processor import BatchProcessor

    # Alias for backward compatibility
    ArtifactExtractor = RobustArtifactExtractor

    __all__ = [
        "RobustKGPipeline",
        "ArtifactExtractor",
        "RobustArtifactExtractor",
        "EnhancedKGBuilder",
        "BatchProcessor"
    ]

except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings

    warnings.warn(f"Some dependencies are missing for algorithm_5: {e}")

    __all__ = []
