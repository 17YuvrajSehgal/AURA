#!/usr/bin/env python3
"""
Example Usage Script for Algorithm 5: Robust Knowledge Graph Pipeline

This script demonstrates various features and usage patterns of the
Robust Knowledge Graph Pipeline including:
- Single artifact processing
- Batch processing
- Configuration management
- Custom queries
- Visualization export
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to path to import algorithm_5
sys.path.append(str(Path(__file__).parent.parent))

try:
    from algorithm_5 import RobustKGPipeline
    from algorithm_5.config import load_config, create_sample_config
except ImportError as e:
    print(f"Error importing algorithm_5: {e}")
    print("Make sure you have installed all dependencies from requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_artifacts(output_dir: str = "./sample_artifacts") -> List[str]:
    """
    Create sample artifacts for demonstration purposes.
    
    Args:
        output_dir: Directory to create sample artifacts
        
    Returns:
        List of created artifact paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    artifact_paths = []
    
    # Create a sample Python project structure
    sample_project = output_path / "sample_python_project"
    sample_project.mkdir(exist_ok=True)
    
    # Create README.md
    readme_content = """# Sample Python Project

This is a sample Python project for demonstrating the Knowledge Graph Pipeline.

## Features
- Data processing utilities
- Configuration management
- Testing framework

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from sample_project import DataProcessor
processor = DataProcessor()
result = processor.process_data("input.csv")
```
"""
    (sample_project / "README.md").write_text(readme_content)
    
    # Create requirements.txt
    requirements = """numpy>=1.21.0
pandas>=1.3.0
pytest>=6.2.0
click>=8.0.0
"""
    (sample_project / "requirements.txt").write_text(requirements)
    
    # Create Python source files
    (sample_project / "src").mkdir(exist_ok=True)
    
    main_py = """#!/usr/bin/env python3
\"\"\"
Main module for the sample project.
\"\"\"

import argparse
import logging
from pathlib import Path
from .data_processor import DataProcessor
from .config import load_config

logger = logging.getLogger(__name__)


def main():
    \"\"\"Main entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Sample Python Project")
    parser.add_argument("input_file", help="Input data file")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--config", "-c", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process data
    processor = DataProcessor(config)
    result = processor.process_data(args.input_file)
    
    # Save result
    if args.output:
        result.to_csv(args.output)
    else:
        print(result)


if __name__ == "__main__":
    main()
"""
    (sample_project / "src" / "__init__.py").write_text("")
    (sample_project / "src" / "main.py").write_text(main_py)
    
    data_processor_py = """\"\"\"
Data processing utilities.
\"\"\"

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    \"\"\"Handles data processing operations.\"\"\"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        \"\"\"
        Initialize DataProcessor.
        
        Args:
            config: Configuration dictionary
        \"\"\"
        self.config = config or {}
        self.processed_count = 0
    
    def process_data(self, input_file: str) -> pd.DataFrame:
        \"\"\"
        Process input data file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            Processed DataFrame
        \"\"\"
        logger.info(f"Processing file: {input_file}")
        
        try:
            # Read data
            if input_file.endswith('.csv'):
                data = pd.read_csv(input_file)
            elif input_file.endswith('.json'):
                data = pd.read_json(input_file)
            else:
                raise ValueError(f"Unsupported file format: {input_file}")
            
            # Apply transformations
            processed_data = self._apply_transformations(data)
            
            self.processed_count += 1
            logger.info(f"Successfully processed {len(processed_data)} records")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            raise
    
    def _apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Apply data transformations.\"\"\"
        # Simple transformations for demonstration
        result = data.copy()
        
        # Add computed columns
        if 'value' in result.columns:
            result['value_squared'] = result['value'] ** 2
            result['value_normalized'] = (result['value'] - result['value'].mean()) / result['value'].std()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        \"\"\"Get processing statistics.\"\"\"
        return {
            'processed_count': self.processed_count,
            'config': self.config
        }
"""
    (sample_project / "src" / "data_processor.py").write_text(data_processor_py)
    
    config_py = """\"\"\"
Configuration management.
\"\"\"

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"
    Load configuration from file.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        Configuration dictionary
    \"\"\"
    default_config = {
        'processing': {
            'batch_size': 1000,
            'parallel': True,
            'workers': 4
        },
        'output': {
            'format': 'csv',
            'compression': None
        }
    }
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            file_config = json.load(f)
        
        # Merge configurations
        default_config.update(file_config)
    
    return default_config
"""
    (sample_project / "src" / "config.py").write_text(config_py)
    
    # Create test files
    (sample_project / "tests").mkdir(exist_ok=True)
    (sample_project / "tests" / "__init__.py").write_text("")
    
    test_data_processor = """\"\"\"
Tests for data processor.
\"\"\"

import pytest
import pandas as pd
from src.data_processor import DataProcessor


class TestDataProcessor:
    \"\"\"Test cases for DataProcessor.\"\"\"
    
    def test_init(self):
        \"\"\"Test DataProcessor initialization.\"\"\"
        processor = DataProcessor()
        assert processor.processed_count == 0
        assert processor.config == {}
    
    def test_init_with_config(self):
        \"\"\"Test DataProcessor initialization with config.\"\"\"
        config = {'test': 'value'}
        processor = DataProcessor(config)
        assert processor.config == config
    
    def test_apply_transformations(self):
        \"\"\"Test data transformations.\"\"\"
        processor = DataProcessor()
        data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        
        result = processor._apply_transformations(data)
        
        assert 'value_squared' in result.columns
        assert 'value_normalized' in result.columns
        assert len(result) == len(data)
    
    def test_get_stats(self):
        \"\"\"Test statistics retrieval.\"\"\"
        processor = DataProcessor()
        stats = processor.get_stats()
        
        assert 'processed_count' in stats
        assert 'config' in stats
        assert stats['processed_count'] == 0
"""
    (sample_project / "tests" / "test_data_processor.py").write_text(test_data_processor)
    
    # Create LICENSE file
    license_content = """MIT License

Copyright (c) 2024 Sample Python Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    (sample_project / "LICENSE").write_text(license_content)
    
    artifact_paths.append(str(sample_project))
    
    logger.info(f"Created sample artifacts in: {output_dir}")
    return artifact_paths


def example_single_artifact_processing():
    """Demonstrate processing a single artifact."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Artifact Processing")
    print("="*60)
    
    # Create sample artifacts
    artifact_paths = create_sample_artifacts()
    
    # Initialize pipeline with configuration
    config = load_config(profile="development")
    
    try:
        with RobustKGPipeline(
            neo4j_uri=config.get("neo4j", "uri"),
            neo4j_user=config.get("neo4j", "user"),
            neo4j_password=config.get("neo4j", "12345678"),
            working_dir=config.get("pipeline", "working_dir"),
            enable_advanced_analysis=True
        ) as pipeline:
            
            # Process single artifact
            result = pipeline.process_single_artifact(
                artifact_path=artifact_paths[0],
                artifact_name="sample_python_project"
            )
            
            if result["success"]:
                print(f"✅ Successfully processed: {result['artifact_name']}")
                print(f"   Processing time: {result.get('processing_time', 0):.2f} seconds")
                
                if result.get("kg_info"):
                    kg_info = result["kg_info"]
                    print(f"   Nodes created: {kg_info.get('nodes_created', 0)}")
                    print(f"   Relationships created: {kg_info.get('relationships_created', 0)}")
                
                # Get graph statistics
                stats = pipeline.get_graph_statistics()
                print(f"   Total graph nodes: {stats.get('total_nodes', 0)}")
                print(f"   Total relationships: {stats.get('total_relationships', 0)}")
                
                # Export visualization
                viz_path = pipeline.export_graph_visualization(
                    format="html"
                )
                if viz_path:
                    print(f"   Visualization saved to: {viz_path}")
                
            else:
                print(f"❌ Failed to process: {result['artifact_name']}")
                print(f"   Error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Pipeline error: {e}")


def example_batch_processing():
    """Demonstrate batch processing of multiple artifacts."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60)
    
    # Create multiple sample artifacts
    sample_dir = "./sample_artifacts"
    create_sample_artifacts(sample_dir)
    
    # Create a second sample project
    sample_project_2 = Path(sample_dir) / "sample_project_2"
    sample_project_2.mkdir(exist_ok=True)
    
    (sample_project_2 / "README.md").write_text("# Sample Project 2\n\nAnother sample project.")
    (sample_project_2 / "main.js").write_text("console.log('Hello from JavaScript!');")
    (sample_project_2 / "package.json").write_text('{"name": "sample-project-2", "version": "1.0.0"}')
    
    try:
        with RobustKGPipeline(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="12345678",
            clear_existing_graph=True  # Clear for clean batch processing
        ) as pipeline:
            
            # Process directory of artifacts
            batch_result = pipeline.process_artifact_directory(
                artifacts_dir=sample_dir,
                file_patterns=["*"]  # Process all directories
            )
            
            if batch_result["success"]:
                summary = batch_result["summary"]
                print(f"✅ Batch processing completed!")
                print(f"   Total artifacts: {summary['total_artifacts']}")
                print(f"   Successful: {summary['successful_artifacts']}")
                print(f"   Failed: {summary['failed_artifacts']}")
                print(f"   Success rate: {summary['success_rate']}")
                print(f"   Total duration: {summary.get('total_duration_seconds', 0):.1f} seconds")
                
                # Show detailed results
                for artifact in batch_result["artifacts_processed"]:
                    status = "✅" if artifact["success"] else "❌"
                    print(f"   {status} {artifact['artifact_name']}: {artifact.get('processing_time', 0):.2f}s")
            
            else:
                print(f"❌ Batch processing failed")
    
    except Exception as e:
        print(f"❌ Batch processing error: {e}")


def example_custom_queries():
    """Demonstrate custom Cypher queries."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Queries")
    print("="*60)
    
    try:
        with RobustKGPipeline(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="12345678"
        ) as pipeline:
            
            # Query 1: Get all artifacts
            print("📊 All artifacts in the database:")
            artifacts = pipeline.query_graph("""
                MATCH (a:Artifact)
                RETURN a.name as artifact_name, a.created_at as created_at
                ORDER BY a.created_at DESC
            """)
            
            for artifact in artifacts:
                print(f"   - {artifact['artifact_name']} (created: {artifact['created_at'][:19]})")
            
            # Query 2: Get code files by programming language
            print("\n📊 Code files by programming language:")
            code_files = pipeline.query_graph("""
                MATCH (a:Artifact)-[:CONTAINS*]->(f:File {type: 'code'})
                RETURN f.extension as extension, count(f) as file_count
                ORDER BY file_count DESC
            """)
            
            for result in code_files:
                print(f"   {result['extension']}: {result['file_count']} files")
            
            # Query 3: Get functions and classes
            print("\n📊 Code structure:")
            code_structure = pipeline.query_graph("""
                MATCH (a:Artifact)-[:CONTAINS*]->(f:File)-[:DEFINES]->(c)
                WHERE c:Function OR c:Class
                RETURN labels(c)[0] as type, count(c) as count
            """)
            
            for result in code_structure:
                print(f"   {result['type']}: {result['count']}")
            
            # Query 4: Get most connected files
            print("\n📊 Most connected files:")
            connected_files = pipeline.query_graph("""
                MATCH (f:File)-[r]-()
                RETURN f.name as filename, count(r) as connections
                ORDER BY connections DESC
                LIMIT 5
            """)
            
            for result in connected_files:
                print(f"   {result['filename']}: {result['connections']} connections")
    
    except Exception as e:
        print(f"❌ Query error: {e}")


def example_configuration_management():
    """Demonstrate configuration management."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Configuration Management")
    print("="*60)
    
    # Create sample configuration file
    print("📁 Creating sample configuration file...")
    create_sample_config("sample_config.json")
    
    # Load configuration with different profiles
    configs = {}
    for profile in ["default", "development", "production"]:
        try:
            config = load_config(profile=profile, config_file="sample_config.json")
            configs[profile] = config
            print(f"✅ Loaded {profile} configuration")
            print(f"   Neo4j URI: {config.get('neo4j', 'uri')}")
            print(f"   Max workers: {config.get('batch', 'max_workers')}")
            print(f"   Working dir: {config.get('pipeline', 'working_dir')}")
        except Exception as e:
            print(f"❌ Failed to load {profile} config: {e}")
    
    # Demonstrate environment variable override
    print("\n🔧 Environment variable configuration:")
    original_uri = os.environ.get("NEO4J_URI")
    os.environ["NEO4J_URI"] = "bolt://custom-neo4j:7687"
    
    try:
        config = load_config()
        print(f"   Neo4j URI from env: {config.get('neo4j', 'uri')}")
    finally:
        # Restore original value
        if original_uri:
            os.environ["NEO4J_URI"] = original_uri
        else:
            os.environ.pop("NEO4J_URI", None)


def example_advanced_analytics():
    """Demonstrate advanced analytics features."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Advanced Analytics")
    print("="*60)
    
    try:
        with RobustKGPipeline(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="12345678"
        ) as pipeline:
            
            # Get artifacts for analysis
            artifacts = pipeline.query_graph("MATCH (a:Artifact) RETURN a.name as name LIMIT 1")
            
            if not artifacts:
                print("❌ No artifacts found. Run single artifact processing first.")
                return
            
            artifact_name = artifacts[0]["name"]
            print(f"📊 Analyzing artifact: {artifact_name}")
            
            # Perform advanced analysis
            analysis = pipeline.kg_builder.perform_advanced_analysis(artifact_name)
            
            if analysis:
                print("✅ Advanced analysis completed!")
                
                # Show network metrics
                if "network_metrics" in analysis:
                    metrics = analysis["network_metrics"]
                    print(f"   Network density: {metrics.get('density', 'N/A')}")
                    print(f"   Network diameter: {metrics.get('diameter', 'N/A')}")
                
                # Show centrality analysis
                if "centrality" in analysis:
                    centrality = analysis["centrality"]
                    most_central = centrality.get("most_central", [])
                    if most_central:
                        print(f"   Most central files: {', '.join(most_central[:3])}")
                
                # Show community detection
                if "communities" in analysis:
                    communities = analysis["communities"]
                    print(f"   Communities detected: {communities.get('communities', 'N/A')}")
                    print(f"   Modularity score: {communities.get('modularity', 'N/A')}")
            
            # Get recommendations
            recommendations = pipeline.get_artifact_recommendations(artifact_name)
            print(f"\n💡 Recommendations for {artifact_name}:")
            
            for category, suggestions in recommendations.items():
                if suggestions:
                    print(f"   {category.title()}:")
                    for suggestion in suggestions:
                        print(f"     - {suggestion}")
    
    except Exception as e:
        print(f"❌ Advanced analytics error: {e}")


def main():
    """Run all examples."""
    print("🚀 Algorithm 5: Robust Knowledge Graph Pipeline - Examples")
    print("=" * 60)
    
    # Check if Neo4j is available
    try:
        import py2neo
        graph = py2neo.Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
        graph.run("RETURN 1")
        print("✅ Neo4j connection successful")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("Please ensure Neo4j is running and accessible.")
        print("You can start Neo4j with Docker:")
        print("  docker run -p 7474:7474 -p 7687:7687 --env NEO4J_AUTH=neo4j/password neo4j")
        return
    
    # Run examples
    try:
        example_single_artifact_processing()
        example_batch_processing()
        example_custom_queries()
        example_configuration_management()
        example_advanced_analytics()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        try:
            if Path("./sample_artifacts").exists():
                shutil.rmtree("./sample_artifacts")
            if Path("sample_config.json").exists():
                Path("sample_config.json").unlink()
            print("🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    main() 