from scripts.algorithm_5 import RobustKGPipeline

# Algorithm 5: Robust Knowledge Graph Pipeline

A comprehensive pipeline for processing software artifacts and creating robust knowledge graphs with advanced analytics capabilities.

## Overview

Algorithm 5 provides a complete solution for analyzing software artifacts (zip files, tar.gz archives, and git repositories) and building comprehensive knowledge graphs using Neo4j. The pipeline includes multi-format extraction, parallel processing, advanced graph analytics, and interactive visualizations.

## Key Features

### 🔧 Multi-Format Artifact Processing
- **ZIP Archives**: `.zip` files
- **TAR Archives**: `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`
- **Git Repositories**: Local cloned directories
- **Nested Archives**: Automatic detection and extraction

### 📊 Knowledge Graph Construction
- **File Structure Analysis**: Directory hierarchies and file relationships
- **Code Analysis**: Functions, classes, imports, and dependencies
- **Documentation Processing**: README files, comments, and documentation sections
- **Dependency Mapping**: Requirements, package.json, pom.xml, etc.
- **Metadata Extraction**: File types, sizes, modification dates

### ⚡ Advanced Analytics
- **Pattern Mining**: Common development patterns and structures
- **Centrality Analysis**: Most important files and components
- **Community Detection**: Related code modules and packages
- **Similarity Metrics**: Compare artifacts and find similar projects
- **Quality Metrics**: Code complexity, documentation coverage

### 🚀 Parallel Processing
- **Batch Processing**: Handle multiple artifacts simultaneously
- **Configurable Workers**: Scale processing based on resources
- **Progress Tracking**: Real-time processing status
- **Error Recovery**: Robust error handling and reporting

### 📈 Visualization & Export
- **Interactive HTML**: Pyvis-based network visualizations
- **JSON Export**: Graph data for external analysis
- **Statistical Reports**: Comprehensive processing summaries
- **Custom Queries**: Cypher query interface

## Installation

### Prerequisites

1. **Neo4j Database**: Install and configure Neo4j
   ```bash
   # Download from https://neo4j.com/download/
   # Or use Docker:
   docker run -p 7474:7474 -p 7687:7687 --env NEO4J_AUTH=neo4j/password neo4j
   ```

2. **Python Dependencies**: Install required packages
   ```bash
   pip install -r scripts/algorithm_5/requirements.txt
   ```

### Required Python Packages
- py2neo>=2021.2.3
- pandas>=1.3.0
- networkx>=2.6
- pyvis>=0.1.9
- matplotlib>=3.4.0

## Quick Start

### 1. Basic Usage

```python
from scripts.algorithm_5 import RobustKGPipeline

# Initialize pipeline
with RobustKGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
) as pipeline:
    
    # Process single artifact
    result = pipeline.process_single_artifact("path/to/artifact.zip")
    print(f"Success: {result['success']}")
    
    # Export visualization
    viz_path = pipeline.export_graph_visualization()
    print(f"Visualization saved to: {viz_path}")
```

### 2. Batch Processing

```python
# Process directory of artifacts
with RobustKGPipeline() as pipeline:
    batch_result = pipeline.process_artifact_directory(
        artifacts_dir="icse_artifacts/",
        file_patterns=["*.zip", "*.tar.gz"]
    )
    
    print(f"Processed: {batch_result['stats']['total_artifacts']} artifacts")
    print(f"Success rate: {batch_result['stats']['overall_success_rate']:.2%}")
```

### 3. Configuration

```python
from scripts.algorithm_5.config import load_config

# Load configuration
config = load_config(profile="development")

# Create pipeline with custom config
pipeline = RobustKGPipeline(
    neo4j_uri=config.get("neo4j", "uri"),
    neo4j_user=config.get("neo4j", "user"),
    neo4j_password=config.get("neo4j", "password"),
    working_dir=config.get("pipeline", "working_dir"),
    enable_advanced_analysis=config.get("pipeline", "enable_advanced_analysis")
)
```

## Configuration

### Environment Variables

```bash
# Neo4j Configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# Pipeline Configuration
export PIPELINE_WORKING_DIR="./output"
export BATCH_MAX_WORKERS="8"
export LOG_LEVEL="INFO"
```

### Configuration File

Create `algorithm_5_config.json`:

```json
{
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "password"
  },
  "pipeline": {
    "working_dir": "./algo_outputs/algorithm_5_output",
    "enable_advanced_analysis": true,
    "timeout_per_artifact": 300
  },
  "batch": {
    "max_workers": 4,
    "use_processes": false
  }
}
```

## Command Line Usage

### Process Single Artifact

```bash
python -m scripts.algorithm_5.robust_kg_pipeline path/to/artifact.zip \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password \
    --output-dir ./output \
    --single-artifact
```

### Process Directory

```bash
python -m scripts.algorithm_5.robust_kg_pipeline icse_artifacts/ \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password \
    --output-dir ./output
```

## Advanced Features

### Custom Queries

```python
# Execute custom Cypher queries
results = pipeline.query_graph("""
    MATCH (a:Artifact)-[:CONTAINS*]->(f:File {type: 'code'})
    RETURN a.name as artifact, count(f) as code_files
    ORDER BY code_files DESC
""")

for result in results:
    print(f"{result['artifact']}: {result['code_files']} code files")
```

### Pattern Analysis

```python
# Get artifact patterns
patterns = pipeline.kg_builder.perform_advanced_analysis("artifact_name")
print("File organization:", patterns["patterns"]["file_organization"])
print("Naming conventions:", patterns["patterns"]["naming_conventions"])
```

### Recommendations

```python
# Get improvement recommendations
recommendations = pipeline.get_artifact_recommendations("artifact_name")
for category, suggestions in recommendations.items():
    print(f"{category.title()}:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
```

## Architecture

### Core Components

1. **RobustKGPipeline**: Main orchestrator class
2. **ArtifactExtractor**: Multi-format extraction engine
3. **EnhancedKGBuilder**: Knowledge graph construction
4. **BatchProcessor**: Parallel processing manager

### Data Flow

```
Artifacts → Extraction → Analysis → Knowledge Graph → Visualization
    ↓           ↓           ↓            ↓              ↓
  ZIP/TAR    File Tree   AST/Deps    Neo4j Nodes   HTML/JSON
```

### Graph Schema

- **Artifact**: Root node for each processed artifact
- **Directory**: Folder structure representation
- **File**: Individual files with metadata
- **Function**: Code functions with signatures
- **Class**: Class definitions with methods
- **Import**: Dependency relationships
- **DocSection**: Documentation sections

## Performance

### Benchmarks

- **Single Artifact**: ~30-60 seconds per medium project
- **Batch Processing**: Scales linearly with worker count
- **Memory Usage**: ~50-200MB per artifact
- **Storage**: ~1-10MB Neo4j data per artifact

### Optimization Tips

1. **Increase Workers**: Set `max_workers` based on CPU cores
2. **Use SSD Storage**: Faster I/O for temporary files
3. **Neo4j Tuning**: Configure memory settings for large datasets
4. **Exclude Patterns**: Skip unnecessary files (node_modules, .git)

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Check Neo4j is running: `http://localhost:7474`
   - Verify credentials and URI
   - Check firewall settings

2. **Memory Errors**
   - Reduce `max_workers`
   - Increase Java heap size for Neo4j
   - Process smaller batches

3. **Extraction Failures**
   - Check file permissions
   - Verify archive integrity
   - Increase timeout settings

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL="DEBUG"

# Or in Python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## API Reference

### RobustKGPipeline

#### Methods

- `process_single_artifact(artifact_path, artifact_name=None)`: Process one artifact
- `process_artifact_directory(artifacts_dir, file_patterns=None)`: Batch process directory
- `get_graph_statistics()`: Get graph metrics
- `export_graph_visualization(output_path=None, format="html")`: Export visualization
- `query_graph(cypher_query, parameters=None)`: Execute custom queries

### Configuration

#### Profiles

- `default`: Standard settings
- `development`: Debug mode, small workers
- `production`: Optimized for performance
- `testing`: Isolated test environment

## Examples

See the `examples/` directory for:
- Basic usage examples
- Advanced analytics demos
- Custom query samples
- Visualization examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is part of the AURA framework. See LICENSE for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{aura_algorithm_5,
  title={Algorithm 5: Robust Knowledge Graph Pipeline},
  author={AURA Framework Team},
  year={2024},
  url={https://github.com/your-repo/AURA}
}
``` 