# Artifact Evaluation System

A comprehensive evaluation system for research artifacts using Knowledge Graphs, LLMs, and Graph Data Science.

## 🎯 Overview

This system evaluates research artifacts (papers, code, datasets) based on multiple criteria to predict their acceptance likelihood for conferences and journals. It uses a combination of:

- **Knowledge Graph Construction** (Neo4j)
- **LLM-based Semantic Analysis** (OpenAI/GPT)
- **Graph Data Science** for pattern mining
- **Multi-dimensional Scoring** across key evaluation criteria

## 📊 Evaluation Criteria

The system evaluates artifacts across 5 key dimensions:

1. **Documentation Quality** (25% weight)
   - README presence and quality
   - License files
   - Documentation sections and structure

2. **Reproducibility** (30% weight)
   - Docker/containerization support
   - Setup instructions clarity
   - Examples and tutorials
   - Build complexity

3. **Availability** (20% weight)
   - Zenodo DOI or persistent archive
   - Data files accessibility
   - Repository structure

4. **Code Structure** (15% weight)
   - Code organization
   - Programming languages
   - File structure depth

5. **Complexity** (10% weight)
   - Setup complexity assessment
   - Repository size
   - Dependency management

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Artifact Evaluation System                   │
├─────────────────────────────────────────────────────────────────┤
│  JSON Analysis    │  Feature           │  Knowledge Graph       │
│  Input            │  Extraction        │  Construction          │
│  ├─ Documentation │  ├─ Has README     │  ├─ Artifact Node      │
│  ├─ Code Files    │  ├─ Has Docker     │  ├─ Documentation      │
│  ├─ Data Files    │  ├─ Zenodo DOI     │  ├─ Code Structure     │
│  └─ Metadata      │  └─ Setup Info     │  └─ Dependencies       │
├─────────────────────────────────────────────────────────────────┤
│  LLM Semantic     │  Scoring &         │  Recommendations &     │
│  Analysis         │  Prediction        │  Reporting             │
│  ├─ Content       │  ├─ Multi-criteria │  ├─ Improvement        │
│  ├─ Classification│  ├─ Weighted Score │  ├─ Comparison         │
│  └─ Quality       │  └─ Acceptance     │  └─ Visualization      │
│     Assessment    │     Likelihood     │                        │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Neo4j Database**
   ```bash
   # Using Docker
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     neo4j:latest
   ```

2. **Python Dependencies**
   ```bash
   pip install py2neo pandas numpy openai pyvis
   ```

3. **OpenAI API Key** (optional but recommended)
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Basic Usage

1. **Single Artifact Evaluation**
   ```python
   from artifact_evaluation_system import ArtifactEvaluationSystem
   
   # Initialize
   evaluator = ArtifactEvaluationSystem(
       neo4j_uri="bolt://localhost:7687",
       neo4j_user="neo4j",
       neo4j_password="password",
       openai_api_key="your-api-key"
   )
   
   # Evaluate artifact
   result = evaluator.evaluate_artifact_from_json("artifact_analysis.json")
   
   if result["success"]:
       print(f"Score: {result['acceptance_prediction']['score']:.3f}")
       print(f"Likelihood: {result['acceptance_prediction']['likelihood']}")
   ```

2. **Batch Evaluation**
   ```bash
   python batch_evaluation.py \
     --input-dir "path/to/json/files" \
     --output-dir "evaluation_results" \
     --openai-api-key "your-api-key"
   ```

3. **Example Usage**
   ```bash
   python example_usage.py
   ```

## 📁 File Structure

```
algorithm_5/
├── artifact_evaluation_system.py    # Main evaluation system
├── evaluation_config.py             # Configuration management
├── enhanced_kg_builder.py           # Knowledge graph builder
├── batch_evaluation.py              # Batch processing
├── example_usage.py                 # Usage examples
└── README_evaluation_system.md      # This file
```

## 🔧 Configuration

### Basic Configuration

```python
from evaluation_config import EvaluationConfig

config = EvaluationConfig(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    openai_api_key="your-api-key",
    
    # Evaluation weights
    evaluation_weights={
        "documentation_quality": 0.25,
        "reproducibility": 0.30,
        "availability": 0.20,
        "code_structure": 0.15,
        "complexity": 0.10
    }
)
```

### Artifact-Specific Configuration

```python
from evaluation_config import ARTIFACT_TYPE_CONFIGS

# For dataset artifacts
dataset_config = ARTIFACT_TYPE_CONFIGS["dataset"]
# Emphasizes availability and documentation over code structure

# For replication studies
replication_config = ARTIFACT_TYPE_CONFIGS["replication"]
# Emphasizes reproducibility and setup clarity
```

### Conference-Specific Configuration

```python
from evaluation_config import CONFERENCE_CONFIGS

# ICSE has higher acceptance thresholds
icse_config = CONFERENCE_CONFIGS["icse"]
# FSE has moderate thresholds
fse_config = CONFERENCE_CONFIGS["fse"]
```

## 📊 Input Format

The system expects JSON files with the following structure (as produced by your existing analysis algorithms):

```json
{
  "success": true,
  "artifact_name": "MyArtifact",
  "documentation_files": [
    {
      "path": "README.md",
      "content": ["# My Artifact", "Description...", "## Installation", "..."]
    }
  ],
  "code_files": [
    {
      "path": "src/main.py",
      "content": ["import os", "def main():", "    pass"]
    }
  ],
  "docker_files": [
    {
      "path": "Dockerfile",
      "content": ["FROM python:3.9", "COPY . /app"]
    }
  ],
  "tree_structure": ["MyArtifact/", "├── README.md", "├── src/", "│   └── main.py"],
  "repo_size_mb": 2.5,
  "analysis_performed": true
}
```

## 📈 Output Format

The system provides comprehensive evaluation results:

```json
{
  "success": true,
  "artifact_name": "MyArtifact",
  "features": {
    "has_readme": true,
    "has_docker": true,
    "has_zenodo_doi": false,
    "setup_complexity": "low",
    "total_files": 15,
    "code_files": 8
  },
  "evaluation_scores": {
    "documentation_quality": 0.85,
    "reproducibility": 0.75,
    "availability": 0.60,
    "code_structure": 0.80,
    "complexity": 0.90
  },
  "acceptance_prediction": {
    "likelihood": "high",
    "confidence": 0.85,
    "score": 0.78
  },
  "recommendations": [
    {
      "category": "availability",
      "priority": "medium",
      "recommendation": "Consider archiving the artifact on Zenodo for persistent access"
    }
  ]
}
```

## 🎨 Visualization and Reports

### HTML Reports

The system generates interactive HTML reports for each artifact:

```python
# Generate HTML report
evaluator.export_evaluation_report(
    artifact_name="MyArtifact",
    output_path="MyArtifact_report.html"
)
```

### Batch Dashboard

For batch evaluations, a comprehensive dashboard is generated:

```python
# Batch processing automatically generates:
# - batch_evaluation_summary.json
# - artifact_comparison.csv
# - batch_evaluation_dashboard.html
```

### Knowledge Graph Visualization

Export interactive network visualizations:

```python
# Export graph visualization
evaluator.kg_builder.export_visualization(
    output_path="artifact_graph.html",
    format="html"
)
```

## 🔍 Advanced Features

### Custom Scoring

You can implement custom scoring functions:

```python
class CustomEvaluationSystem(ArtifactEvaluationSystem):
    def _calculate_custom_score(self, features, semantic_analysis):
        # Your custom scoring logic
        return custom_score
```

### Pattern Mining

The system can detect patterns in successful artifacts:

```python
# Analyze patterns in high-scoring artifacts
patterns = evaluator.kg_builder.perform_advanced_analysis(artifact_name)
```

### Comparison Analysis

Compare multiple artifacts:

```python
comparison = evaluator.compare_artifacts([
    "Artifact1", "Artifact2", "Artifact3"
])
```

## 🛠️ Extending the System

### Adding New Evaluation Criteria

1. **Update ArtifactFeatures dataclass**
   ```python
   @dataclass
   class ArtifactFeatures:
       # ... existing fields ...
       new_criterion: bool = False
   ```

2. **Update feature extraction**
   ```python
   def _extract_features_from_json(self, analysis_data):
       # ... existing extraction ...
       features.new_criterion = self._check_new_criterion(analysis_data)
   ```

3. **Update scoring logic**
   ```python
   def _calculate_evaluation_scores(self, features, semantic_analysis):
       # ... existing scoring ...
       scores["new_criterion"] = self._score_new_criterion(features)
   ```

### Adding New File Types

Extend the file type detection:

```python
def _classify_file_type(self, file_path):
    # Add new file type patterns
    if file_path.suffix.lower() in ['.new_ext']:
        return "new_type"
    # ... existing logic ...
```

## 🧪 Testing

### Unit Tests

```python
# Test feature extraction
def test_feature_extraction():
    evaluator = ArtifactEvaluationSystem()
    features = evaluator._extract_features_from_json(mock_data)
    assert features.has_readme == True
```

### Integration Tests

```python
# Test full evaluation pipeline
def test_evaluation_pipeline():
    evaluator = ArtifactEvaluationSystem()
    result = evaluator.evaluate_artifact_from_json("test_artifact.json")
    assert result["success"] == True
```

## 📋 Examples

### Example 1: Evaluating Your Artifacts

```python
# Using the provided JSON files
evaluator = ArtifactEvaluationSystem()

# Evaluate TXBug artifact
txbug_result = evaluator.evaluate_artifact_from_json(
    "algo_outputs/algorithm_2_output/TXBug-main_analysis.json"
)

# Evaluate Bazel downgrade study
bazel_result = evaluator.evaluate_artifact_from_json(
    "algo_outputs/algorithm_2_output/10460752_analysis.json"
)

# Compare them
comparison = evaluator.compare_artifacts([
    "TXBug-main", "10460752"
])
```

### Example 2: Batch Processing

```bash
# Process all artifacts in a directory
python batch_evaluation.py \
  --input-dir "algo_outputs/algorithm_2_output" \
  --output-dir "evaluation_results" \
  --openai-api-key "$OPENAI_API_KEY"
```

### Example 3: Custom Configuration

```python
# Custom configuration for dataset artifacts
config = EvaluationConfig(
    evaluation_weights={
        "documentation_quality": 0.30,
        "reproducibility": 0.20,
        "availability": 0.35,  # Higher weight for datasets
        "code_structure": 0.10,
        "complexity": 0.05
    }
)

evaluator = ArtifactEvaluationSystem(config=config)
```

## 🐛 Troubleshooting

### Common Issues

1. **Neo4j Connection Error**
   ```
   Error: Failed to connect to Neo4j
   Solution: Ensure Neo4j is running and credentials are correct
   ```

2. **OpenAI API Error**
   ```
   Error: OpenAI API key not found
   Solution: Set OPENAI_API_KEY environment variable
   ```

3. **JSON Parsing Error**
   ```
   Error: Invalid JSON format
   Solution: Ensure JSON files follow expected structure
   ```

### Performance Optimization

- Use batch processing for multiple artifacts
- Enable Neo4j indexes for large datasets
- Configure appropriate Neo4j memory settings
- Use connection pooling for production deployments

## 📚 References

1. **Neo4j Graph Database**: https://neo4j.com/
2. **OpenAI API**: https://openai.com/api/
3. **Graph Data Science**: https://neo4j.com/graph-data-science/
4. **Artifact Evaluation Guidelines**: Research conference guidelines

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔄 Changelog

### Version 1.0.0
- Initial release
- Basic evaluation system
- Knowledge graph construction
- LLM integration
- Batch processing
- HTML reporting

---

For questions or support, please refer to the example usage scripts or create an issue in the repository. 