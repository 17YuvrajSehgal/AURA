# AURA Framework - ICSE 2025 Artifact Evaluation

The AURA (Automated Unified Repository Artifact) Framework is a comprehensive evaluation system designed to assess software artifacts based on ICSE 2025 guidelines. It provides automated evaluation across multiple dimensions using specialized agents and a knowledge graph approach.

## Overview

The framework evaluates artifacts based on the following ICSE 2025 criteria:

1. **Availability** - Public accessibility and archival repository status
2. **Functionality** - Executability, consistency, and verification evidence
3. **Reusability** - Documentation quality and structural reusability
4. **Documentation** - README quality, setup, and usage instructions
5. **Archival Repository** - Suitable repository (Zenodo, FigShare) vs non-archival
6. **Executable Artifacts** - Installation packages and Docker/VM images
7. **Non-executable Artifacts** - Proper packaging and accessibility
8. **License** - Open-source licensing and distribution rights
9. **Setup Instructions** - Clarity and completeness for executable artifacts
10. **Usage Instructions** - Clarity for replicating main results
11. **Iterative Review Process** - Authors' responsiveness to reviewer requests

## Architecture

The framework consists of several specialized evaluation agents:

### Core Components

- **AURAFramework** - Main orchestrator that coordinates all evaluations
- **RepositoryKnowledgeGraphAgent** - Builds and queries Neo4j knowledge graph
- **Evaluation Agents** - Specialized agents for each evaluation dimension

### Evaluation Agents

1. **AccessibilityEvaluationAgent** - Evaluates public access, archival status, and licensing
2. **DocumentationEvaluationAgent** - Assesses README quality, setup, and usage instructions
3. **ExperimentalEvaluationAgent** - Evaluates experimental setup, data availability, and validation
4. **FunctionalityEvaluationAgent** - Assesses executability, consistency, and verification
5. **ReproducibilityEvaluationAgent** - Evaluates reusability, setup clarity, and result replication
6. **UsabilityEvaluationAgent** - Assesses user experience, interfaces, and error handling

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Neo4j database (optional, for advanced features):
```bash
# Install Neo4j Desktop or use Docker
docker run -p 7474:7474 -p 7687:7687 neo4j:latest
```

## Usage

### Basic Usage

```python
from aura_framework import AURAFramework

# Initialize framework
framework = AURAFramework("path/to/artifact.json")

# Evaluate artifact
result = framework.evaluate_artifact()

# Print results
framework.print_results(result)

# Save results
framework.save_results(result, "evaluation_results.json")

# Clean up
framework.close()
```

### Command Line Usage

```bash
python aura_framework.py path/to/artifact.json --output results.json
```

### Advanced Usage with Neo4j

```python
framework = AURAFramework(
    artifact_json_path="path/to/artifact.json",
    neo4j_uri="bolt://localhost:7687"
)
```

## Input Format

The framework expects a JSON file containing artifact information:

```json
{
  "repository_name": "example-repo",
  "repository_url": "https://github.com/example/repo",
  "repository_structure": [
    {
      "name": "README.md",
      "path": "README.md",
      "mime_type": "text/markdown"
    }
  ],
  "documentation_files": [
    {
      "path": "README.md",
      "content": ["# Example", "This is a README"]
    }
  ],
  "code_files": [
    {
      "path": "main.py",
      "content": ["import sys", "print('Hello World')"]
    }
  ],
  "license_files": [
    {
      "path": "LICENSE",
      "content": ["MIT License", "..."]
    }
  ]
}
```

## Output Format

The framework produces comprehensive evaluation results:

```json
{
  "criteria_scores": [
            {
                "dimension": "reproducibility",
      "raw_score": 6.78,
      "normalized_weight": 0.207,
      "llm_evaluated_score": 0.85,
      "justification": "Excellent reproducibility...",
      "evidence": ["Modular code structure found", "Setup instructions found"]
    }
  ],
  "total_weighted_score": 0.78,
  "acceptance_prediction": true,
  "overall_justification": "Total weighted score: 0.780...",
  "recommendations": [
    "Improve README documentation with clear setup and usage instructions"
  ]
}
```

## Evaluation Dimensions

### Reproducibility (Weight: 20.7%)
- Reusability and modular structure
- Setup instructions clarity
- Usage instructions clarity
- Result replication capability

### Documentation (Weight: 15.4%)
- README quality and completeness
- Setup instructions
- Usage instructions
- Comprehensive documentation

### Accessibility (Weight: 13.9%)
- Public accessibility
- Archival repository status
- Licensing compliance

### Usability (Weight: 19.8%)
- User experience and ease of use
- User interface quality
- Error handling and feedback
- Iterative review process support

### Experimental (Weight: 14.8%)
- Experimental setup and requirements
- Data availability and datasets
- Validation and verification evidence
- Non-executable artifact packaging

### Functionality (Weight: 15.5%)
- Executability and main entry points
- Consistency and completeness
- Verification and validation evidence
- Executable artifact preparation

## Acceptance Criteria

The framework uses a weighted scoring system with an acceptance threshold of **0.75**:

- **Score ≥ 0.75**: Artifact is predicted to be ACCEPTED
- **Score < 0.75**: Artifact is predicted to be REJECTED

## Customization

### Modifying Weights

Edit the criteria scores in `aura_integration_data_20250626_013157.json` or modify the `_get_default_criteria()` method.

### Adding New Agents

1. Create a new agent class inheriting from the base pattern
2. Implement the `evaluate()` method
3. Add the agent to the `agents` dictionary in `AURAFramework`

### Custom Acceptance Threshold

Modify the `ACCEPTANCE_THRESHOLD` constant in `aura_framework.py`.

## Knowledge Graph Features

The framework uses Neo4j to build a knowledge graph of the artifact:

- **Repository Structure** - Files, directories, and relationships
- **Code Analysis** - Functions, classes, imports, and dependencies
- **Documentation Links** - README sections and their relationships
- **Badge Support** - ICSE 2025 badge requirements and compliance

### Graph Queries

```python
# Query for specific files
evidence = kg_agent.file_exists("README.md")

# Get file content
content = kg_agent.get_file_content("main.py")

# Check for specific sections
has_setup = kg_agent.readme_has_section("setup")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{aura_framework,
  title={AURA Framework: Automated Unified Repository Artifact Evaluation},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/aura-framework}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

## Roadmap

- [ ] Integration with LLM-based evaluation
- [ ] Support for more artifact types
- [ ] Web-based evaluation interface
- [ ] Batch processing capabilities
- [ ] Integration with conference submission systems 