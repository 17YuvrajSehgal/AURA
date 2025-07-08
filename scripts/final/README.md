# AURA: Artifact Understanding and Research Assessment Framework

AURA is a comprehensive framework for evaluating research artifacts across multiple dimensions using AI-powered analysis, knowledge graphs, and retrieval-augmented generation (RAG).

## Overview

The AURA framework evaluates research artifacts across six key dimensions:

- **Accessibility**: How easy it is to obtain and access the artifact
- **Documentation**: Quality and completeness of documentation
- **Experimental**: Support for experimental evaluation and validation
- **Functionality**: Whether the artifact works as intended
- **Reproducibility**: Ability to reproduce research results
- **Usability**: Ease of use for researchers

## Architecture

The framework consists of several key components:

1. **Knowledge Graph Builder** (`knowledge_graph_builder.py`): Extracts structured information from artifacts and builds a knowledge graph
2. **RAG Retrieval** (`rag_retrieval.py`): Provides contextual retrieval using vector similarity and graph traversal
3. **LangChain Chains** (`langchain_chains.py`): Implements AI-powered evaluation chains for each dimension
4. **AURA Evaluator** (`aura_evaluator.py`): Main orchestrator that coordinates the evaluation pipeline
5. **Configuration** (`config.py`): System configuration and settings
6. **Prompt Templates** (`templates/`): Evaluation prompts for each dimension

## Prerequisites

### Required Python Packages

```bash
pip install langchain openai sentence-transformers faiss-cpu neo4j networkx pandas numpy python-dotenv
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Configuration (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=aura
```

### Optional: Neo4j Setup

For enhanced knowledge graph capabilities, install Neo4j:

1. Download and install Neo4j Desktop or Community Edition
2. Create a new database named "aura"
3. Update the Neo4j configuration in your `.env` file

## Quick Start

### 1. Simple Evaluation

```python
from aura_evaluator import quick_evaluate

# Evaluate an artifact from JSON file
report = quick_evaluate(
    artifact_json_path="path/to/artifact.json",
    use_neo4j=False,  # Use NetworkX for simplicity
    use_rag=True
)

print(f"Overall Rating: {report['overall_rating']:.2f}/5.0")
for dimension, score in report["dimension_scores"].items():
    print(f"{dimension.title()}: {score:.2f}/5.0")
```

### 2. Full Pipeline Evaluation

```python
from aura_evaluator import AURAEvaluator

# Initialize evaluator
evaluator = AURAEvaluator(
    use_neo4j=True,  # Use Neo4j for advanced graph features
    use_rag=True,
    output_dir="evaluation_results"
)

# Evaluate artifact
report = evaluator.evaluate_artifact_from_json(
    artifact_json_path="path/to/artifact.json",
    dimensions=["accessibility", "documentation", "functionality"],  # Specific dimensions
    save_results=True
)

# Get dimension summary
summary = evaluator.get_dimension_summary(report)
print(summary)

evaluator.close()
```

### 3. Batch Evaluation

```python
from aura_evaluator import batch_evaluate

# Evaluate multiple artifacts
artifact_paths = [
    "artifacts/artifact1.json",
    "artifacts/artifact2.json", 
    "artifacts/artifact3.json"
]

results = batch_evaluate(
    artifact_paths=artifact_paths,
    use_neo4j=False,
    use_rag=True
)

# Print batch summary
for artifact_name, report in results.items():
    if "error" not in report:
        print(f"{artifact_name}: {report['overall_rating']:.2f}/5.0")
    else:
        print(f"{artifact_name}: FAILED - {report['error']}")
```

### 4. Conference-Specific Evaluation

AURA supports conference-specific evaluation using guidelines from 20+ major computer science venues including ICSE, ASE, FSE, ASPLOS, and more.

```python
from aura_evaluator import AURAEvaluator
from conference_guidelines_loader import conference_loader

# List available conferences
conferences = conference_loader.get_available_conferences()
print(f"Available conferences: {conferences}")

# Evaluate for ICSE 2025 requirements
evaluator = AURAEvaluator(
    use_neo4j=False,
    use_rag=True,
    conference_name="ICSE"  # Conference-specific guidelines
)

report = evaluator.evaluate_artifact_from_json(
    artifact_json_path="path/to/artifact.json"
)

# The evaluation will now consider ICSE-specific requirements such as:
# - DOI/archival repository requirements for accessibility
# - Specific documentation standards
# - Expected functionality levels
# - Reproducibility requirements
print(f"ICSE Evaluation Rating: {report['overall_rating']:.2f}/5.0")

evaluator.close()
```

#### Comparing Across Conferences

```python
# Compare how the same artifact performs across different venues
conferences_to_test = ["ICSE", "ASE", "FSE", "ASPLOS"]
comparison_results = {}

for conference in conferences_to_test:
    report = quick_evaluate(
        artifact_json_path="path/to/artifact.json",
        conference_name=conference,
        use_neo4j=False
    )
    comparison_results[conference] = report["overall_rating"]

# Print comparison
for conf, rating in comparison_results.items():
    print(f"{conf}: {rating:.2f}/5.0")
```

#### Supported Conferences

Currently supported conferences include:
- **Software Engineering**: ICSE, ASE, FSE, ISSTA
- **Systems**: ASPLOS, ISCA, MICRO, MOBISYS, MOBICOM
- **Programming Languages**: PLDI, ICFP, CGO, PPOPP
- **Human-Computer Interaction**: CHI
- **Security**: Asia CCS
- **And more**: SIGMOD, KDD, ConEXT, Middleware, etc.

Each conference has specific requirements for:
- Repository types (GitHub vs. archival repositories like Zenodo)
- Documentation standards (README requirements, license specifications)
- Reproducibility expectations (Docker, installation packages)
- Experimental validation requirements

## Artifact JSON Format

The framework expects artifact data in a specific JSON format:

```json
{
  "artifact_name": "my_research_artifact",
  "artifact_path": "/path/to/artifact",
  "repo_size_mb": 15.2,
  "extraction_method": "git_clone",
  "success": true,
  "documentation_files": [
    {
      "path": "README.md",
      "content": ["# My Artifact", "Description here..."]
    }
  ],
  "code_files": [
    {
      "path": "main.py",
      "content": ["#!/usr/bin/env python", "import sys", "..."]
    }
  ],
  "data_files": [
    {
      "name": "dataset.csv",
      "path": "data/dataset.csv",
      "size_kb": 1024,
      "mime_type": "text/csv"
    }
  ],
  "docker_files": [...],
  "license_files": [...],
  "build_files": [...],
  "tree_structure": [
    ".",
    "├── README.md",
    "├── main.py",
    "└── data/",
    "    └── dataset.csv"
  ]
}
```

## Configuration

### LLM Configuration

Modify `config.py` to customize the language model:

```python
@dataclass
class LLMConfig:
    model_name: str = "gpt-4-turbo-preview"  # or "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 4000
    api_key: str = os.getenv("OPENAI_API_KEY", "")
```

### Evaluation Dimensions

Customize which dimensions to evaluate:

```python
@dataclass
class AuraConfig:
    sections: List[str] = [
        "accessibility",
        "documentation", 
        "experimental",
        "functionality",
        "reproducibility",
        "usability"
    ]
```

### Custom Prompt Templates

Modify the templates in the `templates/` directory to customize evaluation criteria:

- `accessibility.txt` - Accessibility evaluation prompts
- `documentation.txt` - Documentation quality prompts
- `experimental.txt` - Experimental validation prompts
- `functionality.txt` - Functionality testing prompts
- `reproducibility.txt` - Reproducibility assessment prompts
- `usability.txt` - Usability evaluation prompts

## Output

### Evaluation Report Structure

```json
{
  "artifact_info": {
    "name": "artifact_name",
    "path": "/path/to/artifact",
    "size_mb": 15.2,
    "extraction_method": "git_clone"
  },
  "overall_rating": 3.5,
  "dimension_scores": {
    "accessibility": 4.0,
    "documentation": 3.5,
    "experimental": 3.0,
    "functionality": 4.5,
    "reproducibility": 2.5,
    "usability": 3.5
  },
  "detailed_evaluations": {
    "accessibility": {
      "rating": 4.0,
      "detailed_assessment": {...},
      "strengths": ["Public GitHub repository", "Clear download instructions"],
      "weaknesses": ["No release versioning"],
      "recommendations": ["Add version tags", "Provide DOI"],
      "summary": "Artifact is easily accessible via GitHub..."
    }
  },
  "summary": {
    "total_strengths": 15,
    "total_weaknesses": 8,
    "total_recommendations": 12,
    "dimensions_evaluated": 6,
    "highest_scoring_dimension": "functionality",
    "lowest_scoring_dimension": "reproducibility"
  },
  "evaluation_metadata": {
    "timestamp": "2024-01-15 14:30:22",
    "duration_seconds": 45.2,
    "evaluator_version": "1.0.0",
    "use_neo4j": true,
    "use_rag": true,
    "knowledge_graph_stats": {...}
  }
}
```

## Advanced Usage

### Custom Evaluation Chains

Create custom evaluation chains for specific domains:

```python
from langchain_chains import EvaluationChain, PromptTemplateLoader
from langchain.chat_models import ChatOpenAI

# Create custom template loader
template_loader = PromptTemplateLoader("my_custom_templates/")

# Create custom evaluation chain
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
custom_chain = EvaluationChain(
    dimension="security",
    template_loader=template_loader,
    llm=llm
)

# Run custom evaluation
result = custom_chain.evaluate(artifact_data)
```

### Knowledge Graph Analysis

Access the knowledge graph for custom analysis:

```python
from knowledge_graph_builder import KnowledgeGraphBuilder

# Build knowledge graph
kg_builder = KnowledgeGraphBuilder(use_neo4j=True)
stats = kg_builder.build_from_artifact_json("artifact.json")

# Get graph statistics
graph_stats = kg_builder.get_graph_stats()
print(f"Nodes: {graph_stats['total_nodes']}")
print(f"Relationships: {graph_stats['total_relationships']}")

# Access NetworkX graph (if not using Neo4j)
if hasattr(kg_builder, 'nx_graph'):
    import networkx as nx
    centrality = nx.degree_centrality(kg_builder.nx_graph)
    print("Most central nodes:", sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5])
```

### RAG Retrieval

Use the RAG system independently:

```python
from rag_retrieval import RAGRetriever

# Initialize RAG retriever
rag_retriever = RAGRetriever(kg_builder)

# Retrieve context for specific sections
context = rag_retriever.retrieve_for_section(
    section_type="documentation",
    query="How to install and setup the artifact",
    top_k=5
)

for result in context:
    print(f"Relevance: {result.relevance_score:.2f}")
    print(f"Content: {result.content}")
```

## Demo

Run the built-in demo to test the system:

```python
from aura_evaluator import demo_evaluation

# Run demo with example artifact
demo_evaluation()
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**: Ensure your API key is correctly set in the `.env` file
2. **Neo4j Connection Error**: Check Neo4j is running and credentials are correct
3. **Memory Issues**: For large artifacts, consider using Neo4j instead of NetworkX
4. **Import Errors**: Ensure all required packages are installed

### Performance Optimization

- Use Neo4j for large knowledge graphs (>1000 nodes)
- Disable RAG for faster evaluation: `use_rag=False`
- Limit evaluation dimensions for specific use cases
- Adjust `max_tokens` in LLM config for faster responses

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use AURA in your research, please cite:

```bibtex
@software{aura_evaluation_framework,
  title={AURA: Artifact Understanding and Research Assessment Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/aura}
}
``` 