# README Documentation Generator (Algorithm 3)

## Overview

The README Documentation Generator is an advanced system that automatically generates comprehensive, high-quality README documentation for research artifacts. It leverages cutting-edge technologies including **LangChain**, **Knowledge Graphs**, **RAG (Retrieval Augmented Generation)**, and **Large Language Models** to create contextually rich and technically accurate documentation.

## 🧠 Architecture

The system follows a sophisticated multi-stage pipeline:

```
[User Input or Trigger]
         ↓
[Contextual Graph Search (Cypher)]
         ↓
[Section-Specific Prompt Templates]
         ↓
[LLM Generation]
         ↓
[README.md Assembly]
```

### Core Components

1. **Knowledge Graph Builder** (`knowledge_graph_builder.py`)
   - Extracts structured information from artifact JSON files
   - Creates nodes for artifacts, files, dependencies, commands, and outputs
   - Builds relationships between components
   - Supports both Neo4j and NetworkX backends

2. **RAG Retrieval System** (`rag_retrieval.py`)
   - Combines vector similarity search with graph traversal
   - Uses FAISS for efficient vector operations
   - Implements section-specific retrieval strategies
   - Provides contextually relevant information for each README section

3. **LangChain Orchestrator** (`langchain_chains.py`)
   - Manages prompt templates for different README sections
   - Orchestrates LLM interactions using LangChain
   - Supports parallel section generation
   - Handles cost tracking and error management

4. **Main Generator** (`readme_generator.py`)
   - Coordinates the entire pipeline
   - Provides command-line interface
   - Supports batch processing
   - Handles output formatting and metadata

## 🚀 Features

### ✨ Advanced Generation Capabilities

- **Section-Specific Generation**: Generate individual sections (title, setup, usage, etc.)
- **Full README Generation**: Create complete documentation in one operation
- **Batch Processing**: Process multiple artifacts simultaneously
- **Parallel Generation**: Generate multiple sections concurrently for efficiency
- **Custom Prompts**: Customize generation behavior with custom prompt templates

### 🎯 Intelligent Context Retrieval

- **Hybrid Search**: Combines vector similarity and graph traversal
- **Section-Aware Retrieval**: Different strategies for different README sections
- **Contextual Relevance**: Retrieves information most relevant to each section
- **Multi-Source Context**: Integrates information from multiple file types

### 📊 Knowledge Graph Integration

- **Neo4j Support**: Full Neo4j integration for production use
- **NetworkX Fallback**: Local graph processing for development
- **Rich Relationships**: Captures dependencies, generates outputs, and structural relationships
- **Semantic Search**: Vector embeddings for semantic similarity

### 🔧 Developer-Friendly Features

- **Preview Mode**: Preview context before generation
- **Statistics**: Detailed generation statistics and metrics
- **Customizable**: Modify prompts and generation behavior
- **Error Handling**: Robust error handling and logging
- **Command Line Interface**: Easy-to-use CLI for automation

## 📋 Requirements

### System Requirements

- Python 3.8+
- 8GB+ RAM (recommended for large artifacts)
- OpenAI API key (for LLM generation)
- Optional: Neo4j database for production use

### Software Dependencies

See `requirements.txt` for complete list. Key dependencies:

- `langchain>=0.1.0` - LLM orchestration
- `sentence-transformers>=2.2.0` - Vector embeddings
- `faiss-cpu>=1.7.0` - Vector similarity search
- `neo4j>=5.0.0` - Graph database (optional)
- `networkx>=3.0` - Local graph processing
- `openai>=1.0.0` - LLM API access

## 🔧 Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   Create a `.env` file in the `scripts/algorithm_3/` directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   NEO4J_PASSWORD=your_neo4j_password_here  # Optional
   ```

3. **Optional: Install Neo4j**:
   - Download and install Neo4j Desktop or Community Edition
   - Start Neo4j service
   - Configure connection settings in `config.py`

## 🚀 Usage

### Basic Usage

```bash
# Generate README for a single artifact
python readme_generator.py path/to/artifact.json

# Generate and save to file
python readme_generator.py path/to/artifact.json -o output/README.md

# Generate specific sections only
python readme_generator.py path/to/artifact.json -s title_and_purpose setup usage
```

### Python API

```python
from readme_generator import READMEGenerator

# Initialize generator
generator = READMEGenerator(use_neo4j=False)

# Generate README
readme_content = generator.generate_readme_from_artifact(
    "path/to/artifact.json",
    output_path="output/README.md"
)

print(readme_content)
```

### Batch Processing

```python
# Process multiple artifacts
artifact_paths = ["artifact1.json", "artifact2.json", "artifact3.json"]
results = generator.generate_readme_batch(
    artifact_paths,
    output_dir="output/batch_readmes"
)
```

### Advanced Features

```python
# Preview context for a section
context = generator.preview_section_context("artifact.json", "setup")
print(json.dumps(context, indent=2))

# Customize section prompts
custom_prompt = "Your custom prompt template here..."
generator.customize_section_prompt("title_and_purpose", custom_prompt)

# Get generation statistics
stats = generator.get_generation_statistics()
print(f"Generated {stats['kg_builder_stats']['total_nodes']} nodes")
```

## 🎯 Section Types

The system generates the following README sections:

1. **`title_and_purpose`** - Title and purpose of the artifact
2. **`artifact_available`** - Availability and access information
3. **`artifact_reusable`** - Reusability statement and evidence
4. **`provenance`** - Origin and creation information
5. **`setup`** - Installation and setup instructions
6. **`usage`** - Usage instructions and examples
7. **`outputs`** - Expected outputs and results
8. **`structure`** - Directory structure and organization
9. **`license`** - License information
10. **`attribution`** - Authors and citations

## 🛠️ Configuration

### Main Configuration (`config.py`)

```python
# LLM Configuration
config.llm.model_name = "gpt-4-turbo-preview"
config.llm.temperature = 0.3
config.llm.max_tokens = 4000

# Knowledge Graph Configuration
config.knowledge_graph.uri = "bolt://localhost:7687"
config.knowledge_graph.username = "neo4j"

# Vector Configuration
config.vector.model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

### Prompt Templates

Prompt templates are stored in `templates/` directory:

- `title_purpose.txt` - Title and purpose generation
- `setup.txt` - Setup instructions
- `usage.txt` - Usage instructions
- `outputs.txt` - Output descriptions
- And more...

## 📊 Performance

### Typical Processing Times

- **Small Artifact** (< 1MB): 30-60 seconds
- **Medium Artifact** (1-10MB): 1-3 minutes
- **Large Artifact** (> 10MB): 3-10 minutes

### Cost Estimation

- **GPT-4 Turbo**: ~$0.10-0.50 per README
- **GPT-3.5 Turbo**: ~$0.02-0.10 per README

*Costs depend on artifact complexity and section count*

## 🔍 Examples

### Example 1: Basic Generation

```bash
python readme_generator.py algo_outputs/algorithm_2_output_2/10460752_analysis.json
```

### Example 2: Section-Specific Generation

```bash
python readme_generator.py artifact.json -s setup usage -o setup_usage.md
```

### Example 3: Batch Processing

```bash
python readme_generator.py --batch artifact1.json artifact2.json artifact3.json
```

## 🧪 Demo Script

Run the comprehensive demo to see all features:

```bash
python demo.py
```

The demo includes:
- Basic README generation
- Section-specific generation
- Context preview
- Batch processing
- Custom prompt demonstration

## 📁 Output Structure

Generated files are organized as follows:

```
generated_readmes/
├── artifact_name_README.md          # Main README file
├── artifact_name_README.meta.json   # Generation metadata
└── batch_output/                    # Batch processing results
    ├── artifact1_README.md
    ├── artifact2_README.md
    └── ...
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   - Ensure `OPENAI_API_KEY` is set in environment
   - Check API key validity and credits

2. **Neo4j Connection Error**:
   - Verify Neo4j is running
   - Check connection settings in `config.py`
   - Use `use_neo4j=False` for local-only processing

3. **Memory Issues**:
   - Reduce batch size for large artifacts
   - Use smaller embedding models
   - Close generators after use

4. **Template Loading Error**:
   - Verify template files exist in `templates/` directory
   - Check file permissions
   - Ensure templates have correct format

### Performance Optimization

1. **Use Neo4j for Production**:
   ```python
   generator = READMEGenerator(use_neo4j=True)
   ```

2. **Parallel Section Generation**:
   ```python
   sections = ['title_and_purpose', 'setup', 'usage']
   results = generator.generate_parallel_sections(sections)
   ```

3. **Batch Processing**:
   ```python
   results = generator.generate_readme_batch(artifact_paths)
   ```

## 🔮 Future Enhancements

- **Multi-language Support**: Support for non-English artifacts
- **Custom LLM Backends**: Support for local LLMs and other providers
- **Interactive Web Interface**: Web-based README generation
- **Template Marketplace**: Shareable custom templates
- **Advanced Analytics**: Generation quality metrics and improvement suggestions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 📞 Support

For support, issues, or questions:
- Open an issue on GitHub
- Contact the development team
- Check the troubleshooting section above

---

*Generated by the README Documentation Generator - A sophisticated system for automatic documentation generation using LangChain, Knowledge Graphs, RAG, and LLMs.* 