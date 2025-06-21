# AURA: Unified Artifact Research Assessment Framework

AURA is a comprehensive, modular framework for automated evaluation of research software artifacts according to conference-specific guidelines. It combines qualitative LLM-based assessments with quantitative keyword-based scoring to provide transparent, evidence-based artifact evaluation.

---

## Table of Contents
- [Overview](#overview)
- [Integration with Other Algorithms](#integration-with-other-algorithms)
- [How AURA Works - Step by Step](#how-aura-works---step-by-step)
- [Input Requirements](#input-requirements)
- [Evaluation Dimensions](#evaluation-dimensions)
- [Evaluation Methods](#evaluation-methods)
- [Agent Architecture](#agent-architecture)
- [Output Formats](#output-formats)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [References](#references)

---

## Overview

AURA automates the evaluation of research artifacts (code, data, documentation, etc.) by:
- **Parsing conference guidelines** to extract evaluation criteria
- **Analyzing artifact JSON files** (created by Algorithm 2) for compliance
- **Providing dual evaluation methods**: LLM-based qualitative assessment + keyword-based quantitative scoring
- **Generating comprehensive reports** with evidence-based scores and improvement suggestions

### Key Features
- **Conference-agnostic**: Works with any set of guidelines
- **Modular design**: Each evaluation dimension handled by dedicated agents
- **Transparent reasoning**: Chain-of-thought explanations for all scores
- **Evidence grounding**: LLM evaluations grounded with keyword evidence to prevent hallucination
- **Multiple evaluation modes**: Full, LLM-only, keyword-only, comparison, and grounded evaluation

---

## Integration with Other Algorithms

AURA is designed to work seamlessly with the outputs from other algorithms in the research pipeline:

### Algorithm 1 Integration
**Purpose**: Algorithm 1 extracts evaluation criteria from conference guidelines using NLP techniques
**Output**: CSV file with keyword-based evaluation criteria and weights
**Integration**: AURA's keyword evaluation agent uses this CSV to perform quantitative scoring

**Process**:
1. **Algorithm 1** processes conference guideline markdown files
2. **Extracts keywords** using TF-IDF and semantic similarity
3. **Generates criteria CSV** with dimensions, keywords, and weights
4. **AURA loads** this CSV for keyword-based evaluation

**Example Integration**:
```python
# Algorithm 1 output: algorithm_1_artifact_evaluation_criteria.csv
# AURA loads this file for keyword evaluation
keyword_agent = KeywordEvaluationAgent(criteria_csv_path="algo_outputs/algorithm_1_output/algorithm_1_artifact_evaluation_criteria.csv")
```

### Algorithm 2 Integration
**Purpose**: Algorithm 2 analyzes GitHub repositories and extracts structured information
**Output**: JSON file containing repository structure, file contents, and metadata
**Integration**: AURA's LLM agents use this JSON for semantic analysis and evaluation

**Process**:
1. **Algorithm 2** clones and analyzes GitHub repositories
2. **Extracts file contents** (documentation, code, license files)
3. **Generates structured JSON** with repository information
4. **AURA processes** this JSON for comprehensive evaluation

**Example Integration**:
```python
# Algorithm 2 output: ml-image-classifier_analysis.json
# AURA loads this file for artifact analysis
aura = AURA(
    artifact_json_path="algo_outputs/algorithm_2_output/ml-image-classifier_analysis.json",
    # ... other parameters
)
```

### Data Flow Pipeline
```
Conference Guidelines (MD) → Algorithm 1 → Criteria CSV → AURA Keyword Agent
GitHub Repository → Algorithm 2 → Artifact JSON → AURA LLM Agents
                                                      ↓
                                              Comprehensive Evaluation
```

---

## How AURA Works - Step by Step

### Step 1: Input Processing
1. **Conference Guidelines** (Markdown file): Contains the evaluation criteria for a specific conference
2. **Artifact JSON** (from Algorithm 2): Contains the analyzed repository structure and content
3. **Criteria CSV** (from Algorithm 1): Contains keyword-based evaluation criteria and weights

### Step 2: Agent Initialization
1. **Keyword Agent**: Loads the criteria CSV and artifact JSON
2. **LLM Agents**: Load guidelines, build vector databases, and extract evaluation criteria
3. **Framework Integration**: All agents are connected and ready for evaluation

### Step 3: Evaluation Process
1. **Keyword Analysis**: Quantitative scoring based on keyword presence
2. **LLM Evaluation**: Qualitative assessment with evidence retrieval
3. **Grounding**: LLM evaluations are grounded with keyword evidence
4. **Result Merging**: Both evaluation methods are combined for comprehensive assessment

### Step 4: Output Generation
1. **Detailed Reports**: Per-dimension evaluations with scores and justifications
2. **Comparison Analysis**: Side-by-side comparison of evaluation methods
3. **Improvement Suggestions**: Actionable recommendations for artifact enhancement

---

## Input Requirements

### Required Files

#### 1. Conference Guidelines (Markdown)
**Purpose**: Defines evaluation criteria for a specific conference
**Format**: `.md` file
**Location**: `data/conference_guideline_texts/processed/`
**Example**: `13_icse_2025.md`

**Content Example**:
```markdown
# ICSE 2025 Artifact Evaluation Guidelines

## Documentation Requirements
- Must include a README file with setup instructions
- Should provide API documentation
- Must include usage examples

## Accessibility Requirements
- Artifact must be publicly available
- Dependencies must be clearly listed
- Installation process should be straightforward
```

#### 2. Artifact JSON (from Algorithm 2)
**Purpose**: Contains the analyzed repository structure and content
**Format**: `.json` file
**Location**: `algo_outputs/algorithm_2_output/`
**Example**: `ml-image-classifier_analysis.json`

**Structure**:
```json
{
  "repository_structure": [
    {
      "name": "README.md",
      "path": "README.md",
      "mime_type": "text/markdown",
      "size_kb": 1.23
    }
  ],
  "documentation_files": [
    {
      "path": "README.md",
      "content": ["# My Project", "## Installation", "..."]
    }
  ],
  "code_files": [
    {
      "path": "src/main.py",
      "content": ["import os", "def main():", "..."]
    }
  ],
  "license_files": [
    {
      "path": "LICENSE",
      "content": ["MIT License", "Copyright (c) 2025", "..."]
    }
  ],
  "tree_structure": [
    "ml-image-classifier",
    "├── README.md",
    "├── src/",
    "│   └── main.py",
    "└── LICENSE"
  ]
}
```

**How AURA Processes This JSON**:
- **Documentation Files**: Used by documentation and usability agents for README analysis
- **Code Files**: Analyzed by functionality agent for code quality and structure
- **License Files**: Checked by accessibility agent for licensing compliance
- **Tree Structure**: Used by all agents to understand repository organization
- **Repository Structure**: Provides metadata for file analysis and filtering

#### 3. Criteria CSV (from Algorithm 1)
**Purpose**: Contains keyword-based evaluation criteria and weights
**Format**: `.csv` file
**Location**: `algo_outputs/algorithm_1_output/`
**Example**: `algorithm_1_artifact_evaluation_criteria.csv`

**Structure**:
```csv
dimension,keywords,raw_score,normalized_weight
reproducibility,"users reproduce, repository, artifacts, reproduce results",17.39,0.273
documentation,"configuration, documentation, instructions, setup",6.33,0.100
accessibility,"restrict access, ensuring, public, datasets",10.87,0.171
usability,"installation, demo, repository, include, artifact",13.59,0.213
experimental,"experimentation, experiments, traceability, reproducible",10.10,0.159
functionality,"testing, verifying, functions, test, validate",5.35,0.084
```

**How AURA Uses This CSV**:
- **Keywords**: Extracted and used for text matching in artifact content
- **Raw Scores**: Used for calculating dimension importance
- **Normalized Weights**: Applied to final scoring calculations
- **Dimensions**: Mapped to AURA's evaluation dimensions

---

## Evaluation Dimensions

AURA evaluates artifacts across four primary dimensions:

### 1. Documentation
**Goal**: Assess completeness, clarity, and structure of artifact documentation
**Checks**:
- Presence of required README sections (purpose, setup, usage, provenance)
- Quality and clarity of documentation
- API documentation completeness
- Beginner-friendliness of instructions

**Agent**: `documentation_evaluation_agent.py`

### 2. Usability
**Goal**: Evaluate ease of installation, configuration, and usage
**Checks**:
- Clarity of installation instructions
- Presence of setup scripts (requirements.txt, Dockerfile, etc.)
- Demo availability and functionality
- User interface quality

**Agent**: `usability_evaluation_agent.py`

### 3. Accessibility
**Goal**: Determine community accessibility of the artifact
**Checks**:
- Public availability (GitHub, DOI, Zenodo)
- Dependency clarity and completeness
- Repository structure and organization
- Installation feasibility

**Agent**: `accessibility_evaluation_agent.py`

### 4. Functionality
**Goal**: Assess whether artifact performs as claimed
**Checks**:
- Presence of main execution scripts
- Test coverage and validation
- Output examples and results
- Evidence supporting claimed functionality

**Agent**: `functionality_evaluation_agent.py`

---

## Evaluation Methods

### 1. LLM-based Evaluation
**Approach**: Uses large language models with chain-of-thought reasoning
**Process**:
1. **Guideline Parsing**: Extracts evaluation criteria from conference guidelines
2. **Vector Database**: Builds semantic index of artifact content
3. **Evidence Retrieval**: Retrieves relevant content for each evaluation criterion
4. **Chain-of-Thought**: LLM reasons step-by-step through evaluation process
5. **Grounded Assessment**: Uses keyword evidence to prevent hallucination

**Strengths**:
- Context-aware analysis
- Detailed explanations and suggestions
- Semantic understanding of content
- Specific improvement recommendations

**Output**: Qualitative assessments with detailed reasoning

### 2. Keyword-based Evaluation
**Approach**: Quantitative scoring based on keyword presence in artifact content
**Process**:
1. **Criteria Loading**: Loads evaluation criteria from CSV file
2. **Text Extraction**: Extracts all text content from documentation, code, and license files
3. **Keyword Matching**: Counts occurrences of guideline-derived keywords using word boundaries
4. **Scoring**: Applies log scaling and weighted scoring based on guideline importance
5. **Reporting**: Provides detailed breakdown by dimension with found keywords

**Strengths**:
- Objective and reproducible
- Provides numerical baseline scores
- Fast and efficient
- Consistent across evaluations

**Output**: Numerical scores with dimension breakdown

---

## Agent Architecture

### Core Framework (`aura_framework.py`)
**Purpose**: Orchestrates all evaluation agents and manages results
**Key Functions**:
- Agent initialization and coordination
- Evaluation mode management
- Result merging and comparison
- Error handling and logging

### LLM Agents
Each dimension has a dedicated LLM agent with the following structure:

#### Initialization
```python
def __init__(self, guideline_path, artifact_json_path, conference_name, keyword_agent=None):
    # Load guidelines and artifact
    # Build vector database
    # Extract evaluation criteria
    # Store keyword agent reference for grounding
```

#### Evaluation Process
1. **Criteria Extraction**: Parse guidelines to extract structured evaluation criteria
2. **Vector Database**: Build semantic index of artifact content
3. **Evidence Retrieval**: Retrieve relevant content for evaluation
4. **Prompt Construction**: Build chain-of-thought prompts with keyword grounding
5. **LLM Evaluation**: Generate qualitative assessment with reasoning

#### Grounding Mechanism
```python
def _get_keyword_evidence(self):
    # Get keyword results for this dimension
    # Extract relevant scores and keywords
    # Return evidence for LLM grounding
```

### Keyword Agent (`keyword_evaluation_agent.py`)
**Purpose**: Provides quantitative baseline evaluation
**Key Functions**:
- Load and parse criteria CSV
- Extract text content from artifact JSON
- Perform keyword matching and scoring
- Generate detailed reports

#### Evaluation Process
1. **Text Extraction**: Extract all text content from artifact files
2. **Keyword Matching**: Count keyword occurrences using word boundaries
3. **Scoring**: Apply log scaling and weighted scoring
4. **Reporting**: Generate dimension breakdown and summary

---

## Output Formats

### 1. Full Evaluation Output
```python
{
    "documentation": "LLM evaluation text with detailed reasoning...",
    "usability": "LLM evaluation text with detailed reasoning...",
    "accessibility": "LLM evaluation text with detailed reasoning...",
    "functionality": "LLM evaluation text with detailed reasoning...",
    "keyword_baseline": {
        "overall_score": 4.20,
        "dimensions": [
            {
                "dimension": "reproducibility",
                "raw_score": 105,
                "weighted_score": 1.27,
                "keywords_found": ["repository", "artifacts", "reproduce"]
            },
            # ... other dimensions
        ],
        "summary": "Detailed text summary of keyword evaluation..."
    }
}
```

### 2. Grounded Evaluation Output
```python
{
    "llm_evaluation": "Detailed LLM assessment with reasoning...",
    "keyword_evidence": {
        "raw_score": 12,
        "weighted_score": 0.44,
        "keywords_found": ["repository", "public", "datasets"]
    },
    "grounding_info": "LLM evaluation was grounded with keyword evidence: repository, public, datasets"
}
```

### 3. Comparison Output
```python
{
    "llm_evaluations": {...},
    "keyword_evaluation": {...},
    "comparison_notes": "Detailed comparison of both evaluation methods..."
}
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key (for LLM access)

### Installation Steps

1. **Clone the Repository**
```bash
git clone <repository-url>
cd AURA
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Set Up Environment Variables**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-openai-api-key-here
```

5. **Verify File Structure**
Ensure you have the required input files:
```
AURA/
├── data/
│   └── conference_guideline_texts/
│       └── processed/
│           └── 13_icse_2025.md
├── algo_outputs/
│   ├── algorithm_1_output/
│   │   └── algorithm_1_artifact_evaluation_criteria.csv
│   └── algorithm_2_output/
│       └── ml-image-classifier_analysis.json
└── scripts/
    └── algorithm_4/
        ├── app.py
        ├── aura_framework.py
        └── agents/
```

---

## Usage Guide

### 1. Running the Streamlit App

**Navigate to the algorithm_4 directory:**
```bash
cd scripts/algorithm_4
```

**Launch the app:**
```bash
streamlit run app.py
```

**Access the app:**
Open your browser and go to `http://localhost:8501`

### 2. Configuring the Evaluation

In the Streamlit sidebar, configure:

#### File Paths
- **Conference Guidelines Path**: Path to your conference guidelines markdown file
- **Artifact JSON Path**: Path to the artifact analysis JSON file
- **Criteria CSV Path**: Path to the evaluation criteria CSV file

#### Evaluation Options
- **Dimensions**: Select which aspects to evaluate (documentation, usability, accessibility, functionality)
- **Keyword Evaluation**: Enable/disable keyword-based evaluation
- **Evaluation Mode**: Choose from available evaluation modes

### 3. Evaluation Modes

#### Full Evaluation
- Runs both LLM and keyword-based assessments
- Provides comprehensive evaluation with grounding
- Best for complete artifact assessment

#### LLM Only
- Qualitative evaluation without keyword grounding
- Faster execution
- Good for initial assessment

#### Keyword Only
- Quantitative baseline evaluation
- Fast and objective
- Good for large-scale studies

#### Comparison Mode
- Side-by-side comparison of both methods
- Helps understand differences between approaches
- Good for method validation

#### Grounded Evaluation
- LLM evaluations with keyword evidence integration
- Shows grounding information
- Best for detailed analysis

### 4. Programmatic Usage

```python
from aura_framework import AURA

# Initialize AURA with all components
aura = AURA(
    guideline_path="path/to/guidelines.md",
    artifact_json_path="path/to/artifact.json",
    criteria_csv_path="path/to/criteria.csv"
)

# Full evaluation
results = aura.evaluate()

# Grounded evaluation for specific dimension
grounded_result = aura.get_grounded_evaluation("documentation")

# Comparison
comparison = aura.compare_evaluations()
```

---

## Advanced Features

### 1. Custom Evaluation Criteria
You can modify the criteria CSV file to add new dimensions or adjust weights:

```csv
dimension,keywords,raw_score,normalized_weight
custom_dimension,"custom, keywords, here",10.0,0.200
```

### 2. Custom Prompts
Modify agent prompts in the respective agent files to adjust evaluation focus:

```python
def _build_eval_prompt(self):
    # Customize prompt for your specific needs
    prompt = "Your custom evaluation prompt here..."
    return prompt
```

### 3. Integration with Other Systems
AURA can be integrated into other evaluation pipelines:

```python
# Batch evaluation
artifacts = ["artifact1.json", "artifact2.json", "artifact3.json"]
results = {}
for artifact in artifacts:
    aura = AURA(guideline_path, artifact, criteria_csv_path)
    results[artifact] = aura.evaluate()
```

### 4. Custom Scoring Algorithms
Modify the keyword scoring algorithm in `keyword_evaluation_agent.py`:

```python
def _evaluate_artifact_against_criteria(self):
    # Customize scoring logic
    # Modify log scaling, weights, etc.
    pass
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'scripts.algorithm_4'`
**Solution**: Use relative imports when running from the algorithm_4 directory:
```python
from aura_framework import AURA  # Instead of from scripts.algorithm_4.aura_framework import AURA
```

#### 2. File Path Issues
**Problem**: File not found errors
**Solution**: Use absolute paths or ensure relative paths are correct:
```python
# Use absolute paths
guideline_path = "C:/workplace/AURA/data/conference_guideline_texts/processed/13_icse_2025.md"
```

#### 3. OpenAI API Issues
**Problem**: LLM evaluation failures
**Solution**: Check your API key and quota:
```env
OPENAI_API_KEY=your-valid-api-key-here
```

#### 4. Memory Issues
**Problem**: Large artifacts causing memory problems
**Solution**: Adjust chunk sizes in agent initialization:
```python
agent = DocumentationEvaluationAgent(
    guideline_path, artifact_json_path, conference_name,
    chunk_size=512,  # Reduce from 1024
    chunk_overlap=50  # Reduce from 100
)
```

### Debug Mode
Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## File Structure

```
scripts/algorithm_4/
├── app.py                           # Streamlit web interface
├── aura_framework.py                # Core framework orchestration
├── README.md                        # This comprehensive guide
├── requirements.txt                 # Python dependencies
├── agents/                          # Evaluation agents
│   ├── accessibility_evaluation_agent.py
│   ├── documentation_evaluation_agent.py
│   ├── functionality_evaluation_agent.py
│   ├── usability_evaluation_agent.py
│   └── keyword_evaluation_agent.py
├── chroma_index/                    # Vector database storage
├── accessibility_chroma_index/      # Accessibility-specific vector DB
├── functionality_chroma_index/      # Functionality-specific vector DB
├── usability_chroma_index/          # Usability-specific vector DB
└── guidelines_index/                # Guidelines vector database
```

### Key Files Explained

#### `app.py`
- Streamlit web interface
- Handles user input and result display
- Manages different evaluation modes

#### `aura_framework.py`
- Core framework logic
- Agent initialization and coordination
- Result merging and comparison

#### `agents/`
- Individual evaluation agents
- Each agent handles one evaluation dimension
- Contains LLM prompting and grounding logic

#### `*_chroma_index/`
- Vector database storage for semantic search
- Built automatically during agent initialization
- Cached for performance

---

## References

### Academic References
- [Artifact Evaluation at ACM/IEEE Conferences](https://artifact-eval.org/)
- [Reproducibility in Computer Science](https://www.sigsoft.org/resources/reproducibility.html)

### Technical Documentation
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

### Related Projects
- [AURA Paper/Docs](https://github.com/your-org/aura) (if available)
- [Conference Guidelines Repository](https://github.com/your-org/conference-guidelines)

### Dependencies
- **LangChain**: LLM integration and chain management
- **OpenAI**: Language model access
- **Streamlit**: Web interface
- **Chroma**: Vector database
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

---

## Contributing

We welcome contributions to improve AURA! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black scripts/algorithm_4/
```

---

## Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/your-org/aura/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/aura/discussions)
- **Email**: [your-email@domain.com]

---

**AURA Framework** - Making artifact evaluation transparent, reproducible, and evidence-based. 