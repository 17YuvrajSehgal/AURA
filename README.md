# AURA: Unified Artifact Research Assessment Framework

A comprehensive, automated framework for evaluating research software artifacts according to conference-specific guidelines. AURA combines qualitative LLM-based assessments with quantitative keyword-based scoring to provide transparent, evidence-based artifact evaluation.

## 🎯 Research Problem & Motivation

Research software artifacts (code, data, documentation) submitted with academic papers often suffer from:

- **Inconsistent evaluation standards** across conferences
- **Manual evaluation burden** on conference committees
- **Lack of reproducibility** due to poor artifact quality
- **No unified framework** for artifact assessment
- **Subjective evaluation** leading to inconsistent results

AURA addresses these challenges by providing an automated, evidence-based evaluation system that works across different conferences and evaluation criteria.

## 🏗️ System Architecture

AURA consists of three correlated algorithms working in sequence:

```
Conference Guidelines (MD) → Algorithm 1 → Criteria CSV → AURA Keyword Agent
GitHub Repository → Algorithm 2 → Artifact JSON → AURA LLM Agents
                                                      ↓
                                              Comprehensive Evaluation
```

### Algorithm 1: Automated Criteria Extraction
**Purpose**: Extract and standardize evaluation criteria from conference guidelines

**What it does**:
- Processes conference guideline markdown files (ICSE 2025, FSE 2024, etc.)
- Uses NLP techniques (TF-IDF, semantic similarity, KeyBERT) to identify evaluation dimensions
- Generates standardized CSV with dimensions, keywords, and weights
- **Output**: `algorithm_1_artifact_evaluation_criteria.csv`

**Key dimensions identified**:
- **Reproducibility** (27.3% weight) - most important
- **Usability** (21.3% weight) 
- **Accessibility** (17.1% weight)
- **Experimental** (15.9% weight)
- **Documentation** (10.0% weight)
- **Functionality** (8.4% weight)

### Algorithm 2: Repository Analysis
**Purpose**: Extract structured information from GitHub repositories

**What it does**:
- Clones GitHub repositories automatically
- Analyzes repository structure and extracts file contents
- Categorizes files (documentation, code, license)
- Generates comprehensive JSON representation
- **Output**: `{repository_name}_analysis.json`

### Algorithm 4: AURA Framework (Main Evaluation Engine)
**Purpose**: Comprehensive artifact evaluation using both LLM and keyword-based approaches

**Key innovations**:
- **Evidence grounding**: LLM evaluations grounded with keyword evidence to prevent hallucination
- **Dual evaluation methods**: Qualitative LLM assessment + quantitative keyword scoring
- **Modular agents**: Separate agents for each evaluation dimension
- **Transparent reasoning**: Chain-of-thought explanations for all scores
- **Conference-agnostic**: Works with any set of guidelines

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (for LLM evaluations)
- Git (for repository cloning)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd AURA
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your-openai-api-key-here
```

### Basic Usage

#### Step 1: Extract Evaluation Criteria (Algorithm 1)
```bash
cd scripts/algorithm_1
python improved_algorithm_1.py
```
This processes conference guidelines and generates `algorithm_1_artifact_evaluation_criteria.csv`.

#### Step 2: Analyze Repository (Algorithm 2)
```bash
cd scripts/algorithm_2
python algorithm_2.py
```
Edit the repository URL in the script and run to generate `{repository_name}_analysis.json`.

#### Step 3: Run AURA Evaluation (Algorithm 4)
```bash
cd scripts/algorithm_4
streamlit run app.py
```
Access the web interface at `http://localhost:8501` to run comprehensive evaluations.

## 📊 Evaluation Methods

### 1. LLM-based Evaluation
- **Approach**: Uses large language models with chain-of-thought reasoning
- **Process**: 
  1. Parse conference guidelines to extract evaluation criteria
  2. Build vector database of artifact content
  3. Retrieve relevant content for each criterion
  4. Generate qualitative assessment with detailed reasoning
- **Strengths**: Context-aware analysis, detailed explanations, specific suggestions

### 2. Keyword-based Evaluation
- **Approach**: Quantitative scoring based on keyword presence in artifact content
- **Process**:
  1. Load evaluation criteria from Algorithm 1's CSV
  2. Extract all text content from artifact files
  3. Count keyword occurrences using word boundaries
  4. Apply log scaling and weighted scoring
- **Strengths**: Objective, reproducible, fast, consistent

### 3. Grounded Evaluation (Innovation)
- **Approach**: LLM evaluations grounded with keyword evidence
- **Process**: 
  1. Run keyword analysis to get quantitative evidence
  2. Include keyword evidence in LLM prompts
  3. Guide LLM to be consistent with keyword findings
  4. Prevent hallucination by grounding in actual evidence

## 🎛️ Evaluation Modes

### Full Evaluation
- Runs both LLM and keyword-based assessments
- Provides comprehensive evaluation with grounding
- Best for complete artifact assessment

### LLM Only
- Qualitative evaluation without keyword grounding
- Faster execution
- Good for initial assessment

### Keyword Only
- Quantitative baseline evaluation
- Fast and objective
- Good for large-scale studies

### Comparison Mode
- Side-by-side comparison of both methods
- Helps understand differences between approaches
- Good for method validation

### Grounded Evaluation
- LLM evaluations with keyword evidence integration
- Shows grounding information
- Best for detailed analysis

## 📁 Project Structure

```
AURA/
├── data/
│   └── conference_guideline_texts/
│       └── processed/                    # Conference guideline markdown files
├── scripts/
│   ├── algorithm_1/                     # Criteria extraction
│   │   ├── improved_algorithm_1.py
│   │   └── README.md
│   ├── algorithm_2/                     # Repository analysis
│   │   ├── algorithm_2.py
│   │   └── README.md
│   └── algorithm_4/                     # AURA framework
│       ├── app.py                       # Streamlit web interface
│       ├── aura_framework.py            # Core framework
│       ├── agents/                      # Evaluation agents
│       └── README.md
├── algo_outputs/                        # Generated outputs
│   ├── algorithm_1_output/
│   │   └── algorithm_1_artifact_evaluation_criteria.csv
│   ├── algorithm_2_output/
│   │   └── {repository_name}_analysis.json
│   └── algorithm_4_output/
└── requirements.txt
```

## 🔧 Configuration

### Conference Guidelines
Place conference guideline markdown files in `data/conference_guideline_texts/processed/`
- Example: `13_icse_2025.md`, `9_fse_2024.md`

### Repository Analysis
Configure Algorithm 2 parameters in `scripts/algorithm_2/algorithm_2.py`:
- `max_files_per_dir`: Maximum files per directory (default: 5)
- `max_file_size_kb`: Maximum file size to analyze (default: 2048KB)
- `allowed_extensions`: File types to include

### AURA Framework
Configure evaluation parameters in the Streamlit interface:
- Conference guidelines path
- Artifact JSON path
- Criteria CSV path
- Evaluation dimensions
- Evaluation mode

## 📈 Research Contributions

1. **Automated Criteria Extraction**: First system to automatically extract and standardize evaluation criteria from conference guidelines
2. **Dual Evaluation Approach**: Combines qualitative LLM reasoning with quantitative keyword scoring
3. **Evidence Grounding**: Novel approach to prevent LLM hallucination in artifact evaluation
4. **Conference-Agnostic Design**: Works across different conferences with varying guidelines
5. **Transparent Evaluation**: Provides detailed reasoning and evidence for all scores

## 🎯 Practical Impact

This framework addresses real problems in academic research:

- **Reduces evaluation burden** on conference artifact evaluation committees
- **Improves consistency** across different evaluators and conferences
- **Enhances reproducibility** by ensuring artifacts meet quality standards
- **Provides actionable feedback** to researchers for improving their artifacts
- **Scales evaluation** to handle large numbers of submissions

## 🔬 Technical Stack

### Algorithm 1
- **NLTK**: Text preprocessing and tokenization
- **KeyBERT**: Seed keyword generation
- **SentenceTransformers**: Semantic similarity for keyword expansion
- **TF-IDF**: Term frequency analysis
- **Scikit-learn**: Vectorization and scoring

### Algorithm 2
- **GitPython**: Repository cloning and management
- **AnyTree**: Directory tree generation
- **Mimetypes**: File type detection
- **JSON**: Structured output format

### Algorithm 4
- **LangChain**: LLM integration and chain management
- **OpenAI**: Language model access
- **Chroma**: Vector database for semantic search
- **Streamlit**: Web interface
- **Pandas**: Data manipulation

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're running from the correct directory
   - Check that all dependencies are installed

2. **File Path Issues**
   - Use absolute paths or ensure relative paths are correct
   - Check that input files exist in expected locations

3. **OpenAI API Issues**
   - Verify your API key is valid and has sufficient quota
   - Check network connectivity

4. **Memory Issues**
   - Reduce chunk sizes in agent initialization
   - Process smaller repositories first

### Debug Mode
Enable detailed logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Example Usage

### Basic Evaluation Pipeline

```python
# 1. Extract criteria from conference guidelines
# Run: python scripts/algorithm_1/improved_algorithm_1.py

# 2. Analyze repository
# Run: python scripts/algorithm_2/algorithm_2.py

# 3. Run AURA evaluation
from scripts.algorithm_4.aura_framework import AURA

aura = AURA(
    guideline_path="data/conference_guideline_texts/processed/13_icse_2025.md",
    artifact_json_path="algo_outputs/algorithm_2_output/ml-image-classifier_analysis.json",
    criteria_csv_path="algo_outputs/algorithm_1_output/algorithm_1_artifact_evaluation_criteria.csv"
)

# Full evaluation
results = aura.evaluate()

# Grounded evaluation for specific dimension
grounded_result = aura.get_grounded_evaluation("documentation")
```

### Web Interface
```bash
cd scripts/algorithm_4
streamlit run app.py
```

## 🤝 Contributing

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/your-org/aura/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/aura/discussions)

---

**AURA Framework** - Making artifact evaluation transparent, reproducible, and evidence-based. 🚀
