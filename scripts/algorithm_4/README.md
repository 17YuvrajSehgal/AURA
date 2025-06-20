# AURA: Unified Artifact Research Assessment Framework

AURA is a modular, extensible framework for automated evaluation of research software artifacts according to conference-specific guidelines. It leverages state-of-the-art LLMs and vector search to provide detailed, multi-dimensional assessments of artifact quality, supporting artifact evaluation committees, authors, and reproducibility initiatives.

---

## Table of Contents
- [Overview](#overview)
- [Evaluation Dimensions](#evaluation-dimensions)
  - [Documentation](#documentation)
  - [Usability](#usability)
  - [Accessibility](#accessibility)
  - [Functionality](#functionality)
- [How AURA Works](#how-aura-works)
  - [Architecture](#architecture)
  - [Agent Details](#agent-details)
- [How Results Are Merged and Utilized](#how-results-are-merged-and-utilized)
- [How to Use](#how-to-use)
- [Customization & Extensibility](#customization--extensibility)
- [File Structure](#file-structure)
- [References](#references)

---

## Overview
AURA automates the evaluation of research artifacts (code, data, documentation, etc.) by:
- Parsing conference guidelines to extract evaluation criteria.
- Analyzing the submitted artifact (in JSON format) for compliance.
- Providing detailed, evidence-based scores and suggestions for improvement across multiple dimensions.

AURA is designed to be:
- **Conference-agnostic**: Works with any set of guidelines.
- **Modular**: Each evaluation dimension is handled by a dedicated agent.
- **Transparent**: Chain-of-thought reasoning and evidence are provided for each score.

---

## Evaluation Dimensions
AURA evaluates artifacts along four primary dimensions, each mapped to a dedicated agent:

### Documentation
- **Goal**: Assess the completeness, clarity, and structure of the artifact's documentation (especially the README).
- **Checks**: Presence of required sections (purpose, setup, usage, provenance, etc.) as dictated by the conference guidelines.
- **Agent**: [`documentation_evaluation_agent.py`](agents/documentation_evaluation_agent.py)

### Usability
- **Goal**: Evaluate how easy it is for a user to install, configure, and use the artifact.
- **Checks**: Clarity of installation instructions, presence and completeness of setup scripts, requirements files, Dockerfiles, etc.
- **Agent**: [`usability_evaluation_agent.py`](agents/usability_evaluation_agent.py)

### Accessibility
- **Goal**: Determine whether the artifact is accessible to the community.
- **Checks**: Public availability (e.g., DOI, Zenodo, GitHub), clarity of dependency listings, installability, and repository structure.
- **Agent**: [`accessibility_evaluation_agent.py`](agents/accessibility_evaluation_agent.py)

### Functionality
- **Goal**: Assess whether the artifact does what it claims, can be executed as described, and produces expected results.
- **Checks**: Presence of main scripts, test results, output examples, and evidence supporting claimed functionality.
- **Agent**: [`functionality_evaluation_agent.py`](agents/functionality_evaluation_agent.py)

---

## How AURA Works

### Architecture
- **User Interface**: [`app.py`](app.py) provides a Streamlit-based web UI for inputting paths to guidelines and artifact JSON, selecting evaluation dimensions, and viewing results.
- **Framework Core**: [`aura_framework.py`](aura_framework.py) orchestrates the evaluation by initializing agents and merging their results.
- **Agents**: Each agent (see above) is responsible for one evaluation dimension, using LLMs and vector search to analyze the artifact.

#### Data Flow
1. **Input**: User provides paths to the conference guidelines (Markdown) and artifact analysis (JSON).
2. **Agent Initialization**: Each agent loads the guidelines and artifact, builds a vector database of relevant files, and extracts evaluation criteria using LLMs.
3. **Evaluation**: Each agent constructs a chain-of-thought prompt and queries the LLM, retrieving evidence from the artifact's files.
4. **Scoring & Justification**: Agents output detailed scores, justifications, and suggestions for improvement.
5. **Merging Results**: The `AURA` class collects results from all selected agents and presents them in the UI.

### Agent Details
- **Criteria Extraction**: Agents use LLMs to parse guidelines and extract a structured list of evaluation criteria.
- **Vector Search**: Artifact files are chunked and indexed for semantic retrieval, allowing the LLM to ground its answers in the actual content.
- **Chain-of-Thought Reasoning**: Prompts guide the LLM to reason step-by-step, referencing evidence from the artifact.
- **Logging**: Each agent logs its process and results for transparency and debugging.

---

## How Results Are Merged and Utilized
- The `AURA` class (see [`aura_framework.py`](aura_framework.py)) exposes an `evaluate()` method that:
  - Accepts a list of dimensions (e.g., `["documentation", "usability"]`).
  - Calls the corresponding agent's `evaluate()` method for each dimension.
  - Collects and merges the results into a dictionary, mapping each dimension to its detailed evaluation report.
- The UI displays each dimension's results in a separate section, with scores, justifications, and suggestions.
- Results can be used by:
  - **Artifact authors**: To improve their submissions.
  - **Evaluation committees**: For transparent, reproducible, and evidence-based artifact assessment.
  - **Meta-researchers**: For large-scale studies of artifact quality and reproducibility.

---

## How to Use

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Set up your OpenAI API key (for LLM access) in a `.env` file:
  ```env
  OPENAI_API_KEY=your-key-here
  ```

### Running the App
1. Place your conference guidelines (Markdown) and artifact analysis (JSON) in accessible paths.
2. Launch the Streamlit app:
   ```bash
   streamlit run scripts/algorithm_4/app.py
   ```
3. In the sidebar, specify:
   - Path to guidelines (e.g., `../../data/conference_guideline_texts/processed/13_icse_2025.md`)
   - Path to artifact JSON (e.g., `C:\workplace\AURA\algo_outputs\algorithm_2_output\ml-image-classifier_analysis.json`)
   - Conference name
   - Evaluation dimensions
4. Click **Evaluate** to run the assessment. Results will appear in the main panel.

---

## Customization & Extensibility
- **Add new dimensions**: Implement a new agent following the pattern in the `agents/` directory and update `aura_framework.py`.
- **Change LLM or embeddings**: Modify the agent code to use a different model or vector store.
- **Adapt prompts**: Edit the chain-of-thought prompts in each agent for different evaluation philosophies.
- **Integrate with other UIs**: The `AURA` class can be imported and used in other Python applications or APIs.

---

## File Structure
```
scripts/algorithm_4/
  app.py                # Streamlit UI
  aura_framework.py     # Core framework logic
  agents/
    accessibility_evaluation_agent.py
    documentation_evaluation_agent.py
    functionality_evaluation_agent.py
    usability_evaluation_agent.py
  ...
```

---

## References
- [AURA Paper/Docs (if available)](https://github.com/your-org/aura)
- [Artifact Evaluation at ACM/IEEE Conferences](https://artifact-eval.org/)
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

For questions or contributions, please open an issue or pull request!