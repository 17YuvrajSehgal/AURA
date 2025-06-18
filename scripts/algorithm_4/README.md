# AURA: Agentic Artifact Evaluation and Documentation Pipeline

## Overview

**AURA** (AI-powered Unified Research Artifact Evaluation) is a modular, conference-aware framework for automating research artifact assessment and documentation generation. Leveraging a Neo4j knowledge graph of conference guidelines, multi-agent LLM workflows, and retrieval-augmented evaluation, AURA produces:

- **Reviewer-quality READMEs** tailored to conference requirements.  
- **Automated artifact grading** across configurable dimensions.  
- **Rich Markdown reports** with detailed feedback and scores.

---

## Motivation

Manual review of research artifacts (code, data, documentation) can be slow, inconsistent, and subjective. Conferences define nuanced criteria for reproducibility, usability, and more—but enforcement at scale is challenging. AURA:

- Ensures **strict, evidence-driven evaluation** without assumptions.  
- Automates README generation via **agentic prompt chaining** (author, editor, critic agents).  
- Produces **transparent, weighted scores** aligned to each conference’s priorities.

---

## Architecture Overview

1. **Knowledge Graph Ingestion**  
   - Parse processed `.md` guidelines and criteria CSVs.  
   - Load into Neo4j as:  
     - `(:Conference)-[:REQUIRES_SECTION]->(:Section)`  
     - `(:Conference)-[:USES_DIMENSION]->(:Dimension)`  
     - `(:Dimension)-[:HAS_KEYWORD]->(:Keyword)`

2. **Agentic README Generation**  
   - For each required section:  
     - **Author Agent** drafts based on code, docs, and guidelines.  
     - **Editor Agent** refines clarity and completeness.  
     - **Critic Agent** reviews, scores, and prompts revisions.  
   - Chains context to ensure consistency across sections.

3. **RAG + LLM Artifact Evaluation**  
   - **Retrieve**: Load precomputed analysis JSON (documentation, code, license content).  
   - **Load Criteria**: Query Neo4j for sections, dimensions (with weights & keywords).  
   - **Evaluate** each dimension using GPT-4, guided by strict “Do NOT assume” prompts.  
   - **Weight & Aggregate** scores into a final artifact grade.

4. **Reviewer Report Generation**  
   - Compile:  
     - Final normalized score.  
     - Dimension-level breakdown with feedback.  
     - Full generated README.  
   - Export as Markdown (or convert to HTML/PDF).

---

## Update: Full Repository Evaluation

An extension to AURA now evaluates the **entire artifact** using a precomputed analysis JSON, rather than only the README. The new script `eval_full_repo.py` implements this:
