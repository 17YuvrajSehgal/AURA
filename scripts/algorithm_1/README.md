# Algorithm 1: Automated Extraction of Evaluation Criteria from Conference Guidelines

## Overview

**Purpose:**  
This algorithm automatically extracts and weights artifact evaluation criteria from a diverse set of research conference guidelines.  
It uses NLP and embedding-based techniques to identify the most important assessment dimensions, expands their keyword sets, and generates a normalized scoring table to drive standardized, conference-aware artifact evaluation.

---

## Motivation

- Artifact evaluation criteria vary widely across conferences and are often described in natural language, making manual harmonization difficult.
- Objective, automated extraction of evaluation dimensions and their relative weights enables **consistent, fair, and transparent artifact grading**.

---

## Key Steps

1. **Data Collection**
   - Reads all processed conference guideline `.md` files (e.g., from `data/conference_guideline_texts/processed/`).

2. **Text Preprocessing**
   - Tokenizes, lowercases, and removes stopwords/punctuation from the guidelines using NLTK.

3. **TF-IDF Matrix Computation**
   - Builds a TF-IDF matrix across all preprocessed guideline texts.
   - Extracts the vocabulary/terms for further analysis.

4. **Seed Keyword Generation**
   - For each evaluation dimension (e.g., reproducibility, documentation, accessibility), a set of **seed phrases** is created by running KeyBERT on canonical example sentences.
   - This ensures each dimension starts with a representative core set of keywords.

5. **Semantic Expansion of Keywords**
   - Each seed keyword is semantically expanded using a SentenceTransformer model (`all-MiniLM-L6-v2`).
   - Cosine similarity is computed between the seed and all TF-IDF terms; the top-N most similar terms are added to the keyword set for that dimension.

6. **Raw Scoring**
   - For each dimension, the algorithm sums the TF-IDF scores across all keywords (expanded set) to compute a **raw score** (importance) for that dimension within the corpus.

7. **Weight Normalization**
   - All raw scores are normalized across dimensions, producing a **relative weight** for each dimension.

8. **Tabular Output**
   - The algorithm outputs a table (CSV) listing:
     - Dimension name
     - Associated (expanded) keywords
     - Raw TF-IDF score
     - Normalized weight (sum to 1.0 across all dimensions)

---

## Example Output Table

| dimension       | keywords (sample)         | raw_score | normalized_weight |
|-----------------|--------------------------|-----------|------------------|
| reproducibility | users, reproduce, ...     | 17.4      | 0.27             |
| documentation   | documentation, guide ...  | 6.3       | 0.10             |
| ...             | ...                      | ...       | ...              |

---

## Pipeline Details

### 1. **Preprocessing**
- Uses NLTK to tokenize, lowercase, and remove English stopwords and punctuation.

### 2. **TF-IDF Calculation**
- All guideline texts are vectorized.
- `TfidfVectorizer` from scikit-learn is used to calculate term frequencies.

### 3. **Seed and Expansion**
- **Seed keywords** for each dimension are generated using KeyBERT on dimension-defining example sentences.
- **Expansion:** For each seed, the top-N most semantically similar terms in the TF-IDF vocabulary are added (using embeddings).

### 4. **Scoring**
- For each dimension, sums the TF-IDF scores for all its keywords across the corpus.

### 5. **Normalization**
- Raw scores are divided by the total to yield dimension weights.

### 6. **Saving Results**
- Final dimension/keyword/weight table is saved as `algorithm_1_artifact_evaluation_criteria.csv`.

---

## Usage

**Run the script:**

```bash
python algorithm_1_extract_criteria.py
```
python algorithm_1_extract_criteria.py

**Dependencies:**
- `nltk`
- `sentence-transformers`
- `keybert`
- `scikit-learn`
- `pandas`
- `numpy`

> Ensure your processed conference guideline markdown files are in the correct directory:  
> `data/conference_guideline_texts/processed/`
