import glob
import logging
import os
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess_texts(texts):
    stop_words = set(stopwords.words('english'))
    preprocessed = []

    for i, text in enumerate(texts):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
        cleaned = " ".join(tokens)
        if cleaned.strip():
            preprocessed.append(cleaned)

    return preprocessed


def compute_tfidf_matrix(texts):
    if not texts:
        raise ValueError("No valid documents to process after preprocessing.")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    return tfidf_matrix, terms, vectorizer


def find_similar_terms(seed_term, tfidf_vector, terms, top_n=5):
    if seed_term not in terms:
        return []
    seed_idx = list(terms).index(seed_term)
    seed_vec = tfidf_vector[:, seed_idx].toarray()
    similarities = cosine_similarity(tfidf_vector.T, seed_vec.T).flatten()
    similar_idx = similarities.argsort()[-top_n:]
    return [terms[i] for i in similar_idx]


def get_keyword_score(tfidf_matrix, terms, keywords):
    term_indices = [np.where(terms == k)[0][0] for k in keywords if k in terms]
    return tfidf_matrix[:, term_indices].sum()


def extract_evaluation_criteria(texts, dimension_seeds):
    preprocessed = preprocess_texts(texts)
    tfidf_matrix, terms, vectorizer = compute_tfidf_matrix(preprocessed)

    E = []
    total_score = 0
    raw_scores = {}

    for dim, seeds in dimension_seeds.items():
        keyword_set = set(seeds)
        for seed in seeds:
            keyword_set.update(find_similar_terms(seed, tfidf_matrix, terms))

        score = get_keyword_score(tfidf_matrix, terms, keyword_set)

        raw_scores[dim] = {"keywords": list(keyword_set), "score": score}
        total_score += score

    E_raw = []
    for dim, info in raw_scores.items():
        weight = info["score"] / total_score if total_score > 0 else 0
        E_raw.append({
            "dimension": dim,
            "keywords": info["keywords"],
            "raw_score": info["score"],
            "normalized_weight": weight
        })

    return E_raw


# === Updated: Load .md files from processed directory ===
script_dir = os.path.dirname(os.path.abspath(__file__))
base_processed_dir = os.path.join(script_dir, "..", "..", "data", "conference_guideline_texts", "processed")
pattern = os.path.join(base_processed_dir, "*.md")

all_guideline_files = glob.glob(pattern)
logging.info(f"Found {len(all_guideline_files)} processed .md files.")

# Read the contents
conference_texts = []
for path in all_guideline_files:
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
        conference_texts.append(content)

# === Evaluation criteria seeds ===
seed_keywords = {
    "reproducibility": ["reproduce", "script", "environment", "data"],
    "documentation": ["readme", "manual", "guide"],
    "accessibility": ["available", "open", "access"],
    "usability": ["user", "interface", "demo"],
    "experimental": ["evaluation", "experiment", "result"],
    "functionality": ["feature", "output", "correct"]
}

if not any(conference_texts):
    logging.error("No content found. Please check your input directory.")

evaluation_criteria = extract_evaluation_criteria(conference_texts, seed_keywords)
logging.info(f"Evaluation criteria extracted:\n{evaluation_criteria}")

# Convert evaluation_criteria to DataFrame
df = pd.DataFrame(evaluation_criteria)

# Optional: Convert keyword lists to comma-separated strings for cleaner CSV
df["keywords"] = df["keywords"].apply(lambda kw: ", ".join(kw))

# Save to CSV
output_path = os.path.join(script_dir, "..", "..", "algo_outputs", "algorithm_1_output",
                           "algorithm_1_artifact_evaluation_criteria.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_csv(output_path, index=False)

logging.info(f"Saved evaluation results to: {output_path}")
