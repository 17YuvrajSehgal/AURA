# Enhancement: Integrate Sentence-BERT semantic similarity into extract_evaluation_criteria
import logging
import os
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


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


def find_semantic_similar_terms(seed_term, all_terms, top_n=5):
    seed_embedding = model.encode(seed_term, convert_to_tensor=True)
    term_embeddings = model.encode(all_terms, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(seed_embedding, term_embeddings)[0]
    top_indices = cosine_scores.topk(top_n).indices
    return [all_terms[i] for i in top_indices]


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
            similar_terms = find_semantic_similar_terms(seed, list(terms), top_n=5)
            keyword_set.update(similar_terms)

        score = get_keyword_score(tfidf_matrix, terms, keyword_set)

        raw_scores[dim] = {"keywords": list(keyword_set), "score": score}
        total_score += score

    E_raw = []
    for dim, info in raw_scores.items():
        weight = info["score"] / total_score if total_score > 0 else 0
        E_raw.append({
            "dimension": dim,
            "keywords": ", ".join(info["keywords"]),
            "raw_score": float(info["score"]),
            "normalized_weight": float(weight)
        })

    return pd.DataFrame(E_raw)


# Simulate file input
sample_texts = [
    "This artifact must include a README file and code scripts. Users should be able to reproduce the results using the environment provided.",
    "Ensure the documentation is complete and accessible. The artifact should offer a demo and include reproducibility instructions."
]

seed_keywords = {
    "reproducibility": ["reproduce", "script", "environment", "data"],
    "documentation": ["readme", "manual", "guide"],
    "accessibility": ["available", "open", "access"],
    "usability": ["user", "interface", "demo"],
    "experimental": ["evaluation", "experiment", "result"],
    "functionality": ["feature", "output", "correct"]
}

script_dir = os.path.dirname(os.path.abspath(__file__))

df_output = extract_evaluation_criteria(sample_texts, seed_keywords)
# Save to CSV
output_path = os.path.join(script_dir, "..", "data", "algorithm_1_output", "evaluation_criteria_semantic.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df_output.to_csv(output_path, index=False)
logging.info(f"Saved enhanced evaluation results to: {output_path}")
