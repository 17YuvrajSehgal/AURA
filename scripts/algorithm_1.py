import glob
import logging
import os
import string

import nltk
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
        #logging.info(f"File {i + 1}: {len(tokens)} tokens after cleaning: {cleaned[:100]}")
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

        score = sum(vectorizer.vocabulary_.get(k, 0) for k in keyword_set)
        raw_scores[dim] = {"keywords": list(keyword_set), "score": score}
        total_score += score

    for dim, info in raw_scores.items():
        weight = info["score"] / total_score if total_score > 0 else 0
        E.append((dim, info["keywords"], weight))

    return E


# Base folder where raw guidelines are stored
script_dir = os.path.dirname(os.path.abspath(__file__))
base_raw_dir = os.path.join(script_dir, "..", "data","conference_guideline_texts","raw")
base_processed_dir = os.path.join(script_dir, "..", "data","conference_guideline_texts","processed")

base_directory = base_raw_dir

# Conference subfolders
conference_types = ["acm_conferences", "non_acm_conferences"]

all_guideline_files = []

# Collect all final.txt files from each conference type
for conf_type in conference_types:
    folder = os.path.join(base_directory, conf_type)
    pattern = os.path.join(folder, "**", "final.txt")
    files = glob.glob(pattern, recursive=True)
    logging.info(f"Found {len(files)} files in {conf_type}")
    all_guideline_files.extend(files)

logging.info(f"Total guideline files found: {len(all_guideline_files)}")

# Read the contents
conference_texts = []
for path in all_guideline_files:
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
        #logging.info(f"RAW [{path}]: {repr(content[:100])}")  # Preview first 100 characters
        conference_texts.append(content)

seed_keywords = {
    "reproducibility": ["reproduce", "script", "environment", "data"],
    "documentation": ["readme", "manual", "guide"],
    "accessibility": ["available", "open", "access"],
    "usability": ["user", "interface", "demo"],
    "experimental": ["evaluation", "experiment", "result"],
    "functionality": ["feature", "output", "correct"]
}

if not any(conference_texts):
    logging.warning("No content found. Injecting fallback test content.")
    conference_texts = [
        "This artifact must include source code and a README file. To reproduce results, Docker and scripts are required.",
        "Documentation should be detailed. Reproducibility is important. Code should be tested and runnable.",
    ]

evaluation_criteria = extract_evaluation_criteria(conference_texts, seed_keywords)
logging.info(f"Evaluation criteria: {evaluation_criteria}")

