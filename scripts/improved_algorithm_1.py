# Re-execute necessary setup after environment reset
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
from keybert import KeyBERT

nltk.download('punkt')
nltk.download('stopwords')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(sentence_model)


def preprocess_texts(texts):
    stop_words = set(stopwords.words('english'))
    preprocessed = []

    for text in texts:
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
    seed_embedding = sentence_model.encode(seed_term, convert_to_tensor=True)
    term_embeddings = sentence_model.encode(all_terms, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(seed_embedding, term_embeddings)[0]
    top_indices = cosine_scores.topk(top_n).indices
    return [all_terms[i] for i in top_indices]


def get_keyword_score(tfidf_matrix, terms, keywords):
    term_indices = [np.where(terms == k)[0][0] for k in keywords if k in terms]
    return tfidf_matrix[:, term_indices].sum()


def extract_evaluation_criteria(texts, dimension_seeds):
    preprocessed = preprocess_texts(texts)
    tfidf_matrix, terms, vectorizer = compute_tfidf_matrix(preprocessed)

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


# === Simulated Data ===
sample_texts = [
    "This artifact must include a README file and code scripts. Users should be able to reproduce the results using the environment provided.",
    "Ensure the documentation is complete and accessible. The artifact should offer a demo and include reproducibility instructions."
]

dimension_examples = {
    "reproducibility": [
        "This artifact should allow users to reproduce the results with minimal effort. All scripts, datasets, and environment variables must be provided."
    ],
    "documentation": [
        "A detailed README file with setup instructions and API usage must be included. The documentation should be beginner-friendly."
    ],
    "accessibility": [
        "Ensure all components are publicly available. Do not restrict access to datasets, code, or dependencies."
    ],
    "usability": [
        "The artifact should include a user interface or example demo. The installation process should be straightforward."
    ],
    "experimental": [
        "The paper must be supported by rigorous experiments and statistical evaluation. Include charts, benchmarks, and reproducibility metrics."
    ],
    "functionality": [
        "Code must perform its intended function correctly. Include unit tests or verification examples to validate outputs."
    ]
}


def generate_seed_keywords_from_examples(example_dict, top_n=5):
    seed_dict = {}
    for dim, texts in example_dict.items():
        combined_text = " ".join(texts)
        keywords = kw_model.extract_keywords(combined_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
        seed_dict[dim] = [kw[0] for kw in keywords]
    return seed_dict


auto_seed_keywords = generate_seed_keywords_from_examples(dimension_examples)
df_auto = extract_evaluation_criteria(sample_texts, auto_seed_keywords)

# Save to CSV
script_dir = os.getcwd()
output_path_auto = os.path.join(script_dir, "evaluation_criteria_semantic_auto.csv")
os.makedirs(os.path.dirname(output_path_auto), exist_ok=True)
df_auto.to_csv(output_path_auto, index=False)

print(df_auto.head())
