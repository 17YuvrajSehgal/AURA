import glob
import json
import logging
import os
import string
from datetime import datetime
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Ensure logs directory exists and set up logging
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "..", "..", "algo_outputs", "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(logs_dir, "algorithm_1_execution.log")

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(sentence_model)


def preprocess_texts(texts):
    """Preprocess text documents by removing stopwords and punctuation."""
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
    """Compute TF-IDF matrix from preprocessed texts."""
    if not texts:
        raise ValueError("No valid documents to process after preprocessing.")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    return tfidf_matrix, terms, vectorizer


def find_semantic_similar_terms(seed_term, all_terms, top_n=5):
    """Find semantically similar terms using sentence embeddings."""
    seed_embedding = sentence_model.encode(seed_term, convert_to_tensor=True)
    term_embeddings = sentence_model.encode(all_terms, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(seed_embedding, term_embeddings)[0]
    top_indices = cosine_scores.topk(top_n).indices
    return [all_terms[i] for i in top_indices]


def get_keyword_score(tfidf_matrix, terms, keywords):
    """Calculate keyword score from TF-IDF matrix."""
    term_indices = [np.where(terms == k)[0][0] for k in keywords if k in terms]
    return tfidf_matrix[:, term_indices].sum()


def extract_evaluation_criteria(texts, dimension_seeds):
    """Extract evaluation criteria using TF-IDF and semantic similarity."""
    logger.info("Starting evaluation criteria extraction...")
    
    preprocessed = preprocess_texts(texts)
    logger.info(f"Preprocessed {len(preprocessed)} documents")
    
    tfidf_matrix, terms, vectorizer = compute_tfidf_matrix(preprocessed)
    logger.info(f"Computed TF-IDF matrix with {len(terms)} unique terms")
    
    total_score = 0
    raw_scores = {}
    extraction_details = {}
    
    for dim, seeds in dimension_seeds.items():
        logger.info(f"Processing dimension: {dim}")
        keyword_set = set(seeds)
        similar_terms_by_seed = {}
        
        for seed in seeds:
            similar_terms = find_semantic_similar_terms(seed, list(terms), top_n=5)
            keyword_set.update(similar_terms)
            similar_terms_by_seed[seed] = similar_terms
            logger.debug(f"Seed '{seed}' -> Similar terms: {similar_terms}")
        
        score = get_keyword_score(tfidf_matrix, terms, keyword_set)
        raw_scores[dim] = {"keywords": list(keyword_set), "score": score}
        extraction_details[dim] = {
            "seeds": seeds,
            "similar_terms_by_seed": similar_terms_by_seed,
            "total_keywords": len(keyword_set),
            "score": score
        }
        total_score += score
        logger.info(f"Dimension '{dim}': {len(keyword_set)} keywords, score: {score:.2f}")
    
    # Create DataFrame
    E_raw = []
    for dim, info in raw_scores.items():
        weight = info["score"] / total_score if total_score > 0 else 0
        E_raw.append({
            "dimension": dim,
            "keywords": ", ".join(info["keywords"]),
            "raw_score": float(info["score"]),
            "normalized_weight": float(weight)
        })
    
    df_result = pd.DataFrame(E_raw)
    logger.info(f"Extraction complete. Total score: {total_score:.2f}")
    
    return df_result, extraction_details, {
        "total_documents": len(texts),
        "total_terms": len(terms),
        "total_score": total_score,
        "dimensions_processed": len(dimension_seeds)
    }


def save_outputs(df_result, extraction_details, processing_stats, output_dir):
    """Save outputs in multiple formats with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save CSV with timestamp
    csv_filename = f"algorithm_1_artifact_evaluation_criteria_{timestamp}.csv"
    csv_path = output_path / csv_filename
    df_result.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV results to: {csv_path}")
    
    # 2. Save latest CSV (without timestamp for compatibility)
    latest_csv_path = output_path / "algorithm_1_artifact_evaluation_criteria.csv"
    df_result.to_csv(latest_csv_path, index=False)
    logger.info(f"Saved latest CSV results to: {latest_csv_path}")
    
    # 3. Save detailed JSON with metadata
    json_filename = f"algorithm_1_detailed_results_{timestamp}.json"
    json_path = output_path / json_filename
    
    detailed_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "script_version": "improved_algorithm_1.py",
            "processing_stats": processing_stats,
            "model_info": {
                "sentence_model": "all-MiniLM-L6-v2",
                "keybert_model": "all-MiniLM-L6-v2"
            }
        },
        "evaluation_criteria": df_result.to_dict('records'),
        "extraction_details": extraction_details,
        "summary": {
            "total_dimensions": len(df_result),
            "total_keywords": sum(len(row['keywords'].split(', ')) for _, row in df_result.iterrows()),
            "highest_weight_dimension": df_result.loc[df_result['normalized_weight'].idxmax(), 'dimension'],
            "lowest_weight_dimension": df_result.loc[df_result['normalized_weight'].idxmin(), 'dimension']
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved detailed JSON results to: {json_path}")
    
    # 4. Save processing report
    report_filename = f"algorithm_1_processing_report_{timestamp}.txt"
    report_path = output_path / report_filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("AURA Algorithm 1 - Processing Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script: improved_algorithm_1.py\n\n")
        
        f.write("Processing Statistics:\n")
        f.write(f"- Total documents processed: {processing_stats['total_documents']}\n")
        f.write(f"- Total unique terms: {processing_stats['total_terms']}\n")
        f.write(f"- Total score: {processing_stats['total_score']:.2f}\n")
        f.write(f"- Dimensions processed: {processing_stats['dimensions_processed']}\n\n")
        
        f.write("Evaluation Criteria Summary:\n")
        f.write("-" * 30 + "\n")
        for _, row in df_result.iterrows():
            f.write(f"{row['dimension'].upper()}:\n")
            f.write(f"  - Raw Score: {row['raw_score']:.2f}\n")
            f.write(f"  - Normalized Weight: {row['normalized_weight']:.3f}\n")
            f.write(f"  - Keywords: {row['keywords']}\n\n")
        
        f.write("Files Generated:\n")
        f.write("-" * 20 + "\n")
        f.write(f"- CSV Results: {csv_filename}\n")
        f.write(f"- Latest CSV: algorithm_1_artifact_evaluation_criteria.csv\n")
        f.write(f"- Detailed JSON: {json_filename}\n")
        f.write(f"- Processing Report: {report_filename}\n")
    
    logger.info(f"Saved processing report to: {report_path}")
    
    return {
        "csv_path": str(csv_path),
        "latest_csv_path": str(latest_csv_path),
        "json_path": str(json_path),
        "report_path": str(report_path)
    }


def main():
    """Main execution function."""
    logger.info("Starting AURA Algorithm 1 - Enhanced Evaluation Criteria Extraction")
    
    # === Read actual files instead of sample text ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_processed_dir = os.path.join(script_dir, "..", "..", "data", "conference_guideline_texts", "processed")
    pattern = os.path.join(base_processed_dir, "*.md")

    all_guideline_files = glob.glob(pattern)
    logger.info(f"Found {len(all_guideline_files)} processed .md files: {[os.path.basename(f) for f in all_guideline_files]}")

    conference_texts = []
    file_metadata = []
    
    for path in all_guideline_files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                conference_texts.append(content)
                file_metadata.append({
                    "filename": os.path.basename(path),
                    "size_bytes": len(content),
                    "path": path
                })
                logger.info(f"Loaded: {os.path.basename(path)} ({len(content)} characters)")
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")

    if not any(conference_texts):
        logger.error("No content found. Please check your input directory.")
        return

    # === Dimension examples used to generate seeds ===
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

    logger.info("Generating seed keywords from dimension examples...")
    auto_seed_keywords = generate_seed_keywords_from_examples(dimension_examples)
    logger.info(f"Generated seeds: {auto_seed_keywords}")

    # Extract evaluation criteria
    df_result, extraction_details, processing_stats = extract_evaluation_criteria(
        conference_texts, auto_seed_keywords
    )

    # Save outputs
    output_dir = os.path.join(script_dir, "..", "..", "algo_outputs", "algorithm_1_output")
    saved_files = save_outputs(df_result, extraction_details, processing_stats, output_dir)
    
    # Final summary
    logger.info("=" * 60)
    logger.info("AURA Algorithm 1 - EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed {processing_stats['total_documents']} conference guideline files")
    logger.info(f"Extracted criteria for {processing_stats['dimensions_processed']} dimensions")
    logger.info(f"Total evaluation score: {processing_stats['total_score']:.2f}")
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("Files generated:")
    for file_type, path in saved_files.items():
        logger.info(f"  - {file_type}: {os.path.basename(path)}")
    logger.info("=" * 60)


def generate_seed_keywords_from_examples(example_dict, top_n=5):
    """Generate seed keywords from dimension examples using KeyBERT."""
    seed_dict = {}
    for dim, texts in example_dict.items():
        combined_text = " ".join(texts)
        keywords = kw_model.extract_keywords(
            combined_text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english',
            top_n=top_n
        )
        seed_dict[dim] = [kw[0] for kw in keywords]
        logger.debug(f"Generated seeds for {dim}: {seed_dict[dim]}")
    return seed_dict


if __name__ == "__main__":
    main()
