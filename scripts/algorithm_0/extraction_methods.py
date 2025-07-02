"""
Additional extraction methods for Conference-Specific Algorithm 1.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import NumpyEncoder

logger = logging.getLogger(__name__)


def extract_evaluation_criteria(texts: List[str], dimension_seeds: Dict[str, List[str]], 
                               sentence_model, kw_model) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Extract evaluation criteria using TF-IDF and semantic similarity."""
    logger.info("Starting enhanced evaluation criteria extraction...")

    # Preprocess texts (simplified version)
    preprocessed = [text.lower() for text in texts if text.strip()]
    logger.info(f"Preprocessed {len(preprocessed)} documents")

    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
        max_features=10000
    )
    
    tfidf_matrix = vectorizer.fit_transform(preprocessed)
    terms = vectorizer.get_feature_names_out()
    logger.info(f"Computed TF-IDF matrix with {len(terms)} unique terms")

    total_score = 0
    raw_scores = {}
    extraction_details = {}

    for dim, seeds in dimension_seeds.items():
        logger.info(f"Processing dimension: {dim}")
        
        # Find similar terms using semantic similarity
        all_keywords = set(seeds)
        
        # Expand keywords using semantic similarity
        for seed in seeds:
            if seed in terms:
                # Simple keyword expansion based on TF-IDF presence
                similar_terms = [term for term in terms 
                               if seed.lower() in term.lower() or term.lower() in seed.lower()]
                all_keywords.update(similar_terms[:5])  # Limit expansion

        # Calculate score for this dimension
        dimension_score = 0
        for keyword in all_keywords:
            if keyword in terms:
                keyword_idx = np.where(terms == keyword)[0][0]
                dimension_score += tfidf_matrix[:, keyword_idx].sum()

        raw_scores[dim] = {
            "keywords": list(all_keywords),
            "score": float(dimension_score),
            "count": len(all_keywords)
        }

        extraction_details[dim] = {
            "seeds": seeds,
            "expanded_keywords": list(all_keywords),
            "total_keywords": len(all_keywords),
            "score": float(dimension_score)
        }

        total_score += dimension_score
        logger.info(f"Dimension '{dim}': {len(all_keywords)} keywords, score: {dimension_score:.2f}")

    # Create DataFrame
    E_raw = []
    for dim, info in raw_scores.items():
        weight = info["score"] / total_score if total_score > 0 else 0
        E_raw.append({
            "dimension": dim,
            "keywords": ", ".join(info["keywords"]) if info["keywords"] else "",
            "raw_score": float(info["score"]),
            "normalized_weight": float(weight),
            "keyword_count": info["count"],
            "source_documents": len(texts)
        })

    df_result = pd.DataFrame(E_raw)
    logger.info(f"Enhanced extraction complete. Total score: {total_score:.2f}")

    return df_result, extraction_details, {
        "total_documents": len(texts),
        "total_terms": len(terms),
        "total_score": float(total_score),
        "dimensions_processed": len(dimension_seeds),
        "total_keywords_extracted": sum(len(info["keywords"]) for info in raw_scores.values())
    }


def apply_conference_weighting(df_result: pd.DataFrame, conference_profile: Dict) -> pd.DataFrame:
    """Apply conference-specific emphasis weights."""
    
    emphasis_weights = conference_profile.get("emphasis_weights", {})
    
    # Add conference-adjusted weights
    df_result['conference_adjusted_weight'] = df_result['normalized_weight'].copy()
    
    # Adjust weights based on conference emphasis
    for idx, row in df_result.iterrows():
        dimension = row['dimension']
        if dimension in emphasis_weights:
            # Apply emphasis factor
            emphasis_factor = emphasis_weights[dimension]
            df_result.at[idx, 'conference_adjusted_weight'] = emphasis_factor
    
    # Renormalize weights
    total_adjusted = df_result['conference_adjusted_weight'].sum()
    if total_adjusted > 0:
        df_result['conference_adjusted_weight'] = df_result['conference_adjusted_weight'] / total_adjusted
    
    return df_result


def generate_enhanced_outputs(df_result: pd.DataFrame, extraction_results: Dict,
                            processing_stats: Dict, output_dir: str, 
                            conference_name: str) -> Dict[str, str]:
    """Generate multiple output formats optimized for conference-specific evaluation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # 1. Enhanced CSV
    csv_filename = f"{conference_name.lower()}_criteria_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df_result.to_csv(csv_path, index=False)
    saved_files['enhanced_csv'] = csv_path
    logger.info(f"Saved enhanced CSV to: {csv_path}")

    # 2. AURA Integration Format
    aura_integration = create_aura_integration_format(df_result, conference_name)
    aura_filename = f"{conference_name.lower()}_aura_integration_{timestamp}.json"
    aura_path = os.path.join(output_dir, aura_filename)
    with open(aura_path, 'w', encoding='utf-8') as f:
        json.dump(aura_integration, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    saved_files['aura_integration'] = aura_path
    logger.info(f"Saved AURA integration data to: {aura_path}")

    # 3. Conference-Specific Analysis
    conference_analysis = {
        "conference_metadata": {
            "name": conference_name,
            "timestamp": timestamp,
            "total_documents": processing_stats.get('total_documents', 0),
            "total_keywords": processing_stats.get('total_keywords_extracted', 0)
        },
        "criteria_analysis": df_result.to_dict('records'),
        "extraction_details": extraction_results
    }
    
    analysis_filename = f"{conference_name.lower()}_analysis_{timestamp}.json"
    analysis_path = os.path.join(output_dir, analysis_filename)
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(conference_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    saved_files['conference_analysis'] = analysis_path
    logger.info(f"Saved conference analysis to: {analysis_path}")

    return saved_files


def create_aura_integration_format(df_result: pd.DataFrame, conference_name: str) -> Dict:
    """Create format optimized for AURA framework integration."""
    aura_data = {
        "conference_name": conference_name,
        "structured_criteria": [],
        "grounding_evidence": {},
        "conference_weights": {}
    }

    for _, row in df_result.iterrows():
        dimension = row['dimension']

        # Structured criteria
        criteria_entry = {
            "dimension": dimension,
            "keywords": row['keywords'].split(', ') if row['keywords'] else [],
            "raw_score": row['raw_score'],
            "normalized_weight": row['normalized_weight'],
            "conference_adjusted_weight": row.get('conference_adjusted_weight', row['normalized_weight']),
            "keyword_count": row.get('keyword_count', 0)
        }
        aura_data["structured_criteria"].append(criteria_entry)

        # Grounding evidence
        keywords_list = row['keywords'].split(', ') if row['keywords'] else []
        aura_data["grounding_evidence"][dimension] = {
            "keywords_found": keywords_list,
            "raw_score": row['raw_score'],
            "source_documents": row.get('source_documents', 0)
        }

        # Conference weights
        aura_data["conference_weights"][dimension] = row.get('conference_adjusted_weight', row['normalized_weight'])

    return aura_data 