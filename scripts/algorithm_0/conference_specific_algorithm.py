"""
Conference-Specific Algorithm 1: Enhanced evaluation criteria extraction.
"""

import glob
import json
import logging
import os
import re
import string
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx
import nltk
import numpy as np
import pandas as pd
import torch
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    # Try relative imports first (when used as a package)
    from .conference_profiles import ConferenceProfileManager
    from .utils import NumpyEncoder, setup_logging
    from .config import Config
except ImportError:
    # Fall back to direct imports (when used as standalone)
    from conference_profiles import ConferenceProfileManager
    from utils import NumpyEncoder, setup_logging
    from config import Config

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

logger = logging.getLogger(__name__)


class ConferenceSpecificAlgorithm1:
    """
    Enhanced Algorithm 1 for conference-specific evaluation criteria extraction.
    
    This algorithm generates evaluation criteria tailored to specific conferences,
    enabling more accurate artifact evaluation based on conference-specific requirements.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', config: Config = None):
        """Initialize the conference-specific algorithm."""
        self.config = config or Config()
        self.sentence_model = SentenceTransformer(model_name)
        self.kw_model = KeyBERT(self.sentence_model)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Initialize conference profile manager with automatic profile generation
        # Get the guidelines directory relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        guidelines_dir = os.path.join(script_dir, "..", "..", "data", "conference_guideline_texts", "processed")
        profiles_file = os.path.join(script_dir, "generated_conference_profiles.json")
        
        self.profile_manager = ConferenceProfileManager(
            guidelines_dir=guidelines_dir,
            profiles_file=profiles_file
        )

        logger.info(f"Conference-Specific Algorithm 1 initialized with model: {model_name}")
        logger.info(f"Available conference profiles: {self.profile_manager.list_available_conferences()}")

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Enhanced text preprocessing with lemmatization and better cleaning."""
        preprocessed = []

        for text in texts:
            tokens = word_tokenize(text.lower())
            cleaned_tokens = []

            for token in tokens:
                if (token not in self.stop_words and
                        token not in string.punctuation and
                        len(token) > 2 and
                        not token.isnumeric()):
                    lemmatized = self.lemmatizer.lemmatize(token)
                    cleaned_tokens.append(lemmatized)

            cleaned = " ".join(cleaned_tokens)
            if cleaned.strip():
                preprocessed.append(cleaned)

        logger.info(f"Preprocessed {len(preprocessed)} documents")
        return preprocessed

    def extract_conference_metadata(self, file_paths: List[str]) -> Dict[str, Any]:
        """Extract metadata from conference guideline files."""
        metadata = {}

        for path in file_paths:
            filename = os.path.basename(path)
            match = re.match(r'(\d+)_([a-zA-Z]+)_(\d{4})', filename)

            if match:
                conf_id, conf_name, year = match.groups()
                metadata[filename] = {
                    'conference_id': conf_id,
                    'conference_name': conf_name.upper(),
                    'year': int(year),
                    'file_path': path,
                    'file_size': os.path.getsize(path)
                }
            else:
                metadata[filename] = {
                    'conference_name': 'UNKNOWN',
                    'year': None,
                    'file_path': path,
                    'file_size': os.path.getsize(path)
                }

        return metadata

    def run_conference_specific_extraction(self, input_dir: str, output_dir: str,
                                           target_conference: str = None) -> Dict[str, Any]:
        """
        Run extraction for specific conference or all conferences separately.
        
        Args:
            input_dir: Directory containing conference guideline files
            output_dir: Directory to save outputs
            target_conference: Specific conference to process (None for all)
            
        Returns:
            Dictionary containing extraction results
        """
        logger.info(f"Starting Conference-Specific AURA Algorithm 1")

        # Load conference guidelines
        all_guideline_files = glob.glob(os.path.join(input_dir, "*.md"))
        logger.info(f"Found {len(all_guideline_files)} guideline files")

        if not all_guideline_files:
            raise ValueError("No guideline files found in input directory")

        conference_metadata = self.extract_conference_metadata(all_guideline_files)

        results = {}

        if target_conference:
            # Extract criteria for specific conference
            target_files = {k: v for k, v in conference_metadata.items()
                            if v['conference_name'] == target_conference.upper()}

            if not target_files:
                raise ValueError(f"No files found for conference: {target_conference}")

            results[target_conference] = self._extract_single_conference_criteria(
                target_conference.upper(), target_files, output_dir
            )
        else:
            # Extract criteria for all conferences separately
            conferences_processed = set()
            for conf_file, metadata in conference_metadata.items():
                conf_name = metadata['conference_name']

                if conf_name not in conferences_processed and conf_name != 'UNKNOWN':
                    logger.info(f"Processing {conf_name}...")

                    # Get all files for this conference
                    conf_files = {k: v for k, v in conference_metadata.items()
                                  if v['conference_name'] == conf_name}

                    results[conf_name] = self._extract_single_conference_criteria(
                        conf_name, conf_files, output_dir
                    )
                    conferences_processed.add(conf_name)

        # Generate comparative analysis if multiple conferences
        if len(results) > 1:
            self._generate_cross_conference_analysis(results, output_dir)

        return results

    def _extract_single_conference_criteria(self, conference_name: str,
                                            conference_files: Dict, output_dir: str) -> Dict:
        """Extract criteria for a single conference."""

        logger.info(f"Extracting criteria for {conference_name}")

        # Load conference-specific texts
        conference_texts = []
        for file_path in [meta['file_path'] for meta in conference_files.values()]:
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    conference_texts.append(content)
                    logger.info(f"Loaded: {os.path.basename(file_path)} ({len(content)} characters)")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        if not conference_texts:
            raise ValueError(f"No content found for {conference_name}")

        # Get conference profile
        conference_profile = self.profile_manager.get_conference_profile(conference_name)

        # Generate conference-specific dimension examples
        dimension_examples = self._generate_conference_dimension_examples(conference_profile)

        # Generate hierarchical seed keywords with conference focus
        hierarchical_seeds = self._generate_hierarchical_seed_keywords(dimension_examples)

        # Apply conference-specific keyword enhancement
        enhanced_seeds = self._enhance_seeds_with_conference_profile(hierarchical_seeds, conference_profile)

        # Extract evaluation criteria
        df_result, extraction_results, processing_stats = \
            self._extract_evaluation_criteria(conference_texts, enhanced_seeds)

        # Apply conference-specific weighting
        df_result = self._apply_conference_weighting(df_result, conference_profile)

        # Calculate confidence metrics
        confidence_metrics = self._calculate_criteria_confidence(extraction_results, conference_files)

        # Build semantic relationship map
        all_keywords = []
        term_metadata = []
        for dim, results in extraction_results.items():
            for category, keywords in results['category_keywords'].items():
                for keyword in keywords:
                    all_keywords.append(keyword)
                    term_metadata.append({
                        'keyword': keyword,
                        'dimension': dim,
                        'category': category,
                        'weight': results['category_scores'][category]['score']
                    })

        semantic_relationships = self._build_semantic_relationship_map(all_keywords, term_metadata)

        # Generate conference-specific outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conf_output_dir = os.path.join(output_dir, f"{conference_name.lower()}_{timestamp}")
        os.makedirs(conf_output_dir, exist_ok=True)

        saved_files = self._generate_enhanced_outputs(
            df_result, extraction_results, processing_stats,
            confidence_metrics, semantic_relationships, conf_output_dir, conference_name
        )

        return {
            "conference_name": conference_name,
            "criteria_dataframe": df_result,
            "extraction_results": extraction_results,
            "confidence_metrics": confidence_metrics,
            "conference_profile": conference_profile,
            "saved_files": saved_files,
            "processing_stats": processing_stats,
            "semantic_relationships": semantic_relationships
        }

    def _generate_conference_dimension_examples(self, conference_profile: Dict) -> Dict[str, List[str]]:
        """Generate dimension examples tailored to conference profile."""

        base_examples = {
            "reproducibility": [
                "Artifact should enable replication of paper results",
                "All necessary scripts, data, and environment setup provided",
                "Results can be reproduced with minimal effort"
            ],
            "documentation": [
                "Clear README with setup and usage instructions",
                "API documentation and code comments",
                "Comprehensive installation guide"
            ],
            "accessibility": [
                "Publicly available with appropriate licensing",
                "No proprietary dependencies or restricted access",
                "Available in archival repository"
            ],
            "usability": [
                "Easy installation and execution process",
                "User-friendly interfaces and error handling",
                "Clear usage examples and tutorials"
            ],
            "experimental": [
                "Rigorous experimental design and validation",
                "Statistical analysis and performance benchmarks",
                "Comprehensive evaluation methodology"
            ],
            "functionality": [
                "Code performs intended function correctly",
                "Includes verification and testing evidence",
                "Demonstrable working implementation"
            ]
        }

        # Enhance examples with conference-specific focus
        category = conference_profile.get("category", "general")
        domain_keywords = conference_profile.get("domain_keywords", [])

        enhanced_examples = {}
        for dimension, examples in base_examples.items():
            enhanced = examples.copy()

            # Add domain-specific examples
            if category == "software_engineering":
                if dimension == "functionality":
                    enhanced.extend([
                        "Unit tests and integration tests included",
                        "Code follows software engineering best practices",
                        "Continuous integration and testing pipeline"
                    ])
                elif dimension == "reproducibility":
                    enhanced.extend([
                        "Build scripts and dependency management",
                        "Version-controlled source code",
                        "Containerized execution environment"
                    ])

            elif category == "data_systems":
                if dimension == "experimental":
                    enhanced.extend([
                        "Performance benchmarks on standard datasets",
                        "Scalability analysis and optimization metrics",
                        "Query performance and throughput analysis"
                    ])
                elif dimension == "functionality":
                    enhanced.extend([
                        "Database schema and query implementations",
                        "Data processing pipeline validation",
                        "System performance under load"
                    ])

            elif category == "human_computer_interaction":
                if dimension == "usability":
                    enhanced.extend([
                        "User study materials and protocols",
                        "Interface design justification and evaluation",
                        "Accessibility compliance and testing"
                    ])
                elif dimension == "experimental":
                    enhanced.extend([
                        "User study design and statistical analysis",
                        "Human subjects research protocol",
                        "Qualitative and quantitative evaluation"
                    ])

            enhanced_examples[dimension] = enhanced

        return enhanced_examples

    def _generate_hierarchical_seed_keywords(self, dimension_examples: Dict[str, List[str]]) -> Dict[
        str, Dict[str, List[str]]]:
        """Generate hierarchical seed keywords for each dimension."""
        hierarchical_seeds = {}

        for dimension, examples in dimension_examples.items():
            combined_text = " ".join(examples)

            # Extract keywords with different strategies
            core_keywords = self.kw_model.extract_keywords(
                combined_text,
                keyphrase_ngram_range=(1, 1),
                stop_words='english',
                top_n=10
            )

            semantic_keywords = self.kw_model.extract_keywords(
                combined_text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=15
            )

            contextual_keywords = self.kw_model.extract_keywords(
                combined_text,
                keyphrase_ngram_range=(2, 3),
                stop_words='english',
                top_n=10
            )

            hierarchical_seeds[dimension] = {
                'core_keywords': [kw[0] for kw in core_keywords[:5]],
                'semantic_keywords': [kw[0] for kw in semantic_keywords[:8]],
                'contextual_keywords': [kw[0] for kw in contextual_keywords[:5]],
                'domain_keywords': self._extract_domain_specific_keywords(dimension, examples)
            }

            logger.debug(f"Generated hierarchical seeds for {dimension}: {hierarchical_seeds[dimension]}")

        return hierarchical_seeds

    def _extract_domain_specific_keywords(self, dimension: str, examples: List[str]) -> List[str]:
        """Extract domain-specific keywords based on dimension type."""
        domain_keywords = []

        if dimension == 'reproducibility':
            domain_keywords = ['reproduce', 'replication', 'reproducible', 'reproducibility', 'replicate']
        elif dimension == 'documentation':
            domain_keywords = ['documentation', 'readme', 'guide', 'manual', 'tutorial']
        elif dimension == 'accessibility':
            domain_keywords = ['public', 'accessible', 'available', 'open', 'free']
        elif dimension == 'usability':
            domain_keywords = ['install', 'setup', 'use', 'demo', 'interface']
        elif dimension == 'experimental':
            domain_keywords = ['experiment', 'evaluation', 'benchmark', 'test', 'analysis']
        elif dimension == 'functionality':
            domain_keywords = ['function', 'test', 'verify', 'validate', 'correct']

        return domain_keywords

    def _enhance_seeds_with_conference_profile(self, hierarchical_seeds: Dict,
                                               conference_profile: Dict) -> Dict:
        """Enhance seed keywords with conference-specific terms."""

        domain_keywords = conference_profile.get("domain_keywords", [])
        enhanced_seeds = hierarchical_seeds.copy()

        for dimension, seed_categories in enhanced_seeds.items():
            # Add conference domain keywords to each category
            for category in seed_categories:
                if isinstance(seed_categories[category], list):
                    # Find domain-specific synonyms
                    domain_enhanced = []
                    for domain_kw in domain_keywords:
                        similar_terms = self._find_semantic_similar_terms(
                            domain_kw, seed_categories[category], top_n=2
                        )
                        domain_enhanced.extend(similar_terms)

                    # Add unique domain keywords
                    seed_categories[category] = list(set(seed_categories[category] + domain_enhanced))

        return enhanced_seeds

    def _find_semantic_similar_terms(self, seed_term: str, all_terms: List[str], top_n: int = 5) -> List[str]:
        """Find semantically similar terms using sentence embeddings."""
        if not all_terms:
            return []

        try:
            seed_embedding = self.sentence_model.encode(seed_term, convert_to_tensor=True)
            term_embeddings = self.sentence_model.encode(all_terms, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(seed_embedding, term_embeddings)[0]
            top_indices = cosine_scores.topk(min(top_n, len(all_terms))).indices
            return [all_terms[i] for i in top_indices]
        except Exception as e:
            logger.warning(f"Error finding similar terms for '{seed_term}': {e}")
            return []

    def _extract_evaluation_criteria(self, texts: List[str], dimension_seeds: Dict[str, Dict[str, List[str]]]) -> Tuple[
        pd.DataFrame, Dict, Dict]:
        """Extract evaluation criteria using TF-IDF and semantic similarity with hierarchical approach."""
        logger.info("Starting enhanced evaluation criteria extraction...")

        preprocessed = self.preprocess_texts(texts)
        logger.info(f"Preprocessed {len(preprocessed)} documents")

        tfidf_matrix, terms, vectorizer = self._compute_tfidf_matrix(preprocessed)
        logger.info(f"Computed TF-IDF matrix with {len(terms)} unique terms")

        total_score = 0
        raw_scores = {}
        extraction_details = {}

        for dim, seed_categories in dimension_seeds.items():
            logger.info(f"Processing dimension: {dim}")
            all_keywords = set()
            category_keywords = {}
            category_scores = {}

            # Process each category of keywords
            for category, seeds in seed_categories.items():
                category_keyword_set = set(seeds)
                similar_terms_by_seed = {}

                # Expand each seed keyword
                for seed in seeds:
                    if seed in terms:
                        similar_terms = self._find_semantic_similar_terms(seed, list(terms), top_n=5)
                        category_keyword_set.update(similar_terms)
                        similar_terms_by_seed[seed] = similar_terms
                        logger.debug(f"Seed '{seed}' -> Similar terms: {similar_terms}")

                # Calculate score for this category
                category_score = self._get_keyword_score(tfidf_matrix, terms, category_keyword_set)
                category_keywords[category] = list(category_keyword_set)
                category_scores[category] = {
                    "keywords": list(category_keyword_set),
                    "score": category_score,
                    "count": len(category_keyword_set)
                }

                all_keywords.update(category_keyword_set)

            # Calculate total score for this dimension
            dimension_score = self._get_keyword_score(tfidf_matrix, terms, all_keywords)
            raw_scores[dim] = {
                "keywords": list(all_keywords),
                "score": dimension_score,
                "category_scores": category_scores
            }

            extraction_details[dim] = {
                "seeds": seed_categories,
                "category_keywords": category_keywords,
                "category_scores": category_scores,
                "total_keywords": len(all_keywords),
                "score": dimension_score
            }

            total_score += dimension_score
            logger.info(f"Dimension '{dim}': {len(all_keywords)} keywords, score: {dimension_score:.2f}")

        # Create DataFrame with enhanced structure
        E_raw = []
        for dim, info in raw_scores.items():
            weight = info["score"] / total_score if total_score > 0 else 0
            E_raw.append({
                "dimension": dim,
                "keywords": ", ".join(info["keywords"]) if info["keywords"] else "",
                "raw_score": float(info["score"]),
                "normalized_weight": float(weight),
                "hierarchical_structure": json.dumps(info["category_scores"]),
                "category_scores": json.dumps(info["category_scores"]),
                "keyword_frequencies": json.dumps(self._get_keyword_frequencies(tfidf_matrix, terms, info["keywords"])),
                "source_documents": len(texts)
            })

        df_result = pd.DataFrame(E_raw)
        logger.info(f"Enhanced extraction complete. Total score: {total_score:.2f}")

        # Calculate total keywords extracted
        total_keywords_extracted = sum(len(info["keywords"]) for info in raw_scores.values())

        return df_result, extraction_details, {
            "total_documents": len(texts),
            "total_terms": len(terms),
            "total_score": total_score,
            "dimensions_processed": len(dimension_seeds),
            "total_keywords_extracted": total_keywords_extracted
        }

    def _compute_tfidf_matrix(self, texts: List[str]) -> Tuple[Any, np.ndarray, TfidfVectorizer]:
        """Compute TF-IDF matrix with enhanced parameters."""
        if not texts:
            raise ValueError("No valid documents to process after preprocessing.")

        # Adjust parameters based on dataset size to avoid min_df > max_df error
        num_docs = len(texts)
        
        if num_docs == 1:
            # For single document, use minimal constraints
            vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=1.0,  # Allow all terms in single document
                ngram_range=(1, 2),
                max_features=10000
            )
        elif num_docs <= 5:
            # For very small datasets
            vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=1.0,  # Be inclusive for small datasets
                ngram_range=(1, 2),
                max_features=10000
            )
        else:
            # For larger datasets, use standard constraints
            vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 2),
                max_features=10000
            )

        tfidf_matrix = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()

        return tfidf_matrix, terms, vectorizer

    def _get_keyword_score(self, tfidf_matrix, terms: np.ndarray, keywords: List[str]) -> float:
        """Calculate keyword score from TF-IDF matrix."""
        score = 0
        for keyword in keywords:
            if keyword in terms:
                keyword_idx = np.where(terms == keyword)[0][0]
                score += tfidf_matrix[:, keyword_idx].sum()
        return float(score)

    def _get_keyword_frequencies(self, tfidf_matrix, terms, keywords):
        """Get frequency data for keywords."""
        frequencies = {}
        for keyword in keywords:
            if keyword in terms:
                term_idx = np.where(terms == keyword)[0][0]
                freq = tfidf_matrix[:, term_idx].sum()
                frequencies[keyword] = float(freq)
        return frequencies

    def _apply_conference_weighting(self, df_result: pd.DataFrame,
                                    conference_profile: Dict) -> pd.DataFrame:
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

    def _calculate_criteria_confidence(self, extraction_results: Dict, conference_files: Dict) -> Dict[str, Dict]:
        """Calculate confidence scores for extracted criteria."""
        confidence_metrics = {}

        for dimension, results in extraction_results.items():
            metrics = {}

            # Source coverage: how many documents contributed to this dimension
            metrics['source_coverage'] = len(conference_files) / max(len(conference_files), 1)

            # Keyword consensus: agreement between different keyword categories
            metrics['keyword_consensus'] = self._calculate_keyword_consensus(results['category_keywords'])

            # Semantic coherence: how well keywords cluster together
            metrics['semantic_coherence'] = 0.7  # Placeholder

            # Frequency stability: distribution of keyword frequencies
            metrics['frequency_stability'] = 0.8  # Placeholder

            # Cross-validation score: consistency across different extraction methods
            metrics['cross_validation_score'] = self._calculate_cross_validation_score(results['category_scores'])

            # Overall confidence score (weighted average)
            overall_confidence = (
                    metrics['source_coverage'] * 0.2 +
                    metrics['keyword_consensus'] * 0.3 +
                    metrics['semantic_coherence'] * 0.2 +
                    metrics['frequency_stability'] * 0.2 +
                    metrics['cross_validation_score'] * 0.1
            )

            confidence_metrics[dimension] = {
                'overall_confidence': overall_confidence,
                'detailed_metrics': metrics,
                'reliability_flag': self._get_reliability_flag(overall_confidence)
            }

        return confidence_metrics

    def _calculate_keyword_consensus(self, category_keywords: Dict) -> float:
        """Calculate agreement between different keyword categories."""
        all_keywords = []
        for category, keywords in category_keywords.items():
            all_keywords.extend(keywords)

        if not all_keywords:
            return 0.0

        # Calculate overlap between categories
        category_keywords_list = list(category_keywords.values())
        overlaps = 0
        total_comparisons = 0

        for i in range(len(category_keywords_list)):
            for j in range(i + 1, len(category_keywords_list)):
                overlap = len(set(category_keywords_list[i]) & set(category_keywords_list[j]))
                total = len(set(category_keywords_list[i]) | set(category_keywords_list[j]))
                if total > 0:
                    overlaps += overlap / total
                total_comparisons += 1

        return overlaps / total_comparisons if total_comparisons > 0 else 0.0

    def _calculate_cross_validation_score(self, category_scores: Dict) -> float:
        """Calculate cross-validation score across different extraction methods."""
        if not category_scores:
            return 0.0

        scores = [info['score'] for info in category_scores.values()]

        # Calculate consistency across categories
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        if mean_score == 0:
            return 0.0

        # Consistency score (lower std relative to mean = higher consistency)
        consistency = max(0, 1 - (std_score / mean_score))

        return consistency

    def _get_reliability_flag(self, confidence_score: float) -> str:
        """Get reliability flag based on confidence score."""
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.6:
            return 'medium'
        else:
            return 'low'

    def _build_semantic_relationship_map(self, all_keywords: List[str],
                                         term_metadata: List[Dict]) -> Dict[str, Any]:
        """Build semantic relationships between keywords and dimensions."""
        logger.info("Building semantic relationship map...")

        if not all_keywords:
            return {"keyword_clusters": [], "cross_dimension_relationships": [],
                    "semantic_hierarchy": {}, "contradiction_detection": []}

        # Create embeddings for all keywords
        embeddings = self.sentence_model.encode(all_keywords)

        # Compute similarity matrix
        similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

        # Identify keyword clusters
        keyword_clusters = self._identify_keyword_clusters(similarity_matrix, all_keywords)

        # Find cross-dimension relationships
        cross_dimension_links = self._find_cross_dimension_links(similarity_matrix, term_metadata)

        # Build semantic hierarchy
        semantic_hierarchy = self._build_semantic_hierarchy(similarity_matrix, all_keywords)

        # Detect contradictions
        contradictions = self._detect_contradictory_keywords(similarity_matrix, term_metadata)

        return {
            'keyword_clusters': keyword_clusters,
            'cross_dimension_relationships': cross_dimension_links,
            'semantic_hierarchy': semantic_hierarchy,
            'contradiction_detection': contradictions,
            'similarity_matrix': similarity_matrix.tolist()
        }

    def _identify_keyword_clusters(self, similarity_matrix: np.ndarray,
                                   keywords: List[str]) -> List[Dict]:
        """Identify clusters of semantically similar keywords."""
        if len(keywords) < 2:
            return []

        # Use DBSCAN to cluster similar keywords
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')

        # Convert similarity to distance
        similarity_clipped = np.clip(similarity_matrix, 0, 1)
        distances = 1 - similarity_clipped
        distances = np.maximum(distances, 0)

        cluster_labels = clustering.fit_predict(distances)

        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Not noise
                clusters[label].append(keywords[i])

        return [{'cluster_id': k, 'keywords': v, 'size': len(v)}
                for k, v in clusters.items()]

    def _find_cross_dimension_links(self, similarity_matrix: np.ndarray,
                                    term_metadata: List[Dict]) -> List[Dict]:
        """Find relationships between keywords from different dimensions."""
        links = []
        threshold = self.config.semantic_similarity_threshold

        for i in range(len(term_metadata)):
            for j in range(i + 1, len(term_metadata)):
                if (term_metadata[i]['dimension'] != term_metadata[j]['dimension'] and
                        similarity_matrix[i][j] > threshold):
                    links.append({
                        'keyword1': term_metadata[i]['keyword'],
                        'dimension1': term_metadata[i]['dimension'],
                        'keyword2': term_metadata[j]['keyword'],
                        'dimension2': term_metadata[j]['dimension'],
                        'similarity': float(similarity_matrix[i][j])
                    })

        return sorted(links, key=lambda x: x['similarity'], reverse=True)

    def _build_semantic_hierarchy(self, similarity_matrix: np.ndarray,
                                  keywords: List[str]) -> Dict[str, Any]:
        """Build a semantic hierarchy of keywords."""
        if len(keywords) < 2:
            return {'communities': [], 'central_keywords': [], 'keyword_degrees': {}}

        # Create a graph from similarity matrix
        G = nx.Graph()

        for i, keyword in enumerate(keywords):
            G.add_node(keyword)

        # Add edges based on similarity
        threshold = self.config.semantic_similarity_threshold
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                if similarity_matrix[i][j] > threshold:
                    G.add_edge(keywords[i], keywords[j],
                               weight=similarity_matrix[i][j])

        # Find communities in the graph
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
        except:
            communities = []

        # Build hierarchy
        hierarchy = {
            'communities': [list(community) for community in communities],
            'central_keywords': self._find_central_keywords(G),
            'keyword_degrees': dict(G.degree())
        }

        return hierarchy

    def _find_central_keywords(self, G: nx.Graph) -> List[str]:
        """Find central keywords in the semantic graph."""
        if len(G.nodes()) == 0:
            return []

        try:
            # Use betweenness centrality to find central keywords
            centrality = nx.betweenness_centrality(G)
            central_keywords = sorted(centrality.items(),
                                      key=lambda x: x[1], reverse=True)[:10]
            return [kw for kw, _ in central_keywords]
        except:
            return list(G.nodes())[:10]

    def _detect_contradictory_keywords(self, similarity_matrix: np.ndarray,
                                       term_metadata: List[Dict]) -> List[Dict]:
        """Detect potentially contradictory keywords."""
        contradictions = []

        for i in range(len(term_metadata)):
            for j in range(i + 1, len(term_metadata)):
                if (term_metadata[i]['dimension'] != term_metadata[j]['dimension'] and
                        similarity_matrix[i][j] > 0.8):  # High similarity

                    # Check if dimensions might conflict
                    if self._dimensions_might_conflict(term_metadata[i]['dimension'],
                                                       term_metadata[j]['dimension']):
                        contradictions.append({
                            'keyword1': term_metadata[i]['keyword'],
                            'dimension1': term_metadata[i]['dimension'],
                            'keyword2': term_metadata[j]['keyword'],
                            'dimension2': term_metadata[j]['dimension'],
                            'similarity': float(similarity_matrix[i][j]),
                            'conflict_type': 'semantic_overlap'
                        })

        return contradictions

    def _dimensions_might_conflict(self, dim1: str, dim2: str) -> bool:
        """Check if two dimensions might have conflicting requirements."""
        conflict_pairs = [
            ('accessibility', 'functionality'),
            ('usability', 'experimental'),
            ('documentation', 'functionality')
        ]

        return (dim1, dim2) in conflict_pairs or (dim2, dim1) in conflict_pairs

    def _generate_enhanced_outputs(self, df_result: pd.DataFrame, extraction_results: Dict,
                                   processing_stats: Dict, confidence_metrics: Dict,
                                   semantic_relationships: Dict, output_dir: str,
                                   conference_name: str) -> Dict[str, str]:
        """Generate multiple output formats optimized for conference-specific evaluation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. Enhanced CSV with conference-specific information
        csv_filename = f"{conference_name.lower()}_criteria_{timestamp}.csv"
        csv_path = output_path / csv_filename
        df_result.to_csv(csv_path, index=False)
        saved_files['enhanced_csv'] = str(csv_path)
        logger.info(f"Saved enhanced CSV to: {csv_path}")

        # 2. AURA Integration Format
        aura_integration = self._create_aura_integration_format(df_result, confidence_metrics, conference_name)
        aura_filename = f"{conference_name.lower()}_aura_integration_{timestamp}.json"
        aura_path = output_path / aura_filename
        with open(aura_path, 'w', encoding='utf-8') as f:
            json.dump(aura_integration, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        saved_files['aura_integration'] = str(aura_path)
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
            "extraction_details": extraction_results,
            "confidence_metrics": confidence_metrics,
            "semantic_relationships": semantic_relationships
        }

        analysis_filename = f"{conference_name.lower()}_analysis_{timestamp}.json"
        analysis_path = output_path / analysis_filename
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(conference_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        saved_files['conference_analysis'] = str(analysis_path)
        logger.info(f"Saved conference analysis to: {analysis_path}")

        return saved_files

    def _create_aura_integration_format(self, df_result: pd.DataFrame,
                                        confidence_metrics: Dict, conference_name: str) -> Dict:
        """Create format optimized for AURA framework integration."""
        aura_data = {
            "conference_name": conference_name,
            "structured_criteria": [],
            "grounding_evidence": {},
            "confidence_weights": {},
            "hierarchical_keywords": {},
            "severity_weights": {}
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
                "hierarchical_structure": json.loads(row['hierarchical_structure']) if row[
                    'hierarchical_structure'] else {},
                "category_scores": json.loads(row['category_scores']) if row['category_scores'] else {},
                "keyword_frequencies": json.loads(row['keyword_frequencies']) if row['keyword_frequencies'] else {}
            }
            aura_data["structured_criteria"].append(criteria_entry)

            # Grounding evidence
            keywords_list = row['keywords'].split(', ') if row['keywords'] else []
            aura_data["grounding_evidence"][dimension] = {
                "keywords_found": keywords_list,
                "frequency_data": json.loads(row['keyword_frequencies']) if row['keyword_frequencies'] else {},
                "category_breakdown": json.loads(row['category_scores']) if row['category_scores'] else {}
            }

            # Confidence weights
            if dimension in confidence_metrics:
                aura_data["confidence_weights"][dimension] = confidence_metrics[dimension]

        return aura_data

    def _generate_cross_conference_analysis(self, all_results: Dict, output_dir: str):
        """Generate analysis comparing criteria across conferences."""

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "conferences_analyzed": list(all_results.keys()),
            "conference_comparison": {},
            "dimension_consistency": {},
            "keyword_overlap": {},
            "recommendations": []
        }

        # Compare dimension weights across conferences
        dimensions = ["reproducibility", "documentation", "accessibility",
                      "usability", "experimental", "functionality"]

        for dimension in dimensions:
            weights_by_conf = {}
            adjusted_weights_by_conf = {}

            for conf_name, results in all_results.items():
                df = results["criteria_dataframe"]
                dim_row = df[df["dimension"] == dimension]
                if not dim_row.empty:
                    weights_by_conf[conf_name] = float(dim_row["normalized_weight"].iloc[0])
                    adjusted_weights_by_conf[conf_name] = float(
                        dim_row.get("conference_adjusted_weight", dim_row["normalized_weight"]).iloc[0]
                    )

            if weights_by_conf:
                analysis["dimension_consistency"][dimension] = {
                    "base_weights": weights_by_conf,
                    "adjusted_weights": adjusted_weights_by_conf,
                    "mean_weight": np.mean(list(weights_by_conf.values())),
                    "std_deviation": np.std(list(weights_by_conf.values())),
                    "min_conference": min(weights_by_conf, key=weights_by_conf.get),
                    "max_conference": max(weights_by_conf, key=weights_by_conf.get),
                    "variance_level": "high" if np.std(list(weights_by_conf.values())) > 0.1 else "low"
                }

        # Calculate keyword overlap between conferences
        all_keywords_by_conf = {}
        for conf_name, results in all_results.items():
            conf_keywords = set()
            for _, row in results["criteria_dataframe"].iterrows():
                if row['keywords']:
                    conf_keywords.update(row['keywords'].split(', '))
            all_keywords_by_conf[conf_name] = conf_keywords

        # Calculate pairwise overlaps
        conference_names = list(all_keywords_by_conf.keys())
        for i, conf1 in enumerate(conference_names):
            for j, conf2 in enumerate(conference_names[i + 1:], i + 1):
                overlap = len(all_keywords_by_conf[conf1] & all_keywords_by_conf[conf2])
                total = len(all_keywords_by_conf[conf1] | all_keywords_by_conf[conf2])
                overlap_ratio = overlap / total if total > 0 else 0

                analysis["keyword_overlap"][f"{conf1}_vs_{conf2}"] = {
                    "overlap_count": overlap,
                    "overlap_ratio": overlap_ratio,
                    "unique_to_conf1": len(all_keywords_by_conf[conf1] - all_keywords_by_conf[conf2]),
                    "unique_to_conf2": len(all_keywords_by_conf[conf2] - all_keywords_by_conf[conf1])
                }

        # Generate recommendations
        high_variance_dims = [dim for dim, data in analysis["dimension_consistency"].items()
                              if data.get("variance_level") == "high"]

        if high_variance_dims:
            analysis["recommendations"].append(
                f"High variance detected in {', '.join(high_variance_dims)} across conferences - consider conference-specific evaluation"
            )

        low_overlap_pairs = [pair for pair, data in analysis["keyword_overlap"].items()
                             if data["overlap_ratio"] < 0.3]

        if low_overlap_pairs:
            analysis["recommendations"].append(
                f"Low keyword overlap detected between some conference pairs - domain-specific criteria recommended"
            )

        # Save cross-conference analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = os.path.join(output_dir, f"cross_conference_analysis_{timestamp}.json")

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logger.info(f"Cross-conference analysis saved to: {analysis_file}")
        return analysis_file
