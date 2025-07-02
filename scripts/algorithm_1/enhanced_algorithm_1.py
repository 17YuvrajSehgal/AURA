import glob
import json
import logging
import os
import re
import string
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

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

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Enhanced logging setup
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "..", "..", "algo_outputs", "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(logs_dir, "enhanced_algorithm_1_execution.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class EnhancedAlgorithm1:
    """
    Enhanced Algorithm 1 for extracting evaluation criteria from conference guidelines.
    Provides hierarchical keywords, and multiple output formats
    optimized for downstream NLP and LLM tasks.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the enhanced algorithm with models and configurations."""
        self.sentence_model = SentenceTransformer(model_name)
        self.kw_model = KeyBERT(self.sentence_model)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Configuration parameters
        self.config = {
            'semantic_similarity_threshold': 0.7,
            'keyword_expansion_top_n': 8,
            'min_keyword_frequency': 2,
            'max_keywords_per_dimension': 50,
        }

        logger.info(f"Enhanced Algorithm 1 initialized with model: {model_name}")

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Enhanced text preprocessing with lemmatization and better cleaning."""
        preprocessed = []

        for text in texts:
            # Tokenize and clean
            tokens = word_tokenize(text.lower())

            # Remove stopwords, punctuation, and short tokens
            cleaned_tokens = []
            for token in tokens:
                if (token not in self.stop_words and
                        token not in string.punctuation and
                        len(token) > 2 and
                        not token.isnumeric()):
                    # Lemmatize the token
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

            # Extract conference info from filename (e.g., "13_icse_2025.md")
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

    def generate_hierarchical_seed_keywords(self, dimension_examples: Dict[str, List[str]]) -> Dict[
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
                'domain_keywords': self._extract_domain_specific_keywords(dimension)
            }

            logger.debug(f"Generated hierarchical seeds for {dimension}: {hierarchical_seeds[dimension]}")

        return hierarchical_seeds

    def _extract_domain_specific_keywords(self, dimension: str) -> List[str]:
        """Extract domain-specific keywords based on dimension type."""
        domain_keywords = []

        # Domain-specific keyword extraction based on dimension
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

    def build_semantic_relationship_map(self, all_keywords: List[str],
                                        term_metadata: List[Dict]) -> Dict[str, Any]:
        """Build semantic relationships between keywords and dimensions."""
        logger.info("Building semantic relationship map...")

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
        # Use DBSCAN to cluster similar keywords
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')

        # Convert similarity to distance, ensuring non-negative values
        # Clip similarity scores to [0, 1] range first, then convert to distance
        similarity_clipped = np.clip(similarity_matrix, 0, 1)
        distances = 1 - similarity_clipped

        # Ensure all distances are non-negative (should be, but double-check)
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
        threshold = self.config['semantic_similarity_threshold']

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
        # Create a graph from similarity matrix
        G = nx.Graph()

        for i, keyword in enumerate(keywords):
            G.add_node(keyword)

        # Add edges based on similarity
        threshold = self.config['semantic_similarity_threshold']
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                if similarity_matrix[i][j] > threshold:
                    G.add_edge(keywords[i], keywords[j],
                               weight=similarity_matrix[i][j])

        # Find communities in the graph
        communities = list(nx.community.greedy_modularity_communities(G))

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

        # Use betweenness centrality to find central keywords
        centrality = nx.betweenness_centrality(G)
        central_keywords = sorted(centrality.items(),
                                  key=lambda x: x[1], reverse=True)[:10]

        return [kw for kw, _ in central_keywords]

    def _detect_contradictory_keywords(self, similarity_matrix: np.ndarray,
                                       term_metadata: List[Dict]) -> List[Dict]:
        """Detect potentially contradictory keywords."""
        contradictions = []

        # Look for keywords that are semantically similar but from different dimensions
        # that might have conflicting requirements
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
            ('accessibility', 'functionality'),  # Public vs. proprietary
            ('usability', 'experimental'),  # Simple vs. complex
            ('documentation', 'functionality')  # Documentation vs. code quality
        ]

        return (dim1, dim2) in conflict_pairs or (dim2, dim1) in conflict_pairs

    def extract_evaluation_criteria(self, texts: List[str], dimension_seeds: Dict[str, Dict[str, List[str]]]) -> Tuple[
        pd.DataFrame, Dict, Dict]:
        """Extract evaluation criteria using TF-IDF and semantic similarity with hierarchical approach."""
        logger.info("Starting enhanced evaluation criteria extraction...")

        preprocessed = self.preprocess_texts(texts)
        logger.info(f"Preprocessed {len(preprocessed)} documents")

        tfidf_matrix, terms, vectorizer = self.compute_tfidf_matrix(preprocessed)
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
                category_score = self.get_keyword_score(tfidf_matrix, terms, category_keyword_set)
                category_keywords[category] = list(category_keyword_set)
                category_scores[category] = {
                    "keywords": list(category_keyword_set),
                    "score": category_score,
                    "count": len(category_keyword_set)
                }

                all_keywords.update(category_keyword_set)

            # Calculate total score for this dimension
            dimension_score = self.get_keyword_score(tfidf_matrix, terms, all_keywords)
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

    def _find_semantic_similar_terms(self, seed_term: str, all_terms: List[str], top_n: int = 5) -> List[str]:
        """Find semantically similar terms using sentence embeddings."""
        try:
            seed_embedding = self.sentence_model.encode(seed_term, convert_to_tensor=True)
            term_embeddings = self.sentence_model.encode(all_terms, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(seed_embedding, term_embeddings)[0]
            top_indices = cosine_scores.topk(top_n).indices
            return [all_terms[i] for i in top_indices]
        except Exception as e:
            logger.warning(f"Error finding similar terms for '{seed_term}': {e}")
            return []

    def compute_tfidf_matrix(self, texts: List[str]) -> Tuple[Any, np.ndarray, TfidfVectorizer]:
        """Compute TF-IDF matrix with enhanced parameters."""
        if not texts:
            raise ValueError("No valid documents to process after preprocessing.")

        vectorizer = TfidfVectorizer(
            min_df=1,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency (remove very common terms)
            ngram_range=(1, 2),  # Include bigrams
            max_features=10000  # Limit features to prevent memory issues
        )

        tfidf_matrix = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()

        return tfidf_matrix, terms, vectorizer

    def get_keyword_score(self, tfidf_matrix, terms: np.ndarray, keywords: List[str]) -> float:
        """Calculate keyword score from TF-IDF matrix."""
        score = 0
        for keyword in keywords:
            if keyword in terms:
                keyword_idx = np.where(terms == keyword)[0][0]
                score += tfidf_matrix[:, keyword_idx].sum()
        return float(score)

    def generate_enhanced_outputs(self, df_result: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """Generate multiple output formats optimized for different downstream tasks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # AURA Integration Format
        aura_integration = self._create_aura_integration_format(df_result)
        aura_filename = f"aura_integration_data_{timestamp}.json"
        aura_path = output_path / aura_filename
        with open(aura_path, 'w', encoding='utf-8') as f:
            json.dump(aura_integration, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        saved_files['aura_integration'] = str(aura_path)
        logger.info(f"Saved AURA integration data to: {aura_path}")

        # Comprehensive Report
        # report_filename = f"enhanced_algorithm_1_report_{timestamp}.txt"
        # report_path = output_path / report_filename
        # self._generate_comprehensive_report(
        #     report_path, df_result, extraction_results, processing_stats
        # )
        # saved_files['comprehensive_report'] = str(report_path)
        # logger.info(f"Saved comprehensive report to: {report_path}")

    def _create_aura_integration_format(self, df_result: pd.DataFrame) -> Dict:
        """Create format optimized for AURA framework integration."""
        aura_data = {
            "structured_criteria": [],
        }

        for _, row in df_result.iterrows():
            dimension = row['dimension']

            # Structured criteria
            criteria_entry = {
                "dimension": dimension,
                "keywords": row['keywords'].split(', '),
                "raw_score": row['raw_score'],
                "normalized_weight": row['normalized_weight'],
                "hierarchical_structure": json.loads(row['hierarchical_structure']),
                "category_scores": json.loads(row['category_scores']),
            }
            aura_data["structured_criteria"].append(criteria_entry)
        return aura_data

    def run_enhanced_extraction(self, input_dir: str, output_dir: str) -> Dict[str, str]:
        """Main method to run the enhanced extraction process."""
        logger.info("Starting Enhanced AURA Algorithm 1")

        # Load conference guidelines
        pattern = os.path.join(input_dir, "*.md")
        all_guideline_files = glob.glob(pattern)
        logger.info(f"Found {len(all_guideline_files)} guideline files")

        if not all_guideline_files:
            raise ValueError("No guideline files found in input directory")

        # Extract conference metadata
        conference_metadata = self.extract_conference_metadata(all_guideline_files)

        # Load and process texts
        conference_texts = []
        for path in all_guideline_files:
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    conference_texts.append(content)
                    logger.info(f"Loaded: {os.path.basename(path)} ({len(content)} characters)")
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")

        if not conference_texts:
            raise ValueError("No content found in guideline files")

        # Define dimension examples
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

        # Generate hierarchical seed keywords
        logger.info("Generating hierarchical seed keywords...")
        hierarchical_seeds = self.generate_hierarchical_seed_keywords(dimension_examples)

        # Extract evaluation criteria
        logger.info("Extracting evaluation criteria...")
        df_result, extraction_results, processing_stats = \
            self.extract_evaluation_criteria(conference_texts, hierarchical_seeds)

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

        semantic_relationships = self.build_semantic_relationship_map(all_keywords, term_metadata)

        # Generate outputs
        logger.info("Generating enhanced outputs...")
        saved_files = self.generate_enhanced_outputs(
            df_result, output_dir
        )

        # Final summary
        logger.info("=" * 70)
        logger.info("ENHANCED AURA ALGORITHM 1 - EXTRACTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Processed {processing_stats['total_documents']} conference guideline files")
        logger.info(f"Extracted criteria for {processing_stats['dimensions_processed']} dimensions")
        logger.info(f"Total evaluation score: {processing_stats['total_score']:.2f}")
        logger.info(f"Output files saved to: {output_dir}")

        return saved_files


def main():
    """Main execution function for enhanced Algorithm 1."""
    logger.info("Starting Enhanced AURA Algorithm 1")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "..", "..", "data", "conference_guideline_texts", "processed")
    output_dir = os.path.join(script_dir, "..", "..", "algo_outputs", "algorithm_1_output")

    # Initialize enhanced algorithm
    enhanced_algo = EnhancedAlgorithm1()

    try:
        # Run enhanced extraction
        saved_files = enhanced_algo.run_enhanced_extraction(input_dir, output_dir)

        logger.info("Enhanced Algorithm 1 completed successfully!")
        return saved_files

    except Exception as e:
        logger.error(f"Error in enhanced extraction: {e}")
        raise


if __name__ == "__main__":
    main()
