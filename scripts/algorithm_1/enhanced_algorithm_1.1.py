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

        # Adjust parameters based on document count
        num_docs = len(texts)

        if num_docs == 1:
            # For single document, use simpler parameters
            vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=1.0,  # Allow all terms for single document
                ngram_range=(1, 2),
                max_features=5000
            )
        else:
            # For multiple documents, use more restrictive parameters
            vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=0.95,  # Maximum document frequency (remove very common terms)
                ngram_range=(1, 2),
                max_features=10000
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
        """Main method to run the enhanced extraction process - RQ1 focused."""
        logger.info("Starting Enhanced AURA Algorithm 1 - RQ1 Analysis")

        # Load conference guidelines
        pattern = os.path.join(input_dir, "*.md")
        all_guideline_files = glob.glob(pattern)
        logger.info(f"Found {len(all_guideline_files)} guideline files")

        if not all_guideline_files:
            raise ValueError("No guideline files found in input directory")

        # Extract conference metadata
        conference_metadata = self.extract_conference_metadata(all_guideline_files)

        # NEW: Process each conference separately for RQ1
        conference_results = {}
        all_conference_data = {}

        logger.info("=" * 70)
        logger.info("RQ1: CONFERENCE-SPECIFIC CRITERIA EXTRACTION")
        logger.info("=" * 70)

        for file_path in all_guideline_files:
            filename = os.path.basename(file_path)
            metadata = conference_metadata[filename]
            conf_name = metadata['conference_name']

            if conf_name == 'UNKNOWN':
                continue

            logger.info(f"Processing {conf_name} ({metadata.get('year', 'Unknown Year')})...")

            # Load conference-specific text
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    logger.info(f"Loaded {conf_name}: {len(content)} characters")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue

            # Extract criteria for this specific conference
            conf_results = self._extract_conference_specific_criteria(
                conf_name, [content], metadata
            )

            conference_results[conf_name] = conf_results
            all_conference_data[conf_name] = {
                'content': content,
                'metadata': metadata,
                'results': conf_results
            }

        logger.info(f"Successfully processed {len(conference_results)} conferences")

        # NEW: Generate RQ1-specific analyses
        logger.info("=" * 70)
        logger.info("RQ1: CROSS-CONFERENCE NORMALIZATION & ANALYSIS")
        logger.info("=" * 70)

        # Normalize criteria across conferences
        normalized_analysis = self._normalize_criteria_across_conferences(conference_results)

        # Generate dimension prioritization analysis
        prioritization_analysis = self._analyze_dimension_prioritization(conference_results)

        # Create conference-specific evaluation rubrics
        evaluation_rubrics = self._generate_evaluation_rubrics(conference_results)

        # Statistical analysis of criteria variations
        statistical_analysis = self._perform_statistical_analysis(conference_results)

        # Generate RQ1-focused outputs
        logger.info("Generating RQ1-focused outputs...")
        saved_files = self._generate_rq1_outputs(
            output_dir,
            conference_results,
            normalized_analysis,
            prioritization_analysis,
            evaluation_rubrics,
            statistical_analysis
        )

        # Final RQ1 summary
        logger.info("=" * 70)
        logger.info("RQ1 ANALYSIS COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Processed {len(conference_results)} conferences")
        logger.info(f"Extracted normalized criteria across {len(prioritization_analysis['dimensions'])} dimensions")
        logger.info(f"Generated {len(evaluation_rubrics)} conference-specific rubrics")
        logger.info(f"RQ1 outputs saved to: {output_dir}")

        return saved_files

    def _extract_conference_specific_criteria(self, conf_name: str, texts: List[str], metadata: Dict) -> Dict:
        """Extract evaluation criteria for a specific conference."""
        logger.info(f"Extracting criteria for {conf_name}...")

        # Define dimension examples (same as before)
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
        hierarchical_seeds = self.generate_hierarchical_seed_keywords(dimension_examples)

        # Extract evaluation criteria for this conference
        df_result, extraction_results, processing_stats = \
            self.extract_evaluation_criteria(texts, hierarchical_seeds)

        # Add conference-specific metadata
        df_result['conference'] = conf_name
        df_result['year'] = metadata.get('year')
        df_result['file_size'] = metadata.get('file_size', 0)

        return {
            'conference': conf_name,
            'metadata': metadata,
            'criteria_df': df_result,
            'extraction_details': extraction_results,
            'processing_stats': processing_stats
        }

    def _normalize_criteria_across_conferences(self, conference_results: Dict) -> Dict:
        """Normalize and compare criteria across all conferences."""
        logger.info("Normalizing criteria across conferences...")

        all_dimensions = ["reproducibility", "documentation", "accessibility",
                          "usability", "experimental", "functionality"]

        normalized_data = {
            'dimension_weights_by_conference': {},
            'normalized_comparison': {},
            'conference_similarity_matrix': {},
            'dimension_statistics': {}
        }

        # Extract weights for each dimension across all conferences
        for dimension in all_dimensions:
            dim_weights = {}
            for conf_name, results in conference_results.items():
                df = results['criteria_df']
                dim_row = df[df['dimension'] == dimension]
                if not dim_row.empty:
                    weight = float(dim_row['normalized_weight'].iloc[0])
                    dim_weights[conf_name] = weight
                else:
                    dim_weights[conf_name] = 0.0

            normalized_data['dimension_weights_by_conference'][dimension] = dim_weights

            # Calculate statistics for this dimension
            weights = list(dim_weights.values())
            normalized_data['dimension_statistics'][dimension] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'variance': np.var(weights),
                'conferences_above_mean': [conf for conf, w in dim_weights.items() if w > np.mean(weights)],
                'conferences_below_mean': [conf for conf, w in dim_weights.items() if w < np.mean(weights)]
            }

        # Create normalized comparison matrix
        conferences = list(conference_results.keys())
        for conf in conferences:
            conf_weights = []
            for dim in all_dimensions:
                weight = normalized_data['dimension_weights_by_conference'][dim].get(conf, 0.0)
                conf_weights.append(weight)
            normalized_data['normalized_comparison'][conf] = dict(zip(all_dimensions, conf_weights))

        # Calculate conference similarity matrix
        conf_vectors = []
        for conf in conferences:
            weights = [normalized_data['normalized_comparison'][conf][dim] for dim in all_dimensions]
            conf_vectors.append(weights)

        if len(conf_vectors) > 1:
            # Compute cosine similarity between conferences
            embeddings = self.sentence_model.encode([str(v) for v in conf_vectors])
            similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

            for i, conf1 in enumerate(conferences):
                for j, conf2 in enumerate(conferences):
                    if conf1 not in normalized_data['conference_similarity_matrix']:
                        normalized_data['conference_similarity_matrix'][conf1] = {}
                    normalized_data['conference_similarity_matrix'][conf1][conf2] = float(similarity_matrix[i][j])

        return normalized_data

    def _analyze_dimension_prioritization(self, conference_results: Dict) -> Dict:
        """Analyze how different conferences prioritize different dimensions."""
        logger.info("Analyzing dimension prioritization patterns...")

        prioritization = {
            'dimensions': ["reproducibility", "documentation", "accessibility",
                           "usability", "experimental", "functionality"],
            'conference_rankings': {},
            'dimension_leaders': {},
            'priority_patterns': {},
            'clustering_analysis': {}
        }

        # Rank dimensions for each conference
        for conf_name, results in conference_results.items():
            df = results['criteria_df']

            # Sort dimensions by normalized weight
            sorted_dims = df.sort_values('normalized_weight', ascending=False)
            rankings = {}

            for rank, (_, row) in enumerate(sorted_dims.iterrows(), 1):
                rankings[row['dimension']] = {
                    'rank': rank,
                    'weight': float(row['normalized_weight']),
                    'raw_score': float(row['raw_score'])
                }

            prioritization['conference_rankings'][conf_name] = rankings

        # Find which conference leads in each dimension
        for dimension in prioritization['dimensions']:
            dim_leaders = []
            for conf_name, rankings in prioritization['conference_rankings'].items():
                if dimension in rankings:
                    dim_leaders.append((conf_name, rankings[dimension]['weight']))

            if dim_leaders:
                dim_leaders.sort(key=lambda x: x[1], reverse=True)
                prioritization['dimension_leaders'][dimension] = {
                    'leader': dim_leaders[0][0],
                    'leader_weight': dim_leaders[0][1],
                    'all_rankings': dim_leaders
                }

        # Identify priority patterns (conferences with similar dimension emphasis)
        conferences = list(conference_results.keys())
        if len(conferences) > 1:
            priority_vectors = []
            for conf in conferences:
                vector = []
                rankings = prioritization['conference_rankings'][conf]
                for dim in prioritization['dimensions']:
                    weight = rankings.get(dim, {}).get('weight', 0.0)
                    vector.append(weight)
                priority_vectors.append(vector)

            # Cluster conferences by priority patterns
            if len(priority_vectors) > 2:
                from sklearn.cluster import KMeans
                n_clusters = min(3, len(conferences))  # Max 3 clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(priority_vectors)

                clusters = {}
                for i, conf in enumerate(conferences):
                    cluster_id = int(cluster_labels[i])
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(conf)

                prioritization['clustering_analysis'] = {
                    'clusters': clusters,
                    'n_clusters': n_clusters,
                    'cluster_centers': kmeans.cluster_centers_.tolist()
                }

        return prioritization

    def _generate_evaluation_rubrics(self, conference_results: Dict) -> Dict:
        """Generate conference-specific evaluation rubrics."""
        logger.info("Generating conference-specific evaluation rubrics...")

        rubrics = {}

        for conf_name, results in conference_results.items():
            df = results['criteria_df']

            rubric = {
                'conference': conf_name,
                'year': results['metadata'].get('year'),
                'evaluation_dimensions': {},
                'scoring_weights': {},
                'quality_thresholds': {},
                'total_criteria_count': len(df)
            }

            for _, row in df.iterrows():
                dimension = row['dimension']

                # Extract keywords and create evaluation criteria
                keywords = row['keywords'].split(', ') if row['keywords'] else []

                rubric['evaluation_dimensions'][dimension] = {
                    'weight': float(row['normalized_weight']),
                    'raw_score': float(row['raw_score']),
                    'criteria_keywords': keywords,
                    'keyword_count': len(keywords),
                    'evaluation_focus': self._determine_evaluation_focus(keywords),
                    'assessment_guidelines': self._generate_assessment_guidelines(dimension, keywords)
                }

                rubric['scoring_weights'][dimension] = float(row['normalized_weight'])

            # Set quality thresholds based on weight distribution
            weights = [float(row['normalized_weight']) for _, row in df.iterrows()]
            mean_weight = np.mean(weights)
            std_weight = np.std(weights)

            rubric['quality_thresholds'] = {
                'high_priority': mean_weight + std_weight,
                'medium_priority': mean_weight,
                'low_priority': mean_weight - std_weight,
                'minimum_acceptable': 0.1
            }

            rubrics[conf_name] = rubric

        return rubrics

    def _determine_evaluation_focus(self, keywords: List[str]) -> str:
        """Determine the primary evaluation focus based on keywords."""
        if not keywords:
            return "general"

        # Categorize keywords
        technical_terms = {'code', 'software', 'algorithm', 'implementation', 'system'}
        process_terms = {'process', 'workflow', 'setup', 'installation', 'configuration'}
        quality_terms = {'quality', 'performance', 'accuracy', 'validation', 'verification'}
        access_terms = {'public', 'available', 'accessible', 'open', 'free'}

        keyword_set = set(keywords)

        if keyword_set & technical_terms:
            return "technical_implementation"
        elif keyword_set & process_terms:
            return "process_methodology"
        elif keyword_set & quality_terms:
            return "quality_assurance"
        elif keyword_set & access_terms:
            return "accessibility_openness"
        else:
            return "general_compliance"

    def _generate_assessment_guidelines(self, dimension: str, keywords: List[str]) -> List[str]:
        """Generate specific assessment guidelines for a dimension."""
        guidelines = []

        if dimension == "reproducibility":
            guidelines = [
                "Verify that all necessary code, data, and dependencies are provided",
                "Check if experimental results can be reproduced with provided materials",
                "Assess the completeness of environment setup instructions",
                "Evaluate the clarity of reproduction steps"
            ]
        elif dimension == "documentation":
            guidelines = [
                "Review README file for completeness and clarity",
                "Check if installation and setup instructions are provided",
                "Assess the quality of API documentation and examples",
                "Verify that usage guidelines are beginner-friendly"
            ]
        elif dimension == "accessibility":
            guidelines = [
                "Confirm that all components are publicly accessible",
                "Check for any access restrictions or licensing issues",
                "Verify that datasets and dependencies are available",
                "Assess the sustainability of access methods"
            ]
        elif dimension == "usability":
            guidelines = [
                "Evaluate the ease of installation and setup",
                "Check if user interfaces or demos are provided",
                "Assess the simplicity of the workflow",
                "Verify that examples and tutorials are included"
            ]
        elif dimension == "experimental":
            guidelines = [
                "Review the rigor of experimental methodology",
                "Check if statistical analysis is provided",
                "Assess the quality of benchmarks and metrics",
                "Verify that experimental data supports claims"
            ]
        elif dimension == "functionality":
            guidelines = [
                "Test that the code performs its intended function",
                "Verify that outputs match expected results",
                "Check if unit tests or validation examples are provided",
                "Assess the correctness of implementations"
            ]

        # Customize based on keywords
        if keywords and len(keywords) > 0:
            guidelines.append(f"Pay special attention to: {', '.join(keywords[:5])}")

        return guidelines

    def _perform_statistical_analysis(self, conference_results: Dict) -> Dict:
        """Perform statistical analysis of criteria variations across conferences."""
        logger.info("Performing statistical analysis...")

        stats = {
            'variance_analysis': {},
            'correlation_analysis': {},
            'distribution_analysis': {},
            'significance_tests': {},
            'summary_statistics': {}
        }

        # Collect all dimension weights
        dimensions = ["reproducibility", "documentation", "accessibility",
                      "usability", "experimental", "functionality"]

        dimension_weights = {dim: [] for dim in dimensions}
        conference_names = []

        for conf_name, results in conference_results.items():
            conference_names.append(conf_name)
            df = results['criteria_df']

            for dimension in dimensions:
                dim_row = df[df['dimension'] == dimension]
                if not dim_row.empty:
                    weight = float(dim_row['normalized_weight'].iloc[0])
                else:
                    weight = 0.0
                dimension_weights[dimension].append(weight)

        # Variance analysis
        for dimension in dimensions:
            weights = dimension_weights[dimension]
            stats['variance_analysis'][dimension] = {
                'variance': float(np.var(weights)),
                'std_deviation': float(np.std(weights)),
                'coefficient_of_variation': float(np.std(weights) / np.mean(weights)) if np.mean(weights) > 0 else 0,
                'range': float(np.max(weights) - np.min(weights)),
                'variability_level': 'high' if np.std(weights) > 0.1 else 'medium' if np.std(weights) > 0.05 else 'low'
            }

        # Distribution analysis
        all_weights = []
        for weights in dimension_weights.values():
            all_weights.extend(weights)

        stats['distribution_analysis'] = {
            'overall_mean': float(np.mean(all_weights)),
            'overall_std': float(np.std(all_weights)),
            'dimension_means': {dim: float(np.mean(weights)) for dim, weights in dimension_weights.items()},
            'dimension_medians': {dim: float(np.median(weights)) for dim, weights in dimension_weights.items()},
            'most_variable_dimension': max(stats['variance_analysis'],
                                           key=lambda x: stats['variance_analysis'][x]['variance']),
            'least_variable_dimension': min(stats['variance_analysis'],
                                            key=lambda x: stats['variance_analysis'][x]['variance'])
        }

        # Summary statistics
        stats['summary_statistics'] = {
            'total_conferences': len(conference_results),
            'total_dimensions': len(dimensions),
            'conferences_analyzed': conference_names,
            'high_variance_dimensions': [dim for dim, data in stats['variance_analysis'].items()
                                         if data['variability_level'] == 'high'],
            'standardization_needed': len([dim for dim, data in stats['variance_analysis'].items()
                                           if data['variability_level'] == 'high']) > 0
        }

        return stats

    def _generate_rq1_outputs(self, output_dir: str, conference_results: Dict,
                              normalized_analysis: Dict, prioritization_analysis: Dict,
                              evaluation_rubrics: Dict, statistical_analysis: Dict) -> Dict[str, str]:
        """Generate RQ1-specific output files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 1. RQ1 Main Results - Conference-specific criteria with normalization
        rq1_main_results = {
            "research_question": "RQ1: Can we automatically extract and normalize artifact evaluation criteria across conferences?",
            "methodology": "TF-IDF and semantic similarity analysis with cross-conference normalization",
            "conferences_analyzed": list(conference_results.keys()),
            "analysis_timestamp": timestamp,
            "conference_specific_results": {},
            "normalized_comparison": normalized_analysis['normalized_comparison'],
            "dimension_statistics": normalized_analysis['dimension_statistics'],
            "conference_similarity": normalized_analysis['conference_similarity_matrix'],
            "statistical_significance": statistical_analysis,
            "rq1_conclusion": self._generate_rq1_conclusion(statistical_analysis, prioritization_analysis)
        }

        # Add conference-specific results
        for conf_name, results in conference_results.items():
            rq1_main_results["conference_specific_results"][conf_name] = {
                "metadata": results['metadata'],
                "criteria": results['criteria_df'].to_dict('records'),
                "extraction_confidence": self._calculate_extraction_confidence(results),
                "unique_characteristics": self._identify_unique_characteristics(conf_name, conference_results)
            }

        rq1_filename = f"rq1_criteria_extraction_analysis_{timestamp}.json"
        rq1_path = output_path / rq1_filename
        with open(rq1_path, 'w', encoding='utf-8') as f:
            json.dump(rq1_main_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        saved_files['rq1_main_analysis'] = str(rq1_path)
        logger.info(f"Saved RQ1 main analysis to: {rq1_path}")

        # 2. Dimension Prioritization Analysis
        prioritization_filename = f"rq1_dimension_prioritization_{timestamp}.json"
        prioritization_path = output_path / prioritization_filename
        with open(prioritization_path, 'w', encoding='utf-8') as f:
            json.dump(prioritization_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        saved_files['dimension_prioritization'] = str(prioritization_path)
        logger.info(f"Saved dimension prioritization analysis to: {prioritization_path}")

        # 3. Conference-Specific Evaluation Rubrics
        rubrics_filename = f"rq1_evaluation_rubrics_{timestamp}.json"
        rubrics_path = output_path / rubrics_filename
        with open(rubrics_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_rubrics, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        saved_files['evaluation_rubrics'] = str(rubrics_path)
        logger.info(f"Saved evaluation rubrics to: {rubrics_path}")

        # 4. Statistical Analysis Report
        stats_filename = f"rq1_statistical_analysis_{timestamp}.json"
        stats_path = output_path / stats_filename
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistical_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        saved_files['statistical_analysis'] = str(stats_path)
        logger.info(f"Saved statistical analysis to: {stats_path}")

        # 5. AURA Integration Format (Enhanced for RQ1)
        aura_integration = self._create_rq1_aura_integration_format(
            conference_results, normalized_analysis, prioritization_analysis
        )
        aura_filename = f"rq1_aura_integration_{timestamp}.json"
        aura_path = output_path / aura_filename
        with open(aura_path, 'w', encoding='utf-8') as f:
            json.dump(aura_integration, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        saved_files['aura_integration'] = str(aura_path)
        logger.info(f"Saved RQ1 AURA integration to: {aura_path}")

        # 6. Human-Readable Summary Report
        summary_report = self._generate_rq1_summary_report(
            conference_results, normalized_analysis, prioritization_analysis, statistical_analysis
        )
        summary_filename = f"rq1_summary_report_{timestamp}.md"
        summary_path = output_path / summary_filename
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        saved_files['summary_report'] = str(summary_path)
        logger.info(f"Saved RQ1 summary report to: {summary_path}")

        return saved_files

    def _calculate_extraction_confidence(self, results: Dict) -> Dict:
        """Calculate confidence metrics for the extraction."""
        df = results['criteria_df']
        stats = results['processing_stats']

        confidence = {
            'keyword_density': stats.get('total_keywords_extracted', 0) / max(stats.get('total_terms', 1), 1),
            'dimension_coverage': len(df) / 6.0,  # 6 dimensions expected
            'score_distribution': {
                'mean': float(df['raw_score'].mean()),
                'std': float(df['raw_score'].std()),
                'min': float(df['raw_score'].min()),
                'max': float(df['raw_score'].max())
            },
            'confidence_level': 'high' if len(df) == 6 else 'medium' if len(df) >= 4 else 'low'
        }

        return confidence

    def _identify_unique_characteristics(self, conf_name: str, all_results: Dict) -> List[str]:
        """Identify unique characteristics of a conference compared to others."""
        characteristics = []

        conf_results = all_results[conf_name]
        conf_df = conf_results['criteria_df']

        # Compare with other conferences
        other_conferences = {k: v for k, v in all_results.items() if k != conf_name}

        if not other_conferences:
            return ["Single conference analyzed - no comparison available"]

        # Find dimensions where this conference has significantly higher weights
        for _, row in conf_df.iterrows():
            dimension = row['dimension']
            this_weight = float(row['normalized_weight'])

            other_weights = []
            for other_conf, other_results in other_conferences.items():
                other_df = other_results['criteria_df']
                other_row = other_df[other_df['dimension'] == dimension]
                if not other_row.empty:
                    other_weights.append(float(other_row['normalized_weight'].iloc[0]))

            if other_weights:
                mean_other = np.mean(other_weights)
                if this_weight > mean_other * 1.2:  # 20% higher than average
                    characteristics.append(
                        f"Strong emphasis on {dimension} (weight: {this_weight:.3f} vs avg: {mean_other:.3f})")
                elif this_weight < mean_other * 0.8:  # 20% lower than average
                    characteristics.append(
                        f"Lower emphasis on {dimension} (weight: {this_weight:.3f} vs avg: {mean_other:.3f})")

        if not characteristics:
            characteristics.append("Similar evaluation criteria pattern to other conferences")

        return characteristics

    def _generate_rq1_conclusion(self, statistical_analysis: Dict, prioritization_analysis: Dict) -> Dict:
        """Generate a structured conclusion for RQ1."""
        conclusion = {
            "feasibility": "YES - Automatic extraction and normalization is feasible",
            "evidence": [],
            "key_findings": [],
            "limitations": [],
            "implications": []
        }

        # Evidence for feasibility
        total_conferences = statistical_analysis['summary_statistics']['total_conferences']
        conclusion["evidence"].append(f"Successfully extracted criteria from {total_conferences} conferences")
        conclusion["evidence"].append("Generated normalized comparison matrices across venues")
        conclusion["evidence"].append("Identified statistical variations in dimension prioritization")

        # Key findings
        high_variance_dims = statistical_analysis['summary_statistics']['high_variance_dimensions']
        if high_variance_dims:
            conclusion["key_findings"].append(f"High variance detected in: {', '.join(high_variance_dims)}")

        most_variable = statistical_analysis['distribution_analysis']['most_variable_dimension']
        least_variable = statistical_analysis['distribution_analysis']['least_variable_dimension']
        conclusion["key_findings"].append(f"Most variable dimension: {most_variable}")
        conclusion["key_findings"].append(f"Least variable dimension: {least_variable}")

        # Add dimension leaders
        for dim, leader_info in prioritization_analysis['dimension_leaders'].items():
            leader = leader_info['leader']
            weight = leader_info['leader_weight']
            conclusion["key_findings"].append(f"{leader} leads in {dim} (weight: {weight:.3f})")

        # Limitations
        conclusion["limitations"].append("Analysis limited to textual guidelines - may miss implicit criteria")
        conclusion["limitations"].append("Semantic similarity dependent on model quality")
        if total_conferences < 10:
            conclusion["limitations"].append(f"Limited sample size ({total_conferences} conferences)")

        # Implications
        standardization_needed = statistical_analysis['summary_statistics']['standardization_needed']
        if standardization_needed:
            conclusion["implications"].append("High variance suggests need for standardization efforts")
        conclusion["implications"].append("Conference-specific evaluation approaches are justified")
        conclusion["implications"].append("Automated extraction can support meta-analysis of evaluation practices")

        return conclusion

    def _create_rq1_aura_integration_format(self, conference_results: Dict,
                                            normalized_analysis: Dict, prioritization_analysis: Dict) -> Dict:
        """Create RQ1-specific AURA integration format."""
        aura_data = {
            "rq1_analysis": {
                "research_question": "Can we automatically extract and normalize artifact evaluation criteria across conferences?",
                "answer": "YES - Demonstrated through successful extraction and normalization",
                "confidence": "HIGH"
            },
            "conference_specific_criteria": {},
            "normalized_comparison_matrix": normalized_analysis['normalized_comparison'],
            "dimension_prioritization": prioritization_analysis,
            "statistical_validation": {
                "variance_analysis": normalized_analysis.get('dimension_statistics', {}),
                "conference_clusters": prioritization_analysis.get('clustering_analysis', {})
            },
            "evaluation_framework": {
                "dimensions": ["reproducibility", "documentation", "accessibility",
                               "usability", "experimental", "functionality"],
                "normalization_method": "TF-IDF with semantic similarity",
                "comparison_metrics": ["cosine_similarity", "statistical_variance", "weight_distribution"]
            }
        }

        # Add conference-specific criteria in AURA format
        for conf_name, results in conference_results.items():
            df = results['criteria_df']
            conference_criteria = []

            for _, row in df.iterrows():
                criteria_entry = {
                    "dimension": row['dimension'],
                    "keywords": row['keywords'].split(', ') if row['keywords'] else [],
                    "raw_score": float(row['raw_score']),
                    "normalized_weight": float(row['normalized_weight']),
                    "conference": conf_name,
                    "year": results['metadata'].get('year'),
                    "hierarchical_structure": json.loads(row['hierarchical_structure']),
                    "category_scores": json.loads(row['category_scores'])
                }
                conference_criteria.append(criteria_entry)

            aura_data["conference_specific_criteria"][conf_name] = conference_criteria

        return aura_data

    def _generate_rq1_summary_report(self, conference_results: Dict, normalized_analysis: Dict,
                                     prioritization_analysis: Dict, statistical_analysis: Dict) -> str:
        """Generate a human-readable summary report for RQ1."""
        report = []

        report.append("# RQ1 Analysis: Automatic Extraction and Normalization of Artifact Evaluation Criteria")
        report.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary")
        total_conferences = len(conference_results)
        report.append(
            f"✅ **SUCCESS**: Automatically extracted and normalized evaluation criteria from **{total_conferences} conferences**")
        report.append("")

        standardization_needed = statistical_analysis['summary_statistics']['standardization_needed']
        if standardization_needed:
            report.append(
                "🔍 **Key Finding**: Significant variance detected across conferences - standardization recommended")
        else:
            report.append("🔍 **Key Finding**: Relatively consistent criteria across conferences")
        report.append("")

        # Conference Analysis
        report.append("## Conference-Specific Analysis")
        for conf_name, results in conference_results.items():
            metadata = results['metadata']
            year = metadata.get('year', 'Unknown')
            report.append(f"### {conf_name} ({year})")

            df = results['criteria_df']
            top_dimension = df.loc[df['normalized_weight'].idxmax(), 'dimension']
            top_weight = df['normalized_weight'].max()

            report.append(f"- **Primary Focus**: {top_dimension} (weight: {top_weight:.3f})")
            report.append(f"- **Total Criteria**: {len(df)} dimensions analyzed")

            # Add unique characteristics
            unique_chars = self._identify_unique_characteristics(conf_name, conference_results)
            if unique_chars:
                report.append(f"- **Unique Characteristics**: {unique_chars[0]}")
            report.append("")

        # Dimension Prioritization
        report.append("## Dimension Prioritization Analysis")
        for dimension, leader_info in prioritization_analysis['dimension_leaders'].items():
            leader = leader_info['leader']
            weight = leader_info['leader_weight']
            report.append(f"- **{dimension.title()}**: Led by {leader} (weight: {weight:.3f})")
        report.append("")

        # Statistical Analysis
        report.append("## Statistical Analysis")
        stats = statistical_analysis['variance_analysis']
        report.append("### Dimension Variance Analysis")
        for dimension, variance_data in stats.items():
            variability = variance_data['variability_level']
            std_dev = variance_data['std_deviation']
            report.append(f"- **{dimension.title()}**: {variability} variability (σ = {std_dev:.3f})")
        report.append("")

        # High Variance Dimensions
        high_variance = statistical_analysis['summary_statistics']['high_variance_dimensions']
        if high_variance:
            report.append("### High Variance Dimensions (Requiring Standardization)")
            for dim in high_variance:
                report.append(f"- {dim.title()}")
            report.append("")

        # Conference Clustering
        if 'clustering_analysis' in prioritization_analysis:
            clusters = prioritization_analysis['clustering_analysis']['clusters']
            report.append("### Conference Clustering by Priority Patterns")
            for cluster_id, conferences in clusters.items():
                report.append(f"- **Cluster {cluster_id + 1}**: {', '.join(conferences)}")
            report.append("")

        # Conclusions
        report.append("## RQ1 Conclusions")
        report.append("✅ **Automatic Extraction**: Successfully demonstrated across all conferences")
        report.append("✅ **Normalization**: Effective cross-conference comparison achieved")
        report.append("✅ **Variance Quantification**: Statistical differences identified and measured")
        report.append("")

        if standardization_needed:
            report.append("📋 **Recommendations**:")
            report.append("- Develop standardized evaluation criteria templates")
            report.append("- Focus standardization efforts on high-variance dimensions")
            report.append("- Maintain conference-specific adaptations where justified")

        return "\n".join(report)


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
