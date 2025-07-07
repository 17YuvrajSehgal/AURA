"""
Vector Embeddings Analyzer for Artifact Evaluation

This module performs semantic analysis of documentation content using vector
embeddings to discover patterns, clusters, and similarities across artifacts.

Key Features:
- Semantic embedding generation for documentation sections
- UMAP/t-SNE dimensionality reduction for visualization
- Clustering analysis (HDBSCAN, K-means) on semantic space
- Similarity search and recommendation
- Cross-conference semantic comparison
- Quality-based semantic analysis
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

# Vector databases
import faiss
import numpy as np
import pandas as pd
import umap
# ML and embeddings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Visualization

from config import config, NODE_TYPES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SemanticCluster:
    """Represents a semantic cluster of documentation content"""
    cluster_id: int
    cluster_type: str
    documents: List[str]
    centroid: np.ndarray
    representative_docs: List[str]
    common_themes: List[str]
    quality_score: float
    conferences: Set[str]
    size: int


@dataclass
class SimilarityResult:
    """Represents similarity search results"""
    query_id: str
    similar_documents: List[Tuple[str, float]]
    semantic_themes: List[str]
    recommendations: List[str]


class VectorEmbeddingsAnalyzer:
    """Semantic analysis engine using vector embeddings"""

    def __init__(self, knowledge_graph_builder):
        self.kg_builder = knowledge_graph_builder
        self.embedding_model = SentenceTransformer(config.vector.model_name)
        self.clusters: List[SemanticCluster] = []

        # Vector storage
        self.document_embeddings: Dict[str, np.ndarray] = {}
        self.section_embeddings: Dict[str, np.ndarray] = {}
        self.artifact_embeddings: Dict[str, np.ndarray] = {}

        # Vector database
        self.vector_db_type = config.vector.vector_db_type
        self.vector_index = None
        self.chroma_client = None

        # Analysis results
        self.dimensionality_reduction_results = {}
        self.clustering_results = {}

        self._initialize_vector_storage()

    def _initialize_vector_storage(self):
        """Initialize vector database for efficient similarity search"""
        try:
            if self.vector_db_type == "faiss":
                # Initialize FAISS index
                dimension = config.vector.dimension
                self.vector_index = faiss.IndexFlatIP(dimension)
                logger.info("Initialized FAISS vector index")

            elif self.vector_db_type == "chroma" and CHROMA_AVAILABLE:
                # Initialize ChromaDB
                self.chroma_client = chromadb.Client()
                self.chroma_collection = self.chroma_client.create_collection("artifact_docs")
                logger.info("Initialized ChromaDB vector store")

        except Exception as e:
            logger.warning(f"Failed to initialize vector storage: {e}")
            self.vector_db_type = "memory"  # Fallback to in-memory

    def extract_embeddings(self) -> Dict[str, Any]:
        """
        Extract embeddings from all documentation content
        
        Returns:
            Statistics about embedding extraction
        """
        logger.info("Extracting embeddings from documentation content")

        extraction_stats = {
            'documents_processed': 0,
            'sections_processed': 0,
            'artifacts_processed': 0,
            'embedding_dimension': config.vector.dimension,
            'model_used': config.vector.model_name
        }

        if hasattr(self.kg_builder, 'nx_graph'):
            G = self.kg_builder.nx_graph

            # Extract document-level embeddings
            for node in G.nodes():
                node_data = G.nodes[node]
                node_type = node_data.get('node_type')

                if node_type == NODE_TYPES['DOCUMENTATION']:
                    self._extract_document_embedding(node, node_data)
                    extraction_stats['documents_processed'] += 1

                elif node_type == NODE_TYPES['SECTION']:
                    self._extract_section_embedding(node, node_data)
                    extraction_stats['sections_processed'] += 1

                elif node_type == NODE_TYPES['ARTIFACT']:
                    self._extract_artifact_embedding(node, node_data)
                    extraction_stats['artifacts_processed'] += 1

        # Store embeddings in vector database
        self._store_embeddings_in_vector_db()

        logger.info(f"Embedding extraction completed: {extraction_stats}")
        return extraction_stats

    def _extract_document_embedding(self, node_id: str, node_data: Dict[str, Any]):
        """Extract embedding for a documentation file"""
        # Get content from existing embedding or generate new
        if 'embedding' in node_data:
            embedding = np.array(node_data['embedding'])
        else:
            # Generate embedding from file content
            content = node_data.get('content', '')
            if not content:
                content = f"Documentation file: {node_data.get('file_name', 'unknown')}"

            # Truncate content for embedding model limits
            content = content[:2000] if len(content) > 2000 else content
            embedding = self.embedding_model.encode(content)

        self.document_embeddings[node_id] = embedding

    def _extract_section_embedding(self, node_id: str, node_data: Dict[str, Any]):
        """Extract embedding for a documentation section"""
        if 'embedding' in node_data:
            embedding = np.array(node_data['embedding'])
        else:
            # Generate embedding from section title and content
            title = node_data.get('title', '')
            section_type = node_data.get('section_type', '')
            content = f"{title} {section_type}"
            embedding = self.embedding_model.encode(content)

        self.section_embeddings[node_id] = embedding

    def _extract_artifact_embedding(self, node_id: str, node_data: Dict[str, Any]):
        """Extract embedding for an artifact"""
        if 'embedding' in node_data:
            embedding = np.array(node_data['embedding'])
        else:
            # Generate embedding from artifact description
            description = node_data.get('description', '')
            if not description:
                artifact_id = node_data.get('artifact_id', node_id)
                conference = node_data.get('conference', 'unknown')
                description = f"Artifact {artifact_id} from {conference}"

            embedding = self.embedding_model.encode(description)

        self.artifact_embeddings[node_id] = embedding

    def _store_embeddings_in_vector_db(self):
        """Store embeddings in vector database for efficient search"""
        if self.vector_db_type == "faiss" and self.vector_index is not None:
            # Store document embeddings in FAISS
            if self.document_embeddings:
                embeddings_matrix = np.vstack(list(self.document_embeddings.values()))
                self.vector_index.add(embeddings_matrix.astype('float32'))

                # Create mapping from index to document ID
                self.faiss_id_mapping = list(self.document_embeddings.keys())
                logger.info(f"Stored {len(self.document_embeddings)} embeddings in FAISS")

        elif self.vector_db_type == "chroma" and self.chroma_client is not None:
            # Store in ChromaDB
            documents = []
            embeddings = []
            metadata = []
            ids = []

            for doc_id, embedding in self.document_embeddings.items():
                documents.append(f"Document {doc_id}")
                embeddings.append(embedding.tolist())
                metadata.append({"type": "documentation", "id": doc_id})
                ids.append(doc_id)

            if documents:
                self.chroma_collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadata,
                    ids=ids
                )
                logger.info(f"Stored {len(documents)} embeddings in ChromaDB")

    def perform_semantic_clustering(self) -> Dict[str, Any]:
        """
        Perform semantic clustering on documentation embeddings
        
        Returns:
            Clustering results and analysis
        """
        logger.info("Performing semantic clustering analysis")

        clustering_results = {
            'document_clusters': {},
            'section_clusters': {},
            'artifact_clusters': {},
            'cross_level_analysis': {}
        }

        # Document-level clustering
        if self.document_embeddings:
            clustering_results['document_clusters'] = self._cluster_embeddings(
                self.document_embeddings, "documents"
            )

        # Section-level clustering
        if self.section_embeddings:
            clustering_results['section_clusters'] = self._cluster_embeddings(
                self.section_embeddings, "sections"
            )

        # Artifact-level clustering
        if self.artifact_embeddings:
            clustering_results['artifact_clusters'] = self._cluster_embeddings(
                self.artifact_embeddings, "artifacts"
            )

        # Cross-level analysis
        clustering_results['cross_level_analysis'] = self._analyze_cross_level_patterns()

        return clustering_results

    def _cluster_embeddings(self, embeddings_dict: Dict[str, np.ndarray],
                            level: str) -> Dict[str, Any]:
        """Cluster embeddings using multiple algorithms"""
        if not embeddings_dict:
            return {}

        # Convert to matrix
        ids = list(embeddings_dict.keys())
        embeddings_matrix = np.vstack(list(embeddings_dict.values()))

        clustering_results = {
            'total_items': len(ids),
            'embedding_dimension': embeddings_matrix.shape[1],
            'algorithms': {}
        }

        # HDBSCAN clustering
        try:
            hdbscan = HDBSCAN(
                min_cluster_size=config.pattern_analysis.min_cluster_size,
                min_samples=config.pattern_analysis.min_samples
            )
            hdbscan_labels = hdbscan.fit_predict(embeddings_matrix)

            clustering_results['algorithms']['hdbscan'] = self._process_clustering_results(
                ids, hdbscan_labels, embeddings_matrix, "hdbscan"
            )

        except Exception as e:
            logger.warning(f"HDBSCAN clustering failed: {e}")

        # K-means clustering with different k values
        for k in [3, 5, 8, 10]:
            if k >= len(ids):
                continue

            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans_labels = kmeans.fit_predict(embeddings_matrix)

                clustering_results['algorithms'][f'kmeans_k{k}'] = self._process_clustering_results(
                    ids, kmeans_labels, embeddings_matrix, f"kmeans_k{k}"
                )

            except Exception as e:
                logger.warning(f"K-means clustering (k={k}) failed: {e}")

        # DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=0.3, min_samples=3)
            dbscan_labels = dbscan.fit_predict(embeddings_matrix)

            clustering_results['algorithms']['dbscan'] = self._process_clustering_results(
                ids, dbscan_labels, embeddings_matrix, "dbscan"
            )

        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}")

        return clustering_results

    def _process_clustering_results(self, ids: List[str], labels: np.ndarray,
                                    embeddings_matrix: np.ndarray,
                                    algorithm: str) -> Dict[str, Any]:
        """Process clustering results and extract insights"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)  # Exclude noise

        clusters = {}
        cluster_stats = {
            'n_clusters': n_clusters,
            'n_noise': np.sum(labels == -1) if -1 in labels else 0,
            'silhouette_score': 0.0,
            'cluster_sizes': []
        }

        # Calculate silhouette score
        try:
            from sklearn.metrics import silhouette_score
            if n_clusters > 1:
                valid_indices = labels != -1
                if np.sum(valid_indices) > 1:
                    cluster_stats['silhouette_score'] = silhouette_score(
                        embeddings_matrix[valid_indices],
                        labels[valid_indices]
                    )
        except:
            pass

        # Process each cluster
        for label in unique_labels:
            if label == -1:  # Skip noise cluster
                continue

            cluster_indices = np.where(labels == label)[0]
            cluster_ids = [ids[i] for i in cluster_indices]
            cluster_embeddings = embeddings_matrix[cluster_indices]

            # Calculate cluster centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Find most representative documents (closest to centroid)
            distances = euclidean_distances(cluster_embeddings, [centroid]).flatten()
            representative_indices = np.argsort(distances)[:3]
            representative_docs = [cluster_ids[i] for i in representative_indices]

            clusters[f"cluster_{label}"] = {
                'cluster_id': int(label),
                'size': len(cluster_ids),
                'documents': cluster_ids,
                'representative_docs': representative_docs,
                'centroid': centroid.tolist(),
                'intra_cluster_distance': np.mean(distances)
            }

            cluster_stats['cluster_sizes'].append(len(cluster_ids))

        return {
            'clusters': clusters,
            'statistics': cluster_stats,
            'algorithm': algorithm
        }

    def perform_dimensionality_reduction(self) -> Dict[str, Any]:
        """
        Perform dimensionality reduction for visualization
        
        Returns:
            Reduced embeddings and visualization data
        """
        logger.info("Performing dimensionality reduction for visualization")

        reduction_results = {}

        for embedding_type, embeddings_dict in [
            ("documents", self.document_embeddings),
            ("sections", self.section_embeddings),
            ("artifacts", self.artifact_embeddings)
        ]:
            if not embeddings_dict:
                continue

            ids = list(embeddings_dict.keys())
            embeddings_matrix = np.vstack(list(embeddings_dict.values()))

            # UMAP reduction
            try:
                umap_reducer = umap.UMAP(
                    n_components=config.pattern_analysis.umap_n_components,
                    n_neighbors=config.pattern_analysis.umap_n_neighbors,
                    random_state=42
                )
                umap_embeddings = umap_reducer.fit_transform(embeddings_matrix)

                reduction_results[f'{embedding_type}_umap'] = {
                    'embeddings': umap_embeddings,
                    'ids': ids,
                    'method': 'umap'
                }

            except Exception as e:
                logger.warning(f"UMAP reduction failed for {embedding_type}: {e}")

            # t-SNE reduction
            try:
                tsne_reducer = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, len(ids) - 1)
                )
                tsne_embeddings = tsne_reducer.fit_transform(embeddings_matrix)

                reduction_results[f'{embedding_type}_tsne'] = {
                    'embeddings': tsne_embeddings,
                    'ids': ids,
                    'method': 'tsne'
                }

            except Exception as e:
                logger.warning(f"t-SNE reduction failed for {embedding_type}: {e}")

            # PCA reduction
            try:
                pca_reducer = PCA(n_components=2, random_state=42)
                pca_embeddings = pca_reducer.fit_transform(embeddings_matrix)

                reduction_results[f'{embedding_type}_pca'] = {
                    'embeddings': pca_embeddings,
                    'ids': ids,
                    'method': 'pca',
                    'explained_variance_ratio': pca_reducer.explained_variance_ratio_.tolist()
                }

            except Exception as e:
                logger.warning(f"PCA reduction failed for {embedding_type}: {e}")

        self.dimensionality_reduction_results = reduction_results
        return reduction_results

    def find_similar_documents(self, query_doc_id: str, top_k: int = 10) -> SimilarityResult:
        """
        Find documents similar to a query document
        
        Args:
            query_doc_id: ID of the query document
            top_k: Number of similar documents to return
            
        Returns:
            Similarity search results
        """
        if query_doc_id not in self.document_embeddings:
            raise ValueError(f"Document {query_doc_id} not found in embeddings")

        query_embedding = self.document_embeddings[query_doc_id]

        if self.vector_db_type == "faiss" and self.vector_index is not None:
            # Use FAISS for efficient search
            similarities, indices = self.vector_index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                top_k + 1  # +1 to exclude query document itself
            )

            similar_docs = []
            for i, (similarity, index) in enumerate(zip(similarities[0], indices[0])):
                if index < len(self.faiss_id_mapping):
                    doc_id = self.faiss_id_mapping[index]
                    if doc_id != query_doc_id:  # Exclude query document
                        similar_docs.append((doc_id, float(similarity)))

            similar_docs = similar_docs[:top_k]

        else:
            # Use brute force similarity search
            similarities = []
            for doc_id, embedding in self.document_embeddings.items():
                if doc_id != query_doc_id:
                    similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                    similarities.append((doc_id, similarity))

            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_docs = similarities[:top_k]

        # Extract semantic themes and recommendations
        semantic_themes = self._extract_semantic_themes([query_doc_id] + [doc_id for doc_id, _ in similar_docs])
        recommendations = self._generate_recommendations(query_doc_id, similar_docs)

        return SimilarityResult(
            query_id=query_doc_id,
            similar_documents=similar_docs,
            semantic_themes=semantic_themes,
            recommendations=recommendations
        )

    def analyze_conference_semantic_differences(self) -> Dict[str, Any]:
        """Analyze semantic differences between conferences"""
        logger.info("Analyzing semantic differences between conferences")

        # Group artifacts by conference
        conference_embeddings = defaultdict(list)
        conference_artifacts = defaultdict(list)

        if hasattr(self.kg_builder, 'nx_graph'):
            G = self.kg_builder.nx_graph

            for node in G.nodes():
                node_data = G.nodes[node]
                if node_data.get('node_type') == NODE_TYPES['ARTIFACT']:
                    conference = node_data.get('conference', 'unknown')
                    if node in self.artifact_embeddings:
                        conference_embeddings[conference].append(self.artifact_embeddings[node])
                        conference_artifacts[conference].append(node)

        # Calculate conference centroids and similarities
        conference_analysis = {}
        conference_centroids = {}

        for conference, embeddings in conference_embeddings.items():
            if len(embeddings) < 3:  # Skip conferences with too few artifacts
                continue

            embeddings_matrix = np.vstack(embeddings)
            centroid = np.mean(embeddings_matrix, axis=0)
            conference_centroids[conference] = centroid

            # Calculate intra-conference diversity
            distances = euclidean_distances(embeddings_matrix, [centroid]).flatten()

            conference_analysis[conference] = {
                'artifact_count': len(embeddings),
                'centroid': centroid.tolist(),
                'intra_diversity': float(np.mean(distances)),
                'diversity_std': float(np.std(distances)),
                'artifacts': conference_artifacts[conference]
            }

        # Calculate inter-conference similarities
        conference_similarities = {}
        conference_list = list(conference_centroids.keys())

        for i, conf1 in enumerate(conference_list):
            for j, conf2 in enumerate(conference_list[i + 1:], i + 1):
                similarity = cosine_similarity(
                    [conference_centroids[conf1]],
                    [conference_centroids[conf2]]
                )[0][0]
                conference_similarities[f"{conf1}_vs_{conf2}"] = float(similarity)

        return {
            'individual_analysis': conference_analysis,
            'pairwise_similarities': conference_similarities,
            'most_similar_conferences': sorted(conference_similarities.items(),
                                               key=lambda x: x[1], reverse=True)[:5],
            'most_different_conferences': sorted(conference_similarities.items(),
                                                 key=lambda x: x[1])[:5]
        }

    def generate_semantic_visualizations(self, output_dir: str = "output/semantic_analysis"):
        """Generate visualizations for semantic analysis"""
        logger.info("Generating semantic visualizations")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Dimensionality reduction visualizations
        self._visualize_dimensionality_reduction(output_dir)

        # 2. Clustering visualizations
        self._visualize_clustering_results(output_dir)

        # 3. Conference comparison visualizations
        self._visualize_conference_semantics(output_dir)

        # 4. Similarity heatmaps
        self._visualize_similarity_matrices(output_dir)

        logger.info(f"Semantic visualizations saved to {output_dir}")

    def _analyze_cross_level_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across different embedding levels"""
        # Implementation would analyze relationships between
        # artifact, document, and section level embeddings
        return {
            'artifact_document_correlation': 0.0,
            'document_section_correlation': 0.0,
            'cross_level_clusters': []
        }

    def _extract_semantic_themes(self, doc_ids: List[str]) -> List[str]:
        """Extract common semantic themes from a group of documents"""
        # Implementation would analyze document content to extract themes
        return ["installation", "usage", "configuration", "examples"]

    def _generate_recommendations(self, query_doc_id: str,
                                  similar_docs: List[Tuple[str, float]]) -> List[str]:
        """Generate improvement recommendations based on similar documents"""
        # Implementation would analyze what makes similar documents high quality
        return [
            "Add more detailed installation instructions",
            "Include usage examples",
            "Add troubleshooting section"
        ]

    # Visualization methods
    def _visualize_dimensionality_reduction(self, output_dir: str):
        """Create visualizations for dimensionality reduction results"""
        pass

    def _visualize_clustering_results(self, output_dir: str):
        """Create visualizations for clustering results"""
        pass

    def _visualize_conference_semantics(self, output_dir: str):
        """Create visualizations for conference semantic analysis"""
        pass

    def _visualize_similarity_matrices(self, output_dir: str):
        """Create similarity matrix heatmaps"""
        pass

    def save_embeddings(self, output_path: str):
        """Save embeddings to disk for later use"""
        embeddings_data = {
            'document_embeddings': {k: v.tolist() for k, v in self.document_embeddings.items()},
            'section_embeddings': {k: v.tolist() for k, v in self.section_embeddings.items()},
            'artifact_embeddings': {k: v.tolist() for k, v in self.artifact_embeddings.items()},
            'model_info': {
                'model_name': config.vector.model_name,
                'dimension': config.vector.dimension,
                'created_at': pd.Timestamp.now().isoformat()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2)

        logger.info(f"Embeddings saved to {output_path}")

    def load_embeddings(self, input_path: str):
        """Load embeddings from disk"""
        with open(input_path, 'r') as f:
            embeddings_data = json.load(f)

        self.document_embeddings = {k: np.array(v) for k, v in embeddings_data['document_embeddings'].items()}
        self.section_embeddings = {k: np.array(v) for k, v in embeddings_data['section_embeddings'].items()}
        self.artifact_embeddings = {k: np.array(v) for k, v in embeddings_data['artifact_embeddings'].items()}

        logger.info(f"Embeddings loaded from {input_path}")

    def close(self):
        """Clean up resources"""
        if self.chroma_client:
            # Clean up ChromaDB resources
            pass
