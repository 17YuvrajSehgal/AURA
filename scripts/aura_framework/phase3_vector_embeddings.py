"""
📊 Phase 3: Vector Embeddings for RAG
Goal: Enable semantic search in addition to graph queries.

Features:
- Embed each section using OpenAI, Bedrock, or Cohere
- Store embeddings in FAISS or Qdrant 
- Link embeddings to corresponding Section nodes
- Enable semantic context retrieval for GenAI agents
- Support hybrid search (vector + graph)
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from tqdm import tqdm

# Vector storage imports
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install faiss-cpu for vector search.")

try:
    import qdrant_client
    from qdrant_client.models import VectorParams, Distance, PointStruct

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant not available. Install qdrant-client for Qdrant support.")

try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available. Install chromadb for Chroma support.")

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Install sentence-transformers.")

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install openai for OpenAI embeddings.")

# Local imports
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRecord:
    """Represents a vector embedding with metadata"""
    id: str
    vector: np.ndarray
    section_id: str
    artifact_id: str
    heading: str
    content: str
    metadata: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SearchResult:
    """Represents a search result with relevance score"""
    embedding_record: EmbeddingRecord
    relevance_score: float
    rank: int


@dataclass
class SemanticCluster:
    """Represents a semantic cluster of embeddings"""
    cluster_id: int
    center: np.ndarray
    members: List[str]  # embedding IDs
    coherence_score: float
    representative_texts: List[str]


class VectorEmbeddingEngine:
    """Advanced Vector Embedding Engine for semantic search and RAG"""

    def __init__(self,
                 vector_db_type: str = "faiss",
                 embedding_model: str = "sentence-transformers",
                 model_name: str = "all-MiniLM-L6-v2",
                 storage_directory: str = "data/embeddings"):
        """
        Initialize the Vector Embedding Engine
        
        Args:
            vector_db_type: Type of vector database (faiss, qdrant, chroma)
            embedding_model: Type of embedding model (sentence-transformers, openai)
            model_name: Specific model name
            storage_directory: Directory to store embeddings and indexes
        """
        self.vector_db_type = vector_db_type
        self.embedding_model_type = embedding_model
        self.model_name = model_name
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self._initialize_embedding_model()

        # Initialize vector database
        self._initialize_vector_database()

        # Storage for embedding records
        self.embedding_records: Dict[str, EmbeddingRecord] = {}
        self.section_to_embedding: Dict[str, str] = {}  # section_id -> embedding_id

        # Clustering results
        self.clusters: List[SemanticCluster] = []

        logger.info(f"Vector Embedding Engine initialized with {vector_db_type} and {embedding_model}")

    def _initialize_embedding_model(self):
        """Initialize the embedding model"""
        if self.embedding_model_type == "sentence-transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ValueError("SentenceTransformers not available")

            self.embedding_model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()

        elif self.embedding_model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI not available")

            self.embedding_model = openai
            if not config.llm.api_key:
                raise ValueError("OpenAI API key not provided")
            openai.api_key = config.llm.api_key
            self.embedding_dimension = 1536  # OpenAI embedding dimension

        else:
            raise ValueError(f"Unsupported embedding model type: {self.embedding_model_type}")

        logger.info(f"Initialized {self.embedding_model_type} model: {self.model_name}")

    def _initialize_vector_database(self):
        """Initialize the vector database"""
        if self.vector_db_type == "faiss":
            if not FAISS_AVAILABLE:
                raise ValueError("FAISS not available")
            self._initialize_faiss()

        elif self.vector_db_type == "qdrant":
            if not QDRANT_AVAILABLE:
                raise ValueError("Qdrant not available")
            self._initialize_qdrant()

        elif self.vector_db_type == "chroma":
            if not CHROMA_AVAILABLE:
                raise ValueError("ChromaDB not available")
            self._initialize_chroma()

        else:
            raise ValueError(f"Unsupported vector database type: {self.vector_db_type}")

    def _initialize_faiss(self):
        """Initialize FAISS index"""
        # Create index after we know the dimension
        self.faiss_index = None
        self.faiss_id_mapping = {}  # faiss_id -> embedding_id
        self.next_faiss_id = 0

    def _initialize_qdrant(self):
        """Initialize Qdrant client"""
        self.qdrant_client = qdrant_client.QdrantClient(":memory:")  # In-memory for demo
        self.collection_name = "artifact_embeddings"

    def _initialize_chroma(self):
        """Initialize ChromaDB client"""
        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.create_collection("artifact_embeddings")

    def extract_embeddings_from_processed_artifacts(self,
                                                    processed_artifacts_dir: str,
                                                    max_artifacts: Optional[int] = None,
                                                    batch_size: int = 32) -> Dict[str, Any]:
        """
        Extract embeddings from all processed artifacts
        
        Args:
            processed_artifacts_dir: Directory containing processed artifact JSON files
            max_artifacts: Maximum number of artifacts to process
            batch_size: Batch size for embedding generation
            
        Returns:
            Statistics about embedding extraction
        """
        artifacts_dir = Path(processed_artifacts_dir)
        artifact_files = list(artifacts_dir.glob("*_processed.json"))

        if max_artifacts:
            artifact_files = artifact_files[:max_artifacts]

        logger.info(f"Extracting embeddings from {len(artifact_files)} processed artifacts")

        stats = {
            'artifacts_processed': 0,
            'sections_processed': 0,
            'embeddings_created': 0,
            'total_tokens': 0,
            'avg_embedding_time': 0,
            'dimension': self.embedding_dimension,
            'model_used': self.model_name
        }

        start_time = datetime.now()
        total_embedding_time = 0

        # Process artifacts and extract sections for embedding
        sections_to_embed = []

        for artifact_file in tqdm(artifact_files, desc="Loading artifacts"):
            try:
                with open(artifact_file, 'r', encoding='utf-8') as f:
                    artifact_data = json.load(f)

                artifact_id = artifact_data['metadata']['artifact_name']

                # Extract sections from documentation files
                for doc_file in artifact_data['documentation_files']:
                    sections = doc_file.get('sections', [])
                    for section_data in sections:
                        # Handle both dict and dataclass objects
                        if hasattr(section_data, '__dict__'):
                            section_dict = section_data.__dict__
                        else:
                            section_dict = section_data
                            
                        section_id = self._generate_section_id(
                            artifact_id,
                            section_dict.get('doc_path', ''),
                            section_dict.get('section_order', 0)
                        )

                        # Prepare text for embedding
                        text_to_embed = self._prepare_text_for_embedding(section_dict)

                        sections_to_embed.append({
                            'section_id': section_id,
                            'artifact_id': artifact_id,
                            'heading': section_dict.get('heading', ''),
                            'content': section_dict.get('content', ''),
                            'text_to_embed': text_to_embed,
                            'metadata': {
                                'doc_path': section_dict.get('doc_path', ''),
                                'section_order': section_dict.get('section_order', 0),
                                'level': section_dict.get('level', 1),
                                'content_length': len(section_dict.get('content', '')),
                                'tools': section_dict.get('tools', []),
                                'entities': section_dict.get('entities', []),
                                'commands_count': len(section_dict.get('commands', [])),
                                'structural_features': section_dict.get('structural_features', {})
                            }
                        })

                stats['artifacts_processed'] += 1

            except Exception as e:
                logger.error(f"Error processing artifact file {artifact_file}: {e}")

        stats['sections_processed'] = len(sections_to_embed)
        logger.info(f"Prepared {len(sections_to_embed)} sections for embedding")

        # Generate embeddings in batches
        for i in tqdm(range(0, len(sections_to_embed), batch_size), desc="Generating embeddings"):
            batch = sections_to_embed[i:i + batch_size]

            batch_start_time = datetime.now()

            # Extract texts for batch embedding
            texts = [section['text_to_embed'] for section in batch]

            # Generate embeddings
            embeddings = self._generate_embeddings_batch(texts)

            batch_embedding_time = (datetime.now() - batch_start_time).total_seconds()
            total_embedding_time += batch_embedding_time

            # Create embedding records and store
            for section, embedding in zip(batch, embeddings):
                embedding_record = EmbeddingRecord(
                    id=self._generate_embedding_id(section['section_id']),
                    vector=embedding,
                    section_id=section['section_id'],
                    artifact_id=section['artifact_id'],
                    heading=section['heading'],
                    content=section['content'][:500],  # Truncate for storage
                    metadata=section['metadata']
                )

                # Store embedding record
                self.embedding_records[embedding_record.id] = embedding_record
                self.section_to_embedding[section['section_id']] = embedding_record.id

                # Add to vector database
                self._add_embedding_to_database(embedding_record)

                stats['embeddings_created'] += 1
                stats['total_tokens'] += len(section['text_to_embed'].split())

        # Calculate final statistics
        total_time = (datetime.now() - start_time).total_seconds()
        stats['total_time'] = total_time
        stats['avg_embedding_time'] = total_embedding_time / stats['sections_processed'] if stats[
                                                                                                'sections_processed'] > 0 else 0

        # Save embeddings to disk
        self._save_embeddings()

        # Initialize vector index for search
        self._finalize_vector_database()

        logger.info(f"Embedding extraction completed: {stats}")
        return stats

    def _prepare_text_for_embedding(self, section_data: Dict) -> str:
        """Prepare section text for embedding"""
        # Handle both dict and dataclass objects
        if hasattr(section_data, '__dict__'):
            section_dict = section_data.__dict__
        else:
            section_dict = section_data
            
        # Combine heading and content
        heading = section_dict.get('heading', '')
        text_parts = [heading] if heading else []

        # Add content (truncated if too long)
        content = section_dict.get('content', '')
        if len(content) > config.processing.chunk_size:
            content = content[:config.processing.chunk_size]
        if content:
            text_parts.append(content)

        # Add context from tools and entities
        tools = section_dict.get('tools', [])
        if tools and isinstance(tools, list):
            text_parts.append(f"Tools: {', '.join(tools)}")

        entities = section_dict.get('entities', [])
        if entities and isinstance(entities, list):
            text_parts.append(f"Related: {', '.join(entities)}")

        return " ".join(text_parts)

    def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        if self.embedding_model_type == "sentence-transformers":
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return [emb for emb in embeddings]

        elif self.embedding_model_type == "openai":
            embeddings = []
            for text in texts:
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = np.array(response['data'][0]['embedding'])
                embeddings.append(embedding)
            return embeddings

        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_type}")

    def _add_embedding_to_database(self, embedding_record: EmbeddingRecord):
        """Add embedding to the vector database"""
        if self.vector_db_type == "faiss":
            self._add_to_faiss(embedding_record)
        elif self.vector_db_type == "qdrant":
            self._add_to_qdrant(embedding_record)
        elif self.vector_db_type == "chroma":
            self._add_to_chroma(embedding_record)

    def _add_to_faiss(self, embedding_record: EmbeddingRecord):
        """Add embedding to FAISS index"""
        # Initialize FAISS index if needed
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)

        # Normalize embedding for cosine similarity
        embedding = embedding_record.vector.copy().astype(np.float32)
        faiss.normalize_L2(embedding.reshape(1, -1))

        # Add to index
        self.faiss_index.add(embedding.reshape(1, -1))

        # Store mapping
        self.faiss_id_mapping[self.next_faiss_id] = embedding_record.id
        self.next_faiss_id += 1

    def _add_to_qdrant(self, embedding_record: EmbeddingRecord):
        """Add embedding to Qdrant"""
        # Create collection if needed
        try:
            self.qdrant_client.get_collection(self.collection_name)
        except:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )

        # Add point
        point = PointStruct(
            id=hash(embedding_record.id) % (2 ** 31),  # Convert string ID to int
            vector=embedding_record.vector.tolist(),
            payload={
                'embedding_id': embedding_record.id,
                'section_id': embedding_record.section_id,
                'artifact_id': embedding_record.artifact_id,
                'heading': embedding_record.heading,
                'content': embedding_record.content,
                **embedding_record.metadata
            }
        )

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def _add_to_chroma(self, embedding_record: EmbeddingRecord):
        """Add embedding to ChromaDB"""
        self.chroma_collection.add(
            embeddings=[embedding_record.vector.tolist()],
            documents=[embedding_record.content],
            metadatas=[{
                'embedding_id': embedding_record.id,
                'section_id': embedding_record.section_id,
                'artifact_id': embedding_record.artifact_id,
                'heading': embedding_record.heading,
                **{k: str(v) for k, v in embedding_record.metadata.items()}  # Convert to strings
            }],
            ids=[embedding_record.id]
        )

    def _finalize_vector_database(self):
        """Finalize vector database for searching"""
        if self.vector_db_type == "faiss" and self.faiss_index:
            # Save FAISS index
            index_path = self.storage_directory / "faiss_index.bin"
            faiss.write_index(self.faiss_index, str(index_path))
            logger.info(f"FAISS index saved to {index_path}")

    def semantic_search(self,
                        query: str,
                        top_k: int = 10,
                        artifact_filter: Optional[str] = None,
                        section_type_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Perform semantic search across all embeddings
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            artifact_filter: Filter by specific artifact ID
            section_type_filter: Filter by section heading pattern
            
        Returns:
            List of search results ordered by relevance
        """
        # Generate query embedding
        query_embedding = self._generate_embeddings_batch([query])[0]

        if self.vector_db_type == "faiss":
            return self._search_faiss(query_embedding, top_k, artifact_filter, section_type_filter)
        elif self.vector_db_type == "qdrant":
            return self._search_qdrant(query_embedding, top_k, artifact_filter, section_type_filter)
        elif self.vector_db_type == "chroma":
            return self._search_chroma(query, top_k, artifact_filter, section_type_filter)
        else:
            raise ValueError(f"Search not implemented for {self.vector_db_type}")

    def _search_faiss(self,
                      query_embedding: np.ndarray,
                      top_k: int,
                      artifact_filter: Optional[str],
                      section_type_filter: Optional[str]) -> List[SearchResult]:
        """Search using FAISS index"""
        if self.faiss_index is None:
            return []

        # Normalize query embedding
        query_vec = query_embedding.copy().astype(np.float32)
        faiss.normalize_L2(query_vec.reshape(1, -1))

        # Search
        scores, indices = self.faiss_index.search(query_vec.reshape(1, -1), top_k * 2)  # Get more for filtering

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue

            embedding_id = self.faiss_id_mapping.get(idx)
            if not embedding_id or embedding_id not in self.embedding_records:
                continue

            embedding_record = self.embedding_records[embedding_id]

            # Apply filters
            if artifact_filter and embedding_record.artifact_id != artifact_filter:
                continue

            if section_type_filter and section_type_filter.lower() not in embedding_record.heading.lower():
                continue

            results.append(SearchResult(
                embedding_record=embedding_record,
                relevance_score=float(score),
                rank=len(results) + 1
            ))

            if len(results) >= top_k:
                break

        return results

    def _search_qdrant(self,
                       query_embedding: np.ndarray,
                       top_k: int,
                       artifact_filter: Optional[str],
                       section_type_filter: Optional[str]) -> List[SearchResult]:
        """Search using Qdrant"""
        # Build filter conditions
        filter_conditions = {}
        if artifact_filter:
            filter_conditions['artifact_id'] = artifact_filter

        # Search
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=filter_conditions if filter_conditions else None
        )

        results = []
        for i, result in enumerate(search_results):
            embedding_id = result.payload['embedding_id']
            if embedding_id in self.embedding_records:
                embedding_record = self.embedding_records[embedding_id]

                # Apply section type filter
                if section_type_filter and section_type_filter.lower() not in embedding_record.heading.lower():
                    continue

                results.append(SearchResult(
                    embedding_record=embedding_record,
                    relevance_score=result.score,
                    rank=i + 1
                ))

        return results

    def _search_chroma(self,
                       query: str,
                       top_k: int,
                       artifact_filter: Optional[str],
                       section_type_filter: Optional[str]) -> List[SearchResult]:
        """Search using ChromaDB"""
        # Build where conditions
        where_conditions = {}
        if artifact_filter:
            where_conditions['artifact_id'] = artifact_filter

        # Search
        search_results = self.chroma_collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_conditions if where_conditions else None
        )

        results = []
        for i, (doc_id, distance) in enumerate(zip(search_results['ids'][0], search_results['distances'][0])):
            if doc_id in self.embedding_records:
                embedding_record = self.embedding_records[doc_id]

                # Apply section type filter
                if section_type_filter and section_type_filter.lower() not in embedding_record.heading.lower():
                    continue

                # Convert distance to similarity score
                similarity_score = 1 / (1 + distance)

                results.append(SearchResult(
                    embedding_record=embedding_record,
                    relevance_score=similarity_score,
                    rank=i + 1
                ))

        return results

    def find_similar_sections(self,
                              section_id: str,
                              top_k: int = 5,
                              exclude_same_artifact: bool = True) -> List[SearchResult]:
        """
        Find similar sections to a given section
        
        Args:
            section_id: ID of the reference section
            top_k: Number of similar sections to return
            exclude_same_artifact: Whether to exclude sections from the same artifact
            
        Returns:
            List of similar sections
        """
        if section_id not in self.section_to_embedding:
            return []

        embedding_id = self.section_to_embedding[section_id]
        embedding_record = self.embedding_records[embedding_id]

        # Use the section content as query
        results = self.semantic_search(
            query=embedding_record.content,
            top_k=top_k + 1  # +1 to exclude the original section
        )

        # Filter results
        filtered_results = []
        for result in results:
            # Skip the original section
            if result.embedding_record.section_id == section_id:
                continue

            # Skip same artifact if requested
            if exclude_same_artifact and result.embedding_record.artifact_id == embedding_record.artifact_id:
                continue

            filtered_results.append(result)

            if len(filtered_results) >= top_k:
                break

        return filtered_results

    def perform_semantic_clustering(self,
                                    n_clusters: int = 10,
                                    min_cluster_size: int = 5) -> List[SemanticCluster]:
        """
        Perform semantic clustering on all embeddings
        
        Args:
            n_clusters: Number of clusters to create
            min_cluster_size: Minimum size for a valid cluster
            
        Returns:
            List of semantic clusters
        """
        if not self.embedding_records:
            return []

        logger.info(f"Performing semantic clustering with {n_clusters} clusters")

        # Collect all embeddings
        embeddings = []
        embedding_ids = []

        for embedding_id, record in self.embedding_records.items():
            embeddings.append(record.vector)
            embedding_ids.append(embedding_id)

        embeddings_array = np.array(embeddings)

        # Perform K-means clustering
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_array)

            # Create cluster objects
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_members = [
                    embedding_ids[i] for i, label in enumerate(cluster_labels)
                    if label == cluster_id
                ]

                if len(cluster_members) < min_cluster_size:
                    continue

                # Calculate cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]

                # Calculate coherence score (average cosine similarity to center)
                coherence_scores = []
                representative_texts = []

                for member_id in cluster_members[:5]:  # Sample first 5 for representatives
                    member_record = self.embedding_records[member_id]

                    # Calculate cosine similarity to center
                    similarity = np.dot(member_record.vector, cluster_center) / (
                            np.linalg.norm(member_record.vector) * np.linalg.norm(cluster_center)
                    )
                    coherence_scores.append(similarity)
                    representative_texts.append(f"{member_record.heading}: {member_record.content[:100]}...")

                coherence_score = np.mean(coherence_scores) if coherence_scores else 0

                cluster = SemanticCluster(
                    cluster_id=cluster_id,
                    center=cluster_center,
                    members=cluster_members,
                    coherence_score=coherence_score,
                    representative_texts=representative_texts
                )

                clusters.append(cluster)

            self.clusters = clusters
            logger.info(f"Created {len(clusters)} semantic clusters")

            return clusters

        except ImportError:
            logger.warning("scikit-learn not available for clustering")
            return []

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about embeddings"""
        if not self.embedding_records:
            return {}

        # Basic statistics
        stats = {
            'total_embeddings': len(self.embedding_records),
            'embedding_dimension': self.embedding_dimension,
            'vector_db_type': self.vector_db_type,
            'embedding_model': self.model_name,
            'storage_size_mb': 0  # TODO: Calculate actual storage size
        }

        # Artifact distribution
        artifact_distribution = {}
        section_types = {}
        content_lengths = []

        for record in self.embedding_records.values():
            # Artifact distribution
            artifact_id = record.artifact_id
            artifact_distribution[artifact_id] = artifact_distribution.get(artifact_id, 0) + 1

            # Section types
            heading_type = record.heading.lower().split()[0] if record.heading else 'unknown'
            section_types[heading_type] = section_types.get(heading_type, 0) + 1

            # Content lengths
            content_lengths.append(record.metadata.get('content_length', 0))

        stats.update({
            'unique_artifacts': len(artifact_distribution),
            'avg_sections_per_artifact': np.mean(list(artifact_distribution.values())),
            'section_types': dict(sorted(section_types.items(), key=lambda x: x[1], reverse=True)[:10]),
            'avg_content_length': np.mean(content_lengths) if content_lengths else 0,
            'total_clusters': len(self.clusters)
        })

        return stats

    def _generate_section_id(self, artifact_id: str, doc_path: str, section_order: int) -> str:
        """Generate unique section ID"""
        combined = f"{artifact_id}:{doc_path}:{section_order}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _generate_embedding_id(self, section_id: str) -> str:
        """Generate unique embedding ID"""
        return f"emb_{section_id}"

    def _save_embeddings(self):
        """Save embeddings to disk"""
        embeddings_file = self.storage_directory / "embeddings.pkl"
        mappings_file = self.storage_directory / "mappings.pkl"

        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embedding_records, f)

        with open(mappings_file, 'wb') as f:
            pickle.dump({
                'section_to_embedding': self.section_to_embedding,
                'faiss_id_mapping': getattr(self, 'faiss_id_mapping', {})
            }, f)

        logger.info(f"Saved {len(self.embedding_records)} embeddings to {embeddings_file}")

    def load_embeddings(self):
        """Load embeddings from disk"""
        embeddings_file = self.storage_directory / "embeddings.pkl"
        mappings_file = self.storage_directory / "mappings.pkl"

        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                self.embedding_records = pickle.load(f)

        if mappings_file.exists():
            with open(mappings_file, 'rb') as f:
                mappings = pickle.load(f)
                self.section_to_embedding = mappings.get('section_to_embedding', {})
                self.faiss_id_mapping = mappings.get('faiss_id_mapping', {})

        # Reload FAISS index
        if self.vector_db_type == "faiss":
            index_path = self.storage_directory / "faiss_index.bin"
            if index_path.exists():
                self.faiss_index = faiss.read_index(str(index_path))
                self.next_faiss_id = len(self.faiss_id_mapping)

        logger.info(f"Loaded {len(self.embedding_records)} embeddings from disk")


def main():
    """Example usage of the Vector Embedding Engine"""
    # Initialize embedding engine
    embedding_engine = VectorEmbeddingEngine(
        vector_db_type="faiss",
        embedding_model="sentence-transformers",
        model_name="all-MiniLM-L6-v2"
    )

    # Extract embeddings from processed artifacts
    stats = embedding_engine.extract_embeddings_from_processed_artifacts(
        "data/processed_artifacts",
        max_artifacts=10
    )

    # Print embedding statistics
    print("\n📊 Phase 3: Vector Embeddings Results")
    print("=" * 50)
    print(f"✅ Embeddings created: {stats['embeddings_created']:,}")
    print(f"📄 Sections processed: {stats['sections_processed']:,}")
    print(f"🏛️  Artifacts processed: {stats['artifacts_processed']}")
    print(f"📏 Embedding dimension: {stats['dimension']}")
    print(f"🤖 Model used: {stats['model_used']}")
    print(f"⏱️  Avg embedding time: {stats['avg_embedding_time']:.4f}s")

    # Perform semantic search example
    search_results = embedding_engine.semantic_search("installation setup docker", top_k=5)
    print(f"\n🔍 Sample search results ({len(search_results)}):")
    for i, result in enumerate(search_results[:3]):
        print(f"  {i + 1}. {result.embedding_record.heading} (score: {result.relevance_score:.3f})")
        print(f"     Artifact: {result.embedding_record.artifact_id}")

    # Perform clustering
    clusters = embedding_engine.perform_semantic_clustering(n_clusters=5)
    print(f"\n🎯 Semantic clusters created: {len(clusters)}")
    for cluster in clusters[:3]:
        print(
            f"  Cluster {cluster.cluster_id}: {len(cluster.members)} members, coherence: {cluster.coherence_score:.3f}")

    # Get final statistics
    final_stats = embedding_engine.get_embedding_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"  - Total embeddings: {final_stats['total_embeddings']:,}")
    print(f"  - Unique artifacts: {final_stats['unique_artifacts']}")
    print(f"  - Avg sections/artifact: {final_stats['avg_sections_per_artifact']:.1f}")
    print(f"  - Top section types: {list(final_stats['section_types'].keys())[:5]}")


if __name__ == "__main__":
    main()
