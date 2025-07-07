"""
RAG (Retrieval Augmented Generation) System for README Documentation Generator

This module implements a hybrid retrieval system that combines vector similarity
search with knowledge graph traversal to find relevant context for README generation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from neo4j import GraphDatabase
import faiss

from config import config, NODE_TYPES, RELATIONSHIP_TYPES
from knowledge_graph_builder import KnowledgeGraphBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Represents a retrieval result with context and relevance score"""
    content: str
    node_type: str
    relevance_score: float
    metadata: Dict[str, Any]
    source_path: str = ""

class RAGRetriever:
    """
    Hybrid retrieval system combining vector similarity and graph traversal
    """
    
    def __init__(self, knowledge_graph_builder: KnowledgeGraphBuilder):
        self.kg_builder = knowledge_graph_builder
        self.embedding_model = SentenceTransformer(config.vector.model_name)
        self.vector_index = None
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        
        # Initialize vector index
        self._build_vector_index()
        
        # Initialize graph connections
        self.use_neo4j = knowledge_graph_builder.use_neo4j
        if self.use_neo4j and hasattr(knowledge_graph_builder, 'driver'):
            self.driver = knowledge_graph_builder.driver
        else:
            self.nx_graph = knowledge_graph_builder.nx_graph
        
        logger.info(f"RAG Retriever initialized with {len(self.kg_builder.nodes)} nodes")
    
    def _build_vector_index(self):
        """Build FAISS vector index from node embeddings"""
        if not self.kg_builder.nodes:
            logger.warning("No nodes available for vector index")
            return
        
        # Collect embeddings and build mapping
        embeddings = []
        node_ids = []
        
        for idx, (node_id, node) in enumerate(self.kg_builder.nodes.items()):
            if 'embedding' in node.properties:
                embeddings.append(node.properties['embedding'])
                node_ids.append(node_id)
                self.node_id_to_idx[node_id] = idx
                self.idx_to_node_id[idx] = node_id
        
        if not embeddings:
            logger.warning("No embeddings found in nodes")
            return
        
        # Create FAISS index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]
        
        # Use IndexFlatIP for inner product similarity
        self.vector_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add embeddings to index
        self.vector_index.add(embeddings_array)
        
        logger.info(f"Built vector index with {len(embeddings)} embeddings")
    
    def retrieve_for_section(self, section_type: str, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve relevant context for a specific README section
        
        Args:
            section_type: Type of section (e.g., 'setup', 'usage', 'title_purpose')
            query: Natural language query describing what information is needed
            top_k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects containing relevant context
        """
        logger.info(f"Retrieving context for section: {section_type}")
        
        # Get section-specific retrieval strategy
        retrieval_strategy = self._get_retrieval_strategy(section_type)
        
        # Combine vector search with graph traversal
        vector_results = self._vector_search(query, top_k)
        graph_results = self._graph_search(section_type, query, top_k)
        
        # Merge and rank results
        combined_results = self._merge_and_rank_results(
            vector_results, 
            graph_results, 
            retrieval_strategy
        )
        
        return combined_results[:top_k]
    
    def _get_retrieval_strategy(self, section_type: str) -> Dict[str, Any]:
        """Get section-specific retrieval strategy"""
        strategies = {
            'title_purpose': {
                'focus_nodes': [NODE_TYPES['ARTIFACT'], NODE_TYPES['SECTION']],
                'focus_relationships': [RELATIONSHIP_TYPES['DESCRIBES']],
                'weight_vector': 0.7,
                'weight_graph': 0.3
            },
            'setup': {
                'focus_nodes': [NODE_TYPES['DEPENDENCY'], NODE_TYPES['COMMAND'], NODE_TYPES['TOOL']],
                'focus_relationships': [RELATIONSHIP_TYPES['DEPENDS_ON'], RELATIONSHIP_TYPES['REQUIRES']],
                'weight_vector': 0.5,
                'weight_graph': 0.5
            },
            'usage': {
                'focus_nodes': [NODE_TYPES['COMMAND'], NODE_TYPES['FILE']],
                'focus_relationships': [RELATIONSHIP_TYPES['GENERATES'], RELATIONSHIP_TYPES['PRODUCES']],
                'weight_vector': 0.4,
                'weight_graph': 0.6
            },
            'outputs': {
                'focus_nodes': [NODE_TYPES['OUTPUT'], NODE_TYPES['DATASET']],
                'focus_relationships': [RELATIONSHIP_TYPES['GENERATES'], RELATIONSHIP_TYPES['PRODUCES']],
                'weight_vector': 0.6,
                'weight_graph': 0.4
            },
            'structure': {
                'focus_nodes': [NODE_TYPES['SECTION']],
                'focus_relationships': [RELATIONSHIP_TYPES['CONTAINS'], RELATIONSHIP_TYPES['PART_OF']],
                'weight_vector': 0.3,
                'weight_graph': 0.7
            }
        }
        
        return strategies.get(section_type, {
            'focus_nodes': list(NODE_TYPES.values()),
            'focus_relationships': list(RELATIONSHIP_TYPES.values()),
            'weight_vector': 0.5,
            'weight_graph': 0.5
        })
    
    def _vector_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform vector similarity search"""
        if not self.vector_index:
            logger.warning("Vector index not available")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.vector_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            node_id = self.idx_to_node_id[idx]
            node = self.kg_builder.nodes[node_id]
            
            # Extract content based on node type
            content = self._extract_node_content(node)
            
            result = RetrievalResult(
                content=content,
                node_type=node.type,
                relevance_score=float(score),
                metadata=node.properties,
                source_path=node.properties.get('path', '')
            )
            results.append(result)
        
        return results
    
    def _graph_search(self, section_type: str, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform graph-based search using relationship traversal"""
        if self.use_neo4j:
            return self._neo4j_graph_search(section_type, query, top_k)
        else:
            return self._networkx_graph_search(section_type, query, top_k)
    
    def _neo4j_graph_search(self, section_type: str, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform graph search using Neo4j"""
        if not hasattr(self, 'driver'):
            return []
        
        try:
            with self.driver.session() as session:
                # Build Cypher query based on section type
                cypher_query = self._build_cypher_query(section_type, top_k)
                
                # Execute query
                result = session.run(cypher_query, query=query)
                
                results = []
                for record in result:
                    node_data = record['n']
                    
                    content = self._extract_content_from_neo4j_node(node_data)
                    
                    result_obj = RetrievalResult(
                        content=content,
                        node_type=list(node_data.labels)[0],
                        relevance_score=1.0,  # Default score, could be improved
                        metadata=dict(node_data),
                        source_path=node_data.get('path', '')
                    )
                    results.append(result_obj)
                
                return results
        
        except Exception as e:
            logger.error(f"Neo4j graph search failed: {e}")
            return []
    
    def _networkx_graph_search(self, section_type: str, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform graph search using NetworkX"""
        if not hasattr(self, 'nx_graph'):
            return []
        
        strategy = self._get_retrieval_strategy(section_type)
        focus_nodes = strategy.get('focus_nodes', [])
        focus_relationships = strategy.get('focus_relationships', [])
        
        results = []
        
        # Find relevant nodes by type
        relevant_nodes = []
        for node_id, node_data in self.nx_graph.nodes(data=True):
            if node_data.get('node_type') in focus_nodes:
                relevant_nodes.append((node_id, node_data))
        
        # Score nodes based on relationships
        for node_id, node_data in relevant_nodes:
            score = 0.0
            
            # Check incoming relationships
            for pred in self.nx_graph.predecessors(node_id):
                edge_data = self.nx_graph.get_edge_data(pred, node_id)
                if edge_data and edge_data.get('relationship_type') in focus_relationships:
                    score += 1.0
            
            # Check outgoing relationships
            for succ in self.nx_graph.successors(node_id):
                edge_data = self.nx_graph.get_edge_data(node_id, succ)
                if edge_data and edge_data.get('relationship_type') in focus_relationships:
                    score += 1.0
            
            if score > 0:
                content = self._extract_content_from_networkx_node(node_data)
                
                result = RetrievalResult(
                    content=content,
                    node_type=node_data.get('node_type', 'Unknown'),
                    relevance_score=score,
                    metadata=node_data,
                    source_path=node_data.get('path', '')
                )
                results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:top_k]
    
    def _build_cypher_query(self, section_type: str, top_k: int) -> str:
        """Build Cypher query for Neo4j graph search"""
        strategy = self._get_retrieval_strategy(section_type)
        focus_nodes = strategy.get('focus_nodes', [])
        focus_relationships = strategy.get('focus_relationships', [])
        
        # Build node type filter
        node_filters = ' OR '.join([f'n:{node_type}' for node_type in focus_nodes])
        
        # Build relationship patterns
        relationship_patterns = []
        for rel_type in focus_relationships:
            relationship_patterns.append(f'(n)-[:{rel_type}]-()')
            relationship_patterns.append(f'()-[:{rel_type}]-(n)')
        
        query = f"""
        MATCH (n)
        WHERE {node_filters}
        OPTIONAL MATCH {' OR '.join(relationship_patterns) if relationship_patterns else '(n)'}
        RETURN n
        LIMIT {top_k}
        """
        
        return query
    
    def _extract_node_content(self, node) -> str:
        """Extract meaningful content from a node"""
        content_parts = []
        
        # Add name/title
        if 'name' in node.properties:
            content_parts.append(f"Name: {node.properties['name']}")
        
        if 'title' in node.properties:
            content_parts.append(f"Title: {node.properties['title']}")
        
        # Add description
        if 'description' in node.properties:
            content_parts.append(f"Description: {node.properties['description']}")
        
        # Add content
        if 'content' in node.properties:
            content = node.properties['content']
            if len(content) > 500:
                content = content[:500] + "..."
            content_parts.append(f"Content: {content}")
        
        # Add command
        if 'command' in node.properties:
            content_parts.append(f"Command: {node.properties['command']}")
        
        # Add path
        if 'path' in node.properties:
            content_parts.append(f"Path: {node.properties['path']}")
        
        return "\n".join(content_parts)
    
    def _extract_content_from_neo4j_node(self, node_data) -> str:
        """Extract content from Neo4j node data"""
        content_parts = []
        
        for key, value in node_data.items():
            if key == 'embedding':
                continue
            if isinstance(value, (str, int, float, bool)):
                content_parts.append(f"{key}: {value}")
        
        return "\n".join(content_parts)
    
    def _extract_content_from_networkx_node(self, node_data) -> str:
        """Extract content from NetworkX node data"""
        content_parts = []
        
        for key, value in node_data.items():
            if key in ['embedding', 'node_type']:
                continue
            if isinstance(value, (str, int, float, bool)):
                content_parts.append(f"{key}: {value}")
        
        return "\n".join(content_parts)
    
    def _merge_and_rank_results(self, vector_results: List[RetrievalResult], 
                               graph_results: List[RetrievalResult], 
                               strategy: Dict[str, Any]) -> List[RetrievalResult]:
        """Merge vector and graph results with weighted ranking"""
        
        weight_vector = strategy.get('weight_vector', 0.5)
        weight_graph = strategy.get('weight_graph', 0.5)
        
        # Combine results
        all_results = {}
        
        # Add vector results
        for result in vector_results:
            key = result.metadata.get('id', result.content[:50])
            if key not in all_results:
                all_results[key] = result
                all_results[key].relevance_score *= weight_vector
            else:
                all_results[key].relevance_score += result.relevance_score * weight_vector
        
        # Add graph results
        for result in graph_results:
            key = result.metadata.get('id', result.content[:50])
            if key not in all_results:
                all_results[key] = result
                all_results[key].relevance_score *= weight_graph
            else:
                all_results[key].relevance_score += result.relevance_score * weight_graph
        
        # Sort by combined relevance score
        sorted_results = sorted(all_results.values(), 
                               key=lambda x: x.relevance_score, 
                               reverse=True)
        
        return sorted_results
    
    def get_section_context(self, section_type: str, additional_filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get comprehensive context for a specific section type
        
        Args:
            section_type: Type of section to get context for
            additional_filters: Additional filters to apply
            
        Returns:
            Dictionary containing structured context information
        """
        context = {
            'section_type': section_type,
            'artifact_info': {},
            'dependencies': [],
            'commands': [],
            'files': [],
            'structure': [],
            'outputs': []
        }
        
        # Get artifact information
        artifact_nodes = [node for node in self.kg_builder.nodes.values() 
                         if node.type == NODE_TYPES['ARTIFACT']]
        
        if artifact_nodes:
            artifact = artifact_nodes[0]
            context['artifact_info'] = {
                'id': artifact.properties.get('name', ''),
                'path': artifact.properties.get('path', ''),
                'size_mb': artifact.properties.get('repo_size_mb', 0),
                'extraction_method': artifact.properties.get('extraction_method', ''),
                'description': artifact.properties.get('description', '')
            }
        
        # Get dependencies
        dep_nodes = [node for node in self.kg_builder.nodes.values() 
                    if node.type == NODE_TYPES['DEPENDENCY']]
        
        for dep_node in dep_nodes:
            context['dependencies'].append({
                'name': dep_node.properties.get('name', ''),
                'type': dep_node.properties.get('type', ''),
                'description': dep_node.properties.get('description', '')
            })
        
        # Get commands
        cmd_nodes = [node for node in self.kg_builder.nodes.values() 
                    if node.type == NODE_TYPES['COMMAND']]
        
        for cmd_node in cmd_nodes:
            context['commands'].append({
                'command': cmd_node.properties.get('command', ''),
                'type': cmd_node.properties.get('type', ''),
                'description': cmd_node.properties.get('description', '')
            })
        
        # Get files
        file_nodes = [node for node in self.kg_builder.nodes.values() 
                     if node.type == NODE_TYPES['FILE']]
        
        for file_node in file_nodes:
            context['files'].append({
                'name': file_node.properties.get('name', ''),
                'path': file_node.properties.get('path', ''),
                'type': file_node.properties.get('type', ''),
                'size': file_node.properties.get('size', 0)
            })
        
        # Get structure information
        section_nodes = [node for node in self.kg_builder.nodes.values() 
                        if node.type == NODE_TYPES['SECTION']]
        
        for section_node in section_nodes:
            if section_node.properties.get('type') == 'directory_structure':
                context['structure'].append({
                    'name': section_node.properties.get('name', ''),
                    'content': section_node.properties.get('content', '')
                })
        
        # Get outputs
        output_nodes = [node for node in self.kg_builder.nodes.values() 
                       if node.type == NODE_TYPES['OUTPUT']]
        dataset_nodes = [node for node in self.kg_builder.nodes.values() 
                        if node.type == NODE_TYPES['DATASET']]
        
        for output_node in output_nodes + dataset_nodes:
            context['outputs'].append({
                'name': output_node.properties.get('name', ''),
                'path': output_node.properties.get('path', ''),
                'type': output_node.properties.get('type', ''),
                'description': output_node.properties.get('description', '')
            })
        
        return context
    
    def close(self):
        """Close connections and clean up resources"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
        logger.info("RAG Retriever closed") 