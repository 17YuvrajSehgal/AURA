"""
Knowledge Graph Builder for README Documentation Generator

This module builds a knowledge graph from artifact JSON files, extracting
nodes and relationships that can be used for RAG-based README generation.
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import re
import hashlib

from neo4j import GraphDatabase
import networkx as nx
from sentence_transformers import SentenceTransformer

from config import config, NODE_TYPES, RELATIONSHIP_TYPES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Node:
    """Represents a node in the knowledge graph"""
    id: str
    type: str
    properties: Dict[str, Any]
    
@dataclass
class Relationship:
    """Represents a relationship in the knowledge graph"""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class KnowledgeGraphBuilder:
    """Builds knowledge graphs from artifact JSON files"""
    
    def __init__(self, use_neo4j: bool = True):
        self.use_neo4j = use_neo4j
        self.nodes: Dict[str, Node] = {}
        self.relationships: List[Relationship] = []
        self.embedding_model = SentenceTransformer(config.vector.model_name)
        
        # Initialize Neo4j connection if requested
        if use_neo4j:
            try:
                self.driver = GraphDatabase.driver(
                    config.knowledge_graph.uri,
                    auth=(config.knowledge_graph.username, config.knowledge_graph.password)
                )
                logger.info("Connected to Neo4j database")
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")
                self.use_neo4j = False
        
        # Fallback to NetworkX for local processing
        if not self.use_neo4j:
            self.nx_graph = nx.DiGraph()
            logger.info("Using NetworkX for local graph processing")
    
    def _generate_node_id(self, node_type: str, identifier: str) -> str:
        """Generate a unique node ID"""
        combined = f"{node_type}:{identifier}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _extract_text_content(self, content: List[str]) -> str:
        """Extract and clean text content from file content arrays"""
        if not content:
            return ""
        
        # Join all lines and clean
        text = "\n".join(content)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def build_from_artifact_json(self, artifact_path: str) -> Dict[str, Any]:
        """
        Build knowledge graph from artifact JSON file
        
        Args:
            artifact_path: Path to the artifact JSON file
            
        Returns:
            Dictionary containing graph statistics and metadata
        """
        try:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                artifact_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load artifact JSON: {e}")
            return {}
        
        # Extract artifact metadata
        artifact_id = artifact_data.get('artifact_name', 'unknown')
        logger.info(f"Building knowledge graph for artifact: {artifact_id}")
        
        # Create main artifact node
        self._create_artifact_node(artifact_data)
        
        # Process different types of files
        self._process_documentation_files(artifact_data.get('documentation_files', []), artifact_id)
        self._process_code_files(artifact_data.get('code_files', []), artifact_id)
        self._process_docker_files(artifact_data.get('docker_files', []), artifact_id)
        self._process_data_files(artifact_data.get('data_files', []), artifact_id)
        self._process_license_files(artifact_data.get('license_files', []), artifact_id)
        self._process_build_files(artifact_data.get('build_files', []), artifact_id)
        
        # Extract tree structure
        self._process_tree_structure(artifact_data.get('tree_structure', []), artifact_id)
        
        # Store graph in Neo4j or NetworkX
        if self.use_neo4j:
            self._store_in_neo4j()
        else:
            self._store_in_networkx()
        
        # Generate embeddings for text content
        self._generate_embeddings()
        
        return {
            'artifact_id': artifact_id,
            'nodes_count': len(self.nodes),
            'relationships_count': len(self.relationships),
            'graph_type': 'Neo4j' if self.use_neo4j else 'NetworkX'
        }
    
    def _create_artifact_node(self, artifact_data: Dict[str, Any]):
        """Create the main artifact node"""
        artifact_id = artifact_data.get('artifact_name', 'unknown')
        
        properties = {
            'name': artifact_id,
            'path': artifact_data.get('artifact_path', ''),
            'extracted_path': artifact_data.get('extracted_path', ''),
            'extraction_method': artifact_data.get('extraction_method', ''),
            'success': artifact_data.get('success', False),
            'repo_size_mb': artifact_data.get('repo_size_mb', 0),
            'analysis_performed': artifact_data.get('analysis_performed', False)
        }
        
        # Add embedding of artifact description
        description = f"Artifact {artifact_id} extracted from {properties['path']}"
        properties['description'] = description
        properties['embedding'] = self.embedding_model.encode(description).tolist()
        
        node = Node(
            id=self._generate_node_id(NODE_TYPES['ARTIFACT'], artifact_id),
            type=NODE_TYPES['ARTIFACT'],
            properties=properties
        )
        
        self.nodes[node.id] = node
        logger.info(f"Created artifact node: {artifact_id}")
    
    def _process_documentation_files(self, doc_files: List[Dict], artifact_id: str):
        """Process documentation files and extract sections"""
        for doc_file in doc_files:
            file_path = doc_file.get('path', '')
            file_content = doc_file.get('content', [])
            
            # Create file node
            file_node_id = self._create_file_node(file_path, file_content, 'documentation', artifact_id)
            
            # Extract sections from README files
            if 'readme' in file_path.lower():
                self._extract_readme_sections(file_content, file_node_id, artifact_id)
    
    def _process_code_files(self, code_files: List[Dict], artifact_id: str):
        """Process code files and extract dependencies/commands"""
        for code_file in code_files:
            file_path = code_file.get('path', '')
            file_content = code_file.get('content', [])
            
            # Create file node
            file_node_id = self._create_file_node(file_path, file_content, 'code', artifact_id)
            
            # Extract commands from shell scripts
            if file_path.endswith('.sh'):
                self._extract_commands_from_shell(file_content, file_node_id, artifact_id)
            
            # Extract dependencies from Python files
            elif file_path.endswith('.py'):
                self._extract_python_dependencies(file_content, file_node_id, artifact_id)
    
    def _process_docker_files(self, docker_files: List[Dict], artifact_id: str):
        """Process Docker files and extract dependencies"""
        for docker_file in docker_files:
            file_path = docker_file.get('path', '')
            file_content = docker_file.get('content', [])
            
            # Create file node
            file_node_id = self._create_file_node(file_path, file_content, 'docker', artifact_id)
            
            # Extract Docker dependencies
            self._extract_docker_dependencies(file_content, file_node_id, artifact_id)
    
    def _process_data_files(self, data_files: List[Dict], artifact_id: str):
        """Process data files"""
        for data_file in data_files:
            file_name = data_file.get('name', '')
            file_path = data_file.get('path', '')
            
            # Create dataset node
            dataset_node_id = self._generate_node_id(NODE_TYPES['DATASET'], file_name)
            
            properties = {
                'name': file_name,
                'path': file_path,
                'mime_type': data_file.get('mime_type', ''),
                'size_kb': data_file.get('size_kb', 0),
                'description': f"Dataset file {file_name}",
                'embedding': self.embedding_model.encode(f"Dataset {file_name}").tolist()
            }
            
            node = Node(
                id=dataset_node_id,
                type=NODE_TYPES['DATASET'],
                properties=properties
            )
            
            self.nodes[node.id] = node
            
            # Create relationship with artifact
            artifact_node_id = self._generate_node_id(NODE_TYPES['ARTIFACT'], artifact_id)
            self.relationships.append(Relationship(
                source_id=artifact_node_id,
                target_id=dataset_node_id,
                type=RELATIONSHIP_TYPES['CONTAINS']
            ))
    
    def _process_license_files(self, license_files: List[Dict], artifact_id: str):
        """Process license files"""
        for license_file in license_files:
            file_path = license_file.get('path', '')
            file_content = license_file.get('content', [])
            
            # Create file node
            self._create_file_node(file_path, file_content, 'license', artifact_id)
    
    def _process_build_files(self, build_files: List[Dict], artifact_id: str):
        """Process build files"""
        for build_file in build_files:
            file_path = build_file.get('path', '')
            file_content = build_file.get('content', [])
            
            # Create file node
            self._create_file_node(file_path, file_content, 'build', artifact_id)
    
    def _process_tree_structure(self, tree_structure: List[str], artifact_id: str):
        """Process tree structure and create directory relationships"""
        if not tree_structure:
            return
        
        # Create structure node
        structure_node_id = self._generate_node_id(NODE_TYPES['SECTION'], 'tree_structure')
        
        tree_text = '\n'.join(tree_structure)
        properties = {
            'name': 'Tree Structure',
            'content': tree_text,
            'type': 'directory_structure',
            'description': f"Directory structure of artifact {artifact_id}",
            'embedding': self.embedding_model.encode(tree_text).tolist()
        }
        
        node = Node(
            id=structure_node_id,
            type=NODE_TYPES['SECTION'],
            properties=properties
        )
        
        self.nodes[node.id] = node
        
        # Create relationship with artifact
        artifact_node_id = self._generate_node_id(NODE_TYPES['ARTIFACT'], artifact_id)
        self.relationships.append(Relationship(
            source_id=artifact_node_id,
            target_id=structure_node_id,
            type=RELATIONSHIP_TYPES['CONTAINS']
        ))
    
    def _create_file_node(self, file_path: str, file_content: List[str], file_type: str, artifact_id: str) -> str:
        """Create a file node and return its ID"""
        file_name = Path(file_path).name
        file_node_id = self._generate_node_id(NODE_TYPES['FILE'], file_path)
        
        content_text = self._extract_text_content(file_content)
        
        properties = {
            'name': file_name,
            'path': file_path,
            'type': file_type,
            'content': content_text,
            'size': len(content_text),
            'description': f"{file_type.title()} file {file_name}",
            'embedding': self.embedding_model.encode(content_text[:1000]).tolist()  # Limit for embedding
        }
        
        node = Node(
            id=file_node_id,
            type=NODE_TYPES['FILE'],
            properties=properties
        )
        
        self.nodes[node.id] = node
        
        # Create relationship with artifact
        artifact_node_id = self._generate_node_id(NODE_TYPES['ARTIFACT'], artifact_id)
        self.relationships.append(Relationship(
            source_id=artifact_node_id,
            target_id=file_node_id,
            type=RELATIONSHIP_TYPES['CONTAINS']
        ))
        
        return file_node_id
    
    def _extract_readme_sections(self, content: List[str], file_node_id: str, artifact_id: str):
        """Extract sections from README content"""
        content_text = '\n'.join(content)
        
        # Find markdown headers
        sections = re.findall(r'^#+\s+(.+)$', content_text, re.MULTILINE)
        
        for section_title in sections:
            section_node_id = self._generate_node_id(NODE_TYPES['SECTION'], section_title)
            
            properties = {
                'title': section_title,
                'type': 'readme_section',
                'source_file': file_node_id,
                'description': f"README section: {section_title}",
                'embedding': self.embedding_model.encode(section_title).tolist()
            }
            
            node = Node(
                id=section_node_id,
                type=NODE_TYPES['SECTION'],
                properties=properties
            )
            
            self.nodes[node.id] = node
            
            # Create relationship with file
            self.relationships.append(Relationship(
                source_id=file_node_id,
                target_id=section_node_id,
                type=RELATIONSHIP_TYPES['CONTAINS']
            ))
    
    def _extract_commands_from_shell(self, content: List[str], file_node_id: str, artifact_id: str):
        """Extract commands from shell scripts"""
        content_text = '\n'.join(content)
        
        # Find docker commands, git commands, etc.
        docker_commands = re.findall(r'docker\s+[^\n]+', content_text)
        git_commands = re.findall(r'git\s+[^\n]+', content_text)
        
        all_commands = docker_commands + git_commands
        
        for i, command in enumerate(all_commands):
            command_node_id = self._generate_node_id(NODE_TYPES['COMMAND'], f"{file_node_id}_{i}")
            
            properties = {
                'command': command.strip(),
                'type': 'shell_command',
                'source_file': file_node_id,
                'description': f"Shell command: {command.strip()[:50]}...",
                'embedding': self.embedding_model.encode(command).tolist()
            }
            
            node = Node(
                id=command_node_id,
                type=NODE_TYPES['COMMAND'],
                properties=properties
            )
            
            self.nodes[node.id] = node
            
            # Create relationship with file
            self.relationships.append(Relationship(
                source_id=file_node_id,
                target_id=command_node_id,
                type=RELATIONSHIP_TYPES['CONTAINS']
            ))
    
    def _extract_python_dependencies(self, content: List[str], file_node_id: str, artifact_id: str):
        """Extract Python dependencies from import statements"""
        content_text = '\n'.join(content)
        
        # Find import statements
        imports = re.findall(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', content_text, re.MULTILINE)
        
        dependencies = set()
        for import_match in imports:
            dep = import_match[0] if import_match[0] else import_match[1]
            if dep and not dep.startswith('.'):  # Skip relative imports
                dependencies.add(dep.split('.')[0])  # Get root package
        
        for dep in dependencies:
            dep_node_id = self._generate_node_id(NODE_TYPES['DEPENDENCY'], dep)
            
            properties = {
                'name': dep,
                'type': 'python_package',
                'description': f"Python dependency: {dep}",
                'embedding': self.embedding_model.encode(f"Python package {dep}").tolist()
            }
            
            node = Node(
                id=dep_node_id,
                type=NODE_TYPES['DEPENDENCY'],
                properties=properties
            )
            
            self.nodes[node.id] = node
            
            # Create relationship with file
            self.relationships.append(Relationship(
                source_id=file_node_id,
                target_id=dep_node_id,
                type=RELATIONSHIP_TYPES['DEPENDS_ON']
            ))
    
    def _extract_docker_dependencies(self, content: List[str], file_node_id: str, artifact_id: str):
        """Extract Docker dependencies from Dockerfile"""
        content_text = '\n'.join(content)
        
        # Find FROM statements
        from_statements = re.findall(r'^FROM\s+(\S+)', content_text, re.MULTILINE)
        
        # Find RUN statements with package installations
        run_statements = re.findall(r'^RUN\s+(.+)', content_text, re.MULTILINE)
        
        # Process FROM statements
        for from_stmt in from_statements:
            dep_node_id = self._generate_node_id(NODE_TYPES['DEPENDENCY'], from_stmt)
            
            properties = {
                'name': from_stmt,
                'type': 'docker_base_image',
                'description': f"Docker base image: {from_stmt}",
                'embedding': self.embedding_model.encode(f"Docker image {from_stmt}").tolist()
            }
            
            node = Node(
                id=dep_node_id,
                type=NODE_TYPES['DEPENDENCY'],
                properties=properties
            )
            
            self.nodes[node.id] = node
            
            # Create relationship with file
            self.relationships.append(Relationship(
                source_id=file_node_id,
                target_id=dep_node_id,
                type=RELATIONSHIP_TYPES['DEPENDS_ON']
            ))
    
    def _store_in_neo4j(self):
        """Store the graph in Neo4j database"""
        if not self.use_neo4j or not hasattr(self, 'driver'):
            return
        
        try:
            with self.driver.session() as session:
                # Clear existing data (optional - be careful in production)
                session.run("MATCH (n) DETACH DELETE n")
                
                # Create nodes
                for node in self.nodes.values():
                    query = f"""
                    CREATE (n:{node.type} {{
                        id: $id,
                        {', '.join([f'{k}: ${k}' for k in node.properties.keys()])}
                    }})
                    """
                    session.run(query, id=node.id, **node.properties)
                
                # Create relationships
                for rel in self.relationships:
                    query = f"""
                    MATCH (a {{id: $source_id}})
                    MATCH (b {{id: $target_id}})
                    CREATE (a)-[r:{rel.type}]->(b)
                    """
                    session.run(query, source_id=rel.source_id, target_id=rel.target_id)
                
                logger.info(f"Stored {len(self.nodes)} nodes and {len(self.relationships)} relationships in Neo4j")
        
        except Exception as e:
            logger.error(f"Failed to store graph in Neo4j: {e}")
            raise
    
    def _store_in_networkx(self):
        """Store the graph in NetworkX format"""
        # Add nodes
        for node in self.nodes.values():
            self.nx_graph.add_node(node.id, **node.properties, node_type=node.type)
        
        # Add relationships
        for rel in self.relationships:
            self.nx_graph.add_edge(
                rel.source_id, 
                rel.target_id, 
                relationship_type=rel.type,
                **rel.properties
            )
        
        logger.info(f"Stored {len(self.nodes)} nodes and {len(self.relationships)} relationships in NetworkX")
    
    def _generate_embeddings(self):
        """Generate embeddings for all text content in the graph"""
        logger.info("Embeddings already generated during node creation")
        
    def close(self):
        """Close database connections"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        node_types = {}
        relationship_types = {}
        
        for node in self.nodes.values():
            node_types[node.type] = node_types.get(node.type, 0) + 1
        
        for rel in self.relationships:
            relationship_types[rel.type] = relationship_types.get(rel.type, 0) + 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_relationships': len(self.relationships),
            'node_types': node_types,
            'relationship_types': relationship_types
        } 