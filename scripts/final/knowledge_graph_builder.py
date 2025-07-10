"""
Knowledge Graph Builder for README Documentation Generator

This module builds a knowledge graph from artifact JSON files, extracting
nodes and relationships that can be used for RAG-based README generation.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from config import config, NODE_TYPES, RELATIONSHIP_TYPES, LANGUAGE_DEPENDENCY_PATTERNS, LICENSE_PATTERNS
from integrated_artifact_analyzer import BUILD_FILE_NAMES, BUILD_FILE_EXTENSIONS

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


def detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in ['.py', '.pyx', '.pyi', '.ipynb']:
        return 'python'
    elif ext in ['.java']:
        return 'java'
    elif ext in ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']:
        return 'javascript'
    elif ext in ['.c', '.cpp', '.h', '.hpp', '.cc']:
        return 'c_cpp'
    elif ext in ['.r', '.R']:
        return 'r'
    elif ext in ['.sh', '.bash', '.zsh', '.fish']:
        return 'bash'
    elif ext in ['.go']:
        return 'go'
    elif ext in ['.rb']:
        return 'ruby'
    elif ext in ['.php']:
        return 'php'
    else:
        return 'unknown'


def _infer_license_type(text: str) -> str:
    """Infer the license type from the text content of a license file"""
    for license_name, pattern in LICENSE_PATTERNS.items():
        if pattern.search(text):
            return license_name
    return "Unknown"


def _infer_build_system(file_path: str, content: str) -> str:
    """Infer the build system type from file name or content"""
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    content_lower = content.lower()

    # Match known build file names
    if file_name in BUILD_FILE_NAMES:
        if file_name == "Makefile":
            return "Make"
        elif file_name == "CMakeLists.txt":
            return "CMake"
        elif file_name in {"build.gradle", "settings.gradle"}:
            return "Gradle"
        elif file_name == "pom.xml":
            return "Maven"
        elif file_name == "package.json":
            return "Node.js (npm)"
        elif file_name == "setup.py":
            return "Python setuptools"
        elif file_name == "pyproject.toml":
            return "Python PEP 517 (poetry/flit)"
        elif file_name == "requirements.txt":
            return "Python requirements"
        elif file_name == "Pipfile":
            return "Python pipenv"
        elif file_name == "go.mod":
            return "Go Modules"
        elif file_name == "Cargo.toml":
            return "Rust Cargo"
        elif file_name == "Gemfile":
            return "Ruby Bundler"
        elif file_name == "composer.json":
            return "PHP Composer"
        elif file_name == "build.sbt":
            return "SBT (Scala)"
        elif file_name == "stack.yaml" or file_name == "cabal.project":
            return "Haskell Stack/Cabal"
        elif file_name == "pubspec.yaml":
            return "Flutter/Dart"
        elif file_name == "Package.swift":
            return "Swift Package Manager"
        elif file_name == "DESCRIPTION":
            return "R Package"
        elif file_name == "NAMESPACE":
            return "R Package Namespace"
        elif file_name == "build.xml":
            return "Apache Ant"

    # Match by known extensions
    if file_ext in BUILD_FILE_EXTENSIONS:
        if file_ext == ".sln":
            return "Visual Studio Solution"
        elif file_ext == ".csproj":
            return "C# .NET Project"

    # Fallback based on content (optional)
    if "cmake_minimum_required" in content_lower or "project(" in content_lower:
        return "CMake"
    if "[tool.poetry]" in content_lower:
        return "Python Poetry"

    return "Unknown"


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
                logger.info(
                    f"Connected to Neo4j database at {config.knowledge_graph.uri}, using database: {config.knowledge_graph.database}")
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
        self._link_sections_semantically(threshold=0.75)

        # Store graph in Neo4j or NetworkX
        if self.use_neo4j:
            self._store_in_neo4j()
        else:
            self._store_in_networkx()

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
            'repo_size_mb': artifact_data.get('repo_size_mb', 0),
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
            if '.md' in file_path.lower():
                self._extract_readme_sections(file_content, file_node_id)

    def _process_code_files(self, code_files: List[Dict], artifact_id: str):
        """Process all code files: store content, extract dependencies/commands"""
        for code_file in code_files:
            file_path = code_file.get('path', '')
            file_content = code_file.get('content', [])

            # Always store content and embedding
            file_node_id = self._create_file_node(file_path, file_content, 'code', artifact_id)

            language = detect_language(file_path)

            # Optional: Still extract shell commands for shell files
            if language == 'bash':
                self._extract_commands_from_shell(file_content, file_node_id)

            self._extract_generic_dependencies(file_content, file_node_id, language)

    def _process_docker_files(self, docker_files: List[Dict], artifact_id: str):
        """Process Docker files and extract dependencies"""
        for docker_file in docker_files:
            file_path = docker_file.get('path', '')
            file_content = docker_file.get('content', [])

            # Create file node
            file_node_id = self._create_file_node(file_path, file_content, 'docker', artifact_id)

            # Extract Docker dependencies
            self._extract_docker_dependencies(file_content, file_node_id)

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
        """Process license files and add License node"""
        for license_file in license_files:
            file_path = license_file.get("path", "")
            file_content = license_file.get("content", [])
            content_text = "\n".join(file_content)

            # Try to guess license type (MIT, GPL, etc.)
            license_type = _infer_license_type(content_text)

            license_node_id = self._generate_node_id(NODE_TYPES["LICENSE"], artifact_id)

            properties = {
                "name": license_type or "Unknown License",
                "path": file_path,
                "text_preview": content_text[:300],  # Truncated for graph metadata
                "full_text": content_text
            }

            license_node = Node(
                id=license_node_id,
                type=NODE_TYPES["LICENSE"],
                properties=properties
            )

            self.nodes[license_node.id] = license_node

            # Link license to artifact
            artifact_node_id = self._generate_node_id(NODE_TYPES["ARTIFACT"], artifact_id)
            self.relationships.append(Relationship(
                source_id=artifact_node_id,
                target_id=license_node_id,
                type=RELATIONSHIP_TYPES["DESCRIBES"]
            ))

    def _process_build_files(self, build_files: List[Dict], artifact_id: str):
        """Process build system configuration files and add Build node"""
        for build_file in build_files:
            file_path = build_file.get("path", "")
            file_content = build_file.get("content", [])
            content_text = "\n".join(file_content)

            # Step 1: Create a File node for the build file
            file_node_id = self._create_file_node(file_path, file_content, 'build', artifact_id)

            # Step 2: Create a Build metadata node
            build_type = _infer_build_system(file_path, content_text)
            build_node_id = self._generate_node_id(NODE_TYPES["BUILD"], file_path)

            properties = {
                "name": build_type or "Unknown Build System",
                "path": file_path,
                "text_preview": content_text[:2000],
                "full_text": content_text
            }

            build_node = Node(
                id=build_node_id,
                type=NODE_TYPES["BUILD"],
                properties=properties
            )

            self.nodes[build_node.id] = build_node

            # Link Build node to Artifact
            artifact_node_id = self._generate_node_id(NODE_TYPES["ARTIFACT"], artifact_id)
            self.relationships.append(Relationship(
                source_id=artifact_node_id,
                target_id=build_node_id,
                type=RELATIONSHIP_TYPES["CONTAINS"]
            ))

            # Optional: Link Build node to its file
            self.relationships.append(Relationship(
                source_id=build_node_id,
                target_id=file_node_id,
                type=RELATIONSHIP_TYPES["DESCRIBES"]
            ))

    def _process_tree_structure(self, tree_structure: List[str], artifact_id: str):
        """Process tree structure and create Structure node"""
        if not tree_structure:
            return

        # Create structure node
        structure_node_id = self._generate_node_id(NODE_TYPES['STRUCTURE'], 'tree_structure')

        tree_text = '\n'.join(tree_structure)
        properties = {
            'name': 'Directory Tree',
            'content': tree_text,
            'type': 'directory_structure',
            'description': f"Directory structure of artifact {artifact_id}",
            'embedding': self.embedding_model.encode(tree_text).tolist()
        }

        node = Node(
            id=structure_node_id,
            type=NODE_TYPES['STRUCTURE'],  # 👈 Use new node type
            properties=properties
        )

        self.nodes[node.id] = node

        # Create relationship from artifact to structure node
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
            'embedding': self.embedding_model.encode(content_text[:10000]).tolist(),
            'language': detect_language(file_path),
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

    def _extract_readme_sections(self, content: List[str], file_node_id: str):
        """Extract hierarchical sections from README content"""
        content_text = '\n'.join(content)
        matches = list(re.finditer(r'^(#+)\s+(.+)$', content_text, re.MULTILINE))

        section_stack = []
        last_index = 0

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.end()

            # Find the content range for this section
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content_text)
            section_content = content_text[start_pos:end_pos].strip()

            section_node_id = self._generate_node_id(NODE_TYPES['SECTION'], f"{file_node_id}_{title}")

            section_node = Node(
                id=section_node_id,
                type=NODE_TYPES['SECTION'],
                properties={
                    'title': title,
                    'type': 'readme_section',
                    'level': level,
                    'source_file': file_node_id,
                    'content': section_content,
                    'description': f"README section: {title}",
                    'embedding': self.embedding_model.encode(section_content[:10000]).tolist()
                }
            )
            self.nodes[section_node.id] = section_node

            # Relationship to file
            self.relationships.append(Relationship(
                source_id=file_node_id,
                target_id=section_node_id,
                type=RELATIONSHIP_TYPES['CONTAINS']
            ))

            # Determine parent section for hierarchy
            while section_stack and section_stack[-1]['level'] >= level:
                section_stack.pop()

            if section_stack:
                parent = section_stack[-1]
                self.relationships.append(Relationship(
                    source_id=parent['id'],
                    target_id=section_node_id,
                    type=RELATIONSHIP_TYPES['PART_OF']
                ))

            section_stack.append({'id': section_node_id, 'level': level})

    def _extract_commands_from_shell(self, content: List[str], file_node_id: str):
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

    def _extract_generic_dependencies(self, content: List[str], file_node_id: str, language: str):
        """Generic dependency extractor based on language-specific regex"""
        content_text = '\n'.join(content)
        patterns = LANGUAGE_DEPENDENCY_PATTERNS.get(language, [])

        found_deps = set()

        for pattern in patterns:
            matches = re.findall(pattern, content_text, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = next((m for m in match if m), None)
                if match:
                    dep = match.strip().split('.')[0]
                    if dep and len(dep) < 100:  # basic sanity
                        found_deps.add(dep)

        for dep in found_deps:
            dep_node_id = self._generate_node_id(NODE_TYPES['DEPENDENCY'], dep)

            properties = {
                'name': dep,
                'type': f'{language}_package',
                'description': f"{language.title()} dependency: {dep}",
                'embedding': self.embedding_model.encode(dep).tolist()
            }

            node = Node(
                id=dep_node_id,
                type=NODE_TYPES['DEPENDENCY'],
                properties=properties
            )

            self.nodes[dep_node_id] = node

            self.relationships.append(Relationship(
                source_id=file_node_id,
                target_id=dep_node_id,
                type=RELATIONSHIP_TYPES['DEPENDS_ON']
            ))

    def _extract_docker_dependencies(self, content: List[str], file_node_id: str):
        """Extract Docker dependencies from Dockerfile"""
        content_text = '\n'.join(content)

        # Find FROM statements
        from_statements = re.findall(r'^FROM\s+(\S+)', content_text, re.MULTILINE)

        # Find RUN statements with package installations
        re.findall(r'^RUN\s+(.+)', content_text, re.MULTILINE)

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

    def add_relationship(self, source_id: str, relationship_type: str, target_id: str):
        """
        Add a relationship to the graph (both in-memory NetworkX or Neo4j).
        """
        if self.use_neo4j:
            query = f"""
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            MERGE (a)-[r:{relationship_type}]->(b)
            """
            with self.driver.session() as session:
                session.run(query, source_id=source_id, target_id=target_id)
        else:
            self.nx_graph.add_edge(source_id, target_id, relationship_type=relationship_type)

    def _link_sections_semantically(self, threshold: float = 0.75):
        """Link README sections to relevant code/data/dependencies based on embedding similarity"""
        section_nodes = [n for n in self.nodes.values() if n.type == NODE_TYPES['SECTION']]
        candidate_types = [NODE_TYPES['FILE'], NODE_TYPES['DEPENDENCY'], NODE_TYPES['DATASET']]

        for section in section_nodes:
            section_embedding = np.array(section.properties.get("embedding", []))
            if section_embedding.size == 0:
                continue

            for target_node in self.nodes.values():
                if target_node.type not in candidate_types:
                    continue

                target_embedding = np.array(target_node.properties.get("embedding", []))
                if target_embedding.size == 0:
                    continue

                sim = cosine_similarity(
                    section_embedding.reshape(1, -1),
                    target_embedding.reshape(1, -1)
                )[0][0]

                if sim >= threshold:
                    if target_node.type == NODE_TYPES['FILE']:
                        rel_type = RELATIONSHIP_TYPES['DESCRIBES']
                    elif target_node.type == NODE_TYPES['DEPENDENCY']:
                        rel_type = RELATIONSHIP_TYPES['MENTIONS']
                    elif target_node.type == NODE_TYPES['DATASET']:
                        rel_type = RELATIONSHIP_TYPES['REFERENCES']
                    else:
                        continue

                    self.relationships.append(Relationship(
                        source_id=section.id,
                        target_id=target_node.id,
                        type=rel_type,
                        properties={"similarity": sim}
                    ))

    def _store_in_neo4j(self):
        """Store the graph in Neo4j database"""
        if not self.use_neo4j or not hasattr(self, 'driver'):
            return

        try:
            with self.driver.session(database=config.knowledge_graph.database) as session:
                # Clear existing data (optional - be careful in production)
                session.run("MATCH (n) DETACH DELETE n")

                # Create nodes
                for node in self.nodes.values():
                    properties = node.properties.copy()
                    properties['node_type'] = node.type  # ✅ Explicitly include node_type

                    if 'path' not in properties:
                        properties['path'] = None  # ✅ Ensure path exists (for uniformity)

                    query = f"""
                    CREATE (n:{node.type} {{
                        id: $id,
                        {', '.join([f'{k}: ${k}' for k in properties.keys()])}
                    }})
                    """
                    session.run(query, id=node.id, **properties)

                # Create relationships
                for rel in self.relationships:
                    query = f"""
                    MATCH (a {{id: $source_id}})
                    MATCH (b {{id: $target_id}})
                    CREATE (a)-[r:{rel.type}]->(b)
                    """
                    session.run(query, source_id=rel.source_id, target_id=rel.target_id)

                logger.info(
                    f"Stored {len(self.nodes)} nodes and {len(self.relationships)} relationships in Neo4j database: {config.knowledge_graph.database}")

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

    def enrich_section_links_with_similarity(self, threshold: float = 0.6):
        """
        Create links from Section nodes to other nodes (File, Dependency, Dataset)
        based on cosine similarity between their embeddings.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        section_nodes = [node for node in self.nodes.values() if node.type == NODE_TYPES['SECTION']]
        target_types = {
            NODE_TYPES['FILE']: RELATIONSHIP_TYPES['DESCRIBES'],
            NODE_TYPES['DEPENDENCY']: RELATIONSHIP_TYPES['MENTIONS'],
            NODE_TYPES['DATASET']: RELATIONSHIP_TYPES['REFERENCES']
        }


        for section in section_nodes:
            if 'embedding' not in section.properties:
                continue

            sec_embedding = np.array(section.properties['embedding']).reshape(1, -1)

            for target_node in self.nodes.values():
                if target_node.type not in target_types:
                    continue
                if 'embedding' not in target_node.properties:
                    continue

                tgt_embedding = np.array(target_node.properties['embedding']).reshape(1, -1)
                score = cosine_similarity(sec_embedding, tgt_embedding)[0][0]

                if score >= threshold:
                    rel_type = target_types[target_node.type]
                    self.add_relationship(section.id, rel_type, target_node.id)


