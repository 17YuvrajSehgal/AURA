"""
📐 Phase 2: Build the Knowledge Graph (Neo4j)
Goal: Build a symbolic structure of each artifact for GenAI and evaluators to query.

Features:
- Define comprehensive node types and relationships
- Load processed artifacts into Neo4j knowledge graph
- Create graph indexes for efficient querying
- Support both Neo4j and NetworkX backends
- Enable graph analytics and traversal
"""

import concurrent.futures
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from tqdm import tqdm

# Neo4j imports
try:
    from neo4j import GraphDatabase, basic_auth
    from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j drivers not available. Install py2neo and neo4j for full functionality.")

# NetworkX fallback
import networkx as nx

# Local imports
from config import config, NODE_TYPES, RELATIONSHIP_TYPES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    type: str
    properties: Dict[str, Any]
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph"""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class GraphStatistics:
    """Statistics about the knowledge graph"""
    total_nodes: int
    total_relationships: int
    node_types: Dict[str, int]
    relationship_types: Dict[str, int]
    artifacts_processed: int
    conferences: Set[str]
    tools: Set[str]
    graph_density: float
    avg_degree: float


class KnowledgeGraphBuilder:
    """Advanced Knowledge Graph Builder for artifact evaluation"""

    def __init__(self,
                 use_neo4j: bool = True,
                 clear_existing: bool = True):
        """
        Initialize the Knowledge Graph Builder
        
        Args:
            use_neo4j: Whether to use Neo4j (True) or NetworkX (False)
            clear_existing: Whether to clear existing graph data
        """
        logger.info("🚀 Initializing Knowledge Graph Builder...")
        logger.info(f"   🎯 Requested backend: {'Neo4j' if use_neo4j else 'NetworkX'}")
        logger.info(f"   🗑️ Clear existing data: {clear_existing}")

        if use_neo4j and not NEO4J_AVAILABLE:
            logger.warning("⚠️  Neo4j libraries not available, falling back to NetworkX")
            logger.warning("   Install py2neo and neo4j packages for Neo4j support")

        self.use_neo4j = use_neo4j and NEO4J_AVAILABLE
        self.nodes: Dict[str, GraphNode] = {}
        self.relationships: List[GraphRelationship] = []

        # Initialize graph backend
        if self.use_neo4j:
            self._initialize_neo4j(clear_existing)
        else:
            self._initialize_networkx()

        final_backend = 'Neo4j' if self.use_neo4j else 'NetworkX'
        logger.info(f"🎉 Knowledge Graph Builder initialized with {final_backend} backend")

    def _initialize_neo4j(self, clear_existing: bool = True):
        """Initialize Neo4j connection and setup"""
        logger.info("🔄 Attempting to connect to Neo4j...")
        logger.info(f"   📍 URI: {config.neo4j.uri}")
        logger.info(f"   👤 Username: {config.neo4j.username}")
        logger.info(f"   🗄️  Database: {config.neo4j.database}")

        try:
            logger.info("🔌 Initializing py2neo Graph connection...")
            # Initialize py2neo Graph
            self.graph = Graph(
                uri=config.neo4j.uri,
                user=config.neo4j.username,
                password=config.neo4j.password,
                name=config.neo4j.database
            )
            logger.info("✅ py2neo Graph connection established")

            logger.info("🔌 Initializing Neo4j driver connection...")
            # Initialize Neo4j driver for advanced operations
            self.driver = GraphDatabase.driver(
                config.neo4j.uri,
                auth=basic_auth(config.neo4j.username, config.neo4j.password)
            )
            logger.info("✅ Neo4j driver connection established")

            logger.info("🧪 Testing Neo4j connection...")
            # Test connection
            with self.driver.session(database=config.neo4j.database) as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                logger.info(f"✅ Neo4j connection test successful: {test_result['test']}")

            # Clear existing data if requested
            if clear_existing:
                logger.info("🗑️ Clearing existing Neo4j data...")
                self._clear_graph()
                logger.info("✅ Neo4j data cleared")

            # Create indexes and constraints
            logger.info("📊 Creating Neo4j schema (indexes and constraints)...")
            self._create_graph_schema()
            logger.info("✅ Neo4j schema created")

            # Initialize matchers
            self.node_matcher = NodeMatcher(self.graph)
            self.relationship_matcher = RelationshipMatcher(self.graph)
            logger.info("✅ Neo4j matchers initialized")

            logger.info(f"🎉 Successfully connected to Neo4j at {config.neo4j.uri}")
            logger.info(f"🗄️  Using database: {config.neo4j.database}")

        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error details: {str(e)}")
            logger.warning("⚠️  Falling back to NetworkX backend...")
            self.use_neo4j = False
            self._initialize_networkx()

    def _initialize_networkx(self):
        """Initialize NetworkX as fallback"""
        logger.info("🔄 Initializing NetworkX backend...")
        self.nx_graph = nx.MultiDiGraph()
        logger.info("✅ NetworkX MultiDiGraph created successfully")
        logger.info("📊 Using NetworkX for graph operations (fallback mode)")

    def _clear_graph(self):
        """Clear all data from the graph"""
        if self.use_neo4j:
            logger.info("🗑️ Clearing all nodes and relationships from Neo4j...")
            with self.driver.session(database=config.neo4j.database) as session:
                result = session.run("MATCH (n) DETACH DELETE n")
                logger.info("✅ Neo4j graph data cleared successfully")
        else:
            logger.info("🗑️ Clearing NetworkX graph...")
            self.nx_graph.clear()
            logger.info("✅ NetworkX graph data cleared successfully")

    def _create_graph_schema(self):
        """Create indexes and constraints for optimal performance"""
        if not self.use_neo4j:
            logger.info("📊 Skipping schema creation (using NetworkX)")
            return

        logger.info("📊 Creating Neo4j schema (constraints and indexes)...")

        schema_queries = [
            # Unique constraints
            "CREATE CONSTRAINT artifact_name IF NOT EXISTS FOR (a:Artifact) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",

            # Indexes for fast lookups
            "CREATE INDEX artifact_conference IF NOT EXISTS FOR (a:Artifact) ON (a.conference)",
            "CREATE INDEX artifact_year IF NOT EXISTS FOR (a:Artifact) ON (a.year)",
            "CREATE INDEX section_heading IF NOT EXISTS FOR (s:Section) ON (s.heading)",
            "CREATE INDEX tool_name IF NOT EXISTS FOR (t:Tool) ON (t.name)",
            "CREATE INDEX command_type IF NOT EXISTS FOR (c:Command) ON (c.type)",
            "CREATE INDEX conference_category IF NOT EXISTS FOR (conf:Conference) ON (conf.category)",

            # Full-text search indexes
            "CREATE FULLTEXT INDEX section_content IF NOT EXISTS FOR (s:Section) ON EACH [s.content]",
            "CREATE FULLTEXT INDEX documentation_content IF NOT EXISTS FOR (d:Documentation) ON EACH [d.content]"
        ]

        created_count = 0
        skipped_count = 0

        with self.driver.session(database=config.neo4j.database) as session:
            for query in schema_queries:
                try:
                    session.run(query)
                    logger.debug(f"✅ Created schema: {query[:50]}...")
                    created_count += 1
                except Exception as e:
                    logger.debug(f"⏭️  Schema creation skipped (may already exist): {str(e)[:50]}...")
                    skipped_count += 1

        logger.info(f"📊 Schema creation complete: {created_count} created, {skipped_count} skipped")

    def build_graph_from_processed_artifacts(self,
                                             processed_artifacts_dir: str,
                                             max_artifacts: Optional[int] = None,
                                             convert_format: bool = True) -> GraphStatistics:
        """
        Build knowledge graph from processed artifacts
        
        Args:
            processed_artifacts_dir: Directory containing artifact JSON files
            max_artifacts: Maximum number of artifacts to process
            convert_format: Whether to convert from analysis format to AURA format
            
        Returns:
            Graph statistics
        """
        artifacts_dir = Path(processed_artifacts_dir)

        if convert_format:
            # Look for analysis JSON files
            artifact_files = list(artifacts_dir.glob("*_analysis.json"))
            if not artifact_files:
                # Fallback to processed files
                artifact_files = list(artifacts_dir.glob("*_processed.json"))
        else:
            artifact_files = list(artifacts_dir.glob("*_processed.json"))

        if max_artifacts:
            artifact_files = artifact_files[:max_artifacts]

        logger.info(f"Building knowledge graph from {len(artifact_files)} artifacts")

        # Import the artifact processor here to avoid circular imports
        from artifact_utils import ArtifactJSONProcessor
        processor = ArtifactJSONProcessor() if convert_format else None

        # Process artifacts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.processing.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_artifact_file, file_path, processor): file_path
                for file_path in artifact_files
            }

            processed_count = 0
            for future in tqdm(concurrent.futures.as_completed(future_to_file),
                               total=len(artifact_files),
                               desc="Building graph"):
                try:
                    future.result()
                    processed_count += 1
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Failed to process {file_path}: {e}")

        logger.info(f"Successfully processed {processed_count}/{len(artifact_files)} artifacts")

        # Create cross-artifact relationships
        self._create_cross_artifact_relationships()

        # Calculate and return statistics
        return self._calculate_graph_statistics()

    def _process_artifact_file(self, artifact_file: Path, processor=None):
        """Process a single artifact file and add to graph"""
        try:
            with open(artifact_file, 'r', encoding='utf-8') as f:
                raw_artifact_data = json.load(f)

            # Convert format if processor is provided
            if processor:
                artifact_data = processor.convert_to_aura_format(raw_artifact_data)
            else:
                artifact_data = raw_artifact_data

            # Create artifact node
            artifact_node_id = self._create_artifact_node(artifact_data)

            # Create conference node
            conference_node_id = self._create_conference_node(artifact_data['metadata'])

            # Create relationship between artifact and conference
            self._create_relationship(
                artifact_node_id, conference_node_id,
                RELATIONSHIP_TYPES['USED_IN'], {}
            )

            # Process documentation files
            for doc_file in artifact_data['documentation_files']:
                doc_node_id = self._create_documentation_node(doc_file, artifact_node_id)

                # Process sections
                sections = doc_file.get('sections', [])
                if sections:
                    for section_data in sections:
                        section_node_id = self._create_section_node(section_data, doc_node_id, artifact_node_id)

                        # Handle both dict and dataclass section_data
                        if hasattr(section_data, '__dict__'):
                            # It's a dataclass, convert to dict
                            section_dict = section_data.__dict__
                        else:
                            # It's already a dict
                            section_dict = section_data

                        # Create tool nodes and relationships
                        tools = section_dict.get('tools', [])
                        for tool in tools:
                            tool_node_id = self._create_tool_node(tool)
                            self._create_relationship(
                                section_node_id, tool_node_id,
                                RELATIONSHIP_TYPES['MENTIONS'], {}
                            )

                        # Create command nodes and relationships
                        commands = section_dict.get('commands', [])
                        for command in commands:
                            heading = section_dict.get('heading', '')
                            command_node_id = self._create_command_node(command, heading)
                            self._create_relationship(
                                section_node_id, command_node_id,
                                RELATIONSHIP_TYPES['CONTAINS'], {}
                            )

                        # Create entity nodes and relationships
                        entities = section_dict.get('entities', [])
                        for entity in entities:
                            entity_node_id = self._create_entity_node(entity)
                            self._create_relationship(
                                section_node_id, entity_node_id,
                                RELATIONSHIP_TYPES['MENTIONS'], {}
                            )

            # Create tool nodes from artifact-level tools
            if 'tools' in artifact_data:
                for tool in artifact_data['tools']:
                    tool_node_id = self._create_tool_node(tool)
                    self._create_relationship(
                        artifact_node_id, tool_node_id,
                        RELATIONSHIP_TYPES['MENTIONS'], {}
                    )

            # Batch commit to graph
            self._commit_batch()

        except Exception as e:
            logger.error(f"Error processing artifact file {artifact_file}: {e}")
            raise

    def _create_artifact_node(self, artifact_data: Dict) -> str:
        """Create artifact node"""
        metadata = artifact_data['metadata']
        node_id = self._generate_node_id(NODE_TYPES['ARTIFACT'], metadata['artifact_name'])

        properties = {
            'name': metadata['artifact_name'],
            'repo_path': metadata['repo_path'],
            'repo_size_mb': metadata['repo_size_mb'],
            'conference': metadata['conference'],
            'year': metadata['year'],
            'extraction_method': metadata['extraction_method'],
            'success': metadata['success'],
            'analysis_performed': metadata['analysis_performed'],
            'total_files': metadata['total_files'],
            'code_files': metadata['code_files'],
            'doc_files': metadata['doc_files'],
            'data_files': metadata['data_files'],
            'has_docker': metadata['has_docker'],
            'has_requirements_txt': metadata['has_requirements_txt'],
            'has_setup_py': metadata['has_setup_py'],
            'has_makefile': metadata['has_makefile'],
            'has_jupyter': metadata['has_jupyter'],
            'has_license': metadata['has_license'],
            'license_type': metadata['license_type'],
            'repository_url': metadata['repository_url'],
            'doi': metadata['doi'],
            'processing_timestamp': artifact_data['processing_timestamp']
        }

        self._add_node(node_id, NODE_TYPES['ARTIFACT'], properties)
        return node_id

    def _create_conference_node(self, metadata: Dict) -> str:
        """Create or update conference node"""
        conference_name = metadata['conference']
        year = metadata['year']
        node_id = self._generate_node_id(NODE_TYPES['CONFERENCE'], f"{conference_name}_{year}")

        # Check if conference already exists
        if self._node_exists(node_id):
            return node_id

        # Determine conference category
        category = self._determine_conference_category(conference_name)

        properties = {
            'name': conference_name,
            'year': year,
            'category': category,
            'full_name': f"{conference_name} {year}"
        }

        # Add category-specific properties
        from config import CONFERENCE_CATEGORIES
        if category in CONFERENCE_CATEGORIES:
            cat_info = CONFERENCE_CATEGORIES[category]
            properties.update({
                'emphasis': cat_info['emphasis'],
                'required_tools': cat_info['required_tools'],
                'documentation_style': cat_info['documentation_style'],
                'avg_sections': cat_info['avg_sections'],
                'quality_threshold': cat_info['quality_threshold']
            })

        self._add_node(node_id, NODE_TYPES['CONFERENCE'], properties)
        return node_id

    def _create_documentation_node(self, doc_file: Dict, artifact_node_id: str) -> str:
        """Create documentation file node"""
        file_path = doc_file.get('path', 'unknown')
        node_id = self._generate_node_id(NODE_TYPES['DOCUMENTATION'], f"{artifact_node_id}_{file_path}")

        properties = {
            'path': file_path,
            'file_type': doc_file.get('file_type', 'unknown'),
            'total_length': doc_file.get('total_length', 0),
            'readability_score': doc_file.get('readability_score', 0.0),
            'sections_count': len(doc_file.get('sections', [])),
            'artifact_id': artifact_node_id
        }

        self._add_node(node_id, NODE_TYPES['DOCUMENTATION'], properties)

        # Create relationship with artifact
        self._create_relationship(
            artifact_node_id, node_id,
            RELATIONSHIP_TYPES['HAS_FILE'], {}
        )

        return node_id

    def _create_section_node(self, section_data: Dict, doc_node_id: str, artifact_node_id: str) -> str:
        """Create section node"""
        # Handle both dict and dataclass objects
        if hasattr(section_data, '__dict__'):
            section_dict = section_data.__dict__
        else:
            section_dict = section_data

        heading = section_dict.get('heading', 'Untitled Section')
        section_order = section_dict.get('section_order', 0)
        node_id = self._generate_node_id(NODE_TYPES['SECTION'], f"{doc_node_id}_{section_order}")

        content = section_dict.get('content', '')
        structural_features = section_dict.get('structural_features', {})

        properties = {
            'heading': heading,
            'content': content[:1000] if content else '',  # Truncate for storage
            'content_length': len(content),
            'doc_path': section_dict.get('doc_path', ''),
            'section_order': section_order,
            'level': section_dict.get('level', 1),
            'commands_count': len(section_dict.get('commands', [])),
            'tools_count': len(section_dict.get('tools', [])),
            'entities_count': len(section_dict.get('entities', [])),
            'images_count': len(section_dict.get('images', [])),
            'references_count': len(section_dict.get('references', [])),
            'bullet_points': structural_features.get('bullet_points', 0),
            'code_blocks': structural_features.get('code_blocks', 0),
            'artifact_id': artifact_node_id,
            'doc_id': doc_node_id
        }

        self._add_node(node_id, NODE_TYPES['SECTION'], properties)

        # Create relationship with documentation
        self._create_relationship(
            doc_node_id, node_id,
            RELATIONSHIP_TYPES['HAS_SECTION'], {}
        )

        return node_id

    def _create_tool_node(self, tool_name: str) -> str:
        """Create or get existing tool node"""
        node_id = self._generate_node_id(NODE_TYPES['TOOL'], tool_name)

        # Check if tool already exists
        if self._node_exists(node_id):
            return node_id

        properties = {
            'name': tool_name,
            'category': self._categorize_tool(tool_name),
            'usage_count': 1
        }

        self._add_node(node_id, NODE_TYPES['TOOL'], properties)
        return node_id

    def _create_command_node(self, command: str, section_heading: str) -> str:
        """Create command node"""
        # Generate shorter ID based on command content hash
        command_hash = hashlib.md5(command.encode()).hexdigest()[:8]
        node_id = self._generate_node_id(NODE_TYPES['COMMAND'], command_hash)

        properties = {
            'command': command,
            'type': self._categorize_command(command),
            'section_context': section_heading,
            'length': len(command)
        }

        self._add_node(node_id, NODE_TYPES['COMMAND'], properties)
        return node_id

    def _create_entity_node(self, entity_name: str) -> str:
        """Create or get existing entity node"""
        node_id = self._generate_node_id(NODE_TYPES['ENTITY'], entity_name)

        # Check if entity already exists
        if self._node_exists(node_id):
            return node_id

        properties = {
            'name': entity_name,
            'type': self._categorize_entity(entity_name),
            'usage_count': 1
        }

        self._add_node(node_id, NODE_TYPES['ENTITY'], properties)
        return node_id

    def _create_cross_artifact_relationships(self):
        """Create relationships between similar artifacts"""
        logger.info("Creating cross-artifact relationships...")

        if self.use_neo4j:
            self._create_neo4j_cross_relationships()
        else:
            self._create_networkx_cross_relationships()

    def _create_neo4j_cross_relationships(self):
        """Create cross-artifact relationships in Neo4j"""
        # Find artifacts with similar tools
        query = """
        MATCH (a1:Artifact)-[:MENTIONS]->(t:Tool)<-[:MENTIONS]-(a2:Artifact)
        WHERE a1.name < a2.name
        WITH a1, a2, count(t) as shared_tools
        WHERE shared_tools >= 2
        CREATE (a1)-[:SIMILAR_TO {shared_tools: shared_tools, type: 'tool_similarity'}]->(a2)
        """

        with self.driver.session() as session:
            session.run(query)

        # Find artifacts from same conference
        query = """
        MATCH (a1:Artifact)-[:USED_IN]->(c:Conference)<-[:USED_IN]-(a2:Artifact)
        WHERE a1.name < a2.name
        CREATE (a1)-[:SIMILAR_TO {type: 'conference_similarity', conference: c.name}]->(a2)
        """

        with self.driver.session() as session:
            session.run(query)

    def _create_networkx_cross_relationships(self):
        """Create cross-artifact relationships in NetworkX"""
        artifacts = [n for n in self.nx_graph.nodes() if
                     self.nx_graph.nodes[n].get('node_type') == NODE_TYPES['ARTIFACT']]

        # Create tool-based similarities
        for i, artifact1 in enumerate(artifacts):
            for artifact2 in artifacts[i + 1:]:
                # Find shared tools
                tools1 = set()
                tools2 = set()

                for neighbor in self.nx_graph.neighbors(artifact1):
                    if self.nx_graph.nodes[neighbor].get('node_type') == NODE_TYPES['TOOL']:
                        tools1.add(neighbor)

                for neighbor in self.nx_graph.neighbors(artifact2):
                    if self.nx_graph.nodes[neighbor].get('node_type') == NODE_TYPES['TOOL']:
                        tools2.add(neighbor)

                shared_tools = tools1.intersection(tools2)
                if len(shared_tools) >= 2:
                    self.nx_graph.add_edge(
                        artifact1, artifact2,
                        relationship_type=RELATIONSHIP_TYPES['SIMILAR_TO'],
                        shared_tools=len(shared_tools),
                        similarity_type='tool_similarity'
                    )

    def _generate_node_id(self, node_type: str, identifier: str) -> str:
        """Generate unique node ID"""
        combined = f"{node_type}:{identifier}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _add_node(self, node_id: str, node_type: str, properties: Dict):
        """Add node to graph"""
        graph_node = GraphNode(id=node_id, type=node_type, properties=properties)
        self.nodes[node_id] = graph_node

    def _create_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict):
        """Create relationship between nodes"""
        relationship = GraphRelationship(
            source_id=source_id,
            target_id=target_id,
            type=rel_type,
            properties=properties
        )
        self.relationships.append(relationship)

    def _node_exists(self, node_id: str) -> bool:
        """Check if node already exists"""
        return node_id in self.nodes

    def _commit_batch(self):
        """Commit current batch to graph database"""
        if not self.nodes and not self.relationships:
            return

        if self.use_neo4j:
            self._commit_to_neo4j()
        else:
            self._commit_to_networkx()

        # Clear batch (create new empty containers to avoid iteration issues)
        self.nodes = {}
        self.relationships = []

    def _commit_to_neo4j(self):
        """Commit batch to Neo4j"""
        if not self.nodes and not self.relationships:
            logger.debug("📭 No data to commit to Neo4j")
            return

        logger.debug(f"💾 Committing batch to Neo4j: {len(self.nodes)} nodes, {len(self.relationships)} relationships")

        try:
            nodes_created = 0
            # Create nodes
            for node in self.nodes.values():
                neo4j_node = Node(node.type, **node.properties, id=node.id)
                self.graph.merge(neo4j_node, node.type, "id")
                nodes_created += 1

            relationships_created = 0
            # Create relationships
            for rel in self.relationships:
                source_node = self.node_matcher.match(label=None, id=rel.source_id).first()
                target_node = self.node_matcher.match(label=None, id=rel.target_id).first()

                if source_node and target_node:
                    neo4j_rel = Relationship(source_node, rel.type, target_node, **rel.properties)
                    self.graph.create(neo4j_rel)
                    relationships_created += 1
                else:
                    logger.warning(f"⚠️  Skipping relationship: missing nodes for {rel.source_id} -> {rel.target_id}")

            logger.debug(f"✅ Neo4j batch committed: {nodes_created} nodes, {relationships_created} relationships")

        except Exception as e:
            logger.error(f"❌ Error committing to Neo4j: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            raise

    def _commit_to_networkx(self):
        """Commit batch to NetworkX"""
        # Add nodes
        for node in self.nodes.values():
            self.nx_graph.add_node(node.id, node_type=node.type, **node.properties)

        # Add relationships
        for rel in self.relationships:
            if rel.source_id in self.nodes and rel.target_id in self.nodes:
                self.nx_graph.add_edge(
                    rel.source_id, rel.target_id,
                    relationship_type=rel.type,
                    **rel.properties
                )

    def _determine_conference_category(self, conference_name: str) -> str:
        """Determine conference category"""
        from config import CONFERENCE_CATEGORIES
        conference_upper = conference_name.upper()

        for category, info in CONFERENCE_CATEGORIES.items():
            if conference_upper in [c.upper() for c in info['conferences']]:
                return category

        return 'other'

    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tool by type"""
        tool_categories = {
            'programming': ['python', 'java', 'javascript', 'c_cpp', 'r'],
            'container': ['docker'],
            'notebook': ['jupyter'],
            'database': ['sql'],
            'shell': ['shell']
        }

        for category, tools in tool_categories.items():
            if tool_name in tools:
                return category

        return 'other'

    def _categorize_command(self, command: str) -> str:
        """Categorize command by type"""
        command_lower = command.lower()

        if any(word in command_lower for word in ['pip install', 'conda install']):
            return 'installation'
        elif any(word in command_lower for word in ['python', 'java -', 'node']):
            return 'execution'
        elif any(word in command_lower for word in ['docker run', 'docker build']):
            return 'container'
        elif any(word in command_lower for word in ['make', 'cmake']):
            return 'build'
        else:
            return 'other'

    def _categorize_entity(self, entity_name: str) -> str:
        """Categorize entity by type"""
        return entity_name  # Entity name is already the category

    def _calculate_graph_statistics(self) -> GraphStatistics:
        """Calculate comprehensive graph statistics"""
        if self.use_neo4j:
            return self._calculate_neo4j_statistics()
        else:
            return self._calculate_networkx_statistics()

    def _calculate_neo4j_statistics(self) -> GraphStatistics:
        """Calculate statistics for Neo4j graph"""
        logger.debug("📊 Calculating Neo4j graph statistics...")

        with self.driver.session(database=config.neo4j.database) as session:
            # Total counts
            logger.debug("   🔢 Counting total nodes...")
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            logger.debug(f"   📊 Total nodes: {node_count}")

            logger.debug("   🔢 Counting total relationships...")
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            logger.debug(f"   🔗 Total relationships: {rel_count}")

            # Node types
            logger.debug("   📋 Analyzing node types...")
            node_types_result = session.run(
                "MATCH (n) RETURN labels(n)[0] as type, count(n) as count"
            )
            node_types = {record["type"]: record["count"] for record in node_types_result}
            logger.debug(f"   📊 Node types found: {list(node_types.keys())}")

            # Relationship types
            logger.debug("   🔗 Analyzing relationship types...")
            rel_types_result = session.run(
                "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
            )
            relationship_types = {record["type"]: record["count"] for record in rel_types_result}
            logger.debug(f"   🔗 Relationship types found: {list(relationship_types.keys())}")

            # Artifacts and conferences
            logger.debug("   🏛️  Counting artifacts...")
            artifacts_count = session.run("MATCH (a:Artifact) RETURN count(a) as count").single()["count"]

            logger.debug("   🎯 Finding conferences...")
            conferences_result = session.run("MATCH (c:Conference) RETURN c.name as name")
            conferences = {record["name"] for record in conferences_result if record["name"]}
            logger.debug(f"   🎯 Conferences found: {list(conferences)}")

            # Tools
            logger.debug("   🔧 Finding tools...")
            tools_result = session.run("MATCH (t:Tool) RETURN t.name as name")
            tools = {record["name"] for record in tools_result if record["name"]}
            logger.debug(f"   🔧 Tools found: {len(tools)} total")

            # Graph metrics
            density = rel_count / (node_count * (node_count - 1)) if node_count > 1 else 0
            avg_degree = (2 * rel_count) / node_count if node_count > 0 else 0
            logger.debug(f"   📈 Graph density: {density:.4f}")
            logger.debug(f"   📊 Average degree: {avg_degree:.2f}")

        logger.debug("✅ Neo4j statistics calculation complete")

        return GraphStatistics(
            total_nodes=node_count,
            total_relationships=rel_count,
            node_types=node_types,
            relationship_types=relationship_types,
            artifacts_processed=artifacts_count,
            conferences=conferences,
            tools=tools,
            graph_density=density,
            avg_degree=avg_degree
        )

    def _calculate_networkx_statistics(self) -> GraphStatistics:
        """Calculate statistics for NetworkX graph"""
        # Node type counts
        node_types = {}
        conferences = set()
        tools = set()
        artifacts_count = 0

        for node, data in self.nx_graph.nodes(data=True):
            node_type = data.get('node_type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

            if node_type == NODE_TYPES['ARTIFACT']:
                artifacts_count += 1
            elif node_type == NODE_TYPES['CONFERENCE']:
                conferences.add(data.get('name', ''))
            elif node_type == NODE_TYPES['TOOL']:
                tools.add(data.get('name', ''))

        # Relationship type counts
        relationship_types = {}
        for _, _, data in self.nx_graph.edges(data=True):
            rel_type = data.get('relationship_type', 'Unknown')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        # Graph metrics
        node_count = self.nx_graph.number_of_nodes()
        edge_count = self.nx_graph.number_of_edges()
        density = nx.density(self.nx_graph) if node_count > 0 else 0
        avg_degree = sum(dict(self.nx_graph.degree()).values()) / node_count if node_count > 0 else 0

        return GraphStatistics(
            total_nodes=node_count,
            total_relationships=edge_count,
            node_types=node_types,
            relationship_types=relationship_types,
            artifacts_processed=artifacts_count,
            conferences=conferences,
            tools=tools,
            graph_density=density,
            avg_degree=avg_degree
        )

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get comprehensive graph summary"""
        stats = self._calculate_graph_statistics()

        return {
            'graph_backend': 'Neo4j' if self.use_neo4j else 'NetworkX',
            'total_nodes': stats.total_nodes,
            'total_relationships': stats.total_relationships,
            'node_types': stats.node_types,
            'relationship_types': stats.relationship_types,
            'artifacts_processed': stats.artifacts_processed,
            'unique_conferences': len(stats.conferences),
            'unique_tools': len(stats.tools),
            'graph_density': stats.graph_density,
            'avg_degree': stats.avg_degree,
            'conferences': list(stats.conferences),
            'tools': list(stats.tools)
        }

    def close(self):
        """Close database connections"""
        logger.info("🔌 Closing Knowledge Graph Builder connections...")

        if self.use_neo4j:
            if hasattr(self, 'driver') and self.driver:
                logger.info("🔌 Closing Neo4j driver connection...")
                self.driver.close()
                logger.info("✅ Neo4j driver connection closed")

            if hasattr(self, 'graph') and self.graph:
                logger.info("✅ py2neo Graph connection released")
        else:
            logger.info("✅ NetworkX graph instance released")

        logger.info("🎉 Knowledge Graph Builder connections closed successfully")


def main():
    """Example usage of the Knowledge Graph Builder"""
    print("\n🚀 Starting Phase 2: Knowledge Graph Construction")
    print("=" * 60)

    # Initialize builder
    try:
        logger.info("🔧 Initializing Knowledge Graph Builder...")
        kg_builder = KnowledgeGraphBuilder(use_neo4j=True, clear_existing=True)
        backend_name = "Neo4j" if kg_builder.use_neo4j else "NetworkX"
        print(f"📊 Using {backend_name} backend")
        logger.info(f"✅ Knowledge Graph Builder initialized with {backend_name}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize graph builder: {e}")
        print(f"⚠️  Failed to initialize graph builder: {e}")
        return

    try:
        logger.info("🔄 Starting artifact processing...")
        # Build graph from processed artifacts
        stats = kg_builder.build_graph_from_processed_artifacts(
            "../../algo_outputs/algorithm_2_output_2",
            max_artifacts=5,  # Limit for testing
            convert_format=True  # Convert from analysis format
        )
        logger.info("✅ Artifact processing completed")

        logger.info("📊 Generating graph summary...")
        # Get summary
        summary = kg_builder.get_graph_summary()
        logger.info("✅ Graph summary generated")

        # Print results
        print("\n📐 Phase 2: Knowledge Graph Construction Results")
        print("=" * 60)
        print(f"🗄️  Backend: {summary['graph_backend']}")
        print(f"📊 Nodes: {summary['total_nodes']:,}")
        print(f"🔗 Relationships: {summary['total_relationships']:,}")
        print(f"🏛️  Artifacts: {summary['artifacts_processed']}")
        print(f"🎯 Conferences: {summary['unique_conferences']}")
        print(f"🔧 Tools: {summary['unique_tools']}")
        print(f"📈 Graph Density: {summary['graph_density']:.4f}")
        print(f"📊 Avg Degree: {summary['avg_degree']:.2f}")

        if summary['node_types']:
            print("\n📋 Node Types:")
            for node_type, count in summary['node_types'].items():
                print(f"  - {node_type}: {count:,}")

        if summary['relationship_types']:
            print("\n🔗 Relationship Types:")
            for rel_type, count in summary['relationship_types'].items():
                print(f"  - {rel_type}: {count:,}")

        if summary['conferences']:
            print("\n🎯 Conferences Found:")
            for conf in summary['conferences']:
                print(f"  - {conf}")

        if summary['tools']:
            print("\n🔧 Tools Found:")
            for tool in list(summary['tools'])[:10]:  # Show first 10 tools
                print(f"  - {tool}")
            if len(summary['tools']) > 10:
                print(f"  ... and {len(summary['tools']) - 10} more")

        print("\n✅ Knowledge Graph construction completed successfully!")
        logger.info("🎉 Phase 2 Knowledge Graph construction completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error during knowledge graph construction: {e}")
        print(f"❌ Error during knowledge graph construction: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close connection
        logger.info("🔌 Closing Knowledge Graph Builder connections...")
        kg_builder.close()
        logger.info("✅ Connections closed")


if __name__ == "__main__":
    main()
