"""
Unified Knowledge Graph Builder for Artifact Evaluation

This module builds a comprehensive knowledge graph from 500+ accepted artifacts
to discover patterns in documentation, structure, and tool usage that correlate
with artifact acceptance.

Key Features:
- Batch processing of multiple artifacts
- Conference-specific metadata integration
- Advanced relationship extraction
- Pattern discovery and clustering
- Graph Data Science integration
"""

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

import networkx as nx
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import config, NODE_TYPES, RELATIONSHIP_TYPES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact"""
    artifact_id: str
    conference: str
    year: int
    acceptance_status: str = "accepted"
    category: str = ""
    submission_type: str = ""
    size_mb: float = 0.0


@dataclass
class DocumentationPattern:
    """Represents a documentation pattern found in artifacts"""
    pattern_id: str
    pattern_type: str
    frequency: int
    conferences: List[str]
    description: str
    quality_score: float


class UnifiedKnowledgeGraphBuilder:
    """Builds unified knowledge graph from multiple accepted artifacts"""

    def __init__(self, use_neo4j: bool = True):
        self.use_neo4j = use_neo4j
        self.artifacts_processed = 0
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.patterns: List[DocumentationPattern] = []

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.vector.model_name)

        # Initialize Neo4j connection
        if use_neo4j:
            try:
                self.driver = GraphDatabase.driver(
                    config.knowledge_graph.uri,
                    auth=(config.knowledge_graph.username, config.knowledge_graph.password)
                )
                self._setup_neo4j_constraints()
                logger.info(f"Connected to Neo4j at {config.knowledge_graph.uri}")
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")
                self.use_neo4j = False

        # Fallback to NetworkX
        if not self.use_neo4j:
            self.nx_graph = nx.DiGraph()
            logger.info("Using NetworkX for local graph processing")

    def build_unified_graph(self, artifacts_directory: str,
                            conference_metadata: Optional[str] = None,
                            max_artifacts: Optional[int] = None) -> Dict[str, Any]:
        """
        Build unified knowledge graph from multiple artifacts
        
        Args:
            artifacts_directory: Directory containing artifact JSON files
            conference_metadata: Optional path to conference metadata file
            max_artifacts: Optional limit on number of artifacts to process
            
        Returns:
            Dictionary with graph statistics and analysis results
        """
        logger.info("Starting unified knowledge graph construction")

        # Load conference metadata if provided
        conference_data = {}
        if conference_metadata and Path(conference_metadata).exists():
            with open(conference_metadata, 'r') as f:
                conference_data = json.load(f)

        # Find all artifact JSON files
        artifacts_dir = Path(artifacts_directory)
        artifact_files = list(artifacts_dir.glob("**/*.json"))

        if max_artifacts:
            artifact_files = artifact_files[:max_artifacts]

        logger.info(f"Found {len(artifact_files)} artifact files to process")

        # Process artifacts in batches
        batch_size = config.data.batch_size
        results = {
            'total_artifacts': len(artifact_files),
            'processed_artifacts': 0,
            'failed_artifacts': 0,
            'unique_conferences': set(),
            'unique_tools': set(),
            'patterns_discovered': 0
        }

        # Process in parallel batches
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []

            for i in range(0, len(artifact_files), batch_size):
                batch = artifact_files[i:i + batch_size]
                future = executor.submit(self._process_artifact_batch, batch, conference_data)
                futures.append(future)

            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                try:
                    batch_result = future.result()
                    results['processed_artifacts'] += batch_result['processed']
                    results['failed_artifacts'] += batch_result['failed']
                    results['unique_conferences'].update(batch_result['conferences'])
                    results['unique_tools'].update(batch_result['tools'])

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    results['failed_artifacts'] += batch_size

        # Store graph in Neo4j or NetworkX
        if self.use_neo4j:
            self._store_unified_graph_neo4j()
        else:
            self._store_unified_graph_networkx()

        # Discover patterns
        self._discover_documentation_patterns()
        results['patterns_discovered'] = len(self.patterns)

        # Compute graph statistics
        results.update(self._compute_graph_statistics())

        logger.info(f"Unified graph construction completed: {results}")

        return results

    def _process_artifact_batch(self, artifact_files: List[Path],
                                conference_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of artifact files"""
        batch_result = {
            'processed': 0,
            'failed': 0,
            'conferences': set(),
            'tools': set()
        }

        for artifact_file in artifact_files:
            try:
                with open(artifact_file, 'r', encoding='utf-8') as f:
                    artifact_data = json.load(f)

                # Extract artifact metadata
                metadata = self._extract_artifact_metadata(artifact_data, conference_data)

                # Process the artifact
                self._process_single_artifact(artifact_data, metadata)

                batch_result['processed'] += 1
                batch_result['conferences'].add(metadata.conference)

                # Track tools used
                tools = self._extract_tools_from_artifact(artifact_data)
                batch_result['tools'].update(tools)

            except Exception as e:
                logger.error(f"Failed to process {artifact_file}: {e}")
                batch_result['failed'] += 1

        return batch_result

    def _extract_artifact_metadata(self, artifact_data: Dict[str, Any],
                                   conference_data: Dict[str, Any]) -> ArtifactMetadata:
        """Extract metadata from artifact data"""
        artifact_id = artifact_data.get('artifact_name', 'unknown')

        # Try to infer conference from various sources
        conference = "unknown"
        year = 2024

        # Check if conference is in the artifact path or data
        artifact_path = artifact_data.get('artifact_path', '')
        for conf_name in conference_data.keys():
            if conf_name.lower() in artifact_path.lower():
                conference = conf_name
                break

        # Check conference metadata
        if artifact_id in conference_data:
            conf_info = conference_data[artifact_id]
            conference = conf_info.get('conference', conference)
            year = conf_info.get('year', year)

        return ArtifactMetadata(
            artifact_id=artifact_id,
            conference=conference,
            year=year,
            size_mb=artifact_data.get('repo_size_mb', 0.0)
        )

    def _process_single_artifact(self, artifact_data: Dict[str, Any],
                                 metadata: ArtifactMetadata):
        """Process a single artifact and add to graph"""
        artifact_id = metadata.artifact_id

        # Create artifact node
        self._create_artifact_node(artifact_data, metadata)

        # Create conference node if not exists
        self._create_conference_node(metadata.conference, metadata.year)

        # Process documentation files
        self._process_documentation_files(
            artifact_data.get('documentation_files', []),
            artifact_id
        )

        # Process code files and extract patterns
        self._process_code_files(
            artifact_data.get('code_files', []),
            artifact_id
        )

        # Process build and dependency files
        self._process_build_files(
            artifact_data.get('build_files', []) + artifact_data.get('docker_files', []),
            artifact_id
        )

        # Extract quality indicators
        self._extract_quality_indicators(artifact_data, artifact_id)

        # Create artifact-conference relationship
        self._add_relationship(
            source_type=NODE_TYPES['ARTIFACT'],
            source_id=artifact_id,
            target_type=NODE_TYPES['CONFERENCE'],
            target_id=metadata.conference,
            relationship_type=RELATIONSHIP_TYPES['SUBMITTED_TO']
        )

    def _create_artifact_node(self, artifact_data: Dict[str, Any],
                              metadata: ArtifactMetadata):
        """Create an artifact node with comprehensive properties"""
        properties = {
            'artifact_id': metadata.artifact_id,
            'conference': metadata.conference,
            'year': metadata.year,
            'size_mb': metadata.size_mb,
            'acceptance_status': metadata.acceptance_status,
            'extraction_method': artifact_data.get('extraction_method', ''),
            'success': artifact_data.get('success', False),
            'analysis_performed': artifact_data.get('analysis_performed', False),
            'processed_timestamp': datetime.now().isoformat()
        }

        # Add artifact description for embedding
        description = f"Research artifact {metadata.artifact_id} from {metadata.conference} {metadata.year}"
        properties['description'] = description
        properties['embedding'] = self.embedding_model.encode(description).tolist()

        self._add_node(
            node_type=NODE_TYPES['ARTIFACT'],
            node_id=metadata.artifact_id,
            properties=properties
        )

    def _create_conference_node(self, conference: str, year: int):
        """Create or update conference node"""
        conference_id = f"{conference}_{year}"

        # Check if conference category exists
        category = "unknown"
        for cat_name, cat_info in config.conference.conference_categories.items():
            if conference in cat_info['conferences']:
                category = cat_name
                break

        properties = {
            'name': conference,
            'year': year,
            'category': category,
            'full_name': f"{conference} {year}",
            'description': f"{conference} conference in {year}",
        }

        if category != "unknown":
            cat_info = config.conference.conference_categories[category]
            properties.update({
                'emphasis': cat_info['emphasis'],
                'required_tools': cat_info['required_tools'],
                'documentation_style': cat_info['documentation_style']
            })

        properties['embedding'] = self.embedding_model.encode(properties['description']).tolist()

        self._add_node(
            node_type=NODE_TYPES['CONFERENCE'],
            node_id=conference_id,
            properties=properties
        )

    def _process_documentation_files(self, doc_files: List[Dict], artifact_id: str):
        """Process documentation files and extract sections"""
        for doc_file in doc_files:
            file_path = doc_file.get('path', '')
            file_content = doc_file.get('content', [])

            # Create documentation node
            doc_node_id = self._create_documentation_node(file_path, file_content, artifact_id)

            # Extract sections and quality metrics
            if 'readme' in file_path.lower():
                self._extract_readme_structure(file_content, doc_node_id, artifact_id)
            elif 'install' in file_path.lower():
                self._extract_installation_patterns(file_content, doc_node_id, artifact_id)

    def _create_documentation_node(self, file_path: str, file_content: List[str],
                                   artifact_id: str) -> str:
        """Create a documentation node"""
        doc_id = f"{artifact_id}_{Path(file_path).stem}"
        content_text = '\n'.join(file_content) if file_content else ""

        properties = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'content_length': len(content_text),
            'line_count': len(file_content) if file_content else 0,
            'artifact_id': artifact_id,
            'description': f"Documentation file {Path(file_path).name}",
            'embedding': self.embedding_model.encode(content_text[:2000]).tolist()
        }

        # Analyze content quality
        quality_metrics = self._analyze_documentation_quality(content_text)
        properties.update(quality_metrics)

        self._add_node(
            node_type=NODE_TYPES['DOCUMENTATION'],
            node_id=doc_id,
            properties=properties
        )

        # Create relationship with artifact
        self._add_relationship(
            source_type=NODE_TYPES['ARTIFACT'],
            source_id=artifact_id,
            target_type=NODE_TYPES['DOCUMENTATION'],
            target_id=doc_id,
            relationship_type=RELATIONSHIP_TYPES['HAS_DOCUMENTATION']
        )

        return doc_id

    def _analyze_documentation_quality(self, content: str) -> Dict[str, Any]:
        """Analyze documentation quality metrics"""
        metrics = {
            'has_headers': bool(re.search(r'^#+\s+', content, re.MULTILINE)),
            'has_code_blocks': bool(re.search(r'```', content)),
            'has_numbered_lists': bool(re.search(r'^\d+\.', content, re.MULTILINE)),
            'has_bullet_points': bool(re.search(r'^[\*\-]\s+', content, re.MULTILINE)),
            'has_links': bool(re.search(r'\[.*\]\(.*\)', content)),
            'has_images': bool(re.search(r'!\[.*\]\(.*\)', content)),
            'word_count': len(content.split()) if content else 0,
            'readability_score': self._calculate_readability_score(content)
        }

        # Calculate overall quality score
        quality_score = 0
        for indicator, value in metrics.items():
            if indicator in ['has_headers', 'has_code_blocks', 'has_numbered_lists']:
                quality_score += 2 if value else 0
            elif indicator in ['has_bullet_points', 'has_links']:
                quality_score += 1 if value else 0

        metrics['quality_score'] = min(quality_score, 10)  # Cap at 10

        return metrics

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate a simple readability score"""
        if not text:
            return 0.0

        words = text.split()
        sentences = re.split(r'[.!?]+', text)

        if len(sentences) == 0:
            return 0.0

        avg_words_per_sentence = len(words) / len(sentences)

        # Simple heuristic: ideal is 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            return 1.0
        elif avg_words_per_sentence < 15:
            return avg_words_per_sentence / 15
        else:
            return max(0.1, 1.0 - (avg_words_per_sentence - 20) / 30)

    def _extract_readme_structure(self, content: List[str], doc_node_id: str, artifact_id: str):
        """Extract README structure and create section nodes"""
        content_text = '\n'.join(content)

        # Find all headers
        headers = re.findall(r'^(#+)\s+(.+)$', content_text, re.MULTILINE)

        for level_markers, title in headers:
            level = len(level_markers)
            section_id = f"{doc_node_id}_{hashlib.md5(title.encode()).hexdigest()[:8]}"

            # Classify section type
            section_type = self._classify_section_type(title.lower())

            properties = {
                'title': title,
                'level': level,
                'section_type': section_type,
                'artifact_id': artifact_id,
                'description': f"README section: {title}",
                'embedding': self.embedding_model.encode(title).tolist()
            }

            self._add_node(
                node_type=NODE_TYPES['SECTION'],
                node_id=section_id,
                properties=properties
            )

            # Create relationship with documentation
            self._add_relationship(
                source_type=NODE_TYPES['DOCUMENTATION'],
                source_id=doc_node_id,
                target_type=NODE_TYPES['SECTION'],
                target_id=section_id,
                relationship_type=RELATIONSHIP_TYPES['HAS_SECTION']
            )

    def _classify_section_type(self, title: str) -> str:
        """Classify the type of a documentation section"""
        title_lower = title.lower()

        section_patterns = {
            'installation': ['install', 'setup', 'getting started', 'requirements'],
            'usage': ['usage', 'how to use', 'example', 'tutorial', 'quickstart'],
            'configuration': ['config', 'settings', 'environment', 'env'],
            'api': ['api', 'reference', 'documentation', 'functions'],
            'license': ['license', 'licensing', 'legal'],
            'contributing': ['contribut', 'development', 'build'],
            'troubleshooting': ['troubleshoot', 'faq', 'problem', 'issue'],
            'citation': ['cite', 'citation', 'reference', 'bibtex']
        }

        for section_type, patterns in section_patterns.items():
            if any(pattern in title_lower for pattern in patterns):
                return section_type

        return 'other'

    def _extract_tools_from_artifact(self, artifact_data: Dict[str, Any]) -> Set[str]:
        """Extract tools mentioned in the artifact"""
        tools = set()

        # Check docker files
        if artifact_data.get('docker_files'):
            tools.add('docker')

        # Check for specific file types
        all_files = (
                artifact_data.get('code_files', []) +
                artifact_data.get('build_files', []) +
                artifact_data.get('documentation_files', [])
        )

        for file_info in all_files:
            file_path = file_info.get('path', '').lower()

            if 'requirements.txt' in file_path or 'pip' in file_path:
                tools.add('pip')
            elif 'environment.yml' in file_path or 'conda' in file_path:
                tools.add('conda')
            elif 'package.json' in file_path:
                tools.add('npm')
            elif 'pom.xml' in file_path:
                tools.add('maven')
            elif 'build.gradle' in file_path:
                tools.add('gradle')
            elif 'makefile' in file_path:
                tools.add('make')
            elif 'cmake' in file_path:
                tools.add('cmake')

        return tools

    def _discover_documentation_patterns(self):
        """Discover common patterns in documentation across artifacts"""
        logger.info("Discovering documentation patterns")

        # Pattern 1: Common section sequences
        self._find_section_sequence_patterns()

        # Pattern 2: Tool usage patterns
        self._find_tool_usage_patterns()

        # Pattern 3: Quality indicator patterns
        self._find_quality_patterns()

        logger.info(f"Discovered {len(self.patterns)} documentation patterns")

    def _find_section_sequence_patterns(self):
        """Find common sequences of sections in documentation"""
        # This would analyze the graph to find common section orderings
        # For now, implementing a basic version

        common_sequences = [
            ["purpose", "installation", "usage"],
            ["setup", "requirements", "running"],
            ["introduction", "installation", "configuration", "usage"],
            ["overview", "getting started", "examples"]
        ]

        for i, sequence in enumerate(common_sequences):
            pattern = DocumentationPattern(
                pattern_id=f"section_sequence_{i}",
                pattern_type="section_sequence",
                frequency=0,  # Would be calculated from graph analysis
                conferences=[],
                description=f"Common section sequence: {' -> '.join(sequence)}",
                quality_score=0.8
            )
            self.patterns.append(pattern)

    def _find_tool_usage_patterns(self):
        """Find patterns in tool usage across conferences"""
        # This would analyze tool co-occurrence patterns
        pass

    def _find_quality_patterns(self):
        """Find patterns that correlate with high-quality documentation"""
        # This would analyze quality indicators across accepted artifacts
        pass

    def _add_node(self, node_type: str, node_id: str, properties: Dict[str, Any]):
        """Add a node to the graph"""
        node_key = f"{node_type}:{node_id}"
        if node_key not in self.nodes:
            self.nodes[node_key] = {
                'id': node_id,
                'type': node_type,
                'properties': properties
            }

    def _add_relationship(self, source_type: str, source_id: str,
                          target_type: str, target_id: str,
                          relationship_type: str, properties: Dict[str, Any] = None):
        """Add a relationship to the graph"""
        relationship = {
            'source_key': f"{source_type}:{source_id}",
            'target_key': f"{target_type}:{target_id}",
            'type': relationship_type,
            'properties': properties or {}
        }
        self.relationships.append(relationship)

    def _setup_neo4j_constraints(self):
        """Set up Neo4j constraints and indexes"""
        if not self.use_neo4j:
            return

        constraints = [
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (a:{NODE_TYPES['ARTIFACT']}) REQUIRE a.artifact_id IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (c:{NODE_TYPES['CONFERENCE']}) REQUIRE c.name IS UNIQUE",
            f"CREATE INDEX IF NOT EXISTS FOR (d:{NODE_TYPES['DOCUMENTATION']}) ON d.quality_score",
            f"CREATE INDEX IF NOT EXISTS FOR (s:{NODE_TYPES['SECTION']}) ON s.section_type"
        ]

        try:
            with self.driver.session(database=config.knowledge_graph.database) as session:
                for constraint in constraints:
                    session.run(constraint)
            logger.info("Neo4j constraints and indexes created")
        except Exception as e:
            logger.warning(f"Failed to create constraints: {e}")

    def _store_unified_graph_neo4j(self):
        """Store the unified graph in Neo4j"""
        if not self.use_neo4j:
            return

        logger.info("Storing unified graph in Neo4j")

        try:
            with self.driver.session(database=config.knowledge_graph.database) as session:
                # Create nodes
                for node_key, node_data in tqdm(self.nodes.items(), desc="Creating nodes"):
                    node_type = node_data['type']
                    properties = node_data['properties']

                    # Build property string for Cypher query
                    prop_string = ', '.join([f"{k}: ${k}" for k in properties.keys()])

                    query = f"""
                    MERGE (n:{node_type} {{id: $id}})
                    SET n += {{{prop_string}}}
                    """

                    session.run(query, id=node_data['id'], **properties)

                # Create relationships
                for rel in tqdm(self.relationships, desc="Creating relationships"):
                    query = f"""
                    MATCH (a) WHERE a.id = $source_id
                    MATCH (b) WHERE b.id = $target_id
                    MERGE (a)-[r:{rel['type']}]->(b)
                    SET r += $properties
                    """

                    session.run(query,
                                source_id=rel['source_key'].split(':')[1],
                                target_id=rel['target_key'].split(':')[1],
                                properties=rel['properties'])

            logger.info(f"Stored {len(self.nodes)} nodes and {len(self.relationships)} relationships in Neo4j")

        except Exception as e:
            logger.error(f"Failed to store graph in Neo4j: {e}")
            raise

    def _store_unified_graph_networkx(self):
        """Store the unified graph in NetworkX"""
        logger.info("Storing unified graph in NetworkX")

        # Add nodes
        for node_key, node_data in self.nodes.items():
            self.nx_graph.add_node(node_data['id'],
                                   node_type=node_data['type'],
                                   **node_data['properties'])

        # Add relationships
        for rel in self.relationships:
            source_id = rel['source_key'].split(':')[1]
            target_id = rel['target_key'].split(':')[1]

            self.nx_graph.add_edge(source_id, target_id,
                                   relationship_type=rel['type'],
                                   **rel['properties'])

        logger.info(f"Stored {len(self.nodes)} nodes and {len(self.relationships)} relationships in NetworkX")

    def _compute_graph_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive graph statistics"""
        stats = {
            'total_nodes': len(self.nodes),
            'total_relationships': len(self.relationships),
            'node_types': {},
            'relationship_types': {},
            'patterns_discovered': len(self.patterns)
        }

        # Count node types
        for node_data in self.nodes.values():
            node_type = node_data['type']
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1

        # Count relationship types
        for rel in self.relationships:
            rel_type = rel['type']
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1

        return stats

    def get_graph_for_analysis(self):
        """Get the graph for further analysis"""
        if self.use_neo4j:
            return self.driver
        else:
            return self.nx_graph

    def close(self):
        """Close database connections"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()

    # Additional utility methods would go here...
    def _process_code_files(self, code_files: List[Dict], artifact_id: str):
        """Process code files - placeholder for now"""
        pass

    def _process_build_files(self, build_files: List[Dict], artifact_id: str):
        """Process build files - placeholder for now"""
        pass

    def _extract_quality_indicators(self, artifact_data: Dict[str, Any], artifact_id: str):
        """Extract quality indicators - placeholder for now"""
        pass

    def _extract_installation_patterns(self, file_content: List[str], doc_node_id: str, artifact_id: str):
        """Extract installation patterns - placeholder for now"""
        pass
