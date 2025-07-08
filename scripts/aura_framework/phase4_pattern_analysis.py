"""
📊 Phase 4: Analyze Accepted Artifacts
Goal: Discover patterns in accepted artifacts.

Features:
- In-degree analysis on Section nodes → Find most common sections
- PageRank / Centrality → Identify impactful sections  
- Clustering (Node2Vec) → Group artifacts by structure
- Outlier Detection → Flag missing key sections
- Pattern Mining → Discover success patterns
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
# Graph analysis imports
import networkx as nx
import numpy as np

try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Machine learning imports
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from node2vec import Node2Vec

    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False

# Local imports
from config import NODE_TYPES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    """Represents a discovered pattern in artifacts"""
    pattern_type: str
    pattern_id: str
    description: str
    frequency: int
    confidence: float
    examples: List[str]
    artifacts_with_pattern: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionPattern:
    """Represents a common section pattern"""
    heading: str
    frequency: int
    avg_position: float
    common_content_features: Dict[str, int]
    typical_length: int
    associated_tools: List[str]
    success_correlation: float


@dataclass
class StructuralPattern:
    """Represents a structural pattern in artifacts"""
    pattern_name: str
    section_sequence: List[str]
    frequency: int
    success_rate: float
    avg_artifact_size: float
    common_tools: List[str]


@dataclass
class AnalysisResults:
    """Comprehensive analysis results"""
    section_patterns: List[SectionPattern]
    structural_patterns: List[StructuralPattern]
    outliers: List[str]  # artifact IDs
    clusters: Dict[str, List[str]]  # cluster_id -> artifact_ids
    success_factors: Dict[str, float]
    recommendations: List[str]


class PatternAnalysisEngine:
    """Advanced pattern analysis engine for accepted artifacts"""

    def __init__(self,
                 knowledge_graph_builder,
                 vector_embedding_engine=None,
                 use_neo4j: bool = True):
        """
        Initialize the Pattern Analysis Engine
        
        Args:
            knowledge_graph_builder: Knowledge graph builder instance
            vector_embedding_engine: Vector embedding engine instance
            use_neo4j: Whether to use Neo4j for graph analysis
        """
        self.kg_builder = knowledge_graph_builder
        self.vector_engine = vector_embedding_engine
        self.use_neo4j = use_neo4j and NEO4J_AVAILABLE

        # Analysis results
        self.section_patterns: List[SectionPattern] = []
        self.structural_patterns: List[StructuralPattern] = []
        self.artifact_clusters: Dict[str, List[str]] = {}
        self.outliers: List[str] = []

        # Graph analytics results
        self.centrality_scores: Dict[str, float] = {}
        self.pagerank_scores: Dict[str, float] = {}
        self.community_assignments: Dict[str, int] = {}

        logger.info("Pattern Analysis Engine initialized")

    def analyze_accepted_artifacts(self,
                                   min_frequency: int = 3,
                                   outlier_threshold: float = 2.0) -> AnalysisResults:
        """
        Perform comprehensive analysis of accepted artifacts
        
        Args:
            min_frequency: Minimum frequency for pattern inclusion
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Comprehensive analysis results
        """
        logger.info("Starting comprehensive analysis of accepted artifacts")

        # 1. Section frequency analysis
        logger.info("Analyzing section patterns...")
        self.section_patterns = self._analyze_section_patterns(min_frequency)

        # 2. Structural pattern discovery
        logger.info("Discovering structural patterns...")
        self.structural_patterns = self._discover_structural_patterns(min_frequency)

        # 3. Graph centrality analysis
        logger.info("Computing graph centrality metrics...")
        self._compute_centrality_metrics()

        # 4. Community detection
        logger.info("Performing community detection...")
        self._perform_community_detection()

        # 5. Artifact clustering
        logger.info("Clustering artifacts by features...")
        self.artifact_clusters = self._cluster_artifacts()

        # 6. Outlier detection
        logger.info("Detecting outlier artifacts...")
        self.outliers = self._detect_outliers(outlier_threshold)

        # 7. Success factor analysis
        logger.info("Analyzing success factors...")
        success_factors = self._analyze_success_factors()

        # 8. Generate recommendations
        recommendations = self._generate_recommendations()

        # Create comprehensive results
        results = AnalysisResults(
            section_patterns=self.section_patterns,
            structural_patterns=self.structural_patterns,
            outliers=self.outliers,
            clusters=self.artifact_clusters,
            success_factors=success_factors,
            recommendations=recommendations
        )

        logger.info("Pattern analysis completed successfully")
        return results

    def _analyze_section_patterns(self, min_frequency: int) -> List[SectionPattern]:
        """Analyze common section patterns across artifacts"""
        if self.use_neo4j:
            return self._analyze_section_patterns_neo4j(min_frequency)
        else:
            return self._analyze_section_patterns_networkx(min_frequency)

    def _analyze_section_patterns_neo4j(self, min_frequency: int) -> List[SectionPattern]:
        """Analyze section patterns using Neo4j queries"""
        patterns = []

        with self.kg_builder.driver.session() as session:
            # Get section frequency and characteristics - using COALESCE for missing properties
            query = """
            MATCH (s:Section)
            WHERE s.heading IS NOT NULL
            WITH s.heading as heading,
                 avg(coalesce(s.position, s.section_order, 0)) as avg_position,
                 avg(coalesce(s.content_length, size(coalesce(s.content, '')), 0)) as avg_length,
                 avg(coalesce(s.bullet_points, 0)) as avg_bullets,
                 avg(coalesce(s.code_blocks, 0)) as avg_code_blocks,
                 collect(coalesce(s.artifact_id, s.id))[0..10] as sample_artifacts,
                 count(s) as frequency
            ORDER BY frequency DESC
            LIMIT 50
            RETURN heading, frequency, avg_position, avg_length, avg_bullets, avg_code_blocks, sample_artifacts
            """

            results = session.run(query)

            for record in results:
                if record["frequency"] >= min_frequency:
                    # Get associated tools for this section type (multiple relationship attempts)
                    tools_query = """
                    MATCH (s:Section {heading: $heading})
                    OPTIONAL MATCH (s)-[:USES_TOOL]->(t:Tool)
                    OPTIONAL MATCH (s)-[:MENTIONS]->(t2:Tool)
                    WITH collect(DISTINCT t.name) + collect(DISTINCT t2.name) as all_tools
                    UNWIND all_tools as tool_name
                    WITH tool_name WHERE tool_name IS NOT NULL
                    RETURN tool_name, count(*) as tool_frequency
                    ORDER BY tool_frequency DESC
                    LIMIT 5
                    """

                    try:
                        tools_result = session.run(tools_query, heading=record["heading"])
                        associated_tools = [r["tool_name"] for r in tools_result]
                    except:
                        associated_tools = []

                    pattern = SectionPattern(
                        heading=record["heading"],
                        frequency=record["frequency"],
                        avg_position=record["avg_position"] or 0,
                        common_content_features={
                            'avg_bullet_points': record["avg_bullets"] or 0,
                            'avg_code_blocks': record["avg_code_blocks"] or 0
                        },
                        typical_length=int(record["avg_length"] or 0),
                        associated_tools=associated_tools,
                        success_correlation=0.8  # Placeholder - would calculate from acceptance data
                    )

                    patterns.append(pattern)

        return patterns

    def _analyze_section_patterns_networkx(self, min_frequency: int) -> List[SectionPattern]:
        """Analyze section patterns using NetworkX"""
        patterns = []

        if not hasattr(self.kg_builder, 'nx_graph'):
            return patterns

        G = self.kg_builder.nx_graph

        # Collect section data
        section_data = defaultdict(list)

        for node, data in G.nodes(data=True):
            if data.get('node_type') == NODE_TYPES['SECTION']:
                heading = data.get('heading', 'Unknown')
                section_data[heading].append({
                    'position': data.get('section_order', 0),
                    'length': data.get('content_length', 0),
                    'bullets': data.get('bullet_points', 0),
                    'code_blocks': data.get('code_blocks', 0),
                    'artifact_id': data.get('artifact_id', '')
                })

        # Analyze patterns
        for heading, sections in section_data.items():
            if len(sections) >= min_frequency:
                # Calculate averages
                avg_position = np.mean([s['position'] for s in sections])
                avg_length = np.mean([s['length'] for s in sections])
                avg_bullets = np.mean([s['bullets'] for s in sections])
                avg_code = np.mean([s['code_blocks'] for s in sections])

                # Find associated tools (simplified)
                associated_tools = self._find_tools_for_section_type(G, heading)

                pattern = SectionPattern(
                    heading=heading,
                    frequency=len(sections),
                    avg_position=avg_position,
                    common_content_features={
                        'avg_bullet_points': avg_bullets,
                        'avg_code_blocks': avg_code
                    },
                    typical_length=int(avg_length),
                    associated_tools=associated_tools[:5],  # Top 5
                    success_correlation=0.8  # Placeholder
                )

                patterns.append(pattern)

        # Sort by frequency
        patterns.sort(key=lambda x: x.frequency, reverse=True)
        return patterns

    def _discover_structural_patterns(self, min_frequency: int) -> List[StructuralPattern]:
        """Discover common structural patterns in artifacts"""
        if self.use_neo4j:
            return self._discover_structural_patterns_neo4j(min_frequency)
        else:
            return self._discover_structural_patterns_networkx(min_frequency)

    def _discover_structural_patterns_neo4j(self, min_frequency: int) -> List[StructuralPattern]:
        """Discover structural patterns using Neo4j"""
        patterns = []

        with self.kg_builder.driver.session() as session:
            # Get artifact structures (sequence of sections) - simplified query for compatibility
            query = """
            MATCH (a:Artifact)-[:HAS_SECTION]->(s:Section)
            WITH a, collect({heading: s.heading, position: coalesce(s.position, s.section_order, 0)}) as sections,
                 coalesce(a.total_files, 0) as file_count
            WITH a, [section in sections | section.heading] as section_sequence, file_count
            WHERE size(section_sequence) >= 2
            WITH section_sequence, count(*) as frequency, 
                 avg(file_count) as avg_size,
                 collect(a.name)[0..5] as sample_artifacts
            WHERE frequency >= $min_frequency
            RETURN section_sequence, frequency, avg_size, sample_artifacts
            ORDER BY frequency DESC
            LIMIT 20
            """

            results = session.run(query, min_frequency=min_frequency)

            for record in results:
                section_sequence = record["section_sequence"]

                # Get common tools for this pattern (simplified query)
                tools_query = """
                MATCH (a:Artifact)-[:USES_TOOL]->(t:Tool)
                WHERE a.name IN $artifacts
                RETURN t.name as tool_name, count(*) as tool_frequency
                ORDER BY tool_frequency DESC
                LIMIT 5
                """

                try:
                    tools_result = session.run(tools_query, artifacts=record["sample_artifacts"])
                    common_tools = [r["tool_name"] for r in tools_result]
                except:
                    # Fallback if USES_TOOL relationship doesn't exist
                    common_tools = []

                pattern = StructuralPattern(
                    pattern_name=f"Pattern_{len(patterns) + 1}",
                    section_sequence=section_sequence,
                    frequency=record["frequency"],
                    success_rate=0.85,  # Placeholder - would calculate from acceptance data
                    avg_artifact_size=record["avg_size"] or 0,
                    common_tools=common_tools
                )

                patterns.append(pattern)

        return patterns

    def _discover_structural_patterns_networkx(self, min_frequency: int) -> List[StructuralPattern]:
        """Discover structural patterns using NetworkX"""
        patterns = []

        if not hasattr(self.kg_builder, 'nx_graph'):
            return patterns

        G = self.kg_builder.nx_graph

        # Collect artifact structures
        artifact_structures = {}

        for node, data in G.nodes(data=True):
            if data.get('node_type') == NODE_TYPES['ARTIFACT']:
                artifact_id = data.get('name', node)

                # Get sections for this artifact
                sections = []
                for neighbor in G.neighbors(node):
                    neighbor_data = G.nodes[neighbor]
                    if neighbor_data.get('node_type') == NODE_TYPES['SECTION']:
                        sections.append({
                            'heading': neighbor_data.get('heading', ''),
                            'order': neighbor_data.get('section_order', 0)
                        })

                # Sort by order and create sequence
                sections.sort(key=lambda x: x['order'])
                section_sequence = [s['heading'] for s in sections]

                if len(section_sequence) >= 3:
                    artifact_structures[artifact_id] = {
                        'sequence': section_sequence,
                        'size': data.get('total_files', 0)
                    }

        # Find common patterns
        sequence_counter = Counter()
        for artifact_id, structure in artifact_structures.items():
            sequence_key = tuple(structure['sequence'])
            sequence_counter[sequence_key] += 1

        # Create pattern objects
        for sequence, frequency in sequence_counter.items():
            if frequency >= min_frequency:
                # Find artifacts with this pattern
                matching_artifacts = [
                    aid for aid, struct in artifact_structures.items()
                    if tuple(struct['sequence']) == sequence
                ]

                # Calculate average size
                avg_size = np.mean([
                    artifact_structures[aid]['size']
                    for aid in matching_artifacts
                ])

                # Find common tools (simplified)
                common_tools = self._find_tools_for_artifacts(G, matching_artifacts)

                pattern = StructuralPattern(
                    pattern_name=f"Structure_{len(patterns) + 1}",
                    section_sequence=list(sequence),
                    frequency=frequency,
                    success_rate=0.85,  # Placeholder
                    avg_artifact_size=avg_size,
                    common_tools=common_tools[:5]
                )

                patterns.append(pattern)

        # Sort by frequency
        patterns.sort(key=lambda x: x.frequency, reverse=True)
        return patterns

    def _compute_centrality_metrics(self):
        """Compute various centrality metrics for graph analysis"""
        if self.use_neo4j:
            self._compute_centrality_neo4j()
        else:
            self._compute_centrality_networkx()

    def _compute_centrality_neo4j(self):
        """Compute centrality metrics using Neo4j Graph Data Science"""
        with self.kg_builder.driver.session() as session:
            # Create graph projection
            try:
                session.run("""
                CALL gds.graph.project(
                    'artifact_graph',
                    ['Artifact', 'Section', 'Tool'],
                    ['HAS_SECTION', 'MENTIONS', 'CONTAINS']
                )
                """)

                # Compute PageRank
                pagerank_result = session.run("""
                CALL gds.pageRank.stream('artifact_graph')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name as name, score
                ORDER BY score DESC
                LIMIT 50
                """)

                for record in pagerank_result:
                    if record["name"]:
                        self.pagerank_scores[record["name"]] = record["score"]

                # Compute Betweenness Centrality
                betweenness_result = session.run("""
                CALL gds.betweenness.stream('artifact_graph')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name as name, score
                ORDER BY score DESC
                LIMIT 50
                """)

                for record in betweenness_result:
                    if record["name"]:
                        self.centrality_scores[record["name"]] = record["score"]

                # Drop graph projection
                session.run("CALL gds.graph.drop('artifact_graph')")

            except Exception as e:
                logger.warning(f"Neo4j GDS not available: {e}")
                logger.info("Falling back to NetworkX for centrality computation")
                self._compute_centrality_networkx()

    def _compute_centrality_networkx(self):
        """Compute centrality metrics using NetworkX"""
        if not hasattr(self.kg_builder, 'nx_graph'):
            return

        G = self.kg_builder.nx_graph

        try:
            # Compute PageRank
            pagerank = nx.pagerank(G, max_iter=100)
            self.pagerank_scores = pagerank

            # Compute Betweenness Centrality (sample for large graphs)
            if G.number_of_nodes() > 1000:
                # Sample nodes for large graphs
                sample_nodes = list(G.nodes())[:1000]
                betweenness = nx.betweenness_centrality_subset(
                    G, sample_nodes, sample_nodes, normalized=True
                )
            else:
                betweenness = nx.betweenness_centrality(G, normalized=True)

            self.centrality_scores = betweenness

            logger.info(f"Computed centrality for {len(self.pagerank_scores)} nodes")

        except Exception as e:
            logger.error(f"Error computing centrality: {e}")

    def _perform_community_detection(self):
        """Perform community detection to find artifact groups"""
        if not hasattr(self.kg_builder, 'nx_graph'):
            return

        G = self.kg_builder.nx_graph

        try:
            # Use Louvain algorithm for community detection
            import community as community_louvain

            # Convert to undirected graph for community detection
            G_undirected = G.to_undirected()

            # Detect communities
            communities = community_louvain.best_partition(G_undirected)
            self.community_assignments = communities

            logger.info(f"Detected {len(set(communities.values()))} communities")

        except ImportError:
            logger.warning("python-louvain not available for community detection")
        except Exception as e:
            logger.error(f"Error in community detection: {e}")

    def _cluster_artifacts(self) -> Dict[str, List[str]]:
        """Cluster artifacts based on features"""
        if not hasattr(self.kg_builder, 'nx_graph'):
            return {}

        G = self.kg_builder.nx_graph

        # Extract artifact features
        artifacts = []
        feature_vectors = []

        for node, data in G.nodes(data=True):
            if data.get('node_type') == NODE_TYPES['ARTIFACT']:
                artifacts.append(data.get('name', node))

                # Create feature vector
                features = [
                    data.get('total_files', 0),
                    data.get('code_files', 0),
                    data.get('doc_files', 0),
                    data.get('repo_size_mb', 0),
                    int(data.get('has_docker', False)),
                    int(data.get('has_requirements_txt', False)),
                    int(data.get('has_setup_py', False)),
                    int(data.get('has_jupyter', False)),
                    int(data.get('has_license', False))
                ]

                feature_vectors.append(features)

        if len(feature_vectors) < 3:
            return {}

        try:
            # Standardize features
            if SKLEARN_AVAILABLE:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(feature_vectors)

                # Determine optimal number of clusters
                max_clusters = min(10, len(artifacts) // 2)
                if max_clusters < 2:
                    return {}

                best_score = -1
                best_k = 2

                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    cluster_labels = kmeans.fit_predict(features_scaled)
                    score = silhouette_score(features_scaled, cluster_labels)

                    if score > best_score:
                        best_score = score
                        best_k = k

                # Final clustering
                kmeans = KMeans(n_clusters=best_k, random_state=42)
                cluster_labels = kmeans.fit_predict(features_scaled)

                # Organize results
                clusters = defaultdict(list)
                for artifact, label in zip(artifacts, cluster_labels):
                    clusters[f"cluster_{label}"].append(artifact)

                return dict(clusters)

        except Exception as e:
            logger.error(f"Error in artifact clustering: {e}")

        return {}

    def _detect_outliers(self, threshold: float) -> List[str]:
        """Detect outlier artifacts using statistical methods"""
        if not hasattr(self.kg_builder, 'nx_graph'):
            return []

        G = self.kg_builder.nx_graph

        outliers = []

        # Collect artifact metrics
        artifact_metrics = {}

        for node, data in G.nodes(data=True):
            if data.get('node_type') == NODE_TYPES['ARTIFACT']:
                artifact_id = data.get('name', node)

                # Count sections
                section_count = sum(
                    1 for neighbor in G.neighbors(node)
                    if G.nodes[neighbor].get('node_type') == NODE_TYPES['SECTION']
                )

                # Count tools
                tool_count = sum(
                    1 for neighbor in G.neighbors(node)
                    if G.nodes[neighbor].get('node_type') == NODE_TYPES['TOOL']
                )

                artifact_metrics[artifact_id] = {
                    'section_count': section_count,
                    'tool_count': tool_count,
                    'total_files': data.get('total_files', 0),
                    'repo_size_mb': data.get('repo_size_mb', 0)
                }

        # Detect outliers for each metric
        for metric_name in ['section_count', 'tool_count', 'total_files', 'repo_size_mb']:
            values = [metrics[metric_name] for metrics in artifact_metrics.values()]

            if not values:
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val == 0:
                continue

            for artifact_id, metrics in artifact_metrics.items():
                z_score = abs(metrics[metric_name] - mean_val) / std_val

                if z_score > threshold:
                    outliers.append(artifact_id)

        return list(set(outliers))  # Remove duplicates

    def _analyze_success_factors(self) -> Dict[str, float]:
        """Analyze factors that correlate with success"""
        success_factors = {}

        # Since all artifacts in our dataset are accepted, we'll analyze
        # which features are most common (proxy for success factors)

        if not hasattr(self.kg_builder, 'nx_graph'):
            return success_factors

        G = self.kg_builder.nx_graph

        # Count feature frequencies
        feature_counts = {
            'has_docker': 0,
            'has_requirements_txt': 0,
            'has_setup_py': 0,
            'has_jupyter': 0,
            'has_license': 0,
            'total_artifacts': 0
        }

        for node, data in G.nodes(data=True):
            if data.get('node_type') == NODE_TYPES['ARTIFACT']:
                feature_counts['total_artifacts'] += 1

                for feature in ['has_docker', 'has_requirements_txt', 'has_setup_py', 'has_jupyter', 'has_license']:
                    if data.get(feature, False):
                        feature_counts[feature] += 1

        # Calculate success factors as percentages
        if feature_counts['total_artifacts'] > 0:
            for feature in ['has_docker', 'has_requirements_txt', 'has_setup_py', 'has_jupyter', 'has_license']:
                success_factors[feature] = feature_counts[feature] / feature_counts['total_artifacts']

        # Add section-based success factors
        section_success = {}
        for pattern in self.section_patterns:
            # Normalize by total artifacts
            if feature_counts['total_artifacts'] > 0:
                section_success[f"has_{pattern.heading.lower().replace(' ', '_')}_section"] = \
                    pattern.frequency / feature_counts['total_artifacts']

        success_factors.update(section_success)

        return success_factors

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Recommendations based on most common sections
        if self.section_patterns:
            top_sections = [p.heading for p in self.section_patterns[:5]]
            recommendations.append(
                f"Include these essential sections: {', '.join(top_sections)}"
            )

        # Recommendations based on structural patterns
        if self.structural_patterns:
            best_pattern = max(self.structural_patterns, key=lambda x: x.success_rate)
            recommendations.append(
                f"Follow this successful structure: {' → '.join(best_pattern.section_sequence[:5])}"
            )

        # Recommendations based on common tools
        common_tools = set()
        for pattern in self.structural_patterns[:3]:
            common_tools.update(pattern.common_tools)

        if common_tools:
            recommendations.append(
                f"Consider using these commonly successful tools: {', '.join(list(common_tools)[:5])}"
            )

        # Recommendations for outliers
        if self.outliers:
            recommendations.append(
                f"Review these outlier artifacts for potential issues: {', '.join(self.outliers[:3])}"
            )

        return recommendations

    def _find_tools_for_section_type(self, G: nx.Graph, section_heading: str) -> List[str]:
        """Find tools commonly associated with a section type"""
        tool_counter = Counter()

        for node, data in G.nodes(data=True):
            if (data.get('node_type') == NODE_TYPES['SECTION'] and
                    data.get('heading') == section_heading):

                # Find connected tools
                for neighbor in G.neighbors(node):
                    neighbor_data = G.nodes[neighbor]
                    if neighbor_data.get('node_type') == NODE_TYPES['TOOL']:
                        tool_name = neighbor_data.get('name', '')
                        if tool_name:
                            tool_counter[tool_name] += 1

        return [tool for tool, _ in tool_counter.most_common(10)]

    def _find_tools_for_artifacts(self, G: nx.Graph, artifact_ids: List[str]) -> List[str]:
        """Find tools commonly used by a set of artifacts"""
        tool_counter = Counter()

        for node, data in G.nodes(data=True):
            if (data.get('node_type') == NODE_TYPES['ARTIFACT'] and
                    data.get('name') in artifact_ids):

                # Find connected tools
                for neighbor in G.neighbors(node):
                    neighbor_data = G.nodes[neighbor]
                    if neighbor_data.get('node_type') == NODE_TYPES['TOOL']:
                        tool_name = neighbor_data.get('name', '')
                        if tool_name:
                            tool_counter[tool_name] += 1

        return [tool for tool, _ in tool_counter.most_common(10)]

    def visualize_patterns(self, output_dir: str = "reports/patterns"):
        """Create visualizations of discovered patterns"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Section frequency plot
            if self.section_patterns:
                plt.figure(figsize=(12, 6))
                sections = [p.heading[:20] for p in self.section_patterns[:10]]
                frequencies = [p.frequency for p in self.section_patterns[:10]]

                plt.bar(sections, frequencies)
                plt.title('Most Common Section Types')
                plt.xlabel('Section Heading')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path / 'section_frequencies.png')
                plt.close()

            # 2. Success factors plot
            if hasattr(self, 'success_factors'):
                success_factors = self._analyze_success_factors()
                if success_factors:
                    plt.figure(figsize=(10, 6))
                    factors = list(success_factors.keys())[:10]
                    values = [success_factors[f] for f in factors]

                    plt.bar(factors, values)
                    plt.title('Success Factors (Feature Prevalence)')
                    plt.xlabel('Feature')
                    plt.ylabel('Prevalence')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(output_path / 'success_factors.png')
                    plt.close()

            # 3. Cluster distribution plot
            if self.artifact_clusters:
                plt.figure(figsize=(8, 6))
                cluster_sizes = [len(artifacts) for artifacts in self.artifact_clusters.values()]
                cluster_names = list(self.artifact_clusters.keys())

                plt.pie(cluster_sizes, labels=cluster_names, autopct='%1.1f%%')
                plt.title('Artifact Cluster Distribution')
                plt.savefig(output_path / 'cluster_distribution.png')
                plt.close()

            logger.info(f"Visualizations saved to {output_path}")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

    def export_results(self, output_file: str = "reports/pattern_analysis_results.json"):
        """Export analysis results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'timestamp': datetime.now().isoformat(),
            'section_patterns': [
                {
                    'heading': p.heading,
                    'frequency': p.frequency,
                    'avg_position': p.avg_position,
                    'typical_length': p.typical_length,
                    'associated_tools': p.associated_tools,
                    'success_correlation': p.success_correlation
                }
                for p in self.section_patterns
            ],
            'structural_patterns': [
                {
                    'pattern_name': p.pattern_name,
                    'section_sequence': p.section_sequence,
                    'frequency': p.frequency,
                    'success_rate': p.success_rate,
                    'common_tools': p.common_tools
                }
                for p in self.structural_patterns
            ],
            'artifact_clusters': self.artifact_clusters,
            'outliers': self.outliers,
            'top_centrality_nodes': dict(list(self.centrality_scores.items())[:20]),
            'top_pagerank_nodes': dict(list(self.pagerank_scores.items())[:20])
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results exported to {output_path}")


def main():
    """Example usage of the Pattern Analysis Engine"""
    from phase2_knowledge_graph import KnowledgeGraphBuilder

    # Initialize knowledge graph builder
    kg_builder = KnowledgeGraphBuilder(use_neo4j=False)  # Use NetworkX for demo

    # Build graph (assuming it's already built)
    logger.info("Assuming knowledge graph is already built...")

    # Initialize pattern analysis engine
    pattern_engine = PatternAnalysisEngine(
        knowledge_graph_builder=kg_builder,
        use_neo4j=False
    )

    # Perform comprehensive analysis
    results = pattern_engine.analyze_accepted_artifacts(
        min_frequency=2,
        outlier_threshold=2.0
    )

    # Print results
    print("\n📊 Phase 4: Pattern Analysis Results")
    print("=" * 50)
    print(f"✅ Section patterns discovered: {len(results.section_patterns)}")
    print(f"🏗️  Structural patterns found: {len(results.structural_patterns)}")
    print(f"🎯 Artifact clusters: {len(results.clusters)}")
    print(f"⚠️  Outliers detected: {len(results.outliers)}")

    # Show top section patterns
    print(f"\n📋 Top Section Patterns:")
    for i, pattern in enumerate(results.section_patterns[:5]):
        print(f"  {i + 1}. {pattern.heading} (frequency: {pattern.frequency}, pos: {pattern.avg_position:.1f})")

    # Show structural patterns
    print(f"\n🏗️  Top Structural Patterns:")
    for i, pattern in enumerate(results.structural_patterns[:3]):
        print(f"  {i + 1}. {pattern.pattern_name}: {' → '.join(pattern.section_sequence[:4])}")
        print(f"     Frequency: {pattern.frequency}, Success Rate: {pattern.success_rate:.1%}")

    # Show success factors
    print(f"\n💪 Success Factors:")
    for factor, score in list(results.success_factors.items())[:5]:
        print(f"  - {factor}: {score:.1%}")

    # Show recommendations
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(results.recommendations):
        print(f"  {i + 1}. {rec}")

    # Create visualizations and export results
    pattern_engine.visualize_patterns()
    pattern_engine.export_results()


if __name__ == "__main__":
    main()
