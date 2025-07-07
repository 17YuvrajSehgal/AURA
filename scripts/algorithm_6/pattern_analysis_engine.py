"""
Pattern Analysis Engine for Artifact Evaluation

This module uses Graph Data Science (GDS) techniques to discover patterns
in documentation, structure, and tool usage from accepted research artifacts.

Key Features:
- Community detection for grouping similar artifacts
- Centrality analysis to identify important documentation components
- Motif discovery for recurring structural patterns
- Conference-specific pattern analysis
- Statistical correlation analysis
"""

import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

import networkx as nx
import numpy as np
import pandas as pd

# Neo4j Graph Data Science
try:
    from neo4j import GraphDatabase
    from graphdatascience import GraphDataScience

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Neo4j GDS not available, using NetworkX alternatives")

from config import config, NODE_TYPES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatternCluster:
    """Represents a cluster of artifacts with similar patterns"""
    cluster_id: str
    cluster_type: str
    artifacts: List[str]
    conferences: List[str]
    common_features: Dict[str, Any]
    quality_score: float
    size: int


@dataclass
class DocumentationMotif:
    """Represents a recurring documentation motif"""
    motif_id: str
    pattern: List[str]
    frequency: int
    conferences: Set[str]
    quality_correlation: float
    description: str


@dataclass
class CentralityMetrics:
    """Centrality metrics for graph analysis"""
    node_id: str
    node_type: str
    degree_centrality: float
    betweenness_centrality: float
    pagerank: float
    eigenvector_centrality: float
    clustering_coefficient: float


class PatternAnalysisEngine:
    """Graph Data Science engine for pattern discovery"""

    def __init__(self, knowledge_graph_builder):
        self.kg_builder = knowledge_graph_builder
        self.clusters: List[PatternCluster] = []
        self.motifs: List[DocumentationMotif] = []
        self.centrality_metrics: Dict[str, CentralityMetrics] = {}

        # Initialize GDS connection
        self.use_neo4j_gds = config.knowledge_graph.use_gds and NEO4J_AVAILABLE
        if self.use_neo4j_gds:
            try:
                self.gds = GraphDataScience(
                    config.knowledge_graph.uri,
                    auth=(config.knowledge_graph.username, config.knowledge_graph.password),
                    database=config.knowledge_graph.database
                )
                logger.info("Connected to Neo4j Graph Data Science")
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j GDS: {e}")
                self.use_neo4j_gds = False

        # Get graph for analysis
        self.graph = self.kg_builder.get_graph_for_analysis()

    def analyze_documentation_patterns(self) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis of documentation across artifacts
        
        Returns:
            Analysis results including clusters, motifs, and metrics
        """
        logger.info("Starting comprehensive pattern analysis")

        results = {
            'community_analysis': {},
            'centrality_analysis': {},
            'motif_discovery': {},
            'conference_comparison': {},
            'quality_correlation': {},
            'tool_usage_patterns': {},
            'section_sequence_patterns': {}
        }

        # 1. Community Detection
        results['community_analysis'] = self._detect_communities()

        # 2. Centrality Analysis
        results['centrality_analysis'] = self._analyze_centrality()

        # 3. Motif Discovery
        results['motif_discovery'] = self._discover_motifs()

        # 4. Conference-specific Analysis
        results['conference_comparison'] = self._analyze_conference_patterns()

        # 5. Quality Correlation Analysis
        results['quality_correlation'] = self._analyze_quality_correlations()

        # 6. Tool Usage Pattern Analysis
        results['tool_usage_patterns'] = self._analyze_tool_patterns()

        # 7. Section Sequence Analysis
        results['section_sequence_patterns'] = self._analyze_section_sequences()

        logger.info("Pattern analysis completed")
        return results

    def _detect_communities(self) -> Dict[str, Any]:
        """Detect communities in the artifact graph"""
        logger.info("Detecting communities using multiple algorithms")

        if self.use_neo4j_gds:
            return self._detect_communities_neo4j()
        else:
            return self._detect_communities_networkx()

    def _detect_communities_neo4j(self) -> Dict[str, Any]:
        """Community detection using Neo4j GDS"""
        communities = {}

        try:
            # Create GDS graph projection
            graph_name = config.knowledge_graph.gds_graph_name

            # Project the graph
            projection_query = f"""
            CALL gds.graph.project(
                '{graph_name}',
                ['Artifact', 'Documentation', 'Section', 'Tool'],
                ['HAS_DOCUMENTATION', 'HAS_SECTION', 'USES_TOOL', 'SIMILAR_TO']
            )
            """

            self.gds.run_cypher(projection_query)

            # Run multiple community detection algorithms
            algorithms = config.pattern_analysis.community_algorithms

            for algorithm in algorithms:
                if algorithm == 'louvain':
                    result = self.gds.louvain.stream(graph_name)
                elif algorithm == 'leiden':
                    result = self.gds.leiden.stream(graph_name)
                elif algorithm == 'wcc':  # Weakly Connected Components
                    result = self.gds.wcc.stream(graph_name)

                # Process results
                communities[algorithm] = self._process_community_results(result)

            # Clean up GDS graph
            self.gds.graph.drop(graph_name)

        except Exception as e:
            logger.error(f"Neo4j GDS community detection failed: {e}")
            # Fallback to NetworkX
            return self._detect_communities_networkx()

        return communities

    def _detect_communities_networkx(self) -> Dict[str, Any]:
        """Community detection using NetworkX algorithms"""
        communities = {}

        if not hasattr(self.kg_builder, 'nx_graph'):
            logger.warning("NetworkX graph not available")
            return communities

        G = self.kg_builder.nx_graph

        try:
            # Convert to undirected for community detection
            G_undirected = G.to_undirected()

            # Louvain community detection
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G_undirected)
                communities['louvain'] = self._format_networkx_communities(partition)
            except ImportError:
                logger.warning("python-louvain not available")

            # Connected components
            connected_components = list(nx.connected_components(G_undirected))
            communities['connected_components'] = [list(component) for component in connected_components]

            # K-core decomposition
            k_core = nx.k_core(G_undirected)
            communities['k_core'] = list(k_core.nodes())

        except Exception as e:
            logger.error(f"NetworkX community detection failed: {e}")

        return communities

    def _analyze_centrality(self) -> Dict[str, Any]:
        """Analyze centrality metrics to identify important nodes"""
        logger.info("Computing centrality metrics")

        if self.use_neo4j_gds:
            return self._analyze_centrality_neo4j()
        else:
            return self._analyze_centrality_networkx()

    def _analyze_centrality_networkx(self) -> Dict[str, Any]:
        """Compute centrality metrics using NetworkX"""
        centrality_results = {}

        if not hasattr(self.kg_builder, 'nx_graph'):
            return centrality_results

        G = self.kg_builder.nx_graph

        try:
            # Compute various centrality metrics
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G)))
            pagerank = nx.pagerank(G)

            # For undirected version
            G_undirected = G.to_undirected()
            eigenvector_centrality = nx.eigenvector_centrality(G_undirected, max_iter=1000)
            clustering_coefficient = nx.clustering(G_undirected)

            # Store centrality metrics
            for node in G.nodes():
                self.centrality_metrics[node] = CentralityMetrics(
                    node_id=node,
                    node_type=G.nodes[node].get('node_type', 'unknown'),
                    degree_centrality=degree_centrality.get(node, 0),
                    betweenness_centrality=betweenness_centrality.get(node, 0),
                    pagerank=pagerank.get(node, 0),
                    eigenvector_centrality=eigenvector_centrality.get(node, 0),
                    clustering_coefficient=clustering_coefficient.get(node, 0)
                )

            # Identify top nodes by different metrics
            centrality_results = {
                'top_degree': self._get_top_nodes(degree_centrality, 10),
                'top_betweenness': self._get_top_nodes(betweenness_centrality, 10),
                'top_pagerank': self._get_top_nodes(pagerank, 10),
                'top_eigenvector': self._get_top_nodes(eigenvector_centrality, 10),
                'metrics_summary': {
                    'total_nodes': len(G.nodes()),
                    'total_edges': len(G.edges()),
                    'average_clustering': np.mean(list(clustering_coefficient.values())),
                    'graph_density': nx.density(G)
                }
            }

        except Exception as e:
            logger.error(f"Centrality analysis failed: {e}")

        return centrality_results

    def _discover_motifs(self) -> Dict[str, Any]:
        """Discover recurring structural motifs in documentation"""
        logger.info("Discovering documentation motifs")

        motif_results = {
            'section_motifs': [],
            'tool_motifs': [],
            'structure_motifs': [],
            'quality_motifs': []
        }

        # Find section sequence motifs
        section_sequences = self._extract_section_sequences()
        motif_results['section_motifs'] = self._find_frequent_sequences(section_sequences)

        # Find tool usage motifs
        tool_combinations = self._extract_tool_combinations()
        motif_results['tool_motifs'] = self._find_frequent_combinations(tool_combinations)

        # Find structural motifs
        structure_patterns = self._extract_structure_patterns()
        motif_results['structure_motifs'] = structure_patterns

        return motif_results

    def _analyze_conference_patterns(self) -> Dict[str, Any]:
        """Analyze patterns specific to different conferences"""
        logger.info("Analyzing conference-specific patterns")

        conference_patterns = {}

        # Group artifacts by conference
        conference_groups = self._group_artifacts_by_conference()

        for conference, artifacts in conference_groups.items():
            if len(artifacts) < 3:  # Skip conferences with too few artifacts
                continue

            patterns = {
                'artifact_count': len(artifacts),
                'common_sections': self._find_common_sections(artifacts),
                'preferred_tools': self._find_preferred_tools(artifacts),
                'quality_metrics': self._calculate_conference_quality_metrics(artifacts),
                'documentation_style': self._analyze_documentation_style(artifacts)
            }

            conference_patterns[conference] = patterns

        # Compare conferences
        comparison = self._compare_conference_patterns(conference_patterns)

        return {
            'individual_patterns': conference_patterns,
            'comparison': comparison
        }

    def _analyze_quality_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between documentation features and quality"""
        logger.info("Analyzing quality correlations")

        # Extract features and quality scores for all artifacts
        features_data = []

        if hasattr(self.kg_builder, 'nx_graph'):
            G = self.kg_builder.nx_graph

            for node in G.nodes():
                node_data = G.nodes[node]
                if node_data.get('node_type') == NODE_TYPES['DOCUMENTATION']:
                    features = {
                        'artifact_id': node_data.get('artifact_id'),
                        'has_headers': node_data.get('has_headers', False),
                        'has_code_blocks': node_data.get('has_code_blocks', False),
                        'has_numbered_lists': node_data.get('has_numbered_lists', False),
                        'has_bullet_points': node_data.get('has_bullet_points', False),
                        'has_links': node_data.get('has_links', False),
                        'has_images': node_data.get('has_images', False),
                        'word_count': node_data.get('word_count', 0),
                        'quality_score': node_data.get('quality_score', 0),
                        'readability_score': node_data.get('readability_score', 0)
                    }
                    features_data.append(features)

        if not features_data:
            return {'error': 'No documentation data available for correlation analysis'}

        # Convert to DataFrame for analysis
        df = pd.DataFrame(features_data)

        # Calculate correlations
        correlations = {}
        quality_col = 'quality_score'

        for col in df.columns:
            if col != quality_col and col != 'artifact_id':
                try:
                    correlation = df[col].corr(df[quality_col])
                    if not np.isnan(correlation):
                        correlations[col] = correlation
                except:
                    pass

        # Sort by absolute correlation value
        sorted_correlations = sorted(correlations.items(),
                                     key=lambda x: abs(x[1]),
                                     reverse=True)

        return {
            'feature_correlations': correlations,
            'top_positive_correlations': [(k, v) for k, v in sorted_correlations if v > 0][:5],
            'top_negative_correlations': [(k, v) for k, v in sorted_correlations if v < 0][:5],
            'data_summary': {
                'total_documents': len(df),
                'average_quality_score': df[quality_col].mean(),
                'quality_score_std': df[quality_col].std()
            }
        }

    def _analyze_tool_patterns(self) -> Dict[str, Any]:
        """Analyze tool usage patterns across artifacts"""
        logger.info("Analyzing tool usage patterns")

        tool_patterns = {
            'tool_frequency': Counter(),
            'tool_combinations': Counter(),
            'conference_tool_preferences': defaultdict(Counter),
            'tool_quality_correlation': {}
        }

        # Extract tool usage data
        if hasattr(self.kg_builder, 'nx_graph'):
            G = self.kg_builder.nx_graph

            artifact_tools = defaultdict(set)
            artifact_conferences = {}
            artifact_quality = {}

            # Collect tool usage data
            for node in G.nodes():
                node_data = G.nodes[node]
                if node_data.get('node_type') == NODE_TYPES['ARTIFACT']:
                    artifact_id = node
                    artifact_conferences[artifact_id] = node_data.get('conference', 'unknown')

                    # Find connected tools
                    for neighbor in G.neighbors(node):
                        neighbor_data = G.nodes[neighbor]
                        if neighbor_data.get('node_type') == NODE_TYPES['TOOL']:
                            tool_name = neighbor_data.get('name', neighbor)
                            artifact_tools[artifact_id].add(tool_name)
                            tool_patterns['tool_frequency'][tool_name] += 1

            # Analyze tool combinations
            for artifact_id, tools in artifact_tools.items():
                tools_list = sorted(list(tools))
                if len(tools_list) > 1:
                    for i in range(len(tools_list)):
                        for j in range(i + 1, len(tools_list)):
                            combination = (tools_list[i], tools_list[j])
                            tool_patterns['tool_combinations'][combination] += 1

                # Conference preferences
                conference = artifact_conferences.get(artifact_id, 'unknown')
                for tool in tools:
                    tool_patterns['conference_tool_preferences'][conference][tool] += 1

        return tool_patterns

    def _analyze_section_sequences(self) -> Dict[str, Any]:
        """Analyze common sequences of documentation sections"""
        logger.info("Analyzing section sequences")

        section_sequences = []

        if hasattr(self.kg_builder, 'nx_graph'):
            G = self.kg_builder.nx_graph

            # Group sections by documentation file
            doc_sections = defaultdict(list)

            for node in G.nodes():
                node_data = G.nodes[node]
                if node_data.get('node_type') == NODE_TYPES['SECTION']:
                    doc_id = node_data.get('artifact_id', 'unknown')
                    section_type = node_data.get('section_type', 'other')
                    level = node_data.get('level', 1)

                    doc_sections[doc_id].append({
                        'section_type': section_type,
                        'level': level,
                        'title': node_data.get('title', '')
                    })

            # Extract sequences from each document
            for doc_id, sections in doc_sections.items():
                # Sort by level and extract sequence
                sections.sort(key=lambda x: x['level'])
                sequence = [s['section_type'] for s in sections if s['section_type'] != 'other']
                if len(sequence) >= 2:
                    section_sequences.append(sequence)

        # Find frequent sequences
        sequence_patterns = self._find_frequent_sequences(section_sequences)

        return {
            'total_documents': len(section_sequences),
            'frequent_sequences': sequence_patterns,
            'sequence_lengths': [len(seq) for seq in section_sequences],
            'most_common_sections': Counter([section for seq in section_sequences for section in seq])
        }

    def generate_pattern_visualization(self, output_dir: str = "output/visualizations"):
        """Generate visualizations for discovered patterns"""
        logger.info("Generating pattern visualizations")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Community clusters visualization
        self._visualize_communities(output_dir)

        # 2. Centrality metrics visualization
        self._visualize_centrality_metrics(output_dir)

        # 3. Conference comparison visualization
        self._visualize_conference_patterns(output_dir)

        # 4. Quality correlation heatmap
        self._visualize_quality_correlations(output_dir)

        logger.info(f"Visualizations saved to {output_dir}")

    # Helper methods
    def _get_top_nodes(self, centrality_dict: Dict[str, float], n: int) -> List[Tuple[str, float]]:
        """Get top N nodes by centrality score"""
        return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]

    def _extract_section_sequences(self) -> List[List[str]]:
        """Extract section sequences from documentation"""
        # Implementation would extract actual sequences
        return []

    def _find_frequent_sequences(self, sequences: List[List[str]], min_support: int = 3) -> List[Dict]:
        """Find frequently occurring sequences"""
        sequence_counts = Counter(tuple(seq) for seq in sequences)
        frequent = [(seq, count) for seq, count in sequence_counts.items() if count >= min_support]

        return [{'sequence': list(seq), 'frequency': count} for seq, count in frequent]

    def _group_artifacts_by_conference(self) -> Dict[str, List[str]]:
        """Group artifacts by conference"""
        conference_groups = defaultdict(list)

        if hasattr(self.kg_builder, 'nx_graph'):
            G = self.kg_builder.nx_graph

            for node in G.nodes():
                node_data = G.nodes[node]
                if node_data.get('node_type') == NODE_TYPES['ARTIFACT']:
                    conference = node_data.get('conference', 'unknown')
                    conference_groups[conference].append(node)

        return dict(conference_groups)

    # Additional visualization and analysis methods would be implemented here...
    def _visualize_communities(self, output_dir: str):
        """Visualize community clusters"""
        pass

    def _visualize_centrality_metrics(self, output_dir: str):
        """Visualize centrality metrics"""
        pass

    def _visualize_conference_patterns(self, output_dir: str):
        """Visualize conference-specific patterns"""
        pass

    def _visualize_quality_correlations(self, output_dir: str):
        """Visualize quality correlations"""
        pass

    def _format_networkx_communities(self, partition: Dict) -> List[List[str]]:
        """Format NetworkX community partition"""
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        return list(communities.values())

    def _process_community_results(self, result) -> List[Dict]:
        """Process Neo4j GDS community results"""
        # Implementation would process GDS results
        return []

    def _extract_tool_combinations(self) -> List[List[str]]:
        """Extract tool combinations from artifacts"""
        return []

    def _find_frequent_combinations(self, combinations: List[List[str]]) -> List[Dict]:
        """Find frequent tool combinations"""
        return []

    def _extract_structure_patterns(self) -> List[Dict]:
        """Extract structural patterns"""
        return []

    def _find_common_sections(self, artifacts: List[str]) -> List[str]:
        """Find common sections across artifacts"""
        return []

    def _find_preferred_tools(self, artifacts: List[str]) -> List[str]:
        """Find preferred tools for artifacts"""
        return []

    def _calculate_conference_quality_metrics(self, artifacts: List[str]) -> Dict[str, float]:
        """Calculate quality metrics for conference artifacts"""
        return {}

    def _analyze_documentation_style(self, artifacts: List[str]) -> Dict[str, Any]:
        """Analyze documentation style for artifacts"""
        return {}

    def _compare_conference_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Compare patterns across conferences"""
        return {}

    def _analyze_centrality_neo4j(self) -> Dict[str, Any]:
        """Analyze centrality using Neo4j GDS"""
        return {}

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'gds'):
            # Clean up GDS resources
            pass
