#!/usr/bin/env python3
"""
Graph Analytics Engine

Advanced graph data science operations for pattern discovery in artifact knowledge graphs.
Identifies heavy traffic nodes, relationship patterns, and community structures.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
from collections import defaultdict, Counter
import numpy as np

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_kg_builder import EnhancedKGBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphAnalyticsEngine:
    """
    Advanced graph analytics for pattern discovery and relationship analysis.
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "12345678"
    ):
        """Initialize graph analytics engine."""
        self.kg_builder = EnhancedKGBuilder(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )
        self.graph = self.kg_builder.graph
        
        # Analytics results storage
        self.analytics_results = {}
        
    def analyze_heavy_traffic_patterns(self) -> Dict[str, Any]:
        """
        Identify nodes and relationships with heavy traffic (high connectivity).
        
        Returns:
            Dictionary containing heavy traffic analysis
        """
        logger.info("Analyzing heavy traffic patterns...")
        
        analysis = {
            "high_degree_nodes": self._find_high_degree_nodes(),
            "frequent_relationships": self._find_frequent_relationships(),
            "central_nodes": self._calculate_centrality_measures(),
            "relationship_clusters": self._analyze_relationship_clusters(),
            "traffic_patterns": self._analyze_traffic_patterns()
        }
        
        self.analytics_results["heavy_traffic"] = analysis
        return analysis
    
    def _find_high_degree_nodes(self) -> List[Dict[str, Any]]:
        """Find nodes with highest degree (most connections)."""
        query = """
        MATCH (n)
        WITH n, COUNT { (n)-[]-() } as degree
        WHERE degree > 0
        RETURN 
            id(n) as node_id,
            labels(n) as node_labels,
            coalesce(n.name, n.path, n.title, 'unnamed') as node_name,
            degree,
            COUNT { (n)-->() } as out_degree,
            COUNT { (n)<--() } as in_degree
        ORDER BY degree DESC
        LIMIT 50
        """
        
        results = self.graph.run(query).data()
        
        # Categorize by node type
        categorized_results = defaultdict(list)
        for result in results:
            node_type = result["node_labels"][0] if result["node_labels"] else "Unknown"
            categorized_results[node_type].append(result)
        
        return {
            "all_nodes": results,
            "by_category": dict(categorized_results),
            "statistics": {
                "total_high_degree_nodes": len(results),
                "avg_degree": np.mean([r["degree"] for r in results]) if results else 0,
                "max_degree": max([r["degree"] for r in results]) if results else 0
            }
        }
    
    def _find_frequent_relationships(self) -> List[Dict[str, Any]]:
        """Find most frequent relationship types and patterns."""
        query = """
        MATCH (a)-[r]->(b)
        RETURN 
            type(r) as relationship_type,
            labels(a)[0] as source_type,
            labels(b)[0] as target_type,
            COUNT(*) as frequency
        ORDER BY frequency DESC
        LIMIT 30
        """
        
        results = self.graph.run(query).data()
        
        # Analyze relationship patterns
        pattern_analysis = {
            "most_common_relationships": results,
            "relationship_types": Counter([r["relationship_type"] for r in results]),
            "source_target_patterns": defaultdict(int),
            "total_relationships": sum([r["frequency"] for r in results])
        }
        
        # Analyze source-target patterns
        for result in results:
            pattern = f"{result['source_type']}->{result['target_type']}"
            pattern_analysis["source_target_patterns"][pattern] += result["frequency"]
        
        return pattern_analysis
    
    def _calculate_centrality_measures(self) -> Dict[str, Any]:
        """Calculate various centrality measures for important nodes."""
        centrality_results = {}
        
        # Degree Centrality (already calculated in high_degree_nodes)
        # Let's focus on betweenness and closeness centrality using graph algorithms
        
        # PageRank-like analysis for artifact importance
        pagerank_query = """
        MATCH (a:Artifact)
        WITH a, COUNT { (a)-[]-() } as connections
        RETURN 
            a.name as artifact_name,
            connections,
            a.evaluation_score as eval_score,
            a.acceptance_prediction as acceptance
        ORDER BY connections DESC, eval_score DESC
        LIMIT 20
        """
        
        pagerank_results = self.graph.run(pagerank_query).data()
        
        # Identify central documentation patterns
        doc_centrality_query = """
        MATCH (a:Artifact)-[r1]->(d:Documentation)-[r2]->(s:DocSection)
        WITH d, COUNT(DISTINCT a) as artifact_count, COUNT(s) as section_count
        WHERE artifact_count > 1
        RETURN 
            d.path as doc_path,
            d.doc_type as doc_type,
            artifact_count,
            section_count,
            artifact_count * section_count as centrality_score
        ORDER BY centrality_score DESC
        LIMIT 15
        """
        
        doc_centrality = self.graph.run(doc_centrality_query).data()
        
        # Identify central code patterns
        code_centrality_query = """
        MATCH (a:Artifact)-[r]->(c:CodeFile)
        WITH c.language as language, COUNT(DISTINCT a) as artifact_count, COUNT(c) as file_count
        WHERE artifact_count > 1
        RETURN 
            language,
            artifact_count,
            file_count,
            artifact_count * file_count as language_centrality
        ORDER BY language_centrality DESC
        LIMIT 10
        """
        
        code_centrality = self.graph.run(code_centrality_query).data()
        
        centrality_results = {
            "artifact_importance": pagerank_results,
            "documentation_centrality": doc_centrality,
            "code_language_centrality": code_centrality
        }
        
        return centrality_results
    
    def _analyze_relationship_clusters(self) -> Dict[str, Any]:
        """Analyze clusters of related relationships."""
        # Find artifacts that share similar relationship patterns
        similarity_query = """
        MATCH (a1:Artifact)-[r1]->(n)<-[r2]-(a2:Artifact)
        WHERE a1 <> a2 AND type(r1) = type(r2)
        WITH a1, a2, type(r1) as rel_type, COUNT(n) as shared_connections
        WHERE shared_connections > 1
        RETURN 
            a1.name as artifact1,
            a2.name as artifact2,
            rel_type,
            shared_connections
        ORDER BY shared_connections DESC
        LIMIT 20
        """
        
        similarity_results = self.graph.run(similarity_query).data()
        
        # Find common documentation patterns
        doc_pattern_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(d:Documentation)
        WITH d.doc_type as doc_type, COLLECT(DISTINCT a.name) as artifacts
        WHERE SIZE(artifacts) > 2
        RETURN 
            doc_type,
            artifacts,
            SIZE(artifacts) as artifact_count
        ORDER BY artifact_count DESC
        """
        
        doc_patterns = self.graph.run(doc_pattern_query).data()
        
        # Find common dependency patterns
        dep_pattern_query = """
        MATCH (a:Artifact)-[:HAS_DEPENDENCIES]->(d:DependencyFile)
        WITH d.type as dep_type, COLLECT(DISTINCT a.name) as artifacts
        WHERE SIZE(artifacts) > 1
        RETURN 
            dep_type,
            artifacts,
            SIZE(artifacts) as artifact_count
        ORDER BY artifact_count DESC
        """
        
        dep_patterns = self.graph.run(dep_pattern_query).data()
        
        return {
            "artifact_similarity": similarity_results,
            "common_documentation_patterns": doc_patterns,
            "common_dependency_patterns": dep_patterns
        }
    
    def _analyze_traffic_patterns(self) -> Dict[str, Any]:
        """Analyze overall traffic patterns in the graph."""
        # Node type distribution
        node_distribution_query = """
        MATCH (n)
        RETURN 
            labels(n)[0] as node_type,
            COUNT(n) as count
        ORDER BY count DESC
        """
        
        node_distribution = self.graph.run(node_distribution_query).data()
        
        # Relationship type distribution
        rel_distribution_query = """
        MATCH ()-[r]->()
        RETURN 
            type(r) as relationship_type,
            COUNT(r) as count
        ORDER BY count DESC
        """
        
        rel_distribution = self.graph.run(rel_distribution_query).data()
        
        # Path length analysis
        path_analysis_query = """
        MATCH path = (a:Artifact)-[*1..3]->(n)
        RETURN 
            LENGTH(path) as path_length,
            COUNT(path) as path_count
        ORDER BY path_length
        """
        
        path_analysis = self.graph.run(path_analysis_query).data()
        
        return {
            "node_type_distribution": node_distribution,
            "relationship_type_distribution": rel_distribution,
            "path_length_analysis": path_analysis,
            "graph_density": self._calculate_graph_density()
        }
    
    def _calculate_graph_density(self) -> float:
        """Calculate graph density."""
        try:
            node_count_query = "MATCH (n) RETURN COUNT(n) as node_count"
            edge_count_query = "MATCH ()-[r]->() RETURN COUNT(r) as edge_count"
            
            node_count = self.graph.run(node_count_query).data()[0]["node_count"]
            edge_count = self.graph.run(edge_count_query).data()[0]["edge_count"]
            
            if node_count <= 1:
                return 0.0
            
            max_possible_edges = node_count * (node_count - 1)
            density = edge_count / max_possible_edges if max_possible_edges > 0 else 0.0
            
            return density
            
        except Exception as e:
            logger.error(f"Error calculating graph density: {e}")
            return 0.0
    
    def discover_success_patterns(self) -> Dict[str, Any]:
        """
        Discover patterns that correlate with successful artifacts.
        
        Returns:
            Dictionary containing success pattern analysis
        """
        logger.info("Discovering success patterns...")
        
        patterns = {
            "high_score_characteristics": self._analyze_high_score_characteristics(),
            "successful_artifact_patterns": self._find_successful_artifact_patterns(),
            "correlation_analysis": self._perform_correlation_analysis(),
            "predictive_features": self._identify_predictive_features()
        }
        
        self.analytics_results["success_patterns"] = patterns
        return patterns
    
    def _analyze_high_score_characteristics(self) -> Dict[str, Any]:
        """Analyze characteristics of high-scoring artifacts."""
        high_score_query = """
        MATCH (a:Artifact)
        WHERE a.evaluation_score IS NOT NULL AND a.evaluation_score > 0.7
        OPTIONAL MATCH (a)-[r]->(n)
        RETURN 
            a.name as artifact_name,
            a.evaluation_score as score,
            a.has_readme as has_readme,
            a.has_docker as has_docker,
            a.has_zenodo_doi as has_zenodo_doi,
            a.setup_complexity as setup_complexity,
            a.total_files as total_files,
            type(r) as relationship_type,
            labels(n)[0] as connected_node_type,
            count(r) as connection_count
        """
        
        high_score_results = self.graph.run(high_score_query).data()
        
        # Analyze patterns in high-scoring artifacts
        high_score_analysis = {
            "artifacts": [],
            "common_features": defaultdict(int),
            "common_relationships": defaultdict(int),
            "average_characteristics": {}
        }
        
        artifacts_processed = set()
        total_files_list = []
        
        for result in high_score_results:
            artifact_name = result["artifact_name"]
            
            if artifact_name not in artifacts_processed:
                artifacts_processed.add(artifact_name)
                
                artifact_data = {
                    "name": artifact_name,
                    "score": result["score"],
                    "features": {
                        "has_readme": result.get("has_readme", False),
                        "has_docker": result.get("has_docker", False),
                        "has_zenodo_doi": result.get("has_zenodo_doi", False),
                        "setup_complexity": result.get("setup_complexity", "unknown"),
                        "total_files": result.get("total_files", 0)
                    }
                }
                
                high_score_analysis["artifacts"].append(artifact_data)
                
                # Count common features
                for feature, value in artifact_data["features"].items():
                    if isinstance(value, bool) and value:
                        high_score_analysis["common_features"][feature] += 1
                    elif isinstance(value, str) and value != "unknown":
                        high_score_analysis["common_features"][f"{feature}_{value}"] += 1
                
                if isinstance(result.get("total_files"), (int, float)) and result.get("total_files", 0) > 0:
                    total_files_list.append(result["total_files"])
            
            # Count relationship types
            if result.get("relationship_type"):
                high_score_analysis["common_relationships"][result["relationship_type"]] += 1
        
        # Calculate averages and percentages
        num_artifacts = len(high_score_analysis["artifacts"])
        if num_artifacts > 0:
            for feature, count in high_score_analysis["common_features"].items():
                high_score_analysis["common_features"][feature] = {
                    "count": count,
                    "percentage": (count / num_artifacts) * 100
                }
        
        if total_files_list:
            high_score_analysis["average_characteristics"]["avg_total_files"] = np.mean(total_files_list)
            high_score_analysis["average_characteristics"]["median_total_files"] = np.median(total_files_list)
        
        return high_score_analysis
    
    def _find_successful_artifact_patterns(self) -> Dict[str, Any]:
        """Find patterns common among successful artifacts."""
        # Pattern 1: Documentation structure patterns
        doc_structure_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(d:Documentation)-[:CONTAINS]->(s:DocSection)
        WHERE a.evaluation_score > 0.7
        WITH a, d.doc_type as doc_type, COLLECT(s.section_type) as section_types
        RETURN 
            doc_type,
            section_types,
            COUNT(a) as artifact_count
        ORDER BY artifact_count DESC
        """
        
        doc_structure = self.graph.run(doc_structure_query).data()
        
        # Pattern 2: Code organization patterns
        code_org_query = """
        MATCH (a:Artifact)-[:HAS_CODE]->(c:CodeFile)
        WHERE a.evaluation_score > 0.7
        WITH a, c.language as language, COUNT(c) as file_count
        RETURN 
            language,
            AVG(file_count) as avg_files_per_artifact,
            COUNT(a) as artifact_count
        ORDER BY artifact_count DESC
        """
        
        code_org = self.graph.run(code_org_query).data()
        
        # Pattern 3: Dependency patterns
        dep_patterns_query = """
        MATCH (a:Artifact)-[:HAS_DEPENDENCIES]->(d:DependencyFile)
        WHERE a.evaluation_score > 0.7
        RETURN 
            d.type as dependency_type,
            COUNT(DISTINCT a) as artifact_count,
            AVG(a.evaluation_score) as avg_score
        ORDER BY artifact_count DESC
        """
        
        dep_patterns = self.graph.run(dep_patterns_query).data()
        
        return {
            "documentation_structure_patterns": doc_structure,
            "code_organization_patterns": code_org,
            "dependency_patterns": dep_patterns
        }
    
    def _perform_correlation_analysis(self) -> Dict[str, Any]:
        """Perform correlation analysis between graph features and success."""
        correlation_query = """
        MATCH (a:Artifact)
        WHERE a.evaluation_score IS NOT NULL
        OPTIONAL MATCH (a)-[r]->(n)
        WITH a, COUNT(r) as total_connections, 
             COUNT { (a)-[:HAS_DOCUMENTATION]->() } as doc_connections,
             COUNT { (a)-[:HAS_CODE]->() } as code_connections,
             COUNT { (a)-[:HAS_DEPENDENCIES]->() } as dep_connections
        RETURN 
            a.evaluation_score as score,
            total_connections,
            doc_connections,
            code_connections,
            dep_connections,
            a.total_files as total_files,
            a.has_readme as has_readme,
            a.has_docker as has_docker
        """
        
        correlation_data = self.graph.run(correlation_query).data()
        
        if not correlation_data:
            return {"error": "No data available for correlation analysis"}
        
        # Convert to numpy arrays for correlation calculation
        scores = []
        features = {
            "total_connections": [],
            "doc_connections": [],
            "code_connections": [],
            "dep_connections": [],
            "total_files": [],
            "has_readme": [],
            "has_docker": []
        }
        
        for row in correlation_data:
            if row["score"] is not None:
                scores.append(row["score"])
                
                for feature in features.keys():
                    value = row.get(feature, 0)
                    if isinstance(value, bool):
                        value = 1 if value else 0
                    elif value is None:
                        value = 0
                    features[feature].append(value)
        
        # Calculate correlations
        correlations = {}
        for feature, values in features.items():
            if len(values) == len(scores) and len(scores) > 1:
                try:
                    correlation = np.corrcoef(scores, values)[0, 1]
                    correlations[feature] = {
                        "correlation": correlation,
                        "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
                    }
                except:
                    correlations[feature] = {"correlation": 0.0, "strength": "none"}
        
        return {
            "correlations": correlations,
            "sample_size": len(scores),
            "top_correlations": sorted(correlations.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True)[:5]
        }
    
    def _identify_predictive_features(self) -> List[Dict[str, Any]]:
        """Identify features that are most predictive of success."""
        predictive_query = """
        MATCH (a:Artifact)
        WHERE a.evaluation_score IS NOT NULL
        WITH a.evaluation_score > 0.7 as is_successful,
             a.has_readme as has_readme,
             a.has_docker as has_docker,
             a.has_zenodo_doi as has_zenodo_doi,
             a.setup_complexity as setup_complexity,
             COUNT { (a)-[:HAS_DOCUMENTATION]->() } > 0 as has_documentation,
             COUNT { (a)-[:HAS_CODE]->() } > 0 as has_code,
             a.total_files > 10 as has_many_files
        RETURN 
            is_successful,
            COUNT(*) as total_count,
            SUM(CASE WHEN has_readme THEN 1 ELSE 0 END) as readme_count,
            SUM(CASE WHEN has_docker THEN 1 ELSE 0 END) as docker_count,
            SUM(CASE WHEN has_zenodo_doi THEN 1 ELSE 0 END) as zenodo_count,
            SUM(CASE WHEN has_documentation THEN 1 ELSE 0 END) as doc_count,
            SUM(CASE WHEN has_code THEN 1 ELSE 0 END) as code_count,
            SUM(CASE WHEN has_many_files THEN 1 ELSE 0 END) as many_files_count
        ORDER BY is_successful DESC
        """
        
        predictive_data = self.graph.run(predictive_query).data()
        
        if len(predictive_data) < 2:
            return []
        
        successful_data = predictive_data[0] if predictive_data[0]["is_successful"] else None
        unsuccessful_data = predictive_data[1] if len(predictive_data) > 1 and not predictive_data[1]["is_successful"] else None
        
        if not successful_data or not unsuccessful_data:
            return []
        
        features = ["readme", "docker", "zenodo", "doc", "code", "many_files"]
        predictive_features = []
        
        for feature in features:
            success_count = successful_data.get(f"{feature}_count", 0)
            success_total = successful_data.get("total_count", 1)
            unsuccess_count = unsuccessful_data.get(f"{feature}_count", 0)
            unsuccess_total = unsuccessful_data.get("total_count", 1)
            
            success_rate = success_count / success_total if success_total > 0 else 0
            unsuccess_rate = unsuccess_count / unsuccess_total if unsuccess_total > 0 else 0
            
            predictive_power = success_rate - unsuccess_rate
            
            predictive_features.append({
                "feature": feature,
                "success_rate": success_rate,
                "unsuccess_rate": unsuccess_rate,
                "predictive_power": predictive_power,
                "is_predictive": predictive_power > 0.2
            })
        
        # Sort by predictive power
        predictive_features.sort(key=lambda x: x["predictive_power"], reverse=True)
        
        return predictive_features
    
    def generate_pattern_based_rules(self) -> Dict[str, Any]:
        """
        Generate actionable rules based on discovered patterns.
        
        Returns:
            Dictionary containing pattern-based rules and recommendations
        """
        logger.info("Generating pattern-based rules...")
        
        if "heavy_traffic" not in self.analytics_results:
            self.analyze_heavy_traffic_patterns()
        
        if "success_patterns" not in self.analytics_results:
            self.discover_success_patterns()
        
        rules = {
            "critical_success_factors": self._extract_critical_success_factors(),
            "warning_indicators": self._extract_warning_indicators(),
            "optimization_rules": self._generate_optimization_rules(),
            "prediction_rules": self._generate_prediction_rules()
        }
        
        return rules
    
    def _extract_critical_success_factors(self) -> List[Dict[str, Any]]:
        """Extract critical factors for success based on pattern analysis."""
        factors = []
        
        # From heavy traffic analysis
        heavy_traffic = self.analytics_results.get("heavy_traffic", {})
        frequent_rels = heavy_traffic.get("frequent_relationships", {})
        
        if "most_common_relationships" in frequent_rels:
            top_relationships = frequent_rels["most_common_relationships"][:5]
            for rel in top_relationships:
                factors.append({
                    "type": "relationship_pattern",
                    "factor": f"{rel['source_type']}-{rel['relationship_type']}->{rel['target_type']}",
                    "frequency": rel["frequency"],
                    "importance": "high" if rel["frequency"] > 10 else "medium",
                    "rule": f"Artifacts should establish {rel['relationship_type']} relationships with {rel['target_type']} nodes"
                })
        
        # From success patterns
        success_patterns = self.analytics_results.get("success_patterns", {})
        high_score_chars = success_patterns.get("high_score_characteristics", {})
        
        if "common_features" in high_score_chars:
            for feature, data in high_score_chars["common_features"].items():
                if isinstance(data, dict) and data.get("percentage", 0) > 70:
                    factors.append({
                        "type": "feature_requirement",
                        "factor": feature,
                        "prevalence": data["percentage"],
                        "importance": "critical" if data["percentage"] > 90 else "high",
                        "rule": f"Artifacts should have {feature} (present in {data['percentage']:.1f}% of successful artifacts)"
                    })
        
        return factors
    
    def _extract_warning_indicators(self) -> List[Dict[str, Any]]:
        """Extract warning indicators that correlate with poor performance."""
        warnings = []
        
        # From correlation analysis
        success_patterns = self.analytics_results.get("success_patterns", {})
        correlations = success_patterns.get("correlation_analysis", {}).get("correlations", {})
        
        for feature, data in correlations.items():
            correlation = data.get("correlation", 0)
            if correlation < -0.3:  # Negative correlation
                warnings.append({
                    "indicator": feature,
                    "correlation": correlation,
                    "severity": "high" if correlation < -0.5 else "medium",
                    "warning": f"Low {feature} correlates with poor acceptance (r={correlation:.3f})"
                })
        
        return warnings
    
    def _generate_optimization_rules(self) -> List[Dict[str, Any]]:
        """Generate optimization rules based on successful patterns."""
        rules = []
        
        # Documentation optimization rules
        success_patterns = self.analytics_results.get("success_patterns", {})
        doc_patterns = success_patterns.get("successful_artifact_patterns", {}).get("documentation_structure_patterns", [])
        
        for pattern in doc_patterns[:3]:  # Top 3 patterns
            rules.append({
                "category": "documentation",
                "rule": f"Include {pattern['doc_type']} documentation with sections: {', '.join(pattern['section_types'][:3])}",
                "evidence": f"Found in {pattern['artifact_count']} successful artifacts",
                "priority": "high" if pattern['artifact_count'] > 5 else "medium"
            })
        
        # Code organization rules
        code_patterns = success_patterns.get("successful_artifact_patterns", {}).get("code_organization_patterns", [])
        
        for pattern in code_patterns[:3]:
            if pattern['avg_files_per_artifact'] > 0:
                rules.append({
                    "category": "code_organization",
                    "rule": f"For {pattern['language']} projects, aim for {pattern['avg_files_per_artifact']:.1f} files on average",
                    "evidence": f"Pattern from {pattern['artifact_count']} successful artifacts",
                    "priority": "medium"
                })
        
        return rules
    
    def _generate_prediction_rules(self) -> List[Dict[str, Any]]:
        """Generate prediction rules for new artifacts."""
        rules = []
        
        success_patterns = self.analytics_results.get("success_patterns", {})
        predictive_features = success_patterns.get("predictive_features", [])
        
        for feature in predictive_features:
            if feature.get("is_predictive", False):
                rules.append({
                    "feature": feature["feature"],
                    "rule": f"If artifact has {feature['feature']}, acceptance probability increases by {feature['predictive_power']*100:.1f}%",
                    "success_rate": feature["success_rate"],
                    "weight": feature["predictive_power"],
                    "confidence": "high" if feature["predictive_power"] > 0.4 else "medium"
                })
        
        return rules
    
    def export_analytics_report(self, output_path: str) -> str:
        """Export comprehensive analytics report."""
        try:
            report = {
                "metadata": {
                    "analysis_timestamp": logging.Formatter().format(logging.LogRecord(
                        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
                    )),
                    "graph_statistics": self.kg_builder.get_graph_statistics()
                },
                "heavy_traffic_analysis": self.analytics_results.get("heavy_traffic", {}),
                "success_patterns": self.analytics_results.get("success_patterns", {}),
                "pattern_based_rules": self.generate_pattern_based_rules()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Analytics report exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting analytics report: {e}")
            return f"Error: {str(e)}"
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get a concise summary of analytics results."""
        summary = {
            "heavy_traffic_summary": {},
            "success_patterns_summary": {},
            "key_insights": [],
            "recommendations": []
        }
        
        # Heavy traffic summary
        if "heavy_traffic" in self.analytics_results:
            heavy_traffic = self.analytics_results["heavy_traffic"]
            high_degree = heavy_traffic.get("high_degree_nodes", {})
            
            summary["heavy_traffic_summary"] = {
                "total_high_degree_nodes": high_degree.get("statistics", {}).get("total_high_degree_nodes", 0),
                "max_degree": high_degree.get("statistics", {}).get("max_degree", 0),
                "most_connected_node_types": list(high_degree.get("by_category", {}).keys())[:3]
            }
        
        # Success patterns summary
        if "success_patterns" in self.analytics_results:
            success_patterns = self.analytics_results["success_patterns"]
            high_score_chars = success_patterns.get("high_score_characteristics", {})
            
            summary["success_patterns_summary"] = {
                "high_scoring_artifacts": len(high_score_chars.get("artifacts", [])),
                "most_common_features": [
                    f"{k}: {v.get('percentage', 0):.1f}%" 
                    for k, v in list(high_score_chars.get("common_features", {}).items())[:3]
                    if isinstance(v, dict)
                ]
            }
        
        # Key insights
        summary["key_insights"] = [
            f"Analyzed {summary['heavy_traffic_summary'].get('total_high_degree_nodes', 0)} high-connectivity nodes",
            f"Identified {summary['success_patterns_summary'].get('high_scoring_artifacts', 0)} high-scoring artifacts",
            f"Found {len(summary['success_patterns_summary'].get('most_common_features', []))} key success features"
        ]
        
        return summary
    
    def close(self):
        """Close connections and cleanup."""
        if self.kg_builder:
            self.kg_builder.close()


def main():
    """Example usage of the Graph Analytics Engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph Analytics Engine")
    parser.add_argument("--neo4j-password", default="12345678", help="Neo4j password")
    parser.add_argument("--output-dir", default="graph_analytics_results", help="Output directory")
    parser.add_argument("--analyze-patterns", action="store_true", help="Analyze heavy traffic patterns")
    parser.add_argument("--discover-success", action="store_true", help="Discover success patterns")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analytics engine
    analytics_engine = GraphAnalyticsEngine(neo4j_password=args.neo4j_password)
    
    try:
        print("🔍 Graph Analytics Engine")
        print("=" * 40)
        
        if args.analyze_patterns:
            print("\n📊 Analyzing Heavy Traffic Patterns...")
            heavy_traffic = analytics_engine.analyze_heavy_traffic_patterns()
            
            print(f"Found {len(heavy_traffic.get('high_degree_nodes', {}).get('all_nodes', []))} high-degree nodes")
            print(f"Identified {len(heavy_traffic.get('frequent_relationships', {}).get('most_common_relationships', []))} frequent relationship patterns")
        
        if args.discover_success:
            print("\n🎯 Discovering Success Patterns...")
            success_patterns = analytics_engine.discover_success_patterns()
            
            high_score_artifacts = success_patterns.get('high_score_characteristics', {}).get('artifacts', [])
            print(f"Analyzed {len(high_score_artifacts)} high-scoring artifacts")
            
            predictive_features = success_patterns.get('predictive_features', [])
            predictive_count = sum(1 for f in predictive_features if f.get('is_predictive', False))
            print(f"Found {predictive_count} predictive features")
        
        # Generate pattern-based rules
        print("\n📋 Generating Pattern-Based Rules...")
        rules = analytics_engine.generate_pattern_based_rules()
        
        critical_factors = rules.get('critical_success_factors', [])
        print(f"Identified {len(critical_factors)} critical success factors")
        
        for factor in critical_factors[:3]:
            print(f"  • {factor.get('rule', 'N/A')}")
        
        # Export results
        report_path = output_dir / "graph_analytics_report.json"
        analytics_engine.export_analytics_report(str(report_path))
        
        # Get summary
        summary = analytics_engine.get_analytics_summary()
        print(f"\n📈 Analysis Summary:")
        for insight in summary.get('key_insights', []):
            print(f"  • {insight}")
        
        print(f"\n📁 Results exported to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        analytics_engine.close()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 