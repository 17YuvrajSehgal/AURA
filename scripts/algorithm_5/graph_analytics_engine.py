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
        Analyze README-centric patterns and common features across artifacts.
        
        Returns:
            Dictionary containing README-focused analysis
        """
        logger.info("Analyzing README-centric patterns and common features...")
        
        analysis = {
            "readme_section_patterns": self._analyze_readme_section_patterns(),
            "readme_content_connections": self._analyze_readme_content_connections(),
            "common_readme_features": self._find_common_readme_features(),
            "readme_repository_links": self._analyze_readme_repository_links(),
            "universal_artifact_patterns": self._find_universal_patterns()
        }
        
        self.analytics_results["heavy_traffic"] = analysis
        return analysis
    
    def _analyze_readme_section_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in README sections across all artifacts with semantic enrichment."""
        # Enhance DocSection nodes with semantic properties
        self._enhance_docsection_semantics()
        
        # Find most common README sections with semantic properties
        readme_sections_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        RETURN 
            section.section_type as section_type,
            section.heading as heading,
            section.normalized_heading as normalized_heading,
            COUNT(DISTINCT a) as artifact_count,
            COUNT(section) as total_occurrences,
            AVG(section.content_length) as avg_content_length,
            COUNT(DISTINCT CASE WHEN section.has_code_snippet = true THEN a END) as artifacts_with_code,
            COUNT(DISTINCT CASE WHEN section.has_script_reference = true THEN a END) as artifacts_with_scripts,
            COUNT(DISTINCT CASE WHEN section.has_data_link = true THEN a END) as artifacts_with_data,
            COLLECT(DISTINCT a.name)[0..5] as example_artifacts
        ORDER BY artifact_count DESC, total_occurrences DESC
        LIMIT 20
        """
        
        readme_sections = self.graph.run(readme_sections_query).data()
        
        # Find README section prevalence and classify criticality
        total_artifacts_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        RETURN COUNT(DISTINCT a) as total_with_readme
        """
        
        total_with_readme = self.graph.run(total_artifacts_query).data()[0]["total_with_readme"]
        
        # Calculate prevalence percentages and semantic insights
        for section in readme_sections:
            section["prevalence_percentage"] = (section["artifact_count"] / total_with_readme) * 100 if total_with_readme > 0 else 0
            section["is_universal"] = section["prevalence_percentage"] > 80
            section["is_critical"] = section["prevalence_percentage"] > 70
            section["is_recommended"] = section["prevalence_percentage"] > 40
            section["code_snippet_rate"] = (section["artifacts_with_code"] / section["artifact_count"]) * 100 if section["artifact_count"] > 0 else 0
            section["script_reference_rate"] = (section["artifacts_with_scripts"] / section["artifact_count"]) * 100 if section["artifact_count"] > 0 else 0
            section["data_link_rate"] = (section["artifacts_with_data"] / section["artifact_count"]) * 100 if section["artifact_count"] > 0 else 0
        
        # Analyze README archetypes
        readme_archetypes = self._classify_readme_archetypes()
        
        # Analyze section centrality
        section_centrality = self._analyze_section_centrality()
        
        return {
            "total_artifacts_with_readme": total_with_readme,
            "common_sections": readme_sections,
            "universal_sections": [s for s in readme_sections if s["is_universal"]],
            "critical_sections": [s for s in readme_sections if s["is_critical"]],
            "recommended_sections": [s for s in readme_sections if s["is_recommended"]],
            "readme_archetypes": readme_archetypes,
            "section_centrality": section_centrality
        }
    
    def _enhance_docsection_semantics(self):
        """Enhance DocSection nodes with semantic properties."""
        # Add semantic properties to DocSection nodes
        enhancement_query = """
        MATCH (section:DocSection)
        WHERE section.content IS NOT NULL
        WITH section, toLower(section.content) as content_lower, toLower(section.heading) as heading_lower
        SET section.content_length = SIZE(section.content),
            section.normalized_heading = CASE 
                WHEN heading_lower CONTAINS 'install' OR heading_lower CONTAINS 'setup' OR heading_lower CONTAINS 'getting started' OR heading_lower CONTAINS 'quickstart' THEN 'installation'
                WHEN heading_lower CONTAINS 'usage' OR heading_lower CONTAINS 'how to' OR heading_lower CONTAINS 'tutorial' THEN 'usage'
                WHEN heading_lower CONTAINS 'example' OR heading_lower CONTAINS 'demo' THEN 'examples'
                WHEN heading_lower CONTAINS 'result' OR heading_lower CONTAINS 'output' OR heading_lower CONTAINS 'finding' THEN 'results'
                WHEN heading_lower CONTAINS 'citation' OR heading_lower CONTAINS 'cite' OR heading_lower CONTAINS 'bibtex' THEN 'citation'
                WHEN heading_lower CONTAINS 'license' OR heading_lower CONTAINS 'licence' THEN 'license'
                WHEN heading_lower CONTAINS 'contribute' OR heading_lower CONTAINS 'contributing' THEN 'contribution'
                WHEN heading_lower CONTAINS 'requirement' OR heading_lower CONTAINS 'depend' THEN 'requirements'
                WHEN heading_lower CONTAINS 'overview' OR heading_lower CONTAINS 'description' OR heading_lower CONTAINS 'about' THEN 'overview'
                WHEN heading_lower CONTAINS 'reproduc' OR heading_lower CONTAINS 'replicat' THEN 'reproduction'
                ELSE section.heading
            END,
            section.has_code_snippet = (
                content_lower CONTAINS '```' OR 
                content_lower CONTAINS 'python' OR 
                content_lower CONTAINS 'import' OR
                content_lower CONTAINS 'def ' OR
                content_lower CONTAINS 'class ' OR
                content_lower CONTAINS 'function' OR
                content_lower CONTAINS 'bash' OR
                content_lower CONTAINS 'shell'
            ),
            section.has_script_reference = (
                content_lower CONTAINS '.py' OR 
                content_lower CONTAINS '.sh' OR 
                content_lower CONTAINS '.js' OR
                content_lower CONTAINS '.r' OR
                content_lower CONTAINS 'script' OR
                content_lower CONTAINS 'run.py' OR
                content_lower CONTAINS 'main.py'
            ),
            section.has_data_link = (
                content_lower CONTAINS '.csv' OR 
                content_lower CONTAINS '.json' OR 
                content_lower CONTAINS '.xml' OR
                content_lower CONTAINS 'dataset' OR
                content_lower CONTAINS 'data/' OR
                content_lower CONTAINS 'zenodo' OR
                content_lower CONTAINS 'figshare' OR
                content_lower CONTAINS 'doi'
            ),
            section.has_docker_reference = (
                content_lower CONTAINS 'docker' OR
                content_lower CONTAINS 'container' OR
                content_lower CONTAINS 'dockerfile'
            ),
            section.has_reproducibility_terms = (
                content_lower CONTAINS 'reproduc' OR
                content_lower CONTAINS 'replicat' OR
                content_lower CONTAINS 'verify' OR
                content_lower CONTAINS 'validat' OR
                content_lower CONTAINS 'benchmark'
            ),
            section.instructional_quality = CASE
                WHEN content_lower CONTAINS 'step' AND content_lower CONTAINS 'follow' THEN 'step-by-step'
                WHEN content_lower CONTAINS 'copy' AND content_lower CONTAINS 'paste' THEN 'copy-paste-ready'
                WHEN content_lower CONTAINS 'command' AND content_lower CONTAINS 'run' THEN 'command-oriented'
                WHEN SIZE(section.content) > 200 THEN 'detailed'
                WHEN SIZE(section.content) > 50 THEN 'moderate'
                ELSE 'brief'
            END
        """
        
        try:
            self.graph.run(enhancement_query)
            logger.info("Enhanced DocSection nodes with semantic properties")
        except Exception as e:
            logger.warning(f"Error enhancing DocSection semantics: {e}")
    
    def _classify_readme_archetypes(self) -> List[Dict[str, Any]]:
        """Classify README files into archetypes based on their structure and content."""
        archetype_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WITH a, readme, 
             COUNT(section) as total_sections,
             COUNT(CASE WHEN section.normalized_heading = 'installation' THEN 1 END) as install_sections,
             COUNT(CASE WHEN section.normalized_heading = 'usage' THEN 1 END) as usage_sections,
             COUNT(CASE WHEN section.normalized_heading = 'examples' THEN 1 END) as example_sections,
             COUNT(CASE WHEN section.normalized_heading = 'results' THEN 1 END) as result_sections,
             COUNT(CASE WHEN section.normalized_heading = 'reproduction' THEN 1 END) as repro_sections,
             COUNT(CASE WHEN section.has_code_snippet = true THEN 1 END) as code_sections,
             COUNT(CASE WHEN section.has_script_reference = true THEN 1 END) as script_sections,
             COUNT(CASE WHEN section.has_data_link = true THEN 1 END) as data_sections,
             COUNT(CASE WHEN section.has_reproducibility_terms = true THEN 1 END) as reproducibility_sections
        WITH a, readme,
             CASE 
                WHEN repro_sections > 0 AND result_sections > 0 AND script_sections > 0 THEN 'replication-heavy'
                WHEN example_sections > 0 AND usage_sections > 0 AND code_sections > 0 THEN 'tutorial'
                WHEN result_sections > 0 AND data_sections > 0 THEN 'benchmark'
                WHEN total_sections <= 3 THEN 'summary-only'
                WHEN usage_sections > 0 AND install_sections > 0 THEN 'standard'
                ELSE 'other'
             END as archetype,
             total_sections,
             install_sections + usage_sections + example_sections + result_sections + repro_sections as structured_sections,
             reproducibility_sections
        SET readme.archetype = archetype
        RETURN 
            archetype,
            COUNT(a) as artifact_count,
            AVG(total_sections) as avg_sections,
            AVG(structured_sections) as avg_structured_sections,
            AVG(reproducibility_sections) as avg_repro_sections,
            COLLECT(a.name)[0..3] as example_artifacts
        ORDER BY artifact_count DESC
        """
        
        try:
            archetypes = self.graph.run(archetype_query).data()
            return archetypes
        except Exception as e:
            logger.warning(f"Error classifying README archetypes: {e}")
            return []
    
    def _analyze_section_centrality(self) -> Dict[str, Any]:
        """Analyze centrality of different section types in successful artifacts."""
        centrality_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme' AND a.evaluation_score > 0.7
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WITH section.normalized_heading as section_type,
             COUNT(DISTINCT a) as artifacts_count,
             COUNT(section) as total_occurrences,
             AVG(section.content_length) as avg_content_length,
             SUM(CASE WHEN section.has_code_snippet = true THEN 1 ELSE 0 END) as code_snippets,
             SUM(CASE WHEN section.has_script_reference = true THEN 1 ELSE 0 END) as script_references,
             SUM(CASE WHEN section.has_data_link = true THEN 1 ELSE 0 END) as data_links
        WHERE artifacts_count > 0
        RETURN 
            section_type,
            artifacts_count,
            total_occurrences,
            (artifacts_count * 1.0) as centrality_score,
            avg_content_length,
            code_snippets,
            script_references,
            data_links,
            (code_snippets + script_references + data_links) as total_actionable_content
        ORDER BY centrality_score DESC, total_actionable_content DESC
        LIMIT 15
        """
        
        try:
            centrality_results = self.graph.run(centrality_query).data()
            
            # Calculate relative importance scores
            max_centrality = max([r["centrality_score"] for r in centrality_results]) if centrality_results else 1
            for result in centrality_results:
                result["relative_importance"] = (result["centrality_score"] / max_centrality) * 100
                result["actionable_content_ratio"] = (result["total_actionable_content"] / result["total_occurrences"]) * 100 if result["total_occurrences"] > 0 else 0
            
            return {
                "section_rankings": centrality_results,
                "top_critical_sections": [r for r in centrality_results if r["relative_importance"] > 80],
                "most_actionable_sections": sorted(centrality_results, key=lambda x: x["actionable_content_ratio"], reverse=True)[:5]
            }
        except Exception as e:
            logger.warning(f"Error analyzing section centrality: {e}")
            return {"section_rankings": [], "top_critical_sections": [], "most_actionable_sections": []}
    
    def _analyze_readme_content_connections(self) -> Dict[str, Any]:
        """Analyze how README content connects to repository structure."""
        # README to code file references
        readme_code_refs_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        MATCH (readme)-[:REFERENCES]->(code:CodeFile)
        RETURN 
            code.language as language,
            code.file_type as file_type,
            COUNT(DISTINCT a) as artifact_count,
            COUNT(*) as total_references
        ORDER BY artifact_count DESC
        LIMIT 10
        """
        
        readme_code_refs = self.graph.run(readme_code_refs_query).data()
        
        # README to directory structure references
        readme_dir_refs_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        MATCH (readme)-[:MENTIONS]->(dir:Directory)
        RETURN 
            dir.name as directory_name,
            COUNT(DISTINCT a) as artifact_count,
            COUNT(*) as total_mentions
        ORDER BY artifact_count DESC
        LIMIT 15
        """
        
        readme_dir_refs = self.graph.run(readme_dir_refs_query).data()
        
        # README to setup/installation references
        readme_setup_refs_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WHERE toLower(section.heading) CONTAINS 'install' 
           OR toLower(section.heading) CONTAINS 'setup'
           OR toLower(section.heading) CONTAINS 'usage'
           OR toLower(section.heading) CONTAINS 'run'
        RETURN 
            section.heading as setup_heading,
            COUNT(DISTINCT a) as artifact_count,
            (COUNT(DISTINCT a) * 100.0 / COUNT(DISTINCT a)) as prevalence
        ORDER BY artifact_count DESC
        """
        
        readme_setup_refs = self.graph.run(readme_setup_refs_query).data()
        
        return {
            "code_references": readme_code_refs,
            "directory_references": readme_dir_refs,
            "setup_references": readme_setup_refs
        }
    
    def _find_common_readme_features(self) -> Dict[str, Any]:
        """Find features that are common across README files."""
        # Common README keywords/content
        readme_keywords_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WITH section.content as content, COUNT(DISTINCT a) as artifact_count
        WHERE content IS NOT NULL
        UNWIND split(toLower(content), ' ') as word
        WITH word, artifact_count
        WHERE SIZE(word) > 3 AND word IN ['installation', 'setup', 'usage', 'example', 'docker', 'requirements', 'dependencies', 'citation', 'license', 'contribute', 'overview', 'description', 'running', 'scripts']
        RETURN 
            word as keyword,
            COUNT(*) as total_occurrences,
            COUNT(DISTINCT artifact_count) as artifacts_containing
        ORDER BY artifacts_containing DESC, total_occurrences DESC
        LIMIT 15
        """
        
        try:
            readme_keywords = self.graph.run(readme_keywords_query).data()
        except:
            readme_keywords = []
        
        # README structure patterns
        readme_structure_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        WITH a, readme, COUNT { (readme)-[:CONTAINS]->(:DocSection) } as section_count
        RETURN 
            CASE 
                WHEN section_count <= 3 THEN 'minimal'
                WHEN section_count <= 6 THEN 'standard'
                WHEN section_count <= 10 THEN 'detailed'
                ELSE 'comprehensive'
            END as structure_type,
            COUNT(a) as artifact_count,
            AVG(section_count) as avg_sections
        ORDER BY artifact_count DESC
        """
        
        readme_structure = self.graph.run(readme_structure_query).data()
        
        # README length patterns
        readme_length_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme' AND readme.content IS NOT NULL
        WITH a, SIZE(readme.content) as readme_length
        RETURN 
            CASE 
                WHEN readme_length <= 500 THEN 'brief'
                WHEN readme_length <= 2000 THEN 'moderate'
                WHEN readme_length <= 5000 THEN 'detailed'
                ELSE 'extensive'
            END as length_category,
            COUNT(a) as artifact_count,
            AVG(readme_length) as avg_length
        ORDER BY artifact_count DESC
        """
        
        readme_length = self.graph.run(readme_length_query).data()
        
        return {
            "common_keywords": readme_keywords,
            "structure_patterns": readme_structure,
            "length_patterns": readme_length
        }
    
    def _analyze_readme_repository_links(self) -> Dict[str, Any]:
        """Analyze how README content links to repository elements."""
        # Scripts mentioned in README
        readme_scripts_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        MATCH (a)-[:HAS_CODE]->(script:CodeFile)
        WHERE script.language IN ['shell', 'bash', 'python', 'javascript']
           OR script.file_type CONTAINS 'script'
           OR script.path CONTAINS '.sh'
           OR script.path CONTAINS '.py'
           OR script.path CONTAINS '.js'
        WITH a, script
        OPTIONAL MATCH (readme)-[:MENTIONS]->(script)
        RETURN 
            script.language as script_type,
            COUNT(DISTINCT a) as artifacts_with_scripts,
            COUNT(DISTINCT CASE WHEN (readme)-[:MENTIONS]->(script) THEN a END) as artifacts_mentioning_scripts,
            (COUNT(DISTINCT CASE WHEN (readme)-[:MENTIONS]->(script) THEN a END) * 100.0 / COUNT(DISTINCT a)) as mention_rate
        ORDER BY artifacts_with_scripts DESC
        """
        
        readme_scripts = self.graph.run(readme_scripts_query).data()
        
        # Data files mentioned in README
        readme_data_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        MATCH (a)-[:CONTAINS]->(data:File)
        WHERE data.path CONTAINS '.csv' 
           OR data.path CONTAINS '.json'
           OR data.path CONTAINS '.xml'
           OR data.path CONTAINS '.txt'
           OR data.path CONTAINS 'data'
        WITH a, COUNT(data) as data_file_count
        RETURN 
            CASE 
                WHEN data_file_count = 0 THEN 'no_data'
                WHEN data_file_count <= 5 THEN 'few_data_files'
                WHEN data_file_count <= 20 THEN 'moderate_data'
                ELSE 'data_heavy'
            END as data_category,
            COUNT(a) as artifact_count
        ORDER BY artifact_count DESC
        """
        
        readme_data = self.graph.run(readme_data_query).data()
        
        return {
            "script_mentions": readme_scripts,
            "data_file_patterns": readme_data
        }
    
    def _find_universal_patterns(self) -> Dict[str, Any]:
        """Find patterns that appear across ALL or most artifacts, including graph motifs."""
        # Universal file types
        universal_files_query = """
        MATCH (a:Artifact)
        WITH COUNT(a) as total_artifacts
        MATCH (a:Artifact)-[:CONTAINS]->(f:File)
        WITH total_artifacts, f.file_type as file_type, COUNT(DISTINCT a) as artifact_count
        WHERE artifact_count > total_artifacts * 0.5
        RETURN 
            file_type,
            artifact_count,
            total_artifacts,
            (artifact_count * 100.0 / total_artifacts) as prevalence_percentage
        ORDER BY prevalence_percentage DESC
        """
        
        universal_files = self.graph.run(universal_files_query).data()
        
        # Universal directory structures
        universal_dirs_query = """
        MATCH (a:Artifact)
        WITH COUNT(a) as total_artifacts
        MATCH (a:Artifact)-[:CONTAINS]->(d:Directory)
        WITH total_artifacts, d.name as dir_name, COUNT(DISTINCT a) as artifact_count
        WHERE artifact_count > total_artifacts * 0.3
        RETURN 
            dir_name,
            artifact_count,
            total_artifacts,
            (artifact_count * 100.0 / total_artifacts) as prevalence_percentage
        ORDER BY prevalence_percentage DESC
        LIMIT 10
        """
        
        universal_dirs = self.graph.run(universal_dirs_query).data()
        
        # Universal documentation patterns
        universal_docs_query = """
        MATCH (a:Artifact)
        WITH COUNT(a) as total_artifacts
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(d:Documentation)
        WITH total_artifacts, d.doc_type as doc_type, COUNT(DISTINCT a) as artifact_count
        WHERE artifact_count > total_artifacts * 0.4
        RETURN 
            doc_type,
            artifact_count,
            total_artifacts,
            (artifact_count * 100.0 / total_artifacts) as prevalence_percentage
        ORDER BY prevalence_percentage DESC
        """
        
        universal_docs = self.graph.run(universal_docs_query).data()
        
        # Find README graph motifs
        readme_motifs = self._discover_readme_motifs()
        
        return {
            "universal_file_types": universal_files,
            "common_directory_structures": universal_dirs,
            "universal_documentation_types": universal_docs,
            "readme_motifs": readme_motifs,
            "readme_quality_analysis": self._analyze_readme_quality()
        }
    
    def _analyze_readme_quality(self) -> Dict[str, Any]:
        """Analyze and score README quality based on semantic properties."""
        quality_analysis_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme'
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WITH a, readme,
             COUNT(section) as total_sections,
             COUNT(CASE WHEN section.normalized_heading = 'installation' THEN 1 END) as has_installation,
             COUNT(CASE WHEN section.normalized_heading = 'usage' THEN 1 END) as has_usage,
             COUNT(CASE WHEN section.normalized_heading = 'examples' THEN 1 END) as has_examples,
             COUNT(CASE WHEN section.normalized_heading = 'results' THEN 1 END) as has_results,
             COUNT(CASE WHEN section.normalized_heading = 'citation' THEN 1 END) as has_citation,
             COUNT(CASE WHEN section.has_code_snippet = true THEN 1 END) as code_sections,
             COUNT(CASE WHEN section.has_script_reference = true THEN 1 END) as script_sections,
             COUNT(CASE WHEN section.has_data_link = true THEN 1 END) as data_sections,
             COUNT(CASE WHEN section.has_docker_reference = true THEN 1 END) as docker_sections,
             COUNT(CASE WHEN section.has_reproducibility_terms = true THEN 1 END) as repro_sections,
             COUNT(CASE WHEN section.instructional_quality = 'step-by-step' THEN 1 END) as stepwise_sections,
             COUNT(CASE WHEN section.instructional_quality = 'copy-paste-ready' THEN 1 END) as copyready_sections,
             AVG(section.content_length) as avg_content_length,
             SUM(section.content_length) as total_content_length
        WITH a, readme,
             // Calculate quality score components
             CASE WHEN has_installation > 0 THEN 20 ELSE 0 END as install_score,
             CASE WHEN has_usage > 0 THEN 15 ELSE 0 END as usage_score,
             CASE WHEN has_examples > 0 THEN 10 ELSE 0 END as examples_score,
             CASE WHEN has_results > 0 THEN 10 ELSE 0 END as results_score,
             CASE WHEN has_citation > 0 THEN 5 ELSE 0 END as citation_score,
             CASE WHEN code_sections > 0 THEN 15 ELSE 0 END as code_score,
             CASE WHEN script_sections > 0 THEN 10 ELSE 0 END as script_score,
             CASE WHEN data_sections > 0 THEN 5 ELSE 0 END as data_score,
             CASE WHEN docker_sections > 0 THEN 5 ELSE 0 END as docker_score,
             CASE WHEN repro_sections > 0 THEN 5 ELSE 0 END as repro_score,
             CASE WHEN total_content_length > 1000 THEN 10 ELSE total_content_length/100 END as content_score,
             total_sections,
             avg_content_length,
             stepwise_sections,
             copyready_sections
        WITH a, readme,
             (install_score + usage_score + examples_score + results_score + citation_score + 
              code_score + script_score + data_score + docker_score + repro_score + content_score) as raw_quality_score,
             total_sections,
             avg_content_length,
             stepwise_sections,
             copyready_sections
        SET readme.quality_score = CASE WHEN raw_quality_score > 100 THEN 100 ELSE raw_quality_score END,
            readme.quality_grade = CASE 
                WHEN raw_quality_score >= 90 THEN 'A'
                WHEN raw_quality_score >= 80 THEN 'B'
                WHEN raw_quality_score >= 70 THEN 'C'
                WHEN raw_quality_score >= 60 THEN 'D'
                ELSE 'F'
            END,
            readme.instructional_quality = CASE
                WHEN stepwise_sections > 0 OR copyready_sections > 0 THEN 'high'
                WHEN avg_content_length > 200 THEN 'medium'
                ELSE 'low'
            END
        RETURN 
            a.name as artifact_name,
            readme.quality_score as quality_score,
            readme.quality_grade as quality_grade,
            readme.instructional_quality as instructional_quality,
            a.evaluation_score as artifact_score,
            total_sections,
            avg_content_length
        ORDER BY quality_score DESC
        """
        
        try:
            quality_results = self.graph.run(quality_analysis_query).data()
            
            # Calculate quality statistics
            quality_scores = [r["quality_score"] for r in quality_results if r["quality_score"] is not None]
            artifact_scores = [r["artifact_score"] for r in quality_results if r["artifact_score"] is not None]
            
            quality_stats = {
                "total_readmes_analyzed": len(quality_results),
                "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "quality_distribution": {
                    "A": len([r for r in quality_results if r["quality_grade"] == "A"]),
                    "B": len([r for r in quality_results if r["quality_grade"] == "B"]),
                    "C": len([r for r in quality_results if r["quality_grade"] == "C"]),
                    "D": len([r for r in quality_results if r["quality_grade"] == "D"]),
                    "F": len([r for r in quality_results if r["quality_grade"] == "F"])
                },
                "instructional_quality_distribution": {
                    "high": len([r for r in quality_results if r["instructional_quality"] == "high"]),
                    "medium": len([r for r in quality_results if r["instructional_quality"] == "medium"]),
                    "low": len([r for r in quality_results if r["instructional_quality"] == "low"])
                }
            }
            
            # Calculate correlation between README quality and artifact acceptance
            if len(quality_scores) > 1 and len(artifact_scores) > 1:
                # Simple correlation calculation
                n = len(quality_scores)
                sum_x = sum(quality_scores)
                sum_y = sum(artifact_scores)
                sum_xy = sum(q * a for q, a in zip(quality_scores, artifact_scores))
                sum_x2 = sum(q ** 2 for q in quality_scores)
                sum_y2 = sum(a ** 2 for a in artifact_scores)
                
                correlation = (n * sum_xy - sum_x * sum_y) / ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
                quality_stats["readme_artifact_correlation"] = correlation
            else:
                quality_stats["readme_artifact_correlation"] = 0
            
            return {
                "quality_results": quality_results,
                "quality_statistics": quality_stats,
                "top_quality_readmes": [r for r in quality_results if r["quality_score"] >= 80],
                "improvement_candidates": [r for r in quality_results if r["quality_score"] < 60]
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing README quality: {e}")
            return {
                "quality_results": [],
                "quality_statistics": {"total_readmes_analyzed": 0, "avg_quality_score": 0},
                "top_quality_readmes": [],
                "improvement_candidates": []
            }
    
    def _discover_readme_motifs(self) -> Dict[str, Any]:
        """Discover common README graph motifs that correlate with acceptance."""
        motifs = {}
        
        # Motif 1: README -> CONTAINS -> InstallationSection
        installation_motif_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme' AND a.evaluation_score > 0.7
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WHERE section.normalized_heading = 'installation'
        RETURN 
            'readme_installation' as motif_type,
            COUNT(DISTINCT a) as successful_artifacts,
            COUNT(section) as total_occurrences,
            AVG(section.content_length) as avg_content_length,
            SUM(CASE WHEN section.has_code_snippet = true THEN 1 ELSE 0 END) as code_snippets,
            SUM(CASE WHEN section.has_script_reference = true THEN 1 ELSE 0 END) as script_references
        """
        
        # Motif 2: README -> MENTIONS -> DockerFile
        docker_motif_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme' AND a.evaluation_score > 0.7
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WHERE section.has_docker_reference = true
        RETURN 
            'readme_docker' as motif_type,
            COUNT(DISTINCT a) as successful_artifacts,
            COUNT(section) as total_occurrences,
            AVG(section.content_length) as avg_content_length
        """
        
        # Motif 3: README -> CONTAINS -> CitationSection -> CONTAINS -> ZenodoLink
        citation_motif_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme' AND a.evaluation_score > 0.7
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WHERE section.normalized_heading = 'citation' AND section.has_data_link = true
        RETURN 
            'readme_citation_zenodo' as motif_type,
            COUNT(DISTINCT a) as successful_artifacts,
            COUNT(section) as total_occurrences,
            AVG(section.content_length) as avg_content_length
        """
        
        # Motif 4: README -> CONTAINS -> ResultsSection
        results_motif_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme' AND a.evaluation_score > 0.7
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WHERE section.normalized_heading = 'results'
        RETURN 
            'readme_results' as motif_type,
            COUNT(DISTINCT a) as successful_artifacts,
            COUNT(section) as total_occurrences,
            AVG(section.content_length) as avg_content_length,
            SUM(CASE WHEN section.has_data_link = true THEN 1 ELSE 0 END) as data_links
        """
        
        # Motif 5: README -> CONTAINS -> ReproductionSection
        reproduction_motif_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme' AND a.evaluation_score > 0.7
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WHERE section.has_reproducibility_terms = true
        RETURN 
            'readme_reproduction' as motif_type,
            COUNT(DISTINCT a) as successful_artifacts,
            COUNT(section) as total_occurrences,
            AVG(section.content_length) as avg_content_length,
            SUM(CASE WHEN section.has_script_reference = true THEN 1 ELSE 0 END) as script_references
        """
        
        # Execute motif queries
        motif_queries = [
            installation_motif_query,
            docker_motif_query,
            citation_motif_query,
            results_motif_query,
            reproduction_motif_query
        ]
        
        try:
            for query in motif_queries:
                result = self.graph.run(query).data()
                if result:
                    motif_data = result[0]
                    motif_type = motif_data["motif_type"]
                    
                    # Calculate motif strength
                    motif_data["motif_strength"] = (motif_data["successful_artifacts"] / 20) * 100  # Assuming 20 total artifacts
                    motif_data["is_strong_motif"] = motif_data["motif_strength"] > 50
                    
                    motifs[motif_type] = motif_data
        except Exception as e:
            logger.warning(f"Error discovering README motifs: {e}")
        
        # Find composite motifs (artifacts with multiple strong patterns)
        composite_motifs = self._find_composite_motifs()
        
        return {
            "individual_motifs": motifs,
            "composite_motifs": composite_motifs,
            "motif_summary": {
                "total_motifs_found": len(motifs),
                "strong_motifs": [k for k, v in motifs.items() if v.get("is_strong_motif", False)],
                "average_motif_strength": sum(v.get("motif_strength", 0) for v in motifs.values()) / len(motifs) if motifs else 0
            }
        }
    
    def _find_composite_motifs(self) -> List[Dict[str, Any]]:
        """Find artifacts that exhibit multiple strong README motifs."""
        composite_query = """
        MATCH (a:Artifact)-[:HAS_DOCUMENTATION]->(readme:Documentation)
        WHERE toLower(readme.path) CONTAINS 'readme' AND a.evaluation_score > 0.7
        MATCH (readme)-[:CONTAINS]->(section:DocSection)
        WITH a, readme,
             COUNT(CASE WHEN section.normalized_heading = 'installation' THEN 1 END) as has_installation,
             COUNT(CASE WHEN section.has_docker_reference = true THEN 1 END) as has_docker_ref,
             COUNT(CASE WHEN section.normalized_heading = 'citation' AND section.has_data_link = true THEN 1 END) as has_citation_zenodo,
             COUNT(CASE WHEN section.normalized_heading = 'results' THEN 1 END) as has_results,
             COUNT(CASE WHEN section.has_reproducibility_terms = true THEN 1 END) as has_reproduction
        WITH a, readme,
             (CASE WHEN has_installation > 0 THEN 1 ELSE 0 END +
              CASE WHEN has_docker_ref > 0 THEN 1 ELSE 0 END +
              CASE WHEN has_citation_zenodo > 0 THEN 1 ELSE 0 END +
              CASE WHEN has_results > 0 THEN 1 ELSE 0 END +
              CASE WHEN has_reproduction > 0 THEN 1 ELSE 0 END) as motif_count,
             has_installation > 0 as has_install_motif,
             has_docker_ref > 0 as has_docker_motif,
             has_citation_zenodo > 0 as has_citation_motif,
             has_results > 0 as has_results_motif,
             has_reproduction > 0 as has_repro_motif
        WHERE motif_count >= 2
        RETURN 
            a.name as artifact_name,
            a.evaluation_score as score,
            motif_count,
            has_install_motif,
            has_docker_motif,
            has_citation_motif,
            has_results_motif,
            has_repro_motif
        ORDER BY motif_count DESC, score DESC
        LIMIT 10
        """
        
        try:
            composite_results = self.graph.run(composite_query).data()
            
            # Add motif pattern descriptions
            for result in composite_results:
                motif_patterns = []
                if result["has_install_motif"]:
                    motif_patterns.append("installation")
                if result["has_docker_motif"]:
                    motif_patterns.append("docker")
                if result["has_citation_motif"]:
                    motif_patterns.append("citation-zenodo")
                if result["has_results_motif"]:
                    motif_patterns.append("results")
                if result["has_repro_motif"]:
                    motif_patterns.append("reproduction")
                
                result["motif_patterns"] = motif_patterns
                result["motif_pattern_string"] = " + ".join(motif_patterns)
            
            return composite_results
        except Exception as e:
            logger.warning(f"Error finding composite motifs: {e}")
            return []
    
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
        
        # From README-centric analysis
        heavy_traffic = self.analytics_results.get("heavy_traffic", {})
        readme_sections = heavy_traffic.get("readme_section_patterns", {})
        
        # Universal README sections
        universal_sections = readme_sections.get("universal_sections", [])
        for section in universal_sections:
            factors.append({
                "type": "universal_readme_section",
                "factor": f"README section: {section['heading']}",
                "prevalence": section["prevalence_percentage"],
                "importance": "critical",
                "rule": f"README should include '{section['heading']}' section (found in {section['prevalence_percentage']:.1f}% of successful artifacts)"
            })
        
        # Critical README sections  
        critical_sections = readme_sections.get("critical_sections", [])
        for section in critical_sections[:5]:  # Top 5 critical sections
            if section not in universal_sections:  # Avoid duplicates
                factors.append({
                    "type": "critical_readme_section",
                    "factor": f"README section: {section['heading']}",
                    "prevalence": section["prevalence_percentage"],
                    "importance": "high",
                    "rule": f"README should consider including '{section['heading']}' section ({section['prevalence_percentage']:.1f}% prevalence)"
                })
        
        # Universal artifact patterns
        universal_patterns = heavy_traffic.get("universal_artifact_patterns", {})
        universal_files = universal_patterns.get("universal_file_types", [])
        for file_type in universal_files[:3]:  # Top 3 universal file types
            factors.append({
                "type": "universal_file_type",
                "factor": f"File type: {file_type['file_type']}",
                "prevalence": file_type["prevalence_percentage"],
                "importance": "high",
                "rule": f"Artifacts should include {file_type['file_type']} files ({file_type['prevalence_percentage']:.1f}% prevalence)"
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
        """Generate optimization rules based on README-centric patterns."""
        rules = []
        
        # README optimization rules
        heavy_traffic = self.analytics_results.get("heavy_traffic", {})
        readme_features = heavy_traffic.get("common_readme_features", {})
        
        # README structure recommendations
        structure_patterns = readme_features.get("structure_patterns", [])
        for pattern in structure_patterns[:2]:  # Top 2 structure patterns
            rules.append({
                "category": "readme_structure",
                "rule": f"Use {pattern['structure_type']} README structure with ~{pattern['avg_sections']:.0f} sections",
                "evidence": f"Found in {pattern['artifact_count']} successful artifacts",
                "priority": "high" if pattern['artifact_count'] > 5 else "medium"
            })
        
        # README content recommendations
        content_connections = heavy_traffic.get("readme_content_connections", {})
        setup_refs = content_connections.get("setup_references", [])
        
        for setup_ref in setup_refs[:3]:  # Top 3 setup section patterns
            rules.append({
                "category": "readme_content",
                "rule": f"Include clear setup instructions with heading like '{setup_ref['setup_heading']}'",
                "evidence": f"Found in {setup_ref['artifact_count']} successful artifacts",
                "priority": "high"
            })
        
        # Repository organization rules based on common patterns
        universal_patterns = heavy_traffic.get("universal_artifact_patterns", {})
        common_dirs = universal_patterns.get("common_directory_structures", [])
        
        for dir_pattern in common_dirs[:3]:  # Top 3 directory patterns
            if dir_pattern['prevalence_percentage'] > 50:
                rules.append({
                    "category": "repository_structure",
                    "rule": f"Consider including '{dir_pattern['dir_name']}' directory for better organization",
                    "evidence": f"Present in {dir_pattern['prevalence_percentage']:.1f}% of successful artifacts",
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
                "readme_centric_analysis": self.analytics_results.get("heavy_traffic", {}),
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
        
        # README-centric analysis summary
        if "heavy_traffic" in self.analytics_results:
            heavy_traffic = self.analytics_results["heavy_traffic"]
            readme_sections = heavy_traffic.get("readme_section_patterns", {})
            universal_patterns = heavy_traffic.get("universal_artifact_patterns", {})
            
            summary["heavy_traffic_summary"] = {
                "total_artifacts_with_readme": readme_sections.get("total_artifacts_with_readme", 0),
                "universal_readme_sections": len(readme_sections.get("universal_sections", [])),
                "common_readme_sections": len(readme_sections.get("common_sections", [])),
                "universal_file_types": len(universal_patterns.get("universal_file_types", [])),
                "common_directories": len(universal_patterns.get("common_directory_structures", []))
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
            f"Analyzed {summary['heavy_traffic_summary'].get('total_artifacts_with_readme', 0)} artifacts with README files",
            f"Identified {summary['heavy_traffic_summary'].get('universal_readme_sections', 0)} universal README sections",
            f"Found {summary['heavy_traffic_summary'].get('universal_file_types', 0)} universal file types across artifacts",
            f"Discovered {summary['success_patterns_summary'].get('high_scoring_artifacts', 0)} high-scoring artifacts",
            f"Identified {len(summary['success_patterns_summary'].get('most_common_features', []))} key success features"
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