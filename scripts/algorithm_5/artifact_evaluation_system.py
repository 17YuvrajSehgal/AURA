#!/usr/bin/env python3
"""
Artifact Evaluation System

This module provides a comprehensive evaluation system for research artifacts
using Knowledge Graphs, LLMs, and Graph Data Science.

Features:
- JSON analysis file processing
- LLM-based semantic analysis
- Artifact scoring and prediction
- Explainability and recommendations
- Visualization capabilities
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from collections import defaultdict

from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from enhanced_kg_builder import EnhancedKGBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArtifactFeatures:
    """Features extracted for artifact evaluation."""
    # Documentation features
    has_readme: bool = False
    readme_length: int = 0
    has_license: bool = False
    has_changelog: bool = False
    documentation_sections: int = 0
    
    # Reproducibility features
    has_docker: bool = False
    has_requirements: bool = False
    has_setup_instructions: bool = False
    has_examples: bool = False
    has_tests: bool = False
    
    # Availability features
    has_zenodo_doi: bool = False
    has_github_url: bool = False
    has_data_files: bool = False
    
    # Code structure features
    total_files: int = 0
    code_files: int = 0
    programming_languages: List[str] = None
    build_systems: List[str] = None
    
    # Complexity features
    tree_depth: int = 0
    repo_size_mb: float = 0.0
    setup_complexity: str = "unknown"
    
    def __post_init__(self):
        if self.programming_languages is None:
            self.programming_languages = []
        if self.build_systems is None:
            self.build_systems = []


class ArtifactEvaluationSystem:
    """
    Comprehensive artifact evaluation system using Knowledge Graphs and LLMs.
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        openai_api_key: Optional[str] = None,
        clear_existing: bool = False
    ):
        """
        Initialize the Artifact Evaluation System.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key for LLM analysis
            clear_existing: Whether to clear existing graph data
        """
        # Initialize Knowledge Graph Builder
        self.kg_builder = EnhancedKGBuilder(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            clear_existing=clear_existing
        )
        
        self.graph = self.kg_builder.graph
        self.node_matcher = self.kg_builder.node_matcher
        self.relationship_matcher = self.kg_builder.relationship_matcher
        
        # LLM configuration
        self.openai_api_key = openai_api_key
        if openai_api_key:
            try:
                import openai
                openai.api_key = openai_api_key
                self.openai_client = openai
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
                self.openai_client = None
        else:
            self.openai_client = None
        
        # Evaluation criteria weights
        self.evaluation_weights = {
            "documentation_quality": 0.25,
            "reproducibility": 0.30,
            "availability": 0.20,
            "code_structure": 0.15,
            "complexity": 0.10
        }
        
        # Create evaluation-specific indexes
        self._create_evaluation_indexes()
    
    def _create_evaluation_indexes(self):
        """Create indexes for evaluation-specific properties."""
        indexes = [
            "CREATE INDEX evaluation_score IF NOT EXISTS FOR (a:Artifact) ON (a.evaluation_score)",
            "CREATE INDEX artifact_type IF NOT EXISTS FOR (a:Artifact) ON (a.artifact_type)",
            "CREATE INDEX acceptance_prediction IF NOT EXISTS FOR (a:Artifact) ON (a.acceptance_prediction)",
        ]
        
        for index_query in indexes:
            try:
                self.graph.run(index_query)
            except Exception as e:
                logger.debug(f"Index creation skipped: {e}")
    
    def evaluate_artifact_from_json(self, json_file_path: str) -> Dict[str, Any]:
        """
        Evaluate an artifact from JSON analysis file.
        
        Args:
            json_file_path: Path to JSON analysis file
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            if not analysis_data.get("success", False):
                return {
                    "success": False,
                    "error": f"Analysis data indicates failure: {analysis_data.get('error', 'Unknown error')}",
                    "artifact_name": analysis_data.get("artifact_name", "unknown")
                }
            
            artifact_name = analysis_data.get("artifact_name", "unknown")
            logger.info(f"Evaluating artifact: {artifact_name}")
            
            # Extract features from JSON
            features = self._extract_features_from_json(analysis_data)
            
            # Build knowledge graph from analysis data
            kg_result = self._build_kg_from_analysis(analysis_data, features)
            
            # Perform LLM-based semantic analysis
            semantic_analysis = self._perform_semantic_analysis(analysis_data)
            
            # Calculate evaluation scores
            evaluation_scores = self._calculate_evaluation_scores(features, semantic_analysis)
            
            # Predict acceptance likelihood
            acceptance_prediction = self._predict_acceptance(features, evaluation_scores)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(features, evaluation_scores)
            
            # Update artifact node with evaluation results
            self._update_artifact_with_evaluation(
                artifact_name, 
                features, 
                evaluation_scores, 
                acceptance_prediction,
                recommendations
            )
            
            return {
                "success": True,
                "artifact_name": artifact_name,
                "features": asdict(features),
                "semantic_analysis": semantic_analysis,
                "evaluation_scores": evaluation_scores,
                "acceptance_prediction": acceptance_prediction,
                "recommendations": recommendations,
                "kg_result": kg_result
            }
            
        except Exception as e:
            logger.error(f"Error evaluating artifact from JSON: {e}")
            return {
                "success": False,
                "error": str(e),
                "artifact_name": "unknown"
            }
    
    def _extract_features_from_json(self, analysis_data: Dict) -> ArtifactFeatures:
        """Extract features from JSON analysis data."""
        features = ArtifactFeatures()
        
        # Basic artifact info
        features.total_files = len(analysis_data.get("tree_structure", []))
        features.repo_size_mb = analysis_data.get("repo_size_mb", 0.0)
        
        # Documentation features
        doc_files = analysis_data.get("documentation_files", [])
        features.has_readme = any("readme" in f["path"].lower() for f in doc_files)
        features.documentation_sections = sum(len(f.get("content", [])) for f in doc_files)
        
        if features.has_readme:
            readme_content = next((f["content"] for f in doc_files if "readme" in f["path"].lower()), [])
            features.readme_length = sum(len(line) for line in readme_content)
        
        # License and other files
        features.has_license = len(analysis_data.get("license_files", [])) > 0
        
        # Reproducibility features
        features.has_docker = len(analysis_data.get("docker_files", [])) > 0
        features.has_requirements = any("requirements" in f["path"].lower() for f in analysis_data.get("build_files", []))
        
        # Code structure
        features.code_files = len(analysis_data.get("code_files", []))
        
        # Setup complexity analysis
        features.setup_complexity = self._analyze_setup_complexity(analysis_data)
        
        # Tree depth calculation
        features.tree_depth = self._calculate_tree_depth(analysis_data.get("tree_structure", []))
        
        # Availability features
        features.has_data_files = len(analysis_data.get("data_files", [])) > 0
        
        # Check for Zenodo DOI in documentation
        features.has_zenodo_doi = self._check_zenodo_doi(analysis_data)
        
        # Check for setup instructions and examples
        features.has_setup_instructions = self._check_setup_instructions(analysis_data)
        features.has_examples = self._check_examples(analysis_data)
        
        return features
    
    def _analyze_setup_complexity(self, analysis_data: Dict) -> str:
        """Analyze setup complexity based on files and documentation."""
        complexity_score = 0
        
        # Check for multiple build systems
        build_files = analysis_data.get("build_files", [])
        docker_files = analysis_data.get("docker_files", [])
        
        if docker_files:
            complexity_score += 2  # Docker reduces complexity
        
        if build_files:
            complexity_score += len(build_files) * 0.5
        
        # Check documentation for setup instructions
        doc_files = analysis_data.get("documentation_files", [])
        setup_mentions = 0
        for doc in doc_files:
            content = " ".join(doc.get("content", []))
            setup_mentions += len(re.findall(r'\b(install|setup|configuration|prerequisite)\b', content.lower()))
        
        if setup_mentions > 10:
            complexity_score += 3
        elif setup_mentions > 5:
            complexity_score += 2
        elif setup_mentions > 0:
            complexity_score += 1
        
        # Classify complexity
        if complexity_score <= 2:
            return "low"
        elif complexity_score <= 5:
            return "medium"
        else:
            return "high"
    
    def _calculate_tree_depth(self, tree_structure: List[str]) -> int:
        """Calculate maximum depth of tree structure."""
        if not tree_structure:
            return 0
        
        max_depth = 0
        for item in tree_structure:
            if isinstance(item, str):
                depth = item.count("│") + item.count("├") + item.count("└")
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _check_zenodo_doi(self, analysis_data: Dict) -> bool:
        """Check if artifact mentions Zenodo DOI."""
        doc_files = analysis_data.get("documentation_files", [])
        
        for doc in doc_files:
            content = " ".join(doc.get("content", []))
            if re.search(r'zenodo\.org|doi\.org.*zenodo|10\.5281/zenodo', content.lower()):
                return True
        
        return False
    
    def _check_setup_instructions(self, analysis_data: Dict) -> bool:
        """Check if artifact has setup instructions."""
        doc_files = analysis_data.get("documentation_files", [])
        
        for doc in doc_files:
            content = " ".join(doc.get("content", []))
            if re.search(r'(installation|setup|getting started|how to run|usage)', content.lower()):
                return True
        
        return False
    
    def _check_examples(self, analysis_data: Dict) -> bool:
        """Check if artifact has examples."""
        doc_files = analysis_data.get("documentation_files", [])
        
        for doc in doc_files:
            content = " ".join(doc.get("content", []))
            if re.search(r'(example|demo|tutorial|sample)', content.lower()):
                return True
        
        return False
    
    def _build_kg_from_analysis(self, analysis_data: Dict, features: ArtifactFeatures) -> Dict:
        """Build knowledge graph from analysis data."""
        try:
            artifact_name = analysis_data.get("artifact_name", "unknown")
            
            # Create artifact node with evaluation features
            artifact_node = self._create_evaluation_artifact_node(artifact_name, analysis_data, features)
            
            # Create documentation nodes
            doc_nodes = self._create_documentation_nodes(analysis_data, artifact_node)
            
            # Create code structure nodes
            code_nodes = self._create_code_structure_nodes(analysis_data, artifact_node)
            
            # Create dependency nodes
            dep_nodes = self._create_dependency_nodes(analysis_data, artifact_node)
            
            total_nodes = 1 + len(doc_nodes) + len(code_nodes) + len(dep_nodes)
            
            return {
                "success": True,
                "nodes_created": total_nodes,
                "artifact_node": artifact_node
            }
            
        except Exception as e:
            logger.error(f"Error building KG from analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "nodes_created": 0
            }
    
    def _create_evaluation_artifact_node(self, artifact_name: str, analysis_data: Dict, features: ArtifactFeatures) -> Node:
        """Create artifact node with evaluation features."""
        artifact_hash = hashlib.md5(artifact_name.encode()).hexdigest()[:8]
        
        # Determine artifact type
        artifact_type = self._classify_artifact_type(analysis_data)
        
        artifact_node = Node(
            "Artifact",
            name=artifact_name,
            hash=artifact_hash,
            created_at=datetime.now().isoformat(),
            artifact_type=artifact_type,
            
            # Feature properties
            has_readme=features.has_readme,
            readme_length=features.readme_length,
            has_license=features.has_license,
            has_docker=features.has_docker,
            has_zenodo_doi=features.has_zenodo_doi,
            has_setup_instructions=features.has_setup_instructions,
            has_examples=features.has_examples,
            total_files=features.total_files,
            code_files=features.code_files,
            repo_size_mb=features.repo_size_mb,
            setup_complexity=features.setup_complexity,
            tree_depth=features.tree_depth,
            
            # Analysis metadata
            extraction_method=analysis_data.get("extraction_method", "unknown"),
            analysis_performed=analysis_data.get("analysis_performed", False)
        )
        
        # Use merge to avoid duplicates
        existing = self.node_matcher.match("Artifact", name=artifact_name).first()
        if existing:
            existing.update(artifact_node)
            self.graph.push(existing)
            return existing
        else:
            self.graph.create(artifact_node)
            return artifact_node
    
    def _classify_artifact_type(self, analysis_data: Dict) -> str:
        """Classify artifact type based on content."""
        doc_files = analysis_data.get("documentation_files", [])
        code_files = analysis_data.get("code_files", [])
        data_files = analysis_data.get("data_files", [])
        
        # Check documentation for type hints
        doc_content = ""
        for doc in doc_files:
            doc_content += " ".join(doc.get("content", []))
        
        doc_content_lower = doc_content.lower()
        
        if re.search(r'\b(dataset|data|benchmark|corpus)\b', doc_content_lower):
            return "dataset"
        elif re.search(r'\b(replication|reproduction|replicate)\b', doc_content_lower):
            return "replication"
        elif re.search(r'\b(tool|framework|library|software)\b', doc_content_lower):
            return "tool"
        elif code_files and not data_files:
            return "software"
        elif data_files and not code_files:
            return "dataset"
        else:
            return "mixed"
    
    def _create_documentation_nodes(self, analysis_data: Dict, artifact_node: Node) -> List[Node]:
        """Create documentation nodes and relationships."""
        doc_nodes = []
        
        for doc_file in analysis_data.get("documentation_files", []):
            doc_node = Node(
                "Documentation",
                path=doc_file["path"],
                content_length=sum(len(line) for line in doc_file.get("content", [])),
                sections=len(doc_file.get("content", [])),
                doc_type=self._classify_doc_type(doc_file["path"])
            )
            
            self.graph.create(doc_node)
            self.graph.create(Relationship(artifact_node, "HAS_DOCUMENTATION", doc_node))
            doc_nodes.append(doc_node)
        
        return doc_nodes
    
    def _classify_doc_type(self, path: str) -> str:
        """Classify documentation type."""
        path_lower = path.lower()
        
        if "readme" in path_lower:
            return "readme"
        elif "license" in path_lower:
            return "license"
        elif "changelog" in path_lower:
            return "changelog"
        elif "api" in path_lower or "reference" in path_lower:
            return "api"
        else:
            return "general"
    
    def _create_code_structure_nodes(self, analysis_data: Dict, artifact_node: Node) -> List[Node]:
        """Create code structure nodes."""
        code_nodes = []
        
        for code_file in analysis_data.get("code_files", []):
            code_node = Node(
                "CodeFile",
                path=code_file["path"],
                content_length=sum(len(line) for line in code_file.get("content", [])),
                language=self._detect_language(code_file["path"])
            )
            
            self.graph.create(code_node)
            self.graph.create(Relationship(artifact_node, "HAS_CODE", code_node))
            code_nodes.append(code_node)
        
        return code_nodes
    
    def _detect_language(self, path: str) -> str:
        """Detect programming language from file path."""
        ext = Path(path).suffix.lower()
        
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".r": "r",
            ".sh": "shell",
            ".sql": "sql"
        }
        
        return language_map.get(ext, "unknown")
    
    def _create_dependency_nodes(self, analysis_data: Dict, artifact_node: Node) -> List[Node]:
        """Create dependency nodes."""
        dep_nodes = []
        
        # Check for various dependency files
        dependency_indicators = [
            "requirements.txt",
            "package.json",
            "pom.xml",
            "build.gradle",
            "setup.py",
            "pyproject.toml"
        ]
        
        build_files = analysis_data.get("build_files", [])
        for build_file in build_files:
            if any(indicator in build_file.get("path", "") for indicator in dependency_indicators):
                dep_node = Node(
                    "DependencyFile",
                    path=build_file["path"],
                    type=self._classify_dependency_file(build_file["path"])
                )
                
                self.graph.create(dep_node)
                self.graph.create(Relationship(artifact_node, "HAS_DEPENDENCIES", dep_node))
                dep_nodes.append(dep_node)
        
        return dep_nodes
    
    def _classify_dependency_file(self, path: str) -> str:
        """Classify dependency file type."""
        path_lower = path.lower()
        
        if "requirements" in path_lower:
            return "python_requirements"
        elif "package.json" in path_lower:
            return "npm_package"
        elif "pom.xml" in path_lower:
            return "maven_pom"
        elif "build.gradle" in path_lower:
            return "gradle_build"
        elif "setup.py" in path_lower:
            return "python_setup"
        elif "pyproject.toml" in path_lower:
            return "python_pyproject"
        else:
            return "unknown"
    
    def _perform_semantic_analysis(self, analysis_data: Dict) -> Dict[str, Any]:
        """Perform LLM-based semantic analysis."""
        if not self.openai_client:
            logger.warning("OpenAI client not available, skipping semantic analysis")
            return {"available": False, "reason": "OpenAI client not configured"}
        
        try:
            # Extract documentation content
            doc_content = self._extract_documentation_content(analysis_data)
            
            if not doc_content:
                return {"available": False, "reason": "No documentation content available"}
            
            # Analyze with LLM
            analysis_result = self._analyze_with_llm(doc_content)
            
            return {
                "available": True,
                "analysis": analysis_result,
                "content_length": len(doc_content)
            }
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return {"available": False, "reason": str(e)}
    
    def _extract_documentation_content(self, analysis_data: Dict) -> str:
        """Extract documentation content for LLM analysis."""
        doc_content = ""
        
        for doc_file in analysis_data.get("documentation_files", []):
            if "readme" in doc_file["path"].lower():
                doc_content += "\n".join(doc_file.get("content", []))
                break
        
        # If no README, use first documentation file
        if not doc_content and analysis_data.get("documentation_files"):
            first_doc = analysis_data["documentation_files"][0]
            doc_content = "\n".join(first_doc.get("content", []))
        
        return doc_content
    
    def _analyze_with_llm(self, content: str) -> Dict[str, Any]:
        """Analyze content with LLM."""
        prompt = f"""
        Analyze the following research artifact documentation and provide a structured evaluation:

        Documentation:
        {content[:4000]}  # Limit content to avoid token limits

        Please evaluate the following aspects and provide scores (1-10):

        1. Documentation Quality: How clear and comprehensive is the documentation?
        2. Reproducibility: How well does it support reproduction of results?
        3. Usability: How easy would it be for someone to use this artifact?
        4. Availability: How accessible is the artifact and its components?
        5. Technical Soundness: How technically sound does the approach appear?

        Also classify the artifact type (dataset, tool, replication, software, etc.) and identify key features.

        Provide your response in JSON format with the following structure:
        {{
            "documentation_quality": <score>,
            "reproducibility": <score>,
            "usability": <score>,
            "availability": <score>,
            "technical_soundness": <score>,
            "artifact_type": "<type>",
            "key_features": ["<feature1>", "<feature2>", ...],
            "strengths": ["<strength1>", "<strength2>", ...],
            "weaknesses": ["<weakness1>", "<weakness2>", ...],
            "summary": "<brief summary>"
        }}
        """
        
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, return raw content
                return {"raw_response": content}
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_evaluation_scores(self, features: ArtifactFeatures, semantic_analysis: Dict) -> Dict[str, float]:
        """Calculate evaluation scores based on features and semantic analysis."""
        scores = {}
        
        # Documentation Quality Score
        doc_score = 0.0
        if features.has_readme:
            doc_score += 0.4
        if features.readme_length > 500:
            doc_score += 0.3
        if features.has_license:
            doc_score += 0.2
        if features.documentation_sections > 3:
            doc_score += 0.1
        
        # Add LLM analysis if available
        if semantic_analysis.get("available") and "analysis" in semantic_analysis:
            llm_doc_score = semantic_analysis["analysis"].get("documentation_quality", 5) / 10.0
            doc_score = (doc_score + llm_doc_score) / 2
        
        scores["documentation_quality"] = min(doc_score, 1.0)
        
        # Reproducibility Score
        repro_score = 0.0
        if features.has_docker:
            repro_score += 0.3
        if features.has_setup_instructions:
            repro_score += 0.3
        if features.has_examples:
            repro_score += 0.2
        if features.setup_complexity == "low":
            repro_score += 0.2
        elif features.setup_complexity == "medium":
            repro_score += 0.1
        
        # Add LLM analysis if available
        if semantic_analysis.get("available") and "analysis" in semantic_analysis:
            llm_repro_score = semantic_analysis["analysis"].get("reproducibility", 5) / 10.0
            repro_score = (repro_score + llm_repro_score) / 2
        
        scores["reproducibility"] = min(repro_score, 1.0)
        
        # Availability Score
        avail_score = 0.0
        if features.has_zenodo_doi:
            avail_score += 0.4
        if features.has_data_files:
            avail_score += 0.3
        if features.code_files > 0:
            avail_score += 0.3
        
        # Add LLM analysis if available
        if semantic_analysis.get("available") and "analysis" in semantic_analysis:
            llm_avail_score = semantic_analysis["analysis"].get("availability", 5) / 10.0
            avail_score = (avail_score + llm_avail_score) / 2
        
        scores["availability"] = min(avail_score, 1.0)
        
        # Code Structure Score
        structure_score = 0.0
        if features.code_files > 0:
            structure_score += 0.4
        if features.total_files > 5:
            structure_score += 0.2
        if features.tree_depth > 2:
            structure_score += 0.2
        if features.repo_size_mb < 100:  # Reasonable size
            structure_score += 0.2
        
        scores["code_structure"] = min(structure_score, 1.0)
        
        # Complexity Score (inverse - lower complexity is better)
        complexity_score = 1.0
        if features.setup_complexity == "high":
            complexity_score -= 0.5
        elif features.setup_complexity == "medium":
            complexity_score -= 0.2
        
        if features.repo_size_mb > 500:  # Very large repository
            complexity_score -= 0.3
        
        scores["complexity"] = max(complexity_score, 0.0)
        
        return scores
    
    def _predict_acceptance(self, features: ArtifactFeatures, scores: Dict[str, float]) -> Dict[str, Any]:
        """Predict acceptance likelihood based on features and scores."""
        # Calculate weighted score
        weighted_score = sum(
            scores.get(criteria, 0) * weight 
            for criteria, weight in self.evaluation_weights.items()
        )
        
        # Simple threshold-based prediction
        if weighted_score >= 0.8:
            likelihood = "high"
            confidence = 0.9
        elif weighted_score >= 0.6:
            likelihood = "medium"
            confidence = 0.7
        else:
            likelihood = "low"
            confidence = 0.6
        
        return {
            "likelihood": likelihood,
            "confidence": confidence,
            "score": weighted_score,
            "threshold_high": 0.8,
            "threshold_medium": 0.6
        }
    
    def _generate_recommendations(self, features: ArtifactFeatures, scores: Dict[str, float]) -> List[Dict[str, str]]:
        """Generate recommendations for improving the artifact."""
        recommendations = []
        
        # Documentation recommendations
        if scores.get("documentation_quality", 0) < 0.7:
            if not features.has_readme:
                recommendations.append({
                    "category": "documentation",
                    "priority": "high",
                    "recommendation": "Add a comprehensive README file explaining the project"
                })
            if features.readme_length < 500:
                recommendations.append({
                    "category": "documentation",
                    "priority": "medium",
                    "recommendation": "Expand README with more detailed information"
                })
        
        # Reproducibility recommendations
        if scores.get("reproducibility", 0) < 0.7:
            if not features.has_docker:
                recommendations.append({
                    "category": "reproducibility",
                    "priority": "high",
                    "recommendation": "Add Docker configuration for easy environment setup"
                })
            if not features.has_setup_instructions:
                recommendations.append({
                    "category": "reproducibility",
                    "priority": "high",
                    "recommendation": "Add clear setup and installation instructions"
                })
        
        # Availability recommendations
        if scores.get("availability", 0) < 0.7:
            if not features.has_zenodo_doi:
                recommendations.append({
                    "category": "availability",
                    "priority": "medium",
                    "recommendation": "Consider archiving the artifact on Zenodo for persistent access"
                })
        
        return recommendations
    
    def _update_artifact_with_evaluation(
        self, 
        artifact_name: str, 
        features: ArtifactFeatures, 
        scores: Dict[str, float],
        prediction: Dict[str, Any],
        recommendations: List[Dict[str, str]]
    ):
        """Update artifact node with evaluation results."""
        try:
            artifact_node = self.node_matcher.match("Artifact", name=artifact_name).first()
            
            if artifact_node:
                # Update with evaluation results
                artifact_node.update({
                    "evaluation_score": prediction["score"],
                    "acceptance_prediction": prediction["likelihood"],
                    "prediction_confidence": prediction["confidence"],
                    "doc_quality_score": scores.get("documentation_quality", 0),
                    "reproducibility_score": scores.get("reproducibility", 0),
                    "availability_score": scores.get("availability", 0),
                    "code_structure_score": scores.get("code_structure", 0),
                    "complexity_score": scores.get("complexity", 0),
                    "total_recommendations": len(recommendations),
                    "evaluated_at": datetime.now().isoformat()
                })
                
                self.graph.push(artifact_node)
                
                # Create recommendation nodes
                for rec in recommendations:
                    rec_node = Node(
                        "Recommendation",
                        category=rec["category"],
                        priority=rec["priority"],
                        recommendation=rec["recommendation"],
                        created_at=datetime.now().isoformat()
                    )
                    
                    self.graph.create(rec_node)
                    self.graph.create(Relationship(artifact_node, "HAS_RECOMMENDATION", rec_node))
                
                logger.info(f"Updated artifact {artifact_name} with evaluation results")
            
        except Exception as e:
            logger.error(f"Error updating artifact with evaluation: {e}")
    
    def get_evaluation_summary(self, artifact_name: str) -> Dict[str, Any]:
        """Get comprehensive evaluation summary for an artifact."""
        try:
            query = """
            MATCH (a:Artifact {name: $artifact_name})
            OPTIONAL MATCH (a)-[:HAS_RECOMMENDATION]->(r:Recommendation)
            RETURN a, collect(r) as recommendations
            """
            
            result = self.graph.run(query, artifact_name=artifact_name).data()
            
            if not result:
                return {"error": f"Artifact {artifact_name} not found"}
            
            artifact = result[0]["a"]
            recommendations = result[0]["recommendations"]
            
            return {
                "artifact_name": artifact_name,
                "evaluation_score": artifact.get("evaluation_score", 0),
                "acceptance_prediction": artifact.get("acceptance_prediction", "unknown"),
                "prediction_confidence": artifact.get("prediction_confidence", 0),
                "scores": {
                    "documentation_quality": artifact.get("doc_quality_score", 0),
                    "reproducibility": artifact.get("reproducibility_score", 0),
                    "availability": artifact.get("availability_score", 0),
                    "code_structure": artifact.get("code_structure_score", 0),
                    "complexity": artifact.get("complexity_score", 0)
                },
                "recommendations": [
                    {
                        "category": rec.get("category", "unknown"),
                        "priority": rec.get("priority", "unknown"),
                        "recommendation": rec.get("recommendation", "")
                    }
                    for rec in recommendations
                ],
                "artifact_features": {
                    "has_readme": artifact.get("has_readme", False),
                    "has_docker": artifact.get("has_docker", False),
                    "has_zenodo_doi": artifact.get("has_zenodo_doi", False),
                    "setup_complexity": artifact.get("setup_complexity", "unknown"),
                    "artifact_type": artifact.get("artifact_type", "unknown"),
                    "total_files": artifact.get("total_files", 0),
                    "repo_size_mb": artifact.get("repo_size_mb", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting evaluation summary: {e}")
            return {"error": str(e)}
    
    def compare_artifacts(self, artifact_names: List[str]) -> Dict[str, Any]:
        """Compare multiple artifacts based on evaluation scores."""
        try:
            query = """
            MATCH (a:Artifact)
            WHERE a.name IN $artifact_names
            RETURN a.name as name, 
                   a.evaluation_score as score,
                   a.acceptance_prediction as prediction,
                   a.doc_quality_score as doc_score,
                   a.reproducibility_score as repro_score,
                   a.availability_score as avail_score
            ORDER BY a.evaluation_score DESC
            """
            
            results = self.graph.run(query, artifact_names=artifact_names).data()
            
            if not results:
                return {"error": "No artifacts found"}
            
            comparison = {
                "artifacts": results,
                "best_artifact": results[0]["name"] if results else None,
                "average_score": sum(r["score"] or 0 for r in results) / len(results) if results else 0,
                "score_distribution": {
                    "high": len([r for r in results if (r["score"] or 0) >= 0.8]),
                    "medium": len([r for r in results if 0.6 <= (r["score"] or 0) < 0.8]),
                    "low": len([r for r in results if (r["score"] or 0) < 0.6])
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing artifacts: {e}")
            return {"error": str(e)}
    
    def export_evaluation_report(self, artifact_name: str, output_path: str) -> str:
        """Export comprehensive evaluation report."""
        try:
            summary = self.get_evaluation_summary(artifact_name)
            
            if "error" in summary:
                return f"Error: {summary['error']}"
            
            # Create HTML report
            html_report = self._generate_html_report(summary)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            logger.info(f"Evaluation report exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting evaluation report: {e}")
            return f"Error: {str(e)}"
    
    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate HTML evaluation report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Artifact Evaluation Report: {summary['artifact_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; color: #2e7d32; }}
                .section {{ margin: 20px 0; }}
                .score-bar {{ background-color: #e0e0e0; height: 20px; border-radius: 10px; }}
                .score-fill {{ height: 100%; border-radius: 10px; }}
                .high {{ background-color: #4caf50; }}
                .medium {{ background-color: #ff9800; }}
                .low {{ background-color: #f44336; }}
                .recommendations {{ background-color: #fff3e0; padding: 15px; border-radius: 5px; }}
                .rec-item {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ff9800; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Artifact Evaluation Report</h1>
                <h2>{summary['artifact_name']}</h2>
                <div class="score">Overall Score: {summary['evaluation_score']:.2f}</div>
                <div>Acceptance Prediction: {summary['acceptance_prediction'].upper()}</div>
                <div>Confidence: {summary['prediction_confidence']:.2f}</div>
            </div>
            
            <div class="section">
                <h3>Evaluation Scores</h3>
                {self._generate_score_bars(summary['scores'])}
            </div>
            
            <div class="section">
                <h3>Artifact Features</h3>
                {self._generate_features_table(summary['artifact_features'])}
            </div>
            
            <div class="section">
                <h3>Recommendations</h3>
                <div class="recommendations">
                    {self._generate_recommendations_html(summary['recommendations'])}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_score_bars(self, scores: Dict[str, float]) -> str:
        """Generate HTML for score visualization."""
        html = ""
        for criteria, score in scores.items():
            score_class = "high" if score >= 0.8 else "medium" if score >= 0.6 else "low"
            html += f"""
            <div style="margin: 10px 0;">
                <div>{criteria.replace('_', ' ').title()}: {score:.2f}</div>
                <div class="score-bar">
                    <div class="score-fill {score_class}" style="width: {score*100}%;"></div>
                </div>
            </div>
            """
        return html
    
    def _generate_features_table(self, features: Dict[str, Any]) -> str:
        """Generate HTML table for artifact features."""
        html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        
        for key, value in features.items():
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_recommendations_html(self, recommendations: List[Dict[str, str]]) -> str:
        """Generate HTML for recommendations."""
        if not recommendations:
            return "<p>No recommendations available.</p>"
        
        html = ""
        for rec in recommendations:
            priority_color = {"high": "#f44336", "medium": "#ff9800", "low": "#4caf50"}.get(rec["priority"], "#757575")
            html += f"""
            <div class="rec-item" style="border-left-color: {priority_color};">
                <strong>{rec['category'].title()} ({rec['priority']} priority)</strong><br>
                {rec['recommendation']}
            </div>
            """
        
        return html
    
    def close(self):
        """Close connections."""
        if self.kg_builder:
            self.kg_builder.close()


def main():
    """Example usage of the Artifact Evaluation System."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Artifact Evaluation System")
    parser.add_argument("json_file", help="Path to artifact analysis JSON file")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--openai-api-key", help="OpenAI API key for LLM analysis")
    parser.add_argument("--output-report", help="Output path for evaluation report")
    
    args = parser.parse_args()
    
    # Create evaluation system
    evaluator = ArtifactEvaluationSystem(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        openai_api_key=args.openai_api_key
    )
    
    try:
        # Evaluate artifact
        result = evaluator.evaluate_artifact_from_json(args.json_file)
        
        if result["success"]:
            print(f"✅ Evaluation completed for: {result['artifact_name']}")
            print(f"Overall Score: {result['evaluation_scores']}")
            print(f"Acceptance Prediction: {result['acceptance_prediction']}")
            
            # Export report if requested
            if args.output_report:
                report_path = evaluator.export_evaluation_report(
                    result['artifact_name'], 
                    args.output_report
                )
                print(f"📄 Report exported to: {report_path}")
        else:
            print(f"❌ Evaluation failed: {result['error']}")
    
    finally:
        evaluator.close()


if __name__ == "__main__":
    main() 