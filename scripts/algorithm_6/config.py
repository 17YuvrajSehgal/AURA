"""
Configuration for Artifact Evaluation Framework

This module contains all configuration settings for the research artifact
evaluation system including knowledge graphs, ML models, and evaluation criteria.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for Large Language Model"""
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.3
    max_tokens: int = 4000
    api_key: str = os.getenv("OPENAI_API_KEY", "")


@dataclass
class KnowledgeGraphConfig:
    """Configuration for Knowledge Graph (Neo4j)"""
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = os.getenv("NEO4J_DATABASE", "aura")

    # Graph Data Science settings
    use_gds: bool = True
    gds_graph_name: str = "artifact_graph"


@dataclass
class VectorConfig:
    """Configuration for Vector Embeddings"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    similarity_threshold: float = 0.7

    # Vector database settings
    vector_db_type: str = "faiss"  # Options: faiss, chroma, qdrant
    index_type: str = "IndexFlatIP"


@dataclass
class PatternAnalysisConfig:
    """Configuration for Pattern Analysis"""
    # Community detection algorithms
    community_algorithms: List[str] = field(default_factory=lambda: [
        "louvain", "leiden", "walktrap", "fastgreedy"
    ])

    # Centrality metrics to compute
    centrality_metrics: List[str] = field(default_factory=lambda: [
        "degree", "betweenness", "pagerank", "eigenvector"
    ])

    # Clustering parameters
    min_cluster_size: int = 5
    min_samples: int = 3
    umap_n_components: int = 2
    umap_n_neighbors: int = 15


@dataclass
class EvaluationConfig:
    """Configuration for Artifact Evaluation"""
    # Scoring weights for different aspects
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "documentation_completeness": 0.25,
        "structure_quality": 0.20,
        "reproducibility_indicators": 0.20,
        "tool_support": 0.15,
        "citation_quality": 0.10,
        "semantic_clarity": 0.10
    })

    # Required sections for high-quality artifacts
    required_sections: List[str] = field(default_factory=lambda: [
        "purpose", "setup", "usage", "requirements", "installation"
    ])

    # Bonus sections that improve scores
    bonus_sections: List[str] = field(default_factory=lambda: [
        "docker", "testing", "examples", "license", "citation"
    ])

    # Tool support indicators
    supported_tools: List[str] = field(default_factory=lambda: [
        "docker", "conda", "pip", "maven", "gradle", "cmake", "makefile"
    ])


@dataclass
class ConferenceConfig:
    """Configuration for Conference-specific Analysis"""
    # Conference categories and their characteristics
    conference_categories: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "systems": {
            "conferences": ["OSDI", "SOSP", "NSDI", "EuroSys"],
            "emphasis": ["performance", "scalability", "benchmarks"],
            "required_tools": ["docker", "build_scripts"],
            "documentation_style": "technical_detailed"
        },
        "software_engineering": {
            "conferences": ["ICSE", "FSE", "ASE", "ICSME"],
            "emphasis": ["replication", "data_analysis", "tool_demo"],
            "required_tools": ["scripts", "data_processing"],
            "documentation_style": "methodology_focused"
        },
        "programming_languages": {
            "conferences": ["PLDI", "POPL", "OOPSLA", "ICFP"],
            "emphasis": ["formal_verification", "prototype", "benchmarks"],
            "required_tools": ["compilers", "interpreters"],
            "documentation_style": "formal_precise"
        },
        "machine_learning": {
            "conferences": ["NeurIPS", "ICML", "ICLR", "AAAI"],
            "emphasis": ["reproducibility", "datasets", "model_weights"],
            "required_tools": ["jupyter", "conda", "docker"],
            "documentation_style": "experiment_focused"
        }
    })


@dataclass
class DataConfig:
    """Configuration for Data Processing"""
    # Input data paths
    accepted_artifacts_dir: str = "data/accepted_artifacts"
    conference_metadata_file: str = "data/conference_metadata.json"

    # Output paths
    analysis_output_dir: str = "output/analysis"
    models_output_dir: str = "output/models"
    reports_output_dir: str = "output/reports"

    # Processing settings
    batch_size: int = 50
    max_file_size_mb: int = 10
    supported_file_types: List[str] = field(default_factory=lambda: [
        ".md", ".txt", ".py", ".r", ".sh", ".yml", ".yaml",
        ".json", ".csv", ".dockerfile", ".license", ".rst"
    ])


@dataclass
class SystemConfig:
    """Main system configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    pattern_analysis: PatternAnalysisConfig = field(default_factory=PatternAnalysisConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    conference: ConferenceConfig = field(default_factory=ConferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # System settings
    debug_mode: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    cache_enabled: bool = True


# Global configuration instance
config = SystemConfig()

# Node types for knowledge graph
NODE_TYPES = {
    "ARTIFACT": "Artifact",
    "CONFERENCE": "Conference",
    "DOCUMENTATION": "Documentation",
    "SECTION": "Section",
    "TOOL": "Tool",
    "COMMAND": "Command",
    "DEPENDENCY": "Dependency",
    "PATTERN": "Pattern",
    "CLUSTER": "Cluster",
    "METRIC": "Metric",
    "AUTHOR": "Author",
    "REFERENCE": "Reference",
    "DATASET": "Dataset",
    "IMAGE": "Image",
    "ENTITY": "Entity"
}

# Relationship types for knowledge graph
RELATIONSHIP_TYPES = {
    "SUBMITTED_TO": "SUBMITTED_TO",
    "HAS_DOCUMENTATION": "HAS_DOCUMENTATION",
    "HAS_SECTION": "HAS_SECTION",
    "USES_TOOL": "USES_TOOL",
    "CONTAINS_COMMAND": "CONTAINS_COMMAND",
    "DEPENDS_ON": "DEPENDS_ON",
    "SIMILAR_TO": "SIMILAR_TO",
    "BELONGS_TO_CLUSTER": "BELONGS_TO_CLUSTER",
    "HAS_PATTERN": "HAS_PATTERN",
    "AUTHORED_BY": "AUTHORED_BY",
    "CITES": "CITES",
    "INCLUDES_DATASET": "INCLUDES_DATASET",
    "CONTAINS_IMAGE": "CONTAINS_IMAGE",
    "MENTIONS": "MENTIONS",
    "FOLLOWS_TEMPLATE": "FOLLOWS_TEMPLATE"
}

# Documentation quality indicators
QUALITY_INDICATORS = {
    "structure": {
        "has_toc": 2,
        "proper_headers": 3,
        "numbered_steps": 2,
        "bullet_points": 1,
        "code_blocks": 2
    },
    "completeness": {
        "installation_section": 5,
        "usage_examples": 4,
        "requirements_listed": 3,
        "license_info": 2,
        "contact_info": 1
    },
    "reproducibility": {
        "docker_support": 5,
        "conda_environment": 4,
        "requirements_file": 3,
        "build_scripts": 3,
        "test_scripts": 2
    },
    "clarity": {
        "clear_purpose": 4,
        "step_by_step": 3,
        "examples_provided": 3,
        "troubleshooting": 2,
        "faq_section": 1
    }
}

# Conference-specific weights (can be learned from data)
CONFERENCE_WEIGHTS = {
    "ICSE": {"reproducibility": 0.3, "documentation": 0.25, "novelty": 0.25, "impact": 0.2},
    "FSE": {"tool_quality": 0.3, "usability": 0.25, "documentation": 0.25, "innovation": 0.2},
    "ASE": {"automation": 0.3, "effectiveness": 0.25, "documentation": 0.25, "scope": 0.2},
    "ICSME": {"maintainability": 0.3, "documentation": 0.25, "tool_support": 0.25, "validation": 0.2}
}
