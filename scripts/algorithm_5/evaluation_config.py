#!/usr/bin/env python3
"""
Configuration file for the Artifact Evaluation System
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class EvaluationConfig:
    """Configuration for artifact evaluation system."""
    
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "12345678"
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.3
    
    # Evaluation Weights
    evaluation_weights: Dict[str, float] = field(default_factory=lambda: {
        "documentation_quality": 0.25,
        "reproducibility": 0.30,
        "availability": 0.20,
        "code_structure": 0.15,
        "complexity": 0.10
    })
    
    # Scoring Thresholds
    score_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high_acceptance": 0.8,
        "medium_acceptance": 0.6,
        "low_acceptance": 0.4
    })
    
    # Feature Weights for Scoring
    feature_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "documentation_quality": {
            "has_readme": 0.4,
            "readme_length_bonus": 0.3,
            "has_license": 0.2,
            "documentation_sections": 0.1
        },
        "reproducibility": {
            "has_docker": 0.3,
            "has_setup_instructions": 0.3,
            "has_examples": 0.2,
            "low_complexity_bonus": 0.2
        },
        "availability": {
            "has_zenodo_doi": 0.4,
            "has_data_files": 0.3,
            "has_code_files": 0.3
        },
        "code_structure": {
            "has_code_files": 0.4,
            "file_count_bonus": 0.2,
            "tree_depth_bonus": 0.2,
            "reasonable_size_bonus": 0.2
        }
    })
    
    # File Classification Patterns
    file_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "documentation": ["readme", "documentation", "docs", "guide"],
        "license": ["license", "copyright", "legal"],
        "setup": ["setup", "install", "configure", "requirements"],
        "examples": ["example", "demo", "tutorial", "sample"],
        "tests": ["test", "spec", "check"],
        "data": ["data", "dataset", "corpus", "benchmark"]
    })
    
    # Artifact Type Keywords
    artifact_type_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "dataset": ["dataset", "data", "benchmark", "corpus", "collection"],
        "replication": ["replication", "reproduction", "replicate", "reproduce"],
        "tool": ["tool", "framework", "library", "software", "application"],
        "analysis": ["analysis", "study", "experiment", "evaluation"],
        "benchmark": ["benchmark", "evaluation", "comparison", "performance"]
    })
    
    # Quality Indicators
    quality_indicators: Dict[str, List[str]] = field(default_factory=lambda: {
        "high_quality": [
            "doi", "zenodo", "archived", "peer-reviewed",
            "docker", "containerized", "reproducible",
            "documentation", "tutorial", "guide"
        ],
        "reproducibility": [
            "reproducible", "replicable", "setup", "installation",
            "docker", "container", "environment", "dependencies"
        ],
        "availability": [
            "available", "accessible", "download", "repository",
            "github", "gitlab", "zenodo", "figshare"
        ]
    })
    
    # Complexity Indicators
    complexity_indicators: Dict[str, List[str]] = field(default_factory=lambda: {
        "high_complexity": [
            "complex", "advanced", "prerequisite", "dependency",
            "configure", "compilation", "build"
        ],
        "low_complexity": [
            "simple", "easy", "straightforward", "docker",
            "one-click", "automated", "script"
        ]
    })
    
    # Export Settings
    export_settings: Dict[str, any] = field(default_factory=lambda: {
        "html_template": "default",
        "include_visualizations": True,
        "include_recommendations": True,
        "include_comparison": True
    })
    
    # Logging Configuration
    logging_config: Dict[str, any] = field(default_factory=lambda: {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "evaluation.log"
    })
    
    @classmethod
    def from_file(cls, config_path: str) -> 'EvaluationConfig':
        """Load configuration from file."""
        import json
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def to_file(self, config_path: str):
        """Save configuration to file."""
        import json
        
        config_data = {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": self.neo4j_password,
            "openai_api_key": self.openai_api_key,
            "llm_model": self.llm_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "evaluation_weights": self.evaluation_weights,
            "score_thresholds": self.score_thresholds,
            "feature_weights": self.feature_weights,
            "file_patterns": self.file_patterns,
            "artifact_type_keywords": self.artifact_type_keywords,
            "quality_indicators": self.quality_indicators,
            "complexity_indicators": self.complexity_indicators,
            "export_settings": self.export_settings,
            "logging_config": self.logging_config
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_env_config(self) -> 'EvaluationConfig':
        """Get configuration from environment variables."""
        config = EvaluationConfig()
        
        # Override with environment variables if available
        config.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        config.neo4j_user = os.getenv("NEO4J_USER", self.neo4j_user)
        config.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        config.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        
        return config


# Default configuration instance
DEFAULT_CONFIG = EvaluationConfig()

# Configuration for different artifact types
ARTIFACT_TYPE_CONFIGS = {
    "dataset": EvaluationConfig(
        evaluation_weights={
            "documentation_quality": 0.30,
            "reproducibility": 0.20,
            "availability": 0.35,
            "code_structure": 0.10,
            "complexity": 0.05
        }
    ),
    "replication": EvaluationConfig(
        evaluation_weights={
            "documentation_quality": 0.25,
            "reproducibility": 0.40,
            "availability": 0.20,
            "code_structure": 0.10,
            "complexity": 0.05
        }
    ),
    "tool": EvaluationConfig(
        evaluation_weights={
            "documentation_quality": 0.20,
            "reproducibility": 0.25,
            "availability": 0.15,
            "code_structure": 0.25,
            "complexity": 0.15
        }
    )
}

# Conference-specific configurations
CONFERENCE_CONFIGS = {
    "icse": EvaluationConfig(
        score_thresholds={
            "high_acceptance": 0.85,
            "medium_acceptance": 0.65,
            "low_acceptance": 0.45
        }
    ),
    "fse": EvaluationConfig(
        score_thresholds={
            "high_acceptance": 0.80,
            "medium_acceptance": 0.60,
            "low_acceptance": 0.40
        }
    ),
    "ase": EvaluationConfig(
        score_thresholds={
            "high_acceptance": 0.75,
            "medium_acceptance": 0.55,
            "low_acceptance": 0.35
        }
    )
} 