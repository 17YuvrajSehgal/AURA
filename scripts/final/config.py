"""
Configuration file for the README Documentation Generator
"""

import os
from dataclasses import dataclass
from typing import List

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
    password: str = os.getenv("NEO4J_PASSWORD", "12345678")
    database: str = os.getenv("NEO4J_DATABASE", "aura")


@dataclass
class VectorConfig:
    """Configuration for Vector Embeddings"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    index_type: str = "FAISS"


@dataclass
class AuraConfig:
    """Configuration for Aura report generation"""
    sections: List[str] = None
    template_dir: str = "templates"
    output_dir: str = "generated_evaluation"

    def __post_init__(self):
        if self.sections is None:
            self.sections = [
                "accessibility",
                "documentation",
                "experimental",
                "functionality",
                "reproducibility",
                "usability"
            ]


# Weighted scoring system for artifact evaluation
DIMENSION_WEIGHTS = {
    "reproducibility": 0.2623,  # 26.23%
    "documentation": 0.1182,   # 11.82%
    "accessibility": 0.1586,   # 15.86%
    "usability": 0.2967,       # 29.67%
    "experimental": 0.0999,    # 9.99%
    "functionality": 0.0643,   # 6.43%
}

# Acceptance probability thresholds (configurable)
ACCEPTANCE_THRESHOLDS = {
    "excellent": 0.85,      # 85%+ - Very High Chance
    "good": 0.70,          # 70-85% - Good Chance  
    "acceptable": 0.55,    # 55-70% - Moderate Chance
    "needs_improvement": 0.40,  # 40-55% - Low Chance
    # Below 40% - Very Low Chance
}


@dataclass
class SystemConfig:
    """Main system configuration"""
    llm: LLMConfig = LLMConfig()
    knowledge_graph: KnowledgeGraphConfig = KnowledgeGraphConfig()
    vector: VectorConfig = VectorConfig()
    aura: AuraConfig = AuraConfig()

    # File processing settings
    max_file_size_mb: int = 10
    supported_file_types: List[str] = None

    # Processing settings
    batch_size: int = 100
    max_retries: int = 3

    def __post_init__(self):
        if self.supported_file_types is None:
            self.supported_file_types = [
                ".md", ".txt", ".py", ".r", ".sh", ".yml", ".yaml",
                ".json", ".csv", ".dockerfile", ".license"
            ]


# Global configuration instance
config = SystemConfig()

# Node types for knowledge graph
NODE_TYPES = {
    "ARTIFACT": "Artifact",
    "FILE": "File",
    "TOOL": "Tool",
    "COMMAND": "Command",
    "DATASET": "Dataset",
    "OUTPUT": "Output",
    "SECTION": "Section",
    "DEPENDENCY": "Dependency"
}

# Relationship types for knowledge graph
RELATIONSHIP_TYPES = {
    "CONTAINS": "CONTAINS",
    "DEPENDS_ON": "DEPENDS_ON",
    "GENERATES": "GENERATES",
    "DESCRIBES": "DESCRIBES",
    "REQUIRES": "REQUIRES",
    "PRODUCES": "PRODUCES",
    "PART_OF": "PART_OF",
    "REFERENCES": "REFERENCES"
}

# Prompt templates directory structure
PROMPT_TEMPLATES = {
    "accessibility": "accessibility.txt",
    "documentation": "documentation.txt",
    "experimental": "experimental.txt",
    "functionality": "functionality.txt",
    "reproducibility": "reproducibility.txt",
    "usability": "usability.txt",
}
