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
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = os.getenv("NEO4J_DATABASE", "aura")


@dataclass
class VectorConfig:
    """Configuration for Vector Embeddings"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    index_type: str = "FAISS"


@dataclass
class READMEConfig:
    """Configuration for README generation"""
    sections: List[str] = None
    template_dir: str = "templates"
    output_dir: str = "generated_readmes"

    def __post_init__(self):
        if self.sections is None:
            self.sections = [
                "title_and_purpose",
                "artifact_available",
                "artifact_reusable",
                "provenance",
                "setup",
                "usage",
                "outputs",
                "structure",
                "license",
                "attribution"
            ]


@dataclass
class SystemConfig:
    """Main system configuration"""
    llm: LLMConfig = LLMConfig()
    knowledge_graph: KnowledgeGraphConfig = KnowledgeGraphConfig()
    vector: VectorConfig = VectorConfig()
    readme: READMEConfig = READMEConfig()

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
    "title_and_purpose": "title_purpose.txt",
    "artifact_available": "artifact_available.txt",
    "artifact_reusable": "artifact_reusable.txt",
    "provenance": "provenance.txt",
    "setup": "setup.txt",
    "usage": "usage.txt",
    "outputs": "outputs.txt",
    "structure": "structure.txt",
    "license": "license.txt",
    "attribution": "attribution.txt"
}

# Section priorities for README generation
SECTION_PRIORITIES = {
    "title_and_purpose": 1,
    "artifact_available": 2,
    "artifact_reusable": 3,
    "provenance": 4,
    "setup": 5,
    "usage": 6,
    "outputs": 7,
    "structure": 8,
    "license": 9,
    "attribution": 10
}
