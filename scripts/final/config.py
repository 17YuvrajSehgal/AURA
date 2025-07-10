"""
Configuration file for the README Documentation Generator
"""

import os
import re
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
    database: str = os.getenv("NEO4J_DATABASE_AURA", "aura")


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
    "DEPENDENCY": "Dependency",
    "LICENSE": "License",
    "BUILD": "Build",
    "STRUCTURE": "Structure"

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
    'MENTIONS': 'MENTIONS',
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

LANGUAGE_DEPENDENCY_PATTERNS = {
    'python': [r'^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))'],
    'java': [r'import\s+([\w\.]+);'],
    'javascript': [r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]', r'require\([\'"]([^\'"]+)[\'"]\)'],
    'c_cpp': [r'#include\s+[<"]([\w\.\/]+)[">]'],
    'r': [r'library\(["\']?(\w+)["\']?\)', r'require\(["\']?(\w+)["\']?\)'],
    'bash': [r'^\s*(\w+)=.*', r'\b(\w+)\s+.*'],  # Crude, useful for scripts
    'go': [r'import\s+\(?\s*"([\w\/]+)"'],
    'ruby': [r'require\s+[\'"]([^\'"]+)[\'"]'],
    'php': [r'use\s+([\w\\]+);'],
}

LICENSE_PATTERNS = {
    "MIT": re.compile(r'\bmit\b.*license|\bpermission\s+is\s+hereby\s+granted\b', re.IGNORECASE),
    "Apache-2.0": re.compile(r'\bapache\s+license\b.*(version\s+2\.0)?', re.IGNORECASE),
    "GPL-3.0": re.compile(r'\bgnu\s+(general\s+public\s+license|gpl)\b.*(version\s*3)', re.IGNORECASE),
    "GPL-2.0": re.compile(r'\bgnu\s+(general\s+public\s+license|gpl)\b.*(version\s*2)', re.IGNORECASE),
    "AGPL-3.0": re.compile(r'\bgnu\s+affero\s+general\s+public\s+license\b.*(version\s*3)', re.IGNORECASE),
    "LGPL": re.compile(r'\bgnu\s+(lesser|library)\s+general\s+public\s+license\b', re.IGNORECASE),
    "BSD-2-Clause": re.compile(r'\bbsd\s+2[- ]clause\b|\bredistribution\s+and\s+use\s+in\s+source\s+and\s+binary\s+forms\b.*with\s+or\s+without\s+modification', re.IGNORECASE),
    "BSD-3-Clause": re.compile(r'\bbsd\s+3[- ]clause\b|\bneither\s+the\s+name\s+of\b.*\bendorse\b', re.IGNORECASE),
    "MPL-2.0": re.compile(r'\bmozilla\s+public\s+license\b.*(version\s*2\.0)?', re.IGNORECASE),
    "EPL-2.0": re.compile(r'\beclipse\s+public\s+license\b.*(version\s*2\.0)?', re.IGNORECASE),
    "Unlicense": re.compile(r'\bthis\s+is\s+free\s+and\s+unencumbered\s+software\s+released\s+into\s+the\s+public\s+domain\b', re.IGNORECASE),
    "CC0": re.compile(r'\bcreative\s+commons\s+zero\b.*\bcc0\b', re.IGNORECASE),
    "Proprietary": re.compile(r'\bproprietary\b.*(rights\s+reserved|not\s+redistributable)', re.IGNORECASE),
}