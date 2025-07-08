"""
⚙️ AURA Framework Configuration
Configuration settings for the AURA artifact evaluation framework.

This module contains all the configuration parameters, constants, and settings
used across the different phases of the AURA framework.
"""

import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration"""
    # Neo4j Configuration
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "12345678"
    database: str = "aura"

    # Alternative: Use environment variables
    def __post_init__(self):
        self.uri = os.getenv('NEO4J_URI', self.uri)
        self.username = os.getenv('NEO4J_USERNAME', self.username)
        self.password = os.getenv('NEO4J_PASSWORD', self.password)
        self.database = os.getenv('NEO4J_DATABASE', self.database)


@dataclass
class ProcessingConfig:
    """Processing configuration"""
    max_workers: int = 4
    batch_size: int = 10
    timeout_seconds: int = 300
    max_retries: int = 3

    # Memory and performance settings
    max_memory_mb: int = 2048
    chunk_size: int = 1000


@dataclass
class EmbeddingConfig:
    """Vector embedding configuration"""
    model_name: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    backend: str = "faiss"  # Options: faiss, qdrant, chromadb

    # FAISS settings
    faiss_index_type: str = "IndexFlatL2"

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ChromaDB settings
    chromadb_path: str = "./chromadb"


@dataclass
class LLMConfig:
    """Large Language Model configuration"""
    provider: str = "openai"  # Options: openai, anthropic
    model_name: str = "gpt-3.5-turbo"
    api_key: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7

    def __post_init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY', self.api_key)


@dataclass
class Config:
    """Main configuration class"""

    def __init__(self):
        self.neo4j = DatabaseConfig()
        self.processing = ProcessingConfig()
        self.embeddings = EmbeddingConfig()
        self.llm = LLMConfig()


# Global configuration instance
config = Config()

# Node Types for Knowledge Graph
NODE_TYPES = {
    'ARTIFACT': 'Artifact',
    'CONFERENCE': 'Conference',
    'DOCUMENTATION': 'Documentation',
    'SECTION': 'Section',
    'TOOL': 'Tool',
    'COMMAND': 'Command',
    'ENTITY': 'Entity',
    'FILE': 'File',
    'REQUIREMENT': 'Requirement',
    'DEPENDENCY': 'Dependency',
    'AUTHOR': 'Author',
    'PAPER': 'Paper',
    'DATASET': 'Dataset',
    'MODEL': 'Model',
    'FRAMEWORK': 'Framework',
    'API': 'API',
    'DATABASE': 'Database'
}

# Relationship Types for Knowledge Graph
RELATIONSHIP_TYPES = {
    'USED_IN': 'USED_IN',
    'HAS_FILE': 'HAS_FILE',
    'HAS_SECTION': 'HAS_SECTION',
    'MENTIONS': 'MENTIONS',
    'CONTAINS': 'CONTAINS',
    'DEPENDS_ON': 'DEPENDS_ON',
    'SIMILAR_TO': 'SIMILAR_TO',
    'AUTHORED_BY': 'AUTHORED_BY',
    'PUBLISHED_IN': 'PUBLISHED_IN',
    'IMPLEMENTS': 'IMPLEMENTS',
    'USES_DATASET': 'USES_DATASET',
    'USES_MODEL': 'USES_MODEL',
    'REFERENCES': 'REFERENCES',
    'PART_OF': 'PART_OF',
    'REQUIRES': 'REQUIRES',
    'PROVIDES': 'PROVIDES',
    'EXTENDS': 'EXTENDS',
    'EVALUATES': 'EVALUATES'
}

# Conference Categories and Their Characteristics
CONFERENCE_CATEGORIES = {
    'software_engineering': {
        'conferences': ['ICSE', 'FSE', 'ASE', 'ISSTA', 'ICSME'],
        'emphasis': ['reproducibility', 'functionality', 'usability'],
        'required_tools': ['build_system', 'testing_framework'],
        'documentation_style': 'technical',
        'avg_sections': 8,
        'quality_threshold': 0.8
    },
    'systems': {
        'conferences': ['SOSP', 'OSDI', 'NSDI', 'SIGCOMM', 'MOBICOM'],
        'emphasis': ['performance', 'scalability', 'evaluation'],
        'required_tools': ['performance_tools', 'system_tools'],
        'documentation_style': 'experimental',
        'avg_sections': 6,
        'quality_threshold': 0.85
    },
    'human_computer_interaction': {
        'conferences': ['CHI', 'UIST', 'CSCW', 'IUI'],
        'emphasis': ['usability', 'accessibility', 'user_experience'],
        'required_tools': ['ui_tools', 'survey_tools'],
        'documentation_style': 'user_focused',
        'avg_sections': 7,
        'quality_threshold': 0.75
    },
    'machine_learning': {
        'conferences': ['ICML', 'NIPS', 'ICLR', 'AAAI'],
        'emphasis': ['reproducibility', 'experimental_evaluation', 'datasets'],
        'required_tools': ['ml_frameworks', 'data_tools'],
        'documentation_style': 'experimental',
        'avg_sections': 9,
        'quality_threshold': 0.9
    },
    'data_management': {
        'conferences': ['SIGMOD', 'VLDB', 'ICDE', 'PODS'],
        'emphasis': ['performance', 'scalability', 'data_quality'],
        'required_tools': ['database_tools', 'query_tools'],
        'documentation_style': 'data_focused',
        'avg_sections': 7,
        'quality_threshold': 0.8
    },
    'programming_languages': {
        'conferences': ['PLDI', 'POPL', 'OOPSLA', 'ICFP'],
        'emphasis': ['formal_verification', 'implementation', 'theory'],
        'required_tools': ['compiler_tools', 'verification_tools'],
        'documentation_style': 'formal',
        'avg_sections': 6,
        'quality_threshold': 0.85
    },
    'security': {
        'conferences': ['CCS', 'SECURITY', 'S&P', 'NDSS'],
        'emphasis': ['security_analysis', 'vulnerability_assessment', 'evaluation'],
        'required_tools': ['security_tools', 'analysis_tools'],
        'documentation_style': 'security_focused',
        'avg_sections': 8,
        'quality_threshold': 0.85
    },
    'other': {
        'conferences': [],
        'emphasis': ['general'],
        'required_tools': ['basic_tools'],
        'documentation_style': 'general',
        'avg_sections': 5,
        'quality_threshold': 0.7
    }
}

# Tool Categories
TOOL_CATEGORIES = {
    'programming_languages': {
        'tools': ['python', 'java', 'c_cpp', 'javascript', 'r', 'matlab', 'scala', 'go'],
        'weight': 1.0
    },
    'build_systems': {
        'tools': ['maven', 'gradle', 'make', 'cmake', 'bazel', 'sbt'],
        'weight': 0.8
    },
    'containerization': {
        'tools': ['docker', 'kubernetes', 'singularity'],
        'weight': 0.9
    },
    'version_control': {
        'tools': ['git', 'svn', 'mercurial'],
        'weight': 0.6
    },
    'testing': {
        'tools': ['junit', 'pytest', 'jest', 'mocha'],
        'weight': 0.8
    },
    'ml_frameworks': {
        'tools': ['tensorflow', 'pytorch', 'scikit-learn', 'keras'],
        'weight': 1.0
    },
    'data_tools': {
        'tools': ['pandas', 'numpy', 'spark', 'hadoop'],
        'weight': 0.9
    },
    'databases': {
        'tools': ['postgresql', 'mysql', 'mongodb', 'redis'],
        'weight': 0.7
    },
    'web_frameworks': {
        'tools': ['django', 'flask', 'react', 'angular', 'spring'],
        'weight': 0.8
    },
    'ide_editors': {
        'tools': ['vscode', 'intellij', 'eclipse', 'vim'],
        'weight': 0.5
    }
}

# Quality Metrics Weights
QUALITY_WEIGHTS = {
    'documentation_completeness': 0.25,
    'code_quality': 0.20,
    'reproducibility': 0.25,
    'usability': 0.15,
    'accessibility': 0.10,
    'maintenance': 0.05
}

# Evaluation Thresholds
EVALUATION_THRESHOLDS = {
    'excellent': 0.9,
    'good': 0.75,
    'satisfactory': 0.6,
    'needs_improvement': 0.4,
    'poor': 0.0
}

# File Patterns
FILE_PATTERNS = {
    'documentation': [
        r'readme\.md', r'readme\.txt', r'readme\.rst',
        r'install\.md', r'installation\.md',
        r'usage\.md', r'tutorial\.md',
        r'doc/.*\.md', r'docs/.*\.md',
        r'manual\.md', r'guide\.md'
    ],
    'code': [
        r'.*\.py', r'.*\.java', r'.*\.cpp', r'.*\.c',
        r'.*\.js', r'.*\.ts', r'.*\.r', r'.*\.R',
        r'.*\.scala', r'.*\.go', r'.*\.m'
    ],
    'configuration': [
        r'requirements\.txt', r'setup\.py', r'setup\.cfg',
        r'pom\.xml', r'build\.gradle', r'makefile',
        r'dockerfile', r'docker-compose\.yml',
        r'.*\.json', r'.*\.yaml', r'.*\.yml'
    ],
    'data': [
        r'.*\.csv', r'.*\.json', r'.*\.xml',
        r'.*\.xlsx', r'.*\.xls', r'.*\.txt',
        r'.*\.sql', r'.*\.db'
    ],
    'notebooks': [
        r'.*\.ipynb'
    ],
    'scripts': [
        r'.*\.sh', r'.*\.bat', r'.*\.ps1'
    ]
}

# Default Paths
DEFAULT_PATHS = {
    'output_dir': './output',
    'temp_dir': './temp',
    'logs_dir': './logs',
    'cache_dir': './cache',
    'models_dir': './models',
    'data_dir': './data'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file'],
    'file_path': './logs/aura.log'
}

# Phase-specific Configuration
PHASE_CONFIG = {
    'phase1': {
        'max_file_size_mb': 100,
        'supported_formats': ['.zip', '.tar.gz', '.tar', '.rar', '.7z'],
        'timeout_seconds': 600
    },
    'phase2': {
        'max_nodes': 100000,
        'max_relationships': 500000,
        'batch_size': 1000
    },
    'phase3': {
        'chunk_size': 512,
        'overlap': 50,
        'max_chunks_per_document': 100
    },
    'phase4': {
        'min_cluster_size': 3,
        'min_samples': 2,
        'eps': 0.5
    },
    'phase5': {
        'max_concurrent_agents': 3,
        'agent_timeout': 300,
        'max_retries': 2
    },
    'phase6': {
        'score_precision': 3,
        'confidence_threshold': 0.7
    },
    'phase7': {
        'chart_width': 800,
        'chart_height': 600,
        'max_items_per_chart': 20
    },
    'phase8': {
        'max_parallel_processes': 4,
        'checkpoint_interval': 10
    }
}

# Error Messages
ERROR_MESSAGES = {
    'file_not_found': "File not found: {path}",
    'extraction_failed': "Failed to extract artifact: {error}",
    'processing_failed': "Failed to process artifact: {error}",
    'database_connection_failed': "Failed to connect to database: {error}",
    'api_request_failed': "API request failed: {error}",
    'timeout_exceeded': "Operation timed out after {seconds} seconds",
    'invalid_format': "Invalid format for {item}: {format}",
    'insufficient_resources': "Insufficient resources: {resource}",
    'validation_failed': "Validation failed: {reason}"
}

# Success Messages
SUCCESS_MESSAGES = {
    'extraction_complete': "Successfully extracted {count} artifacts",
    'processing_complete': "Successfully processed {count} items",
    'database_connected': "Successfully connected to database",
    'analysis_complete': "Analysis completed successfully",
    'report_generated': "Report generated: {path}",
    'pipeline_complete': "Pipeline completed successfully"
}


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def update_config(**kwargs):
    """Update configuration parameters"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter: {key}")


def validate_config() -> bool:
    """Validate the current configuration"""
    try:
        # Check required environment variables
        required_vars = []

        # Check database connection
        if config.neo4j.uri and config.neo4j.username:
            pass  # Could add actual connection test here

        # Check API keys if needed
        if config.llm.provider == 'openai' and not config.llm.api_key:
            print("Warning: OpenAI API key not set")

        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    cfg = get_config()
    print(f"Neo4j URI: {cfg.neo4j.uri}")
    print(f"Processing workers: {cfg.processing.max_workers}")
    print(f"Embedding model: {cfg.embeddings.model_name}")
    print(f"Configuration valid: {validate_config()}")
