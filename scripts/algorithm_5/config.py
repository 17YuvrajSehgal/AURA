#!/usr/bin/env python3
"""
Configuration module for Algorithm 5: Robust Knowledge Graph Pipeline

This module provides configuration management for the pipeline including:
- Default settings
- Environment variable handling
- Configuration validation
- Profile management
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json


class Config:
    """Configuration management for the Robust KG Pipeline."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        # Neo4j Database Settings
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j", 
            "password": "password",
            "database": "neo4j"
        },
        
        # Pipeline Settings
        "pipeline": {
            "working_dir": "./algo_outputs/algorithm_5_output",
            "temp_dir": "./temp_extractions",
            "enable_advanced_analysis": True,
            "clear_existing_graph": False,
            "max_file_size": 500 * 1024 * 1024,  # 500MB
            "timeout_per_artifact": 300  # 5 minutes
        },
        
        # Batch Processing Settings
        "batch": {
            "max_workers": 4,
            "use_processes": False,
            "chunk_size": 10,
            "enable_progress_tracking": True
        },
        
        # Extraction Settings
        "extraction": {
            "supported_formats": [".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"],
            "exclude_patterns": ["__pycache__", "*.pyc", ".git", ".svn", "node_modules"],
            "max_depth": 20,
            "follow_symlinks": False
        },
        
        # Knowledge Graph Settings
        "knowledge_graph": {
            "enable_code_analysis": True,
            "enable_dependency_analysis": True,
            "enable_documentation_analysis": True,
            "create_indexes": True,
            "batch_size": 1000
        },
        
        # Visualization Settings
        "visualization": {
            "default_format": "html",
            "max_nodes": 500,
            "physics_enabled": True,
            "node_size": 20,
            "edge_length": 100
        },
        
        # Logging Settings
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_handler": True,
            "console_handler": True,
            "log_file": "./algo_outputs/algorithm_5.log"
        },
        
        # Analysis Settings
        "analysis": {
            "enable_pattern_mining": True,
            "enable_metrics_calculation": True,
            "enable_similarity_analysis": True,
            "enable_centrality_analysis": True,
            "community_detection_algorithm": "louvain"
        }
    }
    
    def __init__(self, config_file: Optional[str] = None, profile: str = "default"):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to configuration file
            profile: Configuration profile name
        """
        self.profile = profile
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_environment()
        
        # Validate configuration
        self.validate()
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # Deep merge with default config
                self.config = self._deep_merge(self.config, file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "NEO4J_URI": ("neo4j", "uri"),
            "NEO4J_USER": ("neo4j", "user"),
            "NEO4J_PASSWORD": ("neo4j", "password"),
            "NEO4J_DATABASE": ("neo4j", "database"),
            "PIPELINE_WORKING_DIR": ("pipeline", "working_dir"),
            "PIPELINE_TEMP_DIR": ("pipeline", "temp_dir"),
            "BATCH_MAX_WORKERS": ("batch", "max_workers"),
            "LOG_LEVEL": ("logging", "level"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if key in ["max_workers", "timeout_per_artifact", "max_file_size"]:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif key in ["enable_advanced_analysis", "clear_existing_graph", "use_processes"]:
                    value = value.lower() in ("true", "1", "yes", "on")
                
                self.config[section][key] = value
    
    def get(self, *keys, default=None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            *keys: Configuration path (e.g., "neo4j", "uri")
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        current = self.config
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, *keys, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            *keys: Configuration path
            value: Value to set
        """
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def validate(self):
        """Validate configuration values."""
        # Check required settings
        required_settings = [
            ("neo4j", "uri"),
            ("neo4j", "user"),
            ("pipeline", "working_dir"),
        ]
        
        for section, key in required_settings:
            if not self.get(section, key):
                raise ValueError(f"Required configuration missing: {section}.{key}")
        
        # Validate numeric settings
        numeric_settings = [
            ("batch", "max_workers", 1, 32),
            ("pipeline", "timeout_per_artifact", 30, 3600),
            ("visualization", "max_nodes", 10, 10000),
        ]
        
        for section, key, min_val, max_val in numeric_settings:
            value = self.get(section, key)
            if value is not None and not (min_val <= value <= max_val):
                raise ValueError(f"Configuration {section}.{key} must be between {min_val} and {max_val}")
    
    def save_to_file(self, config_file: str):
        """Save current configuration to file."""
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config to {config_file}: {e}")
    
    def get_neo4j_config(self) -> Dict[str, str]:
        """Get Neo4j connection configuration."""
        return {
            "uri": self.get("neo4j", "uri"),
            "user": self.get("neo4j", "user"),
            "password": self.get("neo4j", "password"),
            "database": self.get("neo4j", "database")
        }
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.config["pipeline"].copy()
    
    def get_batch_config(self) -> Dict[str, Any]:
        """Get batch processing configuration."""
        return self.config["batch"].copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config["logging"].copy()
    
    def _deep_merge(self, base_dict: Dict, override_dict: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base_dict.copy()
        
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_directories(self):
        """Create necessary directories based on configuration."""
        directories = [
            self.get("pipeline", "working_dir"),
            self.get("pipeline", "temp_dir"),
            Path(self.get("logging", "log_file")).parent if self.get("logging", "log_file") else None
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(profile={self.profile}, neo4j_uri={self.get('neo4j', 'uri')})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Config(profile={self.profile}, config={self.config})"


# Configuration profiles for different environments
PROFILES = {
    "development": {
        "neo4j": {"uri": "bolt://localhost:7687"},
        "pipeline": {"clear_existing_graph": True},
        "logging": {"level": "DEBUG"},
        "batch": {"max_workers": 2}
    },
    
    "production": {
        "neo4j": {"uri": "bolt://neo4j-prod:7687"},
        "pipeline": {"clear_existing_graph": False},
        "logging": {"level": "INFO"},
        "batch": {"max_workers": 8}
    },
    
    "testing": {
        "neo4j": {"uri": "bolt://localhost:7688"},
        "pipeline": {"clear_existing_graph": True},
        "logging": {"level": "DEBUG"},
        "batch": {"max_workers": 1}
    }
}


def load_config(profile: str = "default", config_file: Optional[str] = None) -> Config:
    """
    Load configuration for the specified profile.
    
    Args:
        profile: Configuration profile name
        config_file: Optional path to configuration file
        
    Returns:
        Configuration instance
    """
    config = Config(config_file=config_file, profile=profile)
    
    # Apply profile-specific settings
    if profile in PROFILES:
        profile_config = PROFILES[profile]
        config.config = config._deep_merge(config.config, profile_config)
    
    return config


def create_sample_config(output_file: str = "algorithm_5_config.json"):
    """Create a sample configuration file."""
    config = Config()
    config.save_to_file(output_file)
    print(f"Sample configuration saved to: {output_file}")


if __name__ == "__main__":
    # Create sample configuration file
    create_sample_config()
    
    # Demonstrate configuration usage
    config = load_config("development")
    print(f"Neo4j URI: {config.get('neo4j', 'uri')}")
    print(f"Working directory: {config.get('pipeline', 'working_dir')}")
    print(f"Max workers: {config.get('batch', 'max_workers')}") 