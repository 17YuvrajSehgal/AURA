#!/usr/bin/env python3
"""
Setup script for Algorithm 5: Robust Knowledge Graph Pipeline

This script helps users set up the environment and dependencies
for the ICSE artifacts processing pipeline.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"✅ Python version: {sys.version}")
    return True


def install_dependencies():
    """Install required Python packages."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error(f"Requirements file not found: {requirements_file}")
        return False
    
    try:
        logger.info("Installing Python dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def check_neo4j_connection(uri: str = "bolt://localhost:7687", 
                          user: str = "neo4j", 
                          password: str = "password"):
    """Check Neo4j database connection."""
    try:
        from py2neo import Graph
        graph = Graph(uri, auth=(user, password))
        result = graph.run("RETURN 1 as test").data()
        
        if result and result[0]["test"] == 1:
            logger.info("✅ Neo4j connection successful")
            return True
        else:
            logger.warning("⚠️  Neo4j connection returned unexpected result")
            return False
    
    except ImportError:
        logger.error("py2neo package not installed. Run: pip install py2neo")
        return False
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        logger.info("Please ensure Neo4j is running and accessible")
        logger.info("You can start Neo4j with Docker:")
        logger.info("  docker run -p 7474:7474 -p 7687:7687 --env NEO4J_AUTH=neo4j/password neo4j")
        return False


def create_output_directories():
    """Create necessary output directories."""
    directories = [
        "./algo_outputs",
        "./algo_outputs/algorithm_5_output",
        "./temp_extractions",
        "./icse_analysis_output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
    
    return True


def create_sample_config():
    """Create sample configuration file."""
    from .config import create_sample_config
    
    try:
        config_file = "algorithm_5_config.json"
        create_sample_config(config_file)
        logger.info(f"📝 Sample configuration created: {config_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create sample config: {e}")
        return False


def run_basic_test():
    """Run a basic test to verify the installation."""
    try:
        # Test imports
        from . import RobustKGPipeline, ArtifactExtractor, EnhancedKGBuilder, BatchProcessor
        logger.info("✅ Module imports successful")
        
        # Test configuration
        from .config import load_config
        config = load_config()
        logger.info("✅ Configuration loading successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ Basic test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Place your ICSE artifacts in a directory (e.g., 'icse_artifacts/')")
    print("2. Start Neo4j database (if not already running)")
    print("3. Run the processor:")
    print("   python scripts/algorithm_5/icse_artifacts_processor.py icse_artifacts/")
    print("\n📖 For examples and documentation:")
    print("   python scripts/algorithm_5/example_usage.py")
    print("   cat scripts/algorithm_5/README.md")
    
    print("\n🔧 Configuration:")
    print("   Edit 'algorithm_5_config.json' to customize settings")
    print("   Set environment variables for Neo4j credentials")
    
    print("\n📊 For help:")
    print("   python scripts/algorithm_5/icse_artifacts_processor.py --help")
    print("="*60)


def main():
    """Main setup function."""
    print("🚀 Setting up Algorithm 5: Robust Knowledge Graph Pipeline")
    print("="*60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Create directories
    if success and not create_output_directories():
        success = False
    
    # Create sample config
    if success and not create_sample_config():
        success = False
    
    # Test basic functionality
    if success and not run_basic_test():
        success = False
    
    # Check Neo4j (optional - don't fail setup if Neo4j is not available)
    check_neo4j_connection()
    
    if success:
        print_next_steps()
    else:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main() 