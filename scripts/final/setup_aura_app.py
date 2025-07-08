#!/usr/bin/env python3
"""
AURA Streamlit App Setup Script

This script helps set up the AURA evaluation framework web interface
by installing required dependencies and checking the environment.

Usage:
    python setup_aura_app.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    core_deps = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0", 
        "pandas>=2.0.0",
        "GitPython>=3.1.0"
    ]
    
    # AURA framework dependencies (if not already installed)
    aura_deps = [
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "neo4j>=5.0.0",
        "networkx>=3.0.0",
        "python-dotenv>=1.0.0"
    ]
    
    all_deps = core_deps + aura_deps
    
    try:
        for dep in all_deps:
            print(f"Installing {dep}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], check=True, capture_output=True)
        
        print("✅ All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Please install manually using:")
        print("pip install streamlit plotly pandas GitPython langchain openai sentence-transformers faiss-cpu neo4j networkx python-dotenv")
        return False

def check_environment():
    """Check if the environment is properly set up"""
    print("\n🔍 Checking environment...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found - you'll need to set up your OpenAI API key")
        print("   Create a .env file with: OPENAI_API_KEY=your_api_key_here")
    
    # Check if conference guidelines exist
    guidelines_dir = Path("data/conference_guideline_texts/processed")
    if not guidelines_dir.exists():
        guidelines_dir = Path("../data/conference_guideline_texts/processed")
    
    if guidelines_dir.exists():
        guideline_files = list(guidelines_dir.glob("*.md"))
        print(f"✅ Conference guidelines found: {len(guideline_files)} conferences")
    else:
        print("⚠️  Conference guidelines directory not found")
        print("   Make sure the processed conference guidelines are available")
    
    # Check if required AURA files exist
    required_files = [
        "aura_evaluator.py",
        "langchain_chains.py", 
        "knowledge_graph_builder.py",
        "rag_retrieval.py",
        "conference_guidelines_loader.py",
        "integrated_artifact_analyzer.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required AURA framework files found")
        return True

def create_sample_env():
    """Create a sample .env file"""
    env_content = """# AURA Framework Configuration
# OpenAI API Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Configuration (Optional - for advanced deployments)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=aura
"""
    
    env_file = Path(".env.example")
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"✅ Created {env_file}")
    print("   Copy this to .env and add your OpenAI API key")

def main():
    """Main setup process"""
    print("🚀 AURA Framework Web Interface Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check environment
    env_ok = check_environment()
    
    # Create sample .env if it doesn't exist
    if not Path(".env").exists():
        create_sample_env()
    
    print("\n" + "=" * 50)
    if env_ok:
        print("✅ Setup completed successfully!")
        print("\n🚀 To start the AURA web interface:")
        print("   python run_aura_app.py")
        print("\n   Or directly with Streamlit:")
        print("   streamlit run aura_app.py")
    else:
        print("⚠️  Setup completed with warnings")
        print("   Please resolve the issues above before running the app")
        print("\n🚀 Once resolved, start the app with:")
        print("   python run_aura_app.py")
    
    print("\n📚 Documentation: See README.md for detailed usage instructions")
    print("🐛 Issues? Check the GitHub repository for troubleshooting")

if __name__ == "__main__":
    main() 