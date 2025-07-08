#!/usr/bin/env python3
"""
AURA Streamlit App Runner

Simple script to launch the AURA evaluation framework web interface.

Usage:
    python run_aura_app.py
    
This will start the Streamlit server and open the AURA web interface
in your default browser.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the AURA Streamlit app"""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    app_file = script_dir / "aura_app.py"
    
    # Check if the app file exists
    if not app_file.exists():
        print(f"❌ Error: Could not find aura_app.py at {app_file}")
        sys.exit(1)
    
    print("🚀 Starting AURA Evaluation Framework Web Interface...")
    print(f"📁 App location: {app_file}")
    print("🌐 Opening in your default browser...")
    print("\n" + "="*60)
    print("AURA: Artifact Understanding and Research Assessment")
    print("AI-Powered Research Artifact Evaluation Framework")
    print("="*60 + "\n")
    
    try:
        # Launch Streamlit with the app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 AURA app stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting AURA app: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check that all dependencies are available")
        print("3. Ensure you're in the correct directory")
        sys.exit(1)

if __name__ == "__main__":
    main() 