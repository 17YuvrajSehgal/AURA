#!/usr/bin/env python3
"""
Simple runner script for the AURA Framework tests.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_framework import main

if __name__ == "__main__":
    # Path to the artifact JSON file
    artifact_path = "../../algo_outputs/ml-image-classifier_analysis.json"
    
    # Run tests with full evaluation
    print("Running AURA Framework tests...")
    result = main(artifact_path, skip_evaluation=False)
    
    # Exit with the test result
    sys.exit(result) 