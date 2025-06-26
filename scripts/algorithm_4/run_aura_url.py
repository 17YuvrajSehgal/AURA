#!/usr/bin/env python3
"""
Simple runner script for the AURA Framework on a GitHub repository URL.
"""
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_aura_url import main

# Set your GitHub repository URL here
repo_url = "https://github.com/JackyChok/AI_Code_Detection_Education"

if __name__ == "__main__":
    print(f"Running AURA Framework on repository URL: {repo_url}")
    result = main(repo_url)
    sys.exit(result) 