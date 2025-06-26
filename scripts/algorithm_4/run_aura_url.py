#!/usr/bin/env python3
"""
Simple runner script for the AURA Framework on a GitHub repository URL, local directory, or archive file.
- The analysis JSON will be saved in algo_outputs/algorithm_2_output
- The AURA evaluation JSON will be saved in algo_outputs/algorithm_4_output
- You can set artifact_path_or_url to a GitHub URL, a local directory, or a supported archive file (zip/tar/etc)
"""
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_aura_url import main

# Set your artifact path or URL here (can be a GitHub URL, local directory, or archive)
artifact_path_or_url = "../../temp_dir_for_git/icse_artifacts/AI_Code_Detection_Education-main.zip"

if __name__ == "__main__":
    print(f"Running AURA Framework on artifact: {artifact_path_or_url}")
    result = main(artifact_path_or_url)
    sys.exit(result) 