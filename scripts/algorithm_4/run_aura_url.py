#!/usr/bin/env python3
"""
Batch runner script for the AURA Framework on all artifacts in a directory (icse_artifacts).
- The analysis JSON will be saved in algo_outputs/algorithm_2_output
- The AURA evaluation JSON will be saved in algo_outputs/algorithm_4_output
- All subfolders and supported archives in icse_artifacts will be processed
"""
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_aura_url import main

def find_artifact_paths(base_dir):
    """
    Find all subdirectories and supported archives in base_dir.
    Returns a list of absolute paths.
    """
    supported_exts = ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz']
    artifact_paths = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            artifact_paths.append(full_path)
        elif os.path.isfile(full_path) and any(entry.lower().endswith(ext) for ext in supported_exts):
            artifact_paths.append(full_path)
    return artifact_paths

# Set your artifacts base directory here
artifacts_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../temp_dir_for_git/icse_artifacts'))

if __name__ == "__main__":
    print(f"Batch running AURA Framework on all artifacts in: {artifacts_base_dir}")
    artifact_paths = find_artifact_paths(artifacts_base_dir)
    print(f"Found {len(artifact_paths)} artifacts to process.")
    results = []
    for i, artifact_path in enumerate(artifact_paths, 1):
        print(f"\n=== [{i}/{len(artifact_paths)}] Processing: {artifact_path} ===")
        try:
            result = main(artifact_path)
            results.append((artifact_path, result))
        except Exception as e:
            print(f"✗ Exception for {artifact_path}: {e}")
            results.append((artifact_path, 'error'))
    print("\nBatch processing complete.")
    print(f"Processed {len(artifact_paths)} artifacts.")
    for path, res in results:
        print(f"{path}: {'OK' if res == 0 else 'FAILED'}") 