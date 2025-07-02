#!/usr/bin/env python3
"""
Test script for running the AURA Framework on a GitHub repository URL, local directory, or archive file.

Usage:
    python test_aura_url.py <artifact_path_or_url>
    (prints progress and saves outputs in algo_outputs/algorithm_2_output/ and algo_outputs/algorithm_4_output/)
"""
import sys
import os
import subprocess
import json

# Add the algorithm_4 directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aura_framework import AURAFramework

def main(artifact_path_or_url):
    print("AURA Framework Artifact Test Suite")
    print("=" * 50)
    print(f"Using artifact: {artifact_path_or_url}")
    print("=" * 50)

    # Step 1: Run algorithm_2 to analyze the repo/directory/archive and generate artifact JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    algo2_path = os.path.abspath(os.path.join(script_dir, "..", "algorithm_2", "algorithm_2.py"))
    output_dir_algo2 = os.path.abspath(os.path.join(script_dir, "..", "..", "algo_outputs", "algorithm_2_output"))
    output_dir_algo4 = os.path.abspath(os.path.join(script_dir, "..", "..", "algo_outputs", "algorithm_4_output"))
    os.makedirs(output_dir_algo2, exist_ok=True)
    os.makedirs(output_dir_algo4, exist_ok=True)
    repo_name = os.path.basename(artifact_path_or_url.rstrip("/\\")).replace('.zip','').replace('.tar.gz','').replace('.tar','').replace('.tgz','').replace('.tar.bz2','').replace('.tar.xz','')
    artifact_json_path = os.path.join(output_dir_algo2, f"{repo_name}_analysis.json")
    print(f"\n[1/2] Running artifact analysis (algorithm_2) ...")
    try:
        result = subprocess.run([
            sys.executable, algo2_path, artifact_path_or_url, os.path.join('.', 'temp_dir_for_git'), output_dir_algo2
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"✗ Analysis failed: {result.stderr}")
            return 1
        if not os.path.exists(artifact_json_path):
            print(f"✗ Expected output JSON not found: {artifact_json_path}")
            return 1
        print(f"✓ Analysis complete! JSON saved at: {artifact_json_path}")
    except Exception as e:
        print(f"✗ Exception during analysis: {e}")
        return 1

    # Step 2: Run AURA framework evaluation
    print(f"\n[2/2] Running AURA framework evaluation ...")
    try:
        framework = AURAFramework(artifact_json_path, use_llm=False)
        result = framework.evaluate_artifact()
        eval_json_path = os.path.join(output_dir_algo4, f"{repo_name}_aura_evaluation.json")
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(result.dict(), f, indent=2)
        print(f"✓ Evaluation complete! Results saved at: {eval_json_path}")
        print(f"  - Total weighted score: {result.total_weighted_score:.3f}")
        print(f"  - Acceptance prediction: {result.acceptance_prediction}")
        print(f"  - Number of criteria evaluated: {len(result.criteria_scores)}")
        print("\nDone.")
        return 0
    except Exception as e:
        print(f"✗ Exception during evaluation: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_aura_url.py <artifact_path_or_url>")
        sys.exit(1)
    sys.exit(main(sys.argv[1])) 