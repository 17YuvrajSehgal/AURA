#!/usr/bin/env python3
"""
Test script for running the AURA Framework on a GitHub repository URL.

Usage:
    python test_aura_url.py <repo_url>
    (prints progress and saves outputs in algo_outputs/algorithm_2_output/)
"""
import sys
import os
import subprocess
import json

# Add the algorithm_4 directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aura_framework import AURAFramework

def main(repo_url):
    print("AURA Framework URL Test Suite")
    print("=" * 50)
    print(f"Using repository URL: {repo_url}")
    print("=" * 50)

    # Step 1: Run algorithm_2 to analyze the repo and generate artifact JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    algo2_path = os.path.abspath(os.path.join(script_dir, "..", "algorithm_2", "algorithm_2.py"))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "algo_outputs", "algorithm_2_output"))
    os.makedirs(output_dir, exist_ok=True)
    repo_name = repo_url.rstrip("/").split("/")[-1].replace('.git', '')
    artifact_json_path = os.path.join(output_dir, f"{repo_name}_analysis.json")
    print(f"\n[1/2] Running repository analysis (algorithm_2) ...")
    result = subprocess.run([
        sys.executable, algo2_path, repo_url, os.path.join('.', 'temp_dir_for_git'), output_dir
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ Analysis failed: {result.stderr}")
        return 1
    if not os.path.exists(artifact_json_path):
        print(f"✗ Expected output JSON not found: {artifact_json_path}")
        return 1
    print(f"✓ Analysis complete! JSON saved at: {artifact_json_path}")

    # Step 2: Run AURA framework evaluation
    print(f"\n[2/2] Running AURA framework evaluation ...")
    framework = AURAFramework(artifact_json_path, use_llm=False)
    result = framework.evaluate_artifact()
    eval_json_path = os.path.join(output_dir, f"{repo_name}_aura_evaluation.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(result.dict(), f, indent=2)
    print(f"✓ Evaluation complete! Results saved at: {eval_json_path}")
    print(f"  - Total weighted score: {result.total_weighted_score:.3f}")
    print(f"  - Acceptance prediction: {result.acceptance_prediction}")
    print(f"  - Number of criteria evaluated: {len(result.criteria_scores)}")
    print("\nDone.")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_aura_url.py <repo_url>")
        sys.exit(1)
    sys.exit(main(sys.argv[1])) 