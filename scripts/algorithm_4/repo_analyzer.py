"""
Repository Analyzer Module
Wrapper for algorithm_2 functions to make them easily importable in the Streamlit app
"""

import os
import sys
import logging

# Add the algorithm_2 directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_2_dir = os.path.join(current_dir, '..', 'algorithm_2')
sys.path.append(algorithm_2_dir)

try:
    from algorithm_2 import analyze_repository, save_analysis_result
    logging.info("Successfully imported algorithm_2 functions")
except ImportError as e:
    logging.error(f"Failed to import algorithm_2 functions: {e}")
    raise

def analyze_github_repository(repo_url: str, output_dir: str = "../../algo_outputs/algorithm_2_output"):
    """
    Analyze a GitHub repository and save the results
    
    Args:
        repo_url (str): GitHub repository URL
        output_dir (str): Directory to save the analysis results
        
    Returns:
        tuple: (repo_name, artifact_json_path, analysis_result)
    """
    try:
        # Extract repository name from URL
        repo_name = repo_url.rstrip("/").split("/")[-1]
        
        # Analyze the repository
        logging.info(f"Starting analysis of repository: {repo_url}")
        result = analyze_repository(repo_url)
        
        # Save the analysis result
        logging.info(f"Saving analysis results to: {output_dir}")
        save_analysis_result(result, repo_name, output_dir)
        
        # Construct the artifact JSON path
        artifact_json_path = os.path.join(output_dir, f"{repo_name}_analysis.json")
        
        logging.info(f"Analysis completed successfully for {repo_name}")
        return repo_name, artifact_json_path, result
        
    except Exception as e:
        logging.error(f"Error analyzing repository {repo_url}: {e}")
        raise

def get_analysis_summary(result: dict) -> dict:
    """
    Extract summary information from analysis result
    
    Args:
        result (dict): Analysis result from analyze_repository
        
    Returns:
        dict: Summary information
    """
    return {
        "total_files": len(result.get("repository_structure", [])),
        "documentation_files": len(result.get("documentation_files", [])),
        "code_files": len(result.get("code_files", [])),
        "license_files": len(result.get("license_files", [])),
        "tree_structure_lines": len(result.get("tree_structure", []))
    } 