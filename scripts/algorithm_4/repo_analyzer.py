"""
Repository Analyzer Module
Wrapper for algorithm_2 functions to make them easily importable in the Streamlit app
"""

import os
import sys
import logging
import time
from datetime import datetime

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
    Analyze a GitHub repository and save the results with timing data
    
    Args:
        repo_url (str): GitHub repository URL
        output_dir (str): Directory to save the analysis results
        
    Returns:
        tuple: (repo_name, artifact_json_path, analysis_result, timing_data)
    """
    timing_data = {
        "analysis_start_time": datetime.now().isoformat(),
        "analysis_end_time": None,
        "analysis_duration_seconds": None,
        "cloning_start_time": None,
        "cloning_end_time": None,
        "cloning_duration_seconds": None,
        "analysis_processing_start_time": None,
        "analysis_processing_end_time": None,
        "analysis_processing_duration_seconds": None,
        "saving_start_time": None,
        "saving_end_time": None,
        "saving_duration_seconds": None
    }
    
    try:
        # Extract repository name from URL
        repo_name = repo_url.rstrip("/").split("/")[-1]
        
        # Start timing the overall analysis
        analysis_start = time.time()
        timing_data["analysis_start_time"] = datetime.now().isoformat()
        
        # Time the cloning process
        timing_data["cloning_start_time"] = datetime.now().isoformat()
        cloning_start = time.time()
        
        logging.info(f"Starting analysis of repository: {repo_url}")
        
        # Analyze the repository (this includes cloning)
        result = analyze_repository(repo_url)
        
        # End cloning timing
        cloning_end = time.time()
        timing_data["cloning_end_time"] = datetime.now().isoformat()
        timing_data["cloning_duration_seconds"] = round(cloning_end - cloning_start, 2)
        
        # Time the saving process
        timing_data["saving_start_time"] = datetime.now().isoformat()
        saving_start = time.time()
        
        logging.info(f"Saving analysis results to: {output_dir}")
        save_analysis_result(result, repo_name, output_dir)
        
        # End saving timing
        saving_end = time.time()
        timing_data["saving_end_time"] = datetime.now().isoformat()
        timing_data["saving_duration_seconds"] = round(saving_end - saving_start, 2)
        
        # End overall analysis timing
        analysis_end = time.time()
        timing_data["analysis_end_time"] = datetime.now().isoformat()
        timing_data["analysis_duration_seconds"] = round(analysis_end - analysis_start, 2)
        
        # Calculate processing time (analysis minus cloning and saving)
        processing_duration = timing_data["analysis_duration_seconds"] - timing_data["cloning_duration_seconds"] - timing_data["saving_duration_seconds"]
        timing_data["analysis_processing_duration_seconds"] = round(max(0, processing_duration), 2)
        
        # Construct the artifact JSON path
        artifact_json_path = os.path.join(output_dir, f"{repo_name}_analysis.json")
        
        logging.info(f"Analysis completed successfully for {repo_name}")
        logging.info(f"Timing summary: Total={timing_data['analysis_duration_seconds']}s, Cloning={timing_data['cloning_duration_seconds']}s, Processing={timing_data['analysis_processing_duration_seconds']}s, Saving={timing_data['saving_duration_seconds']}s")
        
        return repo_name, artifact_json_path, result, timing_data
        
    except Exception as e:
        # Record end time even if there's an error
        timing_data["analysis_end_time"] = datetime.now().isoformat()
        if timing_data["analysis_start_time"]:
            start_time = datetime.fromisoformat(timing_data["analysis_start_time"])
            end_time = datetime.fromisoformat(timing_data["analysis_end_time"])
            timing_data["analysis_duration_seconds"] = round((end_time - start_time).total_seconds(), 2)
        
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

def format_timing_summary(timing_data: dict) -> str:
    """
    Format timing data into a readable summary
    
    Args:
        timing_data (dict): Timing information from analysis
        
    Returns:
        str: Formatted timing summary
    """
    if not timing_data:
        return "No timing data available"
    
    summary_parts = []
    summary_parts.append(f"Total Analysis Time: {timing_data.get('analysis_duration_seconds', 0)}s")
    
    if timing_data.get('cloning_duration_seconds'):
        summary_parts.append(f"Cloning: {timing_data['cloning_duration_seconds']}s")
    
    if timing_data.get('analysis_processing_duration_seconds'):
        summary_parts.append(f"Processing: {timing_data['analysis_processing_duration_seconds']}s")
    
    if timing_data.get('saving_duration_seconds'):
        summary_parts.append(f"Saving: {timing_data['saving_duration_seconds']}s")
    
    return " | ".join(summary_parts) 