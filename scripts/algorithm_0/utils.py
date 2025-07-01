"""
Utility functions for Conference-Specific Algorithm 1.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy and torch data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def setup_logging(log_level: str = "INFO", log_dir: Optional[str] = None) -> str:
    """Set up enhanced logging for the algorithm."""
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "algo_outputs", "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"conference_algorithm_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    
    return log_file_path


def validate_input_directory(input_dir: str) -> List[str]:
    """Validate input directory and return list of valid guideline files."""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    md_files = []
    for ext in ["*.md", "*.txt"]:
        md_files.extend(Path(input_dir).glob(ext))
    
    if not md_files:
        raise ValueError(f"No markdown or text files found in: {input_dir}")
    
    valid_files = [str(f) for f in md_files if f.stat().st_size > 0]
    return valid_files


def create_output_directory(base_dir: str, conference_name: Optional[str] = None) -> str:
    """
    Create timestamped output directory.
    
    Args:
        base_dir: Base directory for outputs
        conference_name: Optional conference name for subdirectory
        
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if conference_name:
        output_dir = os.path.join(base_dir, f"{conference_name.lower()}_{timestamp}")
    else:
        output_dir = os.path.join(base_dir, f"extraction_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.getLogger(__name__).info(f"Created output directory: {output_dir}")
    return output_dir


def save_json_with_metadata(data: Dict[str, Any], file_path: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save JSON data with optional metadata header.
    
    Args:
        data: Data to save
        file_path: Output file path
        metadata: Optional metadata to include
    """
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "algorithm_version": "1.0.0",
            "file_path": file_path,
            **(metadata or {})
        },
        "data": data
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    logging.getLogger(__name__).info(f"Saved JSON data to: {file_path}")


def load_conference_guidelines(file_paths: List[str]) -> Dict[str, str]:
    """
    Load conference guideline texts from files.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        Dictionary mapping filenames to content
    """
    guidelines = {}
    logger = logging.getLogger(__name__)
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:
                    filename = os.path.basename(file_path)
                    guidelines[filename] = content
                    logger.debug(f"Loaded {filename}: {len(content)} characters")
                else:
                    logger.warning(f"Empty file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    logger.info(f"Successfully loaded {len(guidelines)} guideline files")
    return guidelines


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        data: List of numerical values
        
    Returns:
        Dictionary with statistical measures
    """
    if not data:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
        }
    
    return {
        "count": len(data),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data))
    }


def normalize_conference_name(conference_name: str) -> str:
    """
    Normalize conference name for consistent lookup.
    
    Args:
        conference_name: Raw conference name
        
    Returns:
        Normalized conference name
    """
    # Remove common prefixes/suffixes
    normalized = conference_name.upper().strip()
    
    # Handle common variations
    name_mappings = {
        "INTERNATIONAL CONFERENCE ON SOFTWARE ENGINEERING": "ICSE",
        "ACM SIGMOD": "SIGMOD",
        "HUMAN FACTORS IN COMPUTING SYSTEMS": "CHI",
        "VERY LARGE DATA BASES": "VLDB",
        "PROGRAMMING LANGUAGE DESIGN": "PLDI"
    }
    
    for full_name, short_name in name_mappings.items():
        if full_name in normalized:
            return short_name
    
    return normalized


def extract_conference_from_filename(filename: str) -> Optional[str]:
    """
    Extract conference name from filename using regex patterns.
    
    Args:
        filename: Name of the file
        
    Returns:
        Extracted conference name or None
    """
    import re
    
    # Pattern: number_conference_year.ext
    match = re.match(r'\d+_([a-zA-Z]+)_\d{4}', filename)
    if match:
        return match.group(1).upper()
    
    # Pattern: conference_year.ext
    match = re.match(r'([a-zA-Z]+)_\d{4}', filename)
    if match:
        return match.group(1).upper()
    
    # Pattern: conference.ext
    match = re.match(r'([a-zA-Z]+)', filename)
    if match:
        return match.group(1).upper()
    
    return None


def merge_dictionaries(dict1: Dict, dict2: Dict, strategy: str = "update") -> Dict:
    """
    Merge two dictionaries with different strategies.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary  
        strategy: Merge strategy ("update", "combine", "average")
        
    Returns:
        Merged dictionary
    """
    if strategy == "update":
        result = dict1.copy()
        result.update(dict2)
        return result
    
    elif strategy == "combine":
        result = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key, [])
            val2 = dict2.get(key, [])
            
            if isinstance(val1, list) and isinstance(val2, list):
                result[key] = val1 + val2
            else:
                result[key] = val2 if key in dict2 else val1
                
        return result
    
    elif strategy == "average":
        result = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key, 0)
            val2 = dict2.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                count = (1 if key in dict1 else 0) + (1 if key in dict2 else 0)
                result[key] = (val1 + val2) / count
            else:
                result[key] = val2 if key in dict2 else val1
                
        return result
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def create_summary_report(results: Dict[str, Any], output_path: str) -> str:
    """
    Create a human-readable summary report.
    
    Args:
        results: Algorithm results
        output_path: Path to save the report
        
    Returns:
        Path to the saved report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Conference-Specific Algorithm 1 - Execution Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if isinstance(results, dict):
            if "conference_name" in results:
                f.write(f"**Conference:** {results['conference_name']}\n")
            
            if "processing_stats" in results:
                stats = results["processing_stats"]
                f.write(f"**Documents Processed:** {stats.get('total_documents', 'N/A')}\n")
                f.write(f"**Keywords Extracted:** {stats.get('total_keywords_extracted', 'N/A')}\n")
                f.write(f"**Total Score:** {stats.get('total_score', 'N/A'):.2f}\n\n")
            
            if "criteria_dataframe" in results:
                df = results["criteria_dataframe"]
                f.write("## Extracted Dimensions\n\n")
                
                for _, row in df.iterrows():
                    f.write(f"### {row['dimension'].title()}\n")
                    f.write(f"- **Weight:** {row['normalized_weight']:.3f}\n")
                    if 'conference_adjusted_weight' in row:
                        f.write(f"- **Conference-Adjusted Weight:** {row['conference_adjusted_weight']:.3f}\n")
                    f.write(f"- **Keywords:** {row['keywords'][:200]}...\n\n")
        
        f.write("\n---\n*Generated by AURA Conference-Specific Algorithm 1*")
    
    logging.getLogger(__name__).info(f"Summary report saved to: {output_path}")
    return output_path 