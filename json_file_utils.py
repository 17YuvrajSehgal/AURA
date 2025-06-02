import json
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_analysis_json(filepath: str) -> Dict[str, Any]:
    """
    Load the entire analysis JSON from disk.

    Parameters:
        filepath: Path to the JSON file (e.g. "data/outputs/algorithm_2_output/ml-image-classifier_analysis.json").

    Returns:
        A Python dict parsed from the JSON.
    """
    if not os.path.isfile(filepath):
        logger.error(f"JSON file not found: {filepath}")
        raise FileNotFoundError(f"Cannot load analysis JSON; file does not exist: {filepath}")

    logger.info(f"Reading analysis JSON from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Successfully loaded JSON. Keys: {list(data.keys())}")
    return data


def get_repository_structure(data: Dict[str, Any]) -> List[str]:
    """
    Extract the repository structure. If 'tree_structure' is present, return that.
    Otherwise, build a list of paths from 'repository_structure'.

    Parameters:
        data: The parsed JSON dictionary.

    Returns:
        A list of strings representing either:
          - Lines of the visual tree (if 'tree_structure' exists), or
          - File paths relative to repo root (from 'repository_structure').
    """
    if "tree_structure" in data:
        tree = data["tree_structure"]
        logger.info(f"Using 'tree_structure' field ({len(tree)} lines).")
        return tree
    elif "repository_structure" in data:
        paths = [entry.get("path", "") for entry in data["repository_structure"]]
        logger.info(f"No 'tree_structure' found; returning {len(paths)} file paths from 'repository_structure'.")
        return paths
    else:
        logger.warning("No 'tree_structure' or 'repository_structure' found in JSON.")
        return []


def get_documentation_texts(data: Dict[str, Any]) -> List[str]:
    """
    Extract documentation file contents. Joins each 'content' array into one string.

    Parameters:
        data: The parsed JSON dictionary.

    Returns:
        A list of fully joined strings, one per documentation file.
        If a documentation file has multiple lines in its 'content' list, they are joined with newlines.
    """
    docs = data.get("documentation_files", [])
    results: List[str] = []
    for idx, entry in enumerate(docs):
        content_list = entry.get("content", [])
        if isinstance(content_list, list):
            joined = "\n".join(content_list).strip()
        else:
            # In case content was stored as a single string
            joined = str(content_list)
        results.append(joined)
        logger.info(f"Loaded documentation[{idx}]: path='{entry.get('path', '')}', length={len(joined)} chars.")
    if not results:
        logger.warning("No 'documentation_files' found or empty.")
    return results


def get_code_texts(data: Dict[str, Any]) -> List[str]:
    """
    Extract code file contents. Joins each 'content' array into one string per file.

    Parameters:
        data: The parsed JSON dictionary.

    Returns:
        A list of strings, each representing the entire content of one code file.
    """
    codes = data.get("code_files", [])
    results: List[str] = []
    for idx, entry in enumerate(codes):
        content_list = entry.get("content", [])
        if isinstance(content_list, list):
            joined = "\n".join(content_list).strip()
        else:
            joined = str(content_list)
        results.append(joined)
        logger.info(f"Loaded code_file[{idx}]: path='{entry.get('path', '')}', length={len(joined)} chars.")
    if not results:
        logger.warning("No 'code_files' found or empty.")
    return results


def get_license_texts(data: Dict[str, Any]) -> List[str]:
    """
    Extract license file contents. Joins each 'content' array into one string per license.

    Parameters:
        data: The parsed JSON dictionary.

    Returns:
        A list of strings, each representing the entire license file content.
    """
    licenses = data.get("license_files", [])
    results: List[str] = []
    for idx, entry in enumerate(licenses):
        content_list = entry.get("content", [])
        if isinstance(content_list, list):
            joined = "\n".join(content_list).strip()
        else:
            joined = str(content_list)
        results.append(joined)
        logger.info(f"Loaded license_file[{idx}]: path='{entry.get('path', '')}', length={len(joined)} chars.")
    if not results:
        logger.warning("No 'license_files' found or empty.")
    return results


def get_first_code_snippet(data: Dict[str, Any]) -> Optional[str]:
    """
    Get a small snippet (first few lines) from the first code file, for use in prompts.

    Parameters:
        data: The parsed JSON dictionary.

    Returns:
        A string with the first 20 lines (or fewer if the file is shorter) of the first code file.
        Returns None if no code files exist.
    """
    codes = data.get("code_files", [])
    if not codes:
        logger.warning("No code files to extract snippet from.")
        return None
    first_content = codes[0].get("content", [])
    if isinstance(first_content, list):
        snippet_lines = first_content[:20]  # first 20 lines
        snippet = "\n".join(snippet_lines).strip()
    else:
        snippet = str(first_content).splitlines()[:20]
        snippet = "\n".join(snippet)
    logger.info(f"Extracted snippet from first code file (length={len(snippet)} chars).")
    return snippet


def get_all_field_paths(data: Dict[str, Any]) -> List[str]:
    """
    Return a flat list of all file paths (documentation, code, license, structure entries).
    This is useful if you need a single combined list of every referenced path.

    Parameters:
        data: The parsed JSON dictionary.

    Returns:
        A list of file path strings.
    """
    paths: List[str] = []
    for sec in ["repository_structure", "documentation_files", "code_files", "license_files"]:
        entries = data.get(sec, [])
        for entry in entries:
            p = entry.get("path")
            if p:
                paths.append(p)
    if "tree_structure" in data:
        # We might not have explicit paths here, but include them anyway
        paths.extend(data["tree_structure"])
    logger.info(f"Collected total {len(paths)} paths from all sections.")
    return paths
