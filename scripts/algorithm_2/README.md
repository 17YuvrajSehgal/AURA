# Repository Analysis Script

This script provides a comprehensive analysis of GitHub repositories by examining their structure, documentation, code files, and license information. It's designed to help understand the organization and content of software repositories.

## Features

- **Repository Cloning**: Automatically clones GitHub repositories for analysis
- **File Analysis**: Examines different types of files:
  - Documentation files (`.md`, `.rst`, `.txt`)
  - Code files (`.py`, `.java`, `.cpp`, `.js`, `.ts`)
  - License files
- **Directory Structure**: Generates a tree representation of the repository structure
- **File Content Analysis**: Extracts and analyzes file contents while respecting size limits
- **Smart File Selection**: Implements intelligent file selection with configurable limits:
  - Maximum files per directory
  - File size limits
  - Minimum files to include regardless of type

## Requirements

- Python 3.x
- Required Python packages:
  - `gitpython`
  - `anytree`

## Usage

```python
from algorithm_2 import analyze_repository, save_analysis_result

# Analyze a GitHub repository
repo_url = "https://github.com/username/repository"
result = analyze_repository(repo_url)

# Save the analysis results
save_analysis_result(result, "repository_name")
```

## Output Format

The script generates a JSON file containing:

- **Repository Structure**: List of file information including name, path, MIME type, and size
- **Documentation Files**: Content of documentation files
- **Code Files**: Content of code files
- **License Files**: Content of license files
- **Tree Structure**: Visual representation of the repository's directory structure
```json
{
  "repository_structure": [
    {
      "name": "README.md",
      "path": "README.md",
      "mime_type": "text/markdown",
      "size_kb": 1.23
    }
  ],
  "documentation_files": [
    {
      "path": "README.md",
      "content": ["# My Project", "..."]
    },
  ],
  "code_files": [
    {
      "path": "main.py",
      "content": ["import os", "..."]
    }
  ],
  "license_files": [
    {
      "path": "LICENSE",
      "content": ["MIT License", "..."]
    }
  ],
  "tree_structure": [
    "ml-image-classifier",
    "├── README.md",
    "├── main.py",
    "└── LICENSE"
  ]
}

```

## Configuration

The script includes several configurable parameters:

- `DOCUMENT_EXTENSIONS`: File extensions considered as documentation
- `CODE_EXTENSIONS`: File extensions considered as code
- `LICENSE_NAMES`: Common license file names
- `EXCLUDE_DIRS`: Directories to exclude from analysis
- `max_files_per_dir`: Maximum number of files to analyze per directory
- `max_file_size_kb`: Maximum file size to analyze (in KB)

## Error Handling

- Gracefully handles file reading errors
- Skips files that are too large
- Provides logging information for debugging

## Notes

- The script creates a temporary directory for cloned repositories
- Large files (>2MB) are automatically skipped
- Empty lines are removed from file contents
- The script uses UTF-8 encoding with error handling for file reading

## Example

```python
# Example usage
repo_url = "https://github.com/sneh2001patel/ml-image-classifier"
result = analyze_repository(repo_url)
save_analysis_result(result, "ml-image-classifier")
```

This will create a JSON file containing the analysis of the specified repository.
