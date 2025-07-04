# Enhanced Algorithm 2: Comprehensive Repository Analysis and Artifact Structure Extraction

## Overview

Enhanced Algorithm 2 is a sophisticated repository analysis system designed to automatically analyze software repositories, extract structural information, and categorize artifacts for comprehensive understanding. This algorithm serves as a critical component in the AURA framework for automated evaluation of research artifacts by providing detailed insights into repository organization, code quality, documentation coverage, and overall project structure.

The algorithm processes GitHub repositories, local directories, and archive files to generate comprehensive analytical reports that enable automated assessment of software artifacts across different evaluation dimensions.

## Key Features

- **Multi-Source Analysis**: Supports GitHub URLs, local directories, and archive files (ZIP, TAR, etc.)
- **Intelligent File Categorization**: Automatically classifies files into documentation, code, license, and configuration categories
- **Enhanced File Detection**: Recognizes 40+ programming languages and 15+ documentation formats
- **Smart Content Extraction**: Implements intelligent file selection with size limits and priority-based inclusion
- **Tree Structure Generation**: Creates visual directory hierarchy representations
- **Archive Support**: Handles compressed repositories and artifact packages
- **Robust Error Handling**: Gracefully manages encoding issues, large files, and corrupted content
- **AURA Framework Integration**: Seamlessly integrates with the broader AURA research artifact evaluation system

## Technical Methodology

### Why This Approach? The Research Problem
Research artifact evaluation requires understanding the structure, quality, and completeness of software repositories. Manual analysis is time-consuming and inconsistent. Key challenges include:
- Identifying different types of artifacts within repositories
- Understanding project organization and documentation quality
- Assessing code structure and maintainability
- Evaluating reproducibility based on available files
- Comparing repositories across different programming languages and domains

Enhanced Algorithm 2 solves this by automatically extracting and categorizing repository contents, enabling systematic comparison and evaluation of software artifacts.

### 1. Repository Acquisition and Preprocessing

#### Multi-Source Input Handling
**Purpose**: Support diverse artifact submission formats common in academic conferences and research contexts.

**Supported Input Types**:
- **GitHub URLs**: `https://github.com/username/repository`
- **Local Directories**: `/path/to/local/repository`
- **Archive Files**: `.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, `.tar.xz`, `.7z`, `.rar`

**How It Works**:
- **URL Detection**: Validates GitHub URLs and handles various URL formats
- **Git Cloning**: Uses GitPython library for efficient repository cloning
- **Archive Extraction**: Automatically detects and extracts compressed files using shutil
- **Directory Validation**: Ensures input directories exist and are accessible

**Why This Matters**: Research artifacts are submitted in various formats. Supporting multiple input types ensures comprehensive coverage of different submission scenarios.

#### Repository Size Analysis
**Purpose**: Understand the scale and complexity of repositories for appropriate processing strategies.

**Metrics Calculated**:
- **Total Size**: Complete repository size in MB
- **File Count**: Number of files across all categories
- **Directory Depth**: Maximum nesting level of directories
- **Size Distribution**: Distribution of file sizes across different categories

### 2. Enhanced File Detection and Categorization

#### Comprehensive File Type Recognition
**Purpose**: Accurately categorize files to understand repository composition and quality.

**File Categories and Extensions**:

**Documentation Files** (40+ formats):
```python
DOCUMENT_EXTENSIONS = [
    # Markdown and text files
    '.md', '.rst', '.txt', '.rtf',
    # Documentation formats  
    '.tex', '.latex', '.org', '.asciidoc', '.adoc',
    # Web documentation
    '.html', '.htm', '.xml'
]
```

**Code Files** (60+ extensions across 25+ languages):
```python
CODE_EXTENSIONS = [
    # Python: '.py', '.pyx', '.pyi', '.ipynb'
    # JavaScript/TypeScript: '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'
    # Java/JVM: '.java', '.scala', '.kt', '.groovy', '.clj'
    # C/C++: '.c', '.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx'
    # And many more...
]
```

**Configuration and Build Files**:
```python
CONFIG_BUILD_EXTENSIONS = [
    '.gradle', '.maven', '.sbt', '.json', '.yaml', '.yml', '.toml',
    '.dockerfile', '.travis.yml', '.requirements', '.npmrc'
]
```

#### Intelligent File Prioritization
**Purpose**: Ensure important files are always included while managing analysis scope.

**Priority Levels**:
1. **Critical Files**: README, LICENSE, main configuration files
2. **Important Files**: Core source code, primary documentation
3. **Supporting Files**: Tests, examples, additional documentation
4. **Auxiliary Files**: Build artifacts, temporary files

**Selection Algorithm**:
```python
def generate_file_list(root, max_files_per_dir=10, max_file_size_kb=2048):
    # Always include critical files regardless of limits
    # Select important files up to directory limits  
    # Include representative sample of supporting files
    # Skip auxiliary files if space limited
```

### 3. Content Extraction and Processing

#### Multi-Encoding Text Processing
**Purpose**: Handle international repositories with various character encodings.

**Encoding Strategy**:
1. **Primary**: UTF-8 with BOM handling
2. **Fallback**: Latin-1, CP1252 for Windows files
3. **Binary Detection**: Identify and skip binary files
4. **Error Recovery**: Replace undecodable characters gracefully

**Content Cleaning**:
- **Empty Line Removal**: Filter out blank lines for concise content
- **Encoding Normalization**: Standardize character representations
- **Content Validation**: Verify text content quality and readability

#### File Size Management
**Purpose**: Balance comprehensive analysis with processing efficiency.

**Size Limits**:
- **Maximum File Size**: 2MB per file (configurable)
- **Maximum Files per Directory**: 10 files (prioritized selection)
- **Minimum Critical Files**: Always include at least 3 important files
- **Total Repository Limit**: Adaptive based on repository size

### 4. Structural Analysis and Tree Generation

#### Directory Hierarchy Mapping
**Purpose**: Understand repository organization and project structure patterns.

**Tree Generation Process**:
1. **Path Normalization**: Standardize path separators across platforms
2. **Node Construction**: Build hierarchical tree using anytree library
3. **Visual Rendering**: Generate ASCII tree representation
4. **Relationship Mapping**: Identify parent-child directory relationships

**Tree Structure Benefits**:
- **Visual Navigation**: Quick understanding of project layout
- **Organization Assessment**: Evaluate project structure quality
- **Pattern Recognition**: Identify common repository patterns
- **Completeness Checking**: Verify expected directories exist

## Input Requirements

### Supported Input Formats

#### 1. GitHub Repository URLs
**Format**: `https://github.com/username/repository`

**Examples**:
- `https://github.com/sneh2001patel/ml-image-classifier`
- `https://github.com/tensorflow/tensorflow`
- `https://github.com/microsoft/vscode`

**Authentication**: 
- Public repositories: No authentication required
- Private repositories: Requires Git credentials or token setup

#### 2. Local Directory Paths
**Format**: Absolute or relative directory paths

**Examples**:
- `/home/user/projects/my-repository`
- `../research-artifacts/algorithm-implementation`
- `C:\Users\Researcher\Desktop\submitted-artifact`

**Requirements**:
- Directory must exist and be readable
- Should contain typical repository structure
- No specific naming convention required

#### 3. Archive Files
**Supported Formats**:
- **ZIP files**: `.zip`
- **TAR archives**: `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`
- **7-Zip archives**: `.7z`
- **RAR archives**: `.rar` (if rarfile installed)

**Archive Processing**:
- Automatic format detection
- Temporary extraction to analysis directory
- Handles nested archives (single level)
- Preserves directory structure

### Input Validation and Preprocessing

#### Quality Checks
**Repository Structure Validation**:
- Minimum file count (>3 files recommended)
- Basic directory structure presence
- Text file accessibility
- Size reasonableness (under 10GB recommended)

**Content Quality Assessment**:
- Text file encoding validation
- Binary file identification
- Corrupt file detection
- Empty repository handling

#### Preprocessing Steps
1. **Path Sanitization**: Remove invalid characters and normalize paths
2. **Permission Verification**: Ensure read access to all target files
3. **Size Calculation**: Pre-calculate repository size for processing planning
4. **Exclusion Filtering**: Skip system files, caches, and build artifacts

## Output Structure and Analysis Results

### Complete Output Schema

The algorithm generates a comprehensive JSON file with the following structure:

```json
{
  "repository_structure": [
    {
      "name": "filename.ext",
      "path": "relative/path/to/file",  
      "mime_type": "text/python",
      "size_kb": 1.23
    }
  ],
  "documentation_files": [
    {
      "path": "README.md",
      "content": ["# Project Title", "Description line", "..."]
    }
  ],
  "code_files": [
    {
      "path": "src/main.py", 
      "content": ["import os", "def main():", "..."]
    }
  ],
  "license_files": [
    {
      "path": "LICENSE",
      "content": ["MIT License", "Copyright (c) 2024", "..."]
    }
  ],
  "tree_structure": [
    "repository-name",
    "├── README.md",
    "├── src/",
    "│   └── main.py",
    "└── LICENSE"
  ],
  "repo_path": "/path/to/analysis/location",
  "repo_size_mb": 15.67
}
```

### Field Descriptions and Usage

#### Repository Structure Array
**Purpose**: Comprehensive inventory of all analyzed files with metadata.

**Fields**:
- **name**: Original filename with extension
- **path**: Relative path from repository root (platform-normalized)
- **mime_type**: MIME type detection (text/python, image/jpeg, etc.)
- **size_kb**: File size in kilobytes (rounded to 2 decimals)

**Usage**:
- File discovery and navigation
- Size distribution analysis
- File type statistics
- Dependency mapping

#### Documentation Files Array
**Purpose**: Extracted content of all documentation and configuration files.

**Content Processing**:
- Non-empty lines only (blank lines removed)
- UTF-8 encoding with fallback handling
- Maximum file size respected (2MB default)
- Maintains original line structure

**File Types Included**:
- README files (all variants)
- Installation guides and tutorials
- API documentation
- Configuration files (JSON, YAML, TOML)
- Build files (Makefile, setup.py, package.json)

#### Code Files Array
**Purpose**: Source code content for analysis and evaluation.

**Languages Supported**:
- **System Programming**: C, C++, Rust, Go
- **Application Development**: Python, Java, C#, JavaScript, TypeScript
- **Data Science**: R, MATLAB, Julia
- **Web Development**: PHP, Ruby, HTML, CSS
- **Functional Programming**: Haskell, Scala, Clojure
- **And many more...**

**Content Features**:
- Syntax-aware processing
- Import/dependency extraction
- Function and class identification
- Comment preservation

#### License Files Array
**Purpose**: Legal and licensing information extraction.

**Detection Patterns**:
- Standard names: LICENSE, LICENCE, COPYING
- Variations: LICENSE.txt, LICENSE.md, LEGAL
- Content-based detection: Copyright notices, license text
- Multi-license handling: Separate entries for each license file

#### Tree Structure Array
**Purpose**: Visual representation of repository hierarchy.

**Format**: ASCII tree using standard characters:
- `├──` for branch connections
- `│` for vertical continuations  
- `└──` for final branches
- Proper indentation for nested levels

**Benefits**:
- Quick project structure understanding
- Organization quality assessment
- Navigation assistance
- Structure pattern recognition

### Real-World Example: ML Image Classifier Analysis

Here's the complete analysis output for the `ml-image-classifier` repository:

<details>
<summary><strong>Complete ml-image-classifier Analysis Output</strong></summary>

```json
{
  "repository_structure": [
    {
      "name": "LICENSE",
      "path": "LICENSE",
      "mime_type": null,
      "size_kb": 1.06
    },
    {
      "name": "README.md",
      "path": "README.md", 
      "mime_type": null,
      "size_kb": 0.3
    },
    {
      "name": ".gitignore",
      "path": ".gitignore",
      "mime_type": null,
      "size_kb": 3.54
    },
    {
      "name": "1000.jpg",
      "path": "data\\testing\\Cat\\1000.jpg",
      "mime_type": "image/jpeg",
      "size_kb": 25.73
    },
    {
      "name": "10005.jpg", 
      "path": "data\\testing\\Cat\\10005.jpg",
      "mime_type": "image/jpeg",
      "size_kb": 11.96
    },
    {
      "name": "1001.jpg",
      "path": "data\\testing\\Cat\\1001.jpg", 
      "mime_type": "image/jpeg",
      "size_kb": 23.21
    },
    {
      "name": "evaluate.py",
      "path": "src\\evaluate.py",
      "mime_type": "text/x-python",
      "size_kb": 1.02
    },
    {
      "name": "model.py",
      "path": "src\\model.py",
      "mime_type": "text/x-python", 
      "size_kb": 0.76
    },
    {
      "name": "train.py",
      "path": "src\\train.py",
      "mime_type": "text/x-python",
      "size_kb": 1.45
    },
    {
      "name": "test_model.py",
      "path": "tests\\test_model.py",
      "mime_type": "text/x-python",
      "size_kb": 0.75
    }
  ],
  "documentation_files": [
    {
      "path": "README.md",
      "content": [
        "# ML Image Classifier",
        "This repository contains the implementation of our",
        "image classification algorithm.",
        "## Installation", 
        "Install dependencies: `pip install -r requirements.txt`",
        "## Usage",
        "Run `python src/train.py` to train the model.",
        "Run `python src/evaluate.py` to evaluate the model."
      ]
    },
    {
      "path": ".gitignore",
      "content": [
        "# Byte-compiled / optimized / DLL files",
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        "# C extensions", 
        "*.so",
        "model.pth",
        "# Distribution / packaging",
        ".Python",
        "build/",
        "develop-eggs/",
        "dist/"
      ]
    }
  ],
  "code_files": [
    {
      "path": "src\\evaluate.py",
      "content": [
        "import torch",
        "from torch.utils.data import DataLoader",
        "from torchvision import datasets, transforms", 
        "from model import CNN",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")",
        "# Transforms",
        "transform = transforms.Compose(",
        "    [",
        "        transforms.Resize((128, 128)),",
        "        transforms.ToTensor(),",
        "        transforms.Normalize([0.5] * 3, [0.5] * 3),",
        "    ]",
        ")"
      ]
    },
    {
      "path": "src\\model.py", 
      "content": [
        "import torch.nn as nn",
        "class CNN(nn.Module):",
        "    def __init__(self):",
        "        super(CNN, self).__init__()",
        "        self.features = nn.Sequential(",
        "            nn.Conv2d(3, 32, 3, padding=1),",
        "            nn.ReLU(),",
        "            nn.MaxPool2d(2),"
      ]
    }
  ],
  "license_files": [
    {
      "path": "LICENSE",
      "content": [
        "MIT License",
        "Copyright (c) 2025 Sneh Patel", 
        "Permission is hereby granted, free of charge, to any person obtaining a copy",
        "of this software and associated documentation files (the \"Software\"), to deal",
        "in the Software without restriction, including without limitation the rights",
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell"
      ]
    }
  ],
  "tree_structure": [
    "ml-image-classifier",
    "├── .gitignore",
    "├── LICENSE", 
    "├── README.md",
    "├── data",
    "│   ├── testing",
    "│   │   ├── Cat",
    "│   │   │   ├── 1000.jpg",
    "│   │   │   ├── 10005.jpg",
    "│   │   │   └── 1001.jpg",
    "│   │   └── Dog",
    "│   │       ├── 1000.jpg",
    "│   │       ├── 10005.jpg", 
    "│   │       └── 1001.jpg",
    "│   └── training",
    "│       ├── Cat",
    "│       │   ├── 0.jpg",
    "│       │   ├── 1.jpg",
    "│       │   └── 10.jpg",
    "│       └── Dog",
    "│           ├── 0.jpg",
    "│           ├── 1.jpg",
    "│           └── 10.jpg",
    "├── src",
    "│   ├── __init__.py",
    "│   ├── evaluate.py",
    "│   ├── model.py",
    "│   └── train.py",
    "└── tests",
        "├── __init__.py",
        "└── test_model.py"
  ],
  "repo_path": "../../temp_dir_for_git\\ml-image-classifier",
  "repo_size_mb": 1583.66
}
```

</details>

## Usage Instructions

### Prerequisites and Dependencies

#### Required Python Packages
```bash
# Core repository analysis dependencies
pip install gitpython anytree

# Additional recommended packages  
pip install python-magic  # Enhanced MIME type detection
pip install rarfile      # RAR archive support (optional)
```

#### System Requirements
- **Python 3.7+**: Core language requirement
- **Git**: For GitHub repository cloning
- **Memory**: 1-4GB RAM depending on repository size
- **Disk Space**: 2-10GB for temporary repository storage

### Basic Usage

#### Command Line Execution
```bash
# Navigate to algorithm directory
cd scripts/algorithm_2/

# Analyze a GitHub repository (basic usage)
python algorithm_2.py https://github.com/username/repository

# Specify custom output directory
python algorithm_2.py https://github.com/username/repository temp_dir output_dir

# Analyze local directory
python algorithm_2.py /path/to/local/repository

# Analyze archive file
python algorithm_2.py /path/to/artifact.zip
```

#### Python Integration
```python
from algorithm_2 import analyze_repository, save_analysis_result

# Analyze GitHub repository
repo_url = "https://github.com/sneh2001patel/ml-image-classifier"
result = analyze_repository(repo_url)

# Save analysis results
save_analysis_result(result, "ml-image-classifier")

# Access analysis data
print(f"Repository size: {result['repo_size_mb']} MB")
print(f"Total files: {len(result['repository_structure'])}")
print(f"Code files: {len(result['code_files'])}")
print(f"Documentation files: {len(result['documentation_files'])}")
```

### Advanced Configuration

#### Customizing File Selection
```python
# Modify file limits in the algorithm
def analyze_with_custom_limits(repo_url):
    # Edit these parameters in generate_file_list function
    custom_limits = {
        'max_files_per_dir': 20,        # Increase from default 10
        'max_file_size_kb': 4096,       # Increase from default 2048
        'min_files_to_include_anyway': 5  # Increase from default 3
    }
    
    # Apply custom limits (requires code modification)
    result = analyze_repository(repo_url)
    return result
```

#### Adding New File Types
```python
# Extend supported file extensions
CUSTOM_CODE_EXTENSIONS = CODE_EXTENSIONS + [
    '.dart',     # Flutter/Dart
    '.sol',      # Solidity
    '.cairo',    # Cairo
    '.move',     # Move language
]

CUSTOM_DOCUMENT_EXTENSIONS = DOCUMENT_EXTENSIONS + [
    '.wiki',     # Wiki files
    '.confluence', # Confluence exports
    '.notion',   # Notion exports
]
```

#### Archive Handling Configuration
```python
# Custom temporary directory for large repositories
import tempfile

def analyze_large_repository(repo_url):
    # Create custom temporary directory with more space
    with tempfile.TemporaryDirectory(dir="/large_disk/temp") as temp_dir:
        result = analyze_repository(repo_url, temp_base_dir=temp_dir)
        return result
```