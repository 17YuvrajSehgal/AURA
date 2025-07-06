#!/usr/bin/env python3
"""
Integrated Artifact Analyzer - Combines robust extraction with comprehensive analysis

This module handles extraction of various artifact formats AND performs comprehensive
analysis similar to algorithm_2, then saves results to JSON.

Features:
- Robust extraction (ZIP, TAR, etc.) with error handling and recursive nested archive extraction
- Comprehensive repository analysis (file categorization, content analysis)
- Smart filtering: excludes binary/executable files (.jar, .exe, .dll, ML models, etc.)
- Directory tree generation
- JSON output with detailed analysis results
- Git repository cloning support
- Configurable file size limits and exclusion rules
"""

import fnmatch
import json
import logging
import mimetypes
import os
import re
import shutil
import tarfile
import zipfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

import unicodedata
from anytree import Node, RenderTree
from git import Repo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Algorithm_2 Constants - File Type Detection
DOCUMENT_EXTENSIONS = ['.md']

CODE_EXTENSIONS = [
    # Python
    '.py', '.pyx', '.pyi', '.ipynb',
    # JavaScript/TypeScript
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    # Java/JVM languages
    '.java', '.scala', '.kt', '.groovy', '.clj',
    # C/C++
    '.c', '.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx',
    # C#/.NET
    '.cs', '.vb', '.fs',
    # Web languages
    '.php', '.rb', '.go', '.rs', '.swift',
    # Shell/Scripts
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
    # Data/Config as code
    '.sql', '.r', '.R', '.m', '.pl', '.lua', '.dart',
    # Functional languages
    '.hs', '.elm', '.ml', '.f90', '.f95', '.f03',
    # Assembly
    '.asm', '.s',
    # Other
    '.vim', '.el'
]

# === NEW: Executable and Binary File Extensions to Exclude ===
EXECUTABLE_EXTENSIONS = [
    # Java executables and archives
    '.jar', '.war', '.ear', '.class',
    # System executables and libraries
    '.exe', '.dll', '.so', '.dylib', '.app',
    # Compiled objects and libraries
    '.bin', '.obj', '.o', '.a', '.lib',
    # Python compiled files
    '.pyc', '.pyo', '.pyd',
    # Package files
    '.whl', '.egg', '.deb', '.rpm', '.msi', '.pkg', '.dmg',
    # Machine Learning models and data
    '.pkl', '.pickle', '.joblib', '.h5', '.hdf5', '.pt', '.pth', 
    '.model', '.weights', '.ckpt', '.pb', '.onnx', '.tflite',
    '.npz', '.npy', '.mat', '.rds', '.rda',
    # Archives (handled separately but should be excluded from text analysis)
    '.zip', '.tar', '.gz', '.7z', '.rar', '.bz2', '.xz', '.tgz',
    # Document files (binary)
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.odt', '.ods', '.odp', '.rtf',
    # Image files
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico', '.tiff',
    '.webp', '.raw', '.psd', '.ai', '.eps',
    # Media files
    '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac', '.ogg', '.mkv',
    '.webm', '.wmv', '.m4a', '.aac',
    # Database files
    '.db', '.sqlite', '.sqlite3', '.mdb', '.accdb',
    # Disk images
    '.iso', '.img', '.vhd', '.vmdk',
    # Font files
    '.ttf', '.otf', '.woff', '.woff2', '.eot',
    # Other binary formats
    '.swf', '.fla', '.unity', '.blend', '.3ds', '.max'
]

# Large data file extensions (typically contain non-textual data)
LARGE_DATA_EXTENSIONS = [
    '.csv', '.tsv', '.parquet', '.arrow', '.feather', '.avro',
    '.json', '.jsonl', '.xml', '.yaml', '.yml'  # Keep these but with size limits
]

LICENSE_PATTERNS = [
    'license', 'licence', 'license.txt', 'licence.txt',
    'license.md', 'licence.md', 'copying', 'copying.md',
    'copying.txt', 'copyright', 'copyright.txt', 'copyright.md',
    'legal', 'legal.txt', 'legal.md', 'notice', 'notice.txt',
    'notice.md', 'authors', 'authors.txt', 'authors.md',
    'contributors', 'contributors.txt', 'contributors.md',
    'acknowledgments', 'acknowledgments.txt', 'acknowledgments.md',
    'credits', 'credits.txt', 'credits.md'
]

DOCUMENTATION_PATTERNS = [
    'readme', 'read_me', 'read-me',
    'contributing', 'contribute', 'contribution',
    'changelog', 'changes', 'history', 'news', 'releases',
    'install', 'installation', 'setup', 'getting-started',
    'getting_started', 'quickstart', 'quick-start',
    'tutorial', 'guide', 'manual', 'documentation', 'docs',
    'todo', 'roadmap', 'milestones', 'development',
    'architecture', 'design', 'overview', 'summary',
    'faq', 'troubleshooting', 'debugging', 'testing',
    'security', 'code_of_conduct', 'code-of-conduct',
    'maintainers', 'governance', 'support', 'contact',
    'api', 'reference', 'specification', 'spec',
    'requirements', 'dependencies', 'constraints',
    'building', 'build', 'compile', 'deployment',
    'configuration', 'config', 'settings'
]

CONFIG_BUILD_EXTENSIONS = [
    '.gradle', '.maven', '.sbt', '.ant',
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.dockerfile',
    '.travis.yml', '.appveyor.yml', '.gitlab-ci.yml',
    '.requirements', '.pip',
    '.npmrc', '.yarnrc',
    '.cmake', '.make', '.mk', '.pro', '.qbs'
]

CONFIG_FILE_PATTERNS = [
    'makefile', 'makefile.am', 'makefile.in', 'cmakelists.txt',
    'build.gradle', 'pom.xml', 'build.xml', 'setup.py', 'setup.cfg',
    'pyproject.toml', 'requirements.txt', 'pipfile', 'pipfile.lock',
    'package.json', 'package-lock.json', 'yarn.lock', 'composer.json',
    'dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
    '.dockerignore',
    '.travis.yml', '.github', '.gitlab-ci.yml', 'appveyor.yml',
    'jenkins', 'jenkinsfile',
    '.env', '.env.example', 'environment.yml', 'conda.yml',
    'tox.ini', 'pytest.ini', '.coveragerc', '.gitignore', '.gitattributes',
    '.vscode', '.idea', '.project', '.classpath'
]

EXCLUDE_DIRS = [
    '__MACOSX',
    '.git', '.svn', '.hg', '.bzr', 'node_modules', 'bower_components', '.yarn', '.pnp', '.parcel-cache',
    '__pycache__', '.pytest_cache', '.mypy_cache', '.tox', '.coverage', '.hypothesis',
    '.venv', 'venv', 'env', '.env', '.virtualenv', '.conda', '.ipynb_checkpoints',
    'build', 'dist', 'out', 'output', 'target', 'bin', 'obj', '.gradle', '.idea', '.vscode', '.settings',
    '.cache', 'logs', 'log', '.next', '.nuxt', '.expo', '.android', '.ios', '.history', '.npm', '.jspm',
    '.c9', '.cloud9', '.sublime-*', '.nyc_output', '.mocha', '.istanbul', '.jupyter',
    '.Rproj.user', '.Rhistory', '.RData', '.Ruserdata', '.snapshots', '.metadata', '.factorypath',
    '.tmp', '.bak', '.old', '.orig', '.save', '.crash', '.pid', '.seed', '.pid.lock',
    '.yarn-cache', '.yarnrc', '.pnp.*', '.eslintcache', '.pyre', '.pytype', 'data', 'dataset'
]

EXCLUDE_FILES = [
    'DS_Store','.DS_Store', 'Thumbs.db', 'desktop.ini', '.Trash-*', '.AppleDouble', '.LSOverride',
    '*.log', '*.tmp', '*.swp', '*.swo'
]

BUILD_FILES = ['Makefile', 'CMakeLists.txt', 'build.gradle', 'pom.xml', 'package.json', 'requirements.txt']
DOCKER_FILES = ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml']
DATA_EXTENSIONS = ['.csv', '.json', '.xml', '.sql', '.db', '.sqlite']
CONFIG_EXTENSIONS = ['.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf']


class IntegratedArtifactAnalyzer:
    """
    Integrated artifact analyzer that combines robust extraction with comprehensive analysis.
    
    This class extracts artifacts (ZIP, TAR, directories) and performs detailed analysis
    similar to algorithm_2, then saves results to JSON.
    """

    def __init__(self, temp_dir: str = "../../temp_dir_for_git", output_dir: str = "../../algo_outputs/algorithm_2_output",
                 max_file_size: int = 5000 * 1024 * 1024, max_recursion_depth: int = 3, extraction_timeout: int = 300):
        """
        Initialize the IntegratedArtifactAnalyzer.
        
        Args:
            temp_dir: Temporary directory for extractions (default: ../../temp_dir_for_git)
            output_dir: Directory for saving analysis results (default: ../../algorithm_2_output)
            max_file_size: Maximum file size to process (default: 5000MB)
            max_recursion_depth: Maximum depth for recursive archive extraction (default: 3)
            extraction_timeout: Maximum time in seconds for nested extraction (default: 300 = 5 minutes)
        """
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self.max_file_size = max_file_size
        self.max_recursion_depth = max_recursion_depth
        self.extraction_timeout = extraction_timeout
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Windows invalid characters
        self.invalid_chars = r'[<>:"/\\|?*]'
        self.invalid_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
            'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
            'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }

        # Supported archive extensions
        self.archive_extensions = {
            '.zip': self._extract_zip_robust,
            '.tar': self._extract_tar_robust,
            '.tar.gz': self._extract_tar_robust,
            '.tgz': self._extract_tar_robust,
            '.tar.bz2': self._extract_tar_robust,
            '.tar.xz': self._extract_tar_robust,
        }

        # Extraction statistics
        self.extraction_stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'files_renamed': 0,
            'encoding_issues': 0,
            'path_issues': 0,
            'skipped_files': [],
            'renamed_files': [],
            'nested_archives_found': 0,
            'nested_extraction_failures': 0
        }

    def analyze_artifact(
            self,
            artifact_path: str,
            artifact_name: Optional[str] = None,
            force_reextract: bool = False,
            skip_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Extract and analyze an artifact with comprehensive analysis.
        
        Args:
            artifact_path: Path to the artifact (file, directory, or URL)
            artifact_name: Optional custom name for the artifact
            force_reextract: Force re-extraction even if already extracted
            skip_analysis: Skip detailed analysis and just extract
            
        Returns:
            Dictionary containing extraction and analysis results
        """
        artifact_path_obj = Path(artifact_path)

        # Handle different input types
        if self._is_github_url(artifact_path):
            # Clone repository
            logger.info(f"Cloning repository: {artifact_path}")
            extracted_path = self._clone_repository(artifact_path)
            artifact_name = artifact_name or artifact_path.split('/')[-1]
            extraction_result = {
                "success": True,
                "artifact_name": artifact_name,
                "artifact_path": artifact_path,
                "extracted_path": extracted_path,
                "extraction_method": "git_clone",
                "extraction_stats": {"reused_existing": False}
            }
        elif artifact_path_obj.is_dir():
            # Use directory directly
            extracted_path = str(artifact_path_obj)
            artifact_name = artifact_name or artifact_path_obj.name
            extraction_result = {
                "success": True,
                "artifact_name": artifact_name,
                "artifact_path": artifact_path,
                "extracted_path": extracted_path,
                "extraction_method": "directory_direct",
                "extraction_stats": {"reused_existing": False}
            }
        else:
            # Extract archive
            extraction_result = self._extract_artifact(artifact_path, artifact_name, force_reextract)
            if not extraction_result["success"]:
                return extraction_result
            extracted_path = extraction_result["extracted_path"]
            artifact_name = extraction_result["artifact_name"]

        # Perform comprehensive analysis
        if not skip_analysis:
            logger.info(f"Performing comprehensive analysis on: {artifact_name}")
            analysis_result = self._analyze_repository(extracted_path)

            # Merge extraction and analysis results
            result = {
                **extraction_result,
                **analysis_result,
                "analysis_performed": True
            }

            # Save results to JSON
            self._save_analysis_result(result, artifact_name)
        else:
            result = {
                **extraction_result,
                "analysis_performed": False
            }

        return result

    def _is_github_url(self, url: str) -> bool:
        """Check if the URL is a GitHub URL."""
        return isinstance(url, str) and (url.startswith("http://") or url.startswith("https://"))

    def _clone_repository(self, repo_url: str) -> str:
        """Clone a git repository."""
        repo_name = repo_url.rstrip("/").split("/")[-1]
        clone_path = self.temp_dir / f"{repo_name}"

        if clone_path.exists():
            logger.info(f"Repository already cloned at: {clone_path}")
            return str(clone_path)

        logger.info(f"Cloning repository: {repo_url} → {clone_path}")
        try:
            Repo.clone_from(repo_url, clone_path)
            logger.info("Cloned successfully.")
            return str(clone_path)
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise

    def _extract_artifact(
            self,
            artifact_path: str,
            artifact_name: Optional[str] = None,
            force_reextract: bool = False
    ) -> Dict[str, Any]:
        """Extract an artifact using the robust extraction logic."""
        artifact_path = Path(artifact_path)

        if not artifact_path.exists():
            return {
                "success": False,
                "error": f"Artifact not found: {artifact_path}",
                "artifact_path": str(artifact_path)
            }

        if artifact_name is None:
            artifact_name = artifact_path.stem

        # Sanitize artifact name
        artifact_name = self._sanitize_filename(artifact_name)

        # Create extraction directory
        extract_dir = self.temp_dir / f"extracted_{artifact_name}"

        # Check if already extracted
        if extract_dir.exists() and not force_reextract:
            logger.info(f"Artifact already extracted, reusing: {artifact_name}")
            return {
                "success": True,
                "artifact_name": artifact_name,
                "artifact_path": str(artifact_path),
                "extracted_path": str(extract_dir),
                "extraction_method": "reused_existing",
                "extraction_stats": {"reused_existing": True}
            }

        logger.info(f"Extracting artifact: {artifact_name}")

        # Reset extraction statistics
        self.extraction_stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'files_renamed': 0,
            'encoding_issues': 0,
            'path_issues': 0,
            'skipped_files': [],
            'renamed_files': [],
            'nested_archives_found': 0,
            'nested_extraction_failures': 0
        }

        result = {
            "success": False,
            "artifact_name": artifact_name,
            "artifact_path": str(artifact_path),
            "extracted_path": str(extract_dir),
            "extraction_method": None,
            "error": None,
            "extraction_stats": {}
        }

        try:
            if artifact_path.is_dir():
                # Copy directory with robust handling
                self._copy_directory_robust(artifact_path, extract_dir)
                result["extraction_method"] = "directory_copy_robust"
                
                # Perform recursive extraction of nested archives
                self._extract_nested_archives(extract_dir, depth=0)
            elif artifact_path.is_file():
                # Check file size
                file_size = artifact_path.stat().st_size
                if file_size > self.max_file_size:
                    return {
                        **result,
                        "error": f"File too large: {file_size} bytes (max: {self.max_file_size})"
                    }

                # Determine extraction method
                extraction_method = self._get_extraction_method(artifact_path)
                if extraction_method is None:
                    return {
                        **result,
                        "error": f"Unsupported file format: {artifact_path.suffix}"
                    }

                # Extract archive
                extract_dir.mkdir(parents=True, exist_ok=True)
                extraction_method(artifact_path, extract_dir)
                result["extraction_method"] = extraction_method.__name__
                
                # Perform recursive extraction of nested archives
                self._extract_nested_archives(extract_dir, depth=0)

            result["extraction_stats"] = self.extraction_stats.copy()
            result["success"] = True
            logger.info(f"Successfully extracted artifact: {artifact_name}")

        except Exception as e:
            error_msg = f"Extraction failed for {artifact_name}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            result["extraction_stats"] = self.extraction_stats.copy()

            # Clean up on failure
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

        return result

    def _analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """Perform comprehensive repository analysis using algorithm_2 logic."""
        repo_path = Path(repo_path)

        # Calculate repository size
        repo_size_mb = self._calculate_repo_size(repo_path)

        # Enhanced file collection
        all_extensions = set(DOCUMENT_EXTENSIONS + CODE_EXTENSIONS + CONFIG_BUILD_EXTENSIONS)
        file_paths = self._generate_file_list(
            str(repo_path),
            exclude_dirs=EXCLUDE_DIRS,
            max_files_per_dir=10,
            allowed_extensions=all_extensions,
            max_file_size_kb=2048,
            min_files_to_include_anyway=3
        )

        # Initialize collections
        S = []  # Repository structure
        M, C, L, F = [], [], [], []  # Documentation, code, license, config files
        BUILD, DOCKER, DATA, CONFIG = [], [], [], []

        for path in file_paths:
            try:
                # Additional safety check for excluded files
                should_exclude, exclude_reason = self._should_exclude_file(path)
                if should_exclude:
                    logger.debug(f"Skipping excluded file: {path} ({exclude_reason})")
                    continue

                file_info = {
                    "name": os.path.basename(path),
                    "path": os.path.relpath(path, repo_path),
                    "mime_type": mimetypes.guess_type(path)[0],
                    "size_kb": round(os.path.getsize(path) / 1024, 2)
                }
                S.append(file_info)

                # Read file content
                content_lines = self._read_file_content(path)

                # Categorize files
                if self._is_license_file(path):
                    L.append({"path": os.path.relpath(path, repo_path), "content": content_lines})
                elif self._is_documentation_file(path):
                    M.append({"path": os.path.relpath(path, repo_path), "content": content_lines})
                elif self._is_docker_file(path):
                    DOCKER.append({"path": os.path.relpath(path, repo_path), "content": content_lines})
                elif self._is_build_file(path):
                    BUILD.append(file_info)
                elif self._is_code_file(path):
                    C.append({"path": os.path.relpath(path, repo_path), "content": content_lines})
                elif self._is_data_file(path):
                    DATA.append(file_info)
                elif self._is_config_file(path):
                    CONFIG.append(file_info)
                elif self._is_config_build_file(path):
                    M.append({"path": os.path.relpath(path, repo_path), "content": content_lines})

            except Exception as e:
                logger.warning(f"Error processing file {path}: {e}")
                continue

        # Generate a directory tree
        try:
            tree_lines = self._generate_tree_lines(file_paths, str(repo_path))
        except Exception as e:
            logger.error(f"Error generating tree structure: {e}")
            tree_lines = [f"[ERROR GENERATING TREE]: {e}"]

        # Log statistics
        logger.info(f"Analysis complete: {len(S)} total files processed")
        logger.info(f"  - Documentation files: {len(M)}")
        logger.info(f"  - Code files: {len(C)}")
        logger.info(f"  - License files: {len(L)}")
        logger.info(f"  - Build files: {len(BUILD)}")
        logger.info(f"  - Docker files: {len(DOCKER)}")
        logger.info(f"  - Data files: {len(DATA)}")
        logger.info(f"  - Config files: {len(CONFIG)}")

        return {
            #"repository_structure": S,
            "documentation_files": M,
            "code_files": C,
            "license_files": L,
            "tree_structure": tree_lines,
            "repo_path": str(repo_path),
            "repo_size_mb": repo_size_mb,
            "build_files": BUILD,
            "docker_files": DOCKER,
            "data_files": DATA,
            "config_files": CONFIG
        }

    def _save_analysis_result(self, result: dict, artifact_name: str):
        """Save analysis results to JSON file."""
        try:
            json_path = self.output_dir / f"{artifact_name}_analysis.json"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved analysis result to: {json_path}")

            # Log summary statistics
            if result.get("analysis_performed", False):
                stats = result
                logger.info(f"Summary: {len(stats.get('repository_structure', []))} total files, "
                            f"{len(stats.get('documentation_files', []))} documentation, "
                            f"{len(stats.get('code_files', []))} code, "
                            f"{len(stats.get('license_files', []))} license files")

        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")
            raise

    # === File Detection Methods (from algorithm_2) ===

    def _is_documentation_file(self, path: str) -> bool:
        """Documentation file detection: only .md files."""
        ext = os.path.splitext(path.lower())[1]
        return ext == '.md'

    def _is_code_file(self, path: str) -> bool:
        """Enhanced code file detection."""
        ext = os.path.splitext(path.lower())[1]
        return ext in CODE_EXTENSIONS

    def _is_license_file(self, path: str) -> bool:
        """Enhanced license file detection."""
        name = os.path.basename(path).lower()
        base_name = os.path.splitext(name)[0]

        if name in LICENSE_PATTERNS or base_name in LICENSE_PATTERNS:
            return True

        license_keywords = ['license', 'licence', 'copying', 'copyright', 'legal',
                            'notice', 'authors', 'contributors', 'credits', 'acknowledgment']
        return any(keyword in name for keyword in license_keywords)

    def _is_config_build_file(self, path: str) -> bool:
        """Detect configuration and build files."""
        name = os.path.basename(path).lower()
        base_name = os.path.splitext(name)[0]
        ext = os.path.splitext(name)[1]

        if ext in CONFIG_BUILD_EXTENSIONS:
            return True

        if name in CONFIG_FILE_PATTERNS or base_name in CONFIG_FILE_PATTERNS:
            return True

        if any(pattern in path.lower() for pattern in ['.github', '.gitlab', '.vscode', '.idea']):
            return True

        return False

    def _is_build_file(self, path: str) -> bool:
        """Check if file is a build file."""
        name = os.path.basename(path)
        return name in BUILD_FILES

    def _is_docker_file(self, path: str) -> bool:
        """Detect Docker-related files."""
        name = os.path.basename(path).lower()

        if name == '.dockerignore':
            return True
        if name.startswith('docker-compose'):
            return True
        if name.startswith('dockerfile'):
            return True
        if name.startswith('docker') and name != 'docker':
            return True
        return False

    def _is_data_file(self, path: str) -> bool:
        """Check if file is a data file."""
        ext = os.path.splitext(path)[1].lower()
        return ext in DATA_EXTENSIONS

    def _is_config_file(self, path: str) -> bool:
        """Check if file is a configuration file."""
        ext = os.path.splitext(path)[1].lower()
        return ext in CONFIG_EXTENSIONS

    def _is_important_file(self, filename: str) -> bool:
        """Check if a file is considered important."""
        name_lower = filename.lower()
        base_name = os.path.splitext(name_lower)[0]

        important_patterns = ['readme', 'license', 'licence', 'contributing',
                              'changelog', 'makefile', 'dockerfile', 'setup.py',
                              'requirements.txt', 'package.json', 'pom.xml']

        return (name_lower in important_patterns or
                base_name in important_patterns or
                any(pattern in name_lower for pattern in ['readme', 'license', 'licence']))

    def _is_executable_or_binary_file(self, path: str) -> bool:
        """Check if file is an executable or binary file that should be excluded."""
        ext = os.path.splitext(path)[1].lower()
        return ext in EXECUTABLE_EXTENSIONS

    def _is_large_data_file(self, path: str, max_size_mb: float = 10.0) -> bool:
        """Check if file is a large data file that should be excluded or size-limited."""
        ext = os.path.splitext(path)[1].lower()
        
        if ext not in LARGE_DATA_EXTENSIONS:
            return False
            
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            return size_mb > max_size_mb
        except (OSError, IOError):
            return False

    def _is_in_excluded_dir(self, file_path: str) -> bool:
        """Check if file path is within any excluded directory."""
        # Normalize path separators
        normalized_path = file_path.replace('\\', '/')
        path_parts = normalized_path.split('/')
        
        # Check if any part of the path matches excluded directories
        for part in path_parts:
            if part in EXCLUDE_DIRS:
                return True
            
            # Check against patterns with wildcards
            for exclude_pattern in EXCLUDE_DIRS:
                if '*' in exclude_pattern and fnmatch.fnmatch(part, exclude_pattern):
                    return True
        
        return False

    def _should_exclude_file(self, path: str) -> tuple[bool, str]:
        """
        Check if a file should be excluded from analysis.
        
        Returns:
            (should_exclude, reason)
        """
        # Check if it's an executable or binary file
        if self._is_executable_or_binary_file(path):
            return True, "executable/binary file"
        
        # Check if it's a large data file
        if self._is_large_data_file(path):
            return True, "large data file"
            
        # Check file size (general limit)
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb > 50:  # 50MB limit for any file
                return True, f"file too large ({size_mb:.1f}MB)"
        except (OSError, IOError):
            pass
            
        return False, ""

    # === File Processing Methods ===

    def _read_file_content(self, path: str) -> List[str]:
        """Read file content with robust encoding handling."""
        try:
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(path, 'r', encoding=encoding, errors='ignore') as f:
                        raw = f.read()
                        all_lines = raw.splitlines()
                        return [line for line in all_lines if line.strip() != ""]
                except UnicodeDecodeError:
                    continue

            # Binary fallback
            with open(path, 'rb') as f:
                raw_bytes = f.read()
                try:
                    raw = raw_bytes.decode('utf-8', errors='replace')
                    all_lines = raw.splitlines()
                    return [line for line in all_lines if line.strip() != ""]
                except:
                    return [f"[BINARY FILE - COULD NOT READ TEXT CONTENT]"]

        except Exception as e:
            logger.error(f"Failed to read file: {path} — {e}")
            return [f"[ERROR READING FILE]: {e}"]

    def _generate_file_list(
            self,
            root: str,
            exclude_dirs=EXCLUDE_DIRS,
            max_files_per_dir: int = 10,
            allowed_extensions=None,
            max_file_size_kb: int = 2048,
            min_files_to_include_anyway: int = 3
    ) -> List[str]:
        """Enhanced file collection with better coverage."""
        file_paths = []
        excluded_files_count = 0
        excluded_reasons = {}
        
        for dirpath, dirnames, filenames in os.walk(root):
            # Exclude directories
            dirnames[:] = [d for d in dirnames if
                           d not in exclude_dirs and not any(
                               fnmatch.fnmatch(d, pat) for pat in exclude_dirs if '*' in pat)]

            # Filter files
            filtered_filenames = []
            for filename in filenames:
                if filename in EXCLUDE_FILES or any(
                        fnmatch.fnmatch(filename, pat) for pat in EXCLUDE_FILES if '*' in pat):
                    continue
                filtered_filenames.append(filename)
            filenames = filtered_filenames

            # Prioritize files
            important_files = []
            selected = []
            sampled = []

            for filename in sorted(filenames):
                full_path = os.path.join(dirpath, filename)
                ext = os.path.splitext(full_path)[1].lower()

                # # Check if file should be excluded (NEW)
                # should_exclude, exclude_reason = self._should_exclude_file(full_path)
                # if should_exclude:
                #     excluded_files_count += 1
                #     excluded_reasons[exclude_reason] = excluded_reasons.get(exclude_reason, 0) + 1
                #     continue

                try:
                    size_kb = os.path.getsize(full_path) / 1024
                    if size_kb > max_file_size_kb:
                        # excluded_files_count += 1
                        # excluded_reasons["file too large (KB)"] = excluded_reasons.get("file too large (KB)", 0) + 1
                        continue
                except (OSError, IOError):
                    continue

                if self._is_important_file(filename):
                    important_files.append(full_path)
                elif allowed_extensions and ext in allowed_extensions:
                    selected.append(full_path)
                elif len(sampled) < min_files_to_include_anyway:
                    sampled.append(full_path)

                if len(selected) >= max_files_per_dir:
                    break

            file_paths.extend(important_files + selected[:max_files_per_dir] + sampled)

        # Log exclusion statistics
        # if excluded_files_count > 0:
        #     logger.info(f"Excluded {excluded_files_count} files from analysis:")
        #     for reason, count in excluded_reasons.items():
        #         logger.info(f"  - {reason}: {count} files")

        return file_paths

    def _generate_tree_lines(self, file_paths: List[str], root_dir: str) -> List[str]:
        """Generate directory tree structure."""
        logger.info("Generating directory tree structure...")
        root_node = Node(os.path.basename(root_dir))
        node_map = {root_dir: root_node}

        for path in sorted(file_paths):
            try:
                rel_path = os.path.relpath(path, root_dir)
                parts = rel_path.split(os.sep)
                current_path = root_dir
                parent = root_node
                for part in parts:
                    current_path = os.path.join(current_path, part)
                    if current_path not in node_map:
                        node_map[current_path] = Node(part, parent=parent)
                    parent = node_map[current_path]
            except Exception as e:
                logger.warning(f"Error processing path {path}: {e}")
                continue

        lines = []
        try:
            for pre, _, node in RenderTree(root_node):
                lines.append(f"{pre}{node.name}")
        except Exception as e:
            logger.error(f"Error generating tree: {e}")
            lines = [f"[ERROR GENERATING TREE]: {e}"]

        return lines

    def _calculate_repo_size(self, repo_path: Path) -> float:
        """Calculate repository size in MB."""
        total_size = 0
        if repo_path.exists():
            for dirpath, dirnames, filenames in os.walk(repo_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.isfile(fp):
                        try:
                            total_size += os.path.getsize(fp)
                        except (OSError, IOError):
                            continue
        return round(total_size / (1024 * 1024), 2)

    # === Robust Extraction Methods ===

    def _sanitize_filename(self, filename: str, used_names: Set[str] = None) -> str:
        """Sanitize filename for Windows compatibility."""
        if used_names is None:
            used_names = set()

        try:
            filename = unicodedata.normalize('NFKD', filename)
            filename = filename.encode('ascii', 'ignore').decode('ascii')
        except (UnicodeError, UnicodeDecodeError):
            filename = re.sub(r'[^\x00-\x7F]+', '_', filename)

        sanitized = re.sub(self.invalid_chars, '_', filename)

        name_part = sanitized.split('.')[0].upper()
        if name_part in self.invalid_names:
            sanitized = f"_{sanitized}"

        if not sanitized or sanitized.replace('.', '').strip() == '':
            sanitized = 'unnamed_file'

        sanitized = sanitized.strip('. ')
        if not sanitized:
            sanitized = 'unnamed_file'

        original_sanitized = sanitized
        counter = 1
        while sanitized in used_names:
            name, ext = os.path.splitext(original_sanitized)
            sanitized = f"{name}_{counter}{ext}"
            counter += 1

        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = f"{name[:200 - len(ext)]}{ext}"

        used_names.add(sanitized)
        return sanitized

    def _copy_directory_robust(self, src_dir: Path, dst_dir: Path):
        """Copy directory with robust error handling."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        used_paths = set()

        for root, dirs, files in os.walk(src_dir):
            try:
                rel_path = Path(root).relative_to(src_dir)
                
                # Skip if current directory is excluded
                if str(rel_path) != '.' and self._is_in_excluded_dir(str(rel_path)):
                    logger.debug(f"Skipping excluded directory: {rel_path}")
                    dirs.clear()  # Don't recurse into subdirectories
                    continue
                
                if str(rel_path) != '.':
                    sanitized_rel_path = self._sanitize_path(str(rel_path), used_paths)
                    dst_subdir = dst_dir / sanitized_rel_path
                    dst_subdir.mkdir(parents=True, exist_ok=True)
                else:
                    dst_subdir = dst_dir

                for file in files:
                    try:
                        src_file = Path(root) / file
                        sanitized_filename = self._sanitize_filename(file)
                        dst_file = dst_subdir / sanitized_filename

                        if sanitized_filename != file:
                            self.extraction_stats['files_renamed'] += 1
                            self.extraction_stats['renamed_files'].append({
                                'original': str(src_file.relative_to(src_dir)),
                                'sanitized': str(dst_file.relative_to(dst_dir))
                            })

                        shutil.copy2(src_file, dst_file)
                        self.extraction_stats['files_processed'] += 1

                    except (OSError, PermissionError, UnicodeError) as e:
                        self.extraction_stats['files_skipped'] += 1
                        self.extraction_stats['skipped_files'].append({
                            'file': str(Path(root) / file),
                            'error': str(e)
                        })
                        logger.warning(f"Skipped file {file}: {e}")

            except Exception as e:
                logger.warning(f"Error processing directory {root}: {e}")

    def _sanitize_path(self, path: str, used_paths: Set[str] = None) -> str:
        """Sanitize full file path."""
        if used_paths is None:
            used_paths = set()

        parts = Path(path).parts
        sanitized_parts = []

        for part in parts:
            if part in ('/', '\\'):
                continue
            sanitized_part = self._sanitize_filename(part)
            sanitized_parts.append(sanitized_part)

        sanitized_path = '/'.join(sanitized_parts)

        original_path = sanitized_path
        counter = 1
        while sanitized_path in used_paths:
            path_obj = Path(original_path)
            sanitized_path = str(path_obj.parent / f"{path_obj.stem}_{counter}{path_obj.suffix}")
            counter += 1

        used_paths.add(sanitized_path)
        return sanitized_path

    def _get_extraction_method(self, file_path: Path):
        """Determine extraction method for a file."""
        file_str = str(file_path).lower()

        for ext in ['.tar.gz', '.tar.bz2', '.tar.xz']:
            if file_str.endswith(ext):
                return self.archive_extensions[ext]

        suffix = file_path.suffix.lower()
        return self.archive_extensions.get(suffix)

    def _extract_zip_robust(self, archive_path: Path, extract_dir: Path):
        """Extract ZIP archive with robust error handling."""
        used_paths = set()

        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    try:
                        if member.is_dir():
                            continue

                        try:
                            filename = member.filename
                            if member.flag_bits & 0x800 == 0:
                                try:
                                    filename = member.filename.encode('cp437').decode('utf-8')
                                except (UnicodeDecodeError, UnicodeEncodeError):
                                    pass
                        except Exception:
                            filename = f"file_{self.extraction_stats['files_processed']}"
                            self.extraction_stats['encoding_issues'] += 1

                        if os.path.isabs(filename) or ".." in filename:
                            logger.warning(f"Skipping unsafe path: {filename}")
                            self.extraction_stats['files_skipped'] += 1
                            continue
                        
                        # Check if file is in excluded directories
                        if self._is_in_excluded_dir(filename):
                            logger.debug(f"Skipping excluded directory file: {filename}")
                            self.extraction_stats['files_skipped'] += 1
                            continue

                        sanitized_path = self._sanitize_path(filename, used_paths)
                        target_path = extract_dir / sanitized_path

                        if sanitized_path != filename:
                            self.extraction_stats['files_renamed'] += 1
                            self.extraction_stats['renamed_files'].append({
                                'original': filename,
                                'sanitized': sanitized_path
                            })

                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)

                        self.extraction_stats['files_processed'] += 1

                    except Exception as e:
                        self.extraction_stats['files_skipped'] += 1
                        self.extraction_stats['skipped_files'].append({
                            'file': getattr(member, 'filename', 'unknown'),
                            'error': str(e)
                        })
                        logger.warning(f"Skipped ZIP member: {e}")

        except Exception as e:
            logger.error(f"Error opening ZIP file: {e}")
            raise

    def _extract_tar_robust(self, archive_path: Path, extract_dir: Path):
        """Extract TAR archive with robust error handling."""
        used_paths = set()

        try:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                for member in tar_ref.getmembers():
                    try:
                        try:
                            filename = member.name
                            if isinstance(filename, bytes):
                                filename = filename.decode('utf-8', errors='ignore')
                        except Exception:
                            filename = f"item_{self.extraction_stats['files_processed']}"
                            self.extraction_stats['encoding_issues'] += 1

                        if os.path.isabs(filename) or ".." in filename:
                            logger.warning(f"Skipping unsafe path: {filename}")
                            self.extraction_stats['files_skipped'] += 1
                            continue
                        
                        # Check if file is in excluded directories
                        if self._is_in_excluded_dir(filename):
                            logger.debug(f"Skipping excluded directory file: {filename}")
                            self.extraction_stats['files_skipped'] += 1
                            continue

                        sanitized_path = self._sanitize_path(filename, used_paths)
                        target_path = extract_dir / sanitized_path

                        if sanitized_path != filename:
                            self.extraction_stats['files_renamed'] += 1
                            self.extraction_stats['renamed_files'].append({
                                'original': filename,
                                'sanitized': sanitized_path
                            })

                        if member.isdir():
                            try:
                                target_path.mkdir(parents=True, exist_ok=True)
                            except Exception as e:
                                logger.warning(f"Could not create directory {target_path}: {e}")
                                self.extraction_stats['files_skipped'] += 1
                            continue

                        if member.isfile():
                            try:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                with tar_ref.extractfile(member) as source:
                                    if source:
                                        with open(target_path, 'wb') as target:
                                            shutil.copyfileobj(source, target)
                                self.extraction_stats['files_processed'] += 1
                            except Exception as e:
                                self.extraction_stats['files_skipped'] += 1
                                self.extraction_stats['skipped_files'].append({
                                    'file': filename,
                                    'error': str(e)
                                })
                                logger.warning(f"Skipped file {filename}: {e}")

                        elif member.issym() or member.islnk():
                            try:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(target_path, 'w', encoding='utf-8') as f:
                                    f.write(f"# Symbolic link to: {member.linkname}\n")
                                self.extraction_stats['files_processed'] += 1
                            except Exception as e:
                                self.extraction_stats['files_skipped'] += 1
                                logger.warning(f"Skipped symlink {filename}: {e}")

                    except Exception as e:
                        self.extraction_stats['files_skipped'] += 1
                        self.extraction_stats['skipped_files'].append({
                            'file': getattr(member, 'name', 'unknown'),
                            'error': str(e)
                        })
                        logger.warning(f"Skipped TAR member: {e}")

        except Exception as e:
            logger.error(f"Error opening TAR file: {e}")
            raise

    def _is_archive_file(self, file_path: Path) -> bool:
        """Check if a file is a supported archive format."""
        file_str = str(file_path).lower()
        
        # Check for compound extensions first
        for ext in ['.tar.gz', '.tar.bz2', '.tar.xz', '.tgz']:
            if file_str.endswith(ext):
                return True
        
        # Check single extensions
        return file_path.suffix.lower() in ['.zip', '.tar', '.7z', '.rar', '.gz', '.bz2', '.xz']

    def _extract_nested_archives(self, directory: Path, depth: int = 0, processed_archives: set = None, start_time: float = None):
        """
        Recursively extract nested archives found in the directory.
        
        Args:
            directory: Directory to scan for nested archives
            depth: Current recursion depth
            processed_archives: Set of already processed archive paths (by content hash)
            start_time: Start time for timeout tracking
        """
        if depth >= self.max_recursion_depth:
            logger.warning(f"Maximum recursion depth ({self.max_recursion_depth}) reached, skipping deeper extraction")
            return
        
        if processed_archives is None:
            processed_archives = set()
        
        if start_time is None:
            start_time = time.time()
        
        # Check timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > self.extraction_timeout:
            logger.warning(f"Nested extraction timeout ({self.extraction_timeout}s) reached, stopping extraction")
            return
        
        archives_found = []
        
        # Find all archive files in the directory
        for item in directory.rglob('*'):
            if item.is_file() and self._is_archive_file(item):
                try:
                    # Skip if file is too large
                    if item.stat().st_size > self.max_file_size:
                        logger.warning(f"Skipping large nested archive: {item.name} ({item.stat().st_size} bytes)")
                        continue
                    
                    # Create a unique identifier for this archive based on size and relative path
                    # This helps prevent extracting the same archive multiple times
                    archive_id = f"{item.name}_{item.stat().st_size}_{item.stat().st_mtime}"
                    
                    if archive_id in processed_archives:
                        logger.debug(f"Archive already processed: {item.name}")
                        continue
                    
                    archives_found.append((item, archive_id))
                except (OSError, IOError):
                    continue
        
        # Limit the number of archives processed at each depth to prevent infinite loops
        max_archives_per_depth = 50
        if len(archives_found) > max_archives_per_depth:
            logger.warning(f"Found {len(archives_found)} archives at depth {depth}, limiting to {max_archives_per_depth}")
            archives_found = archives_found[:max_archives_per_depth]
        
        # Extract each found archive
        for i, (archive_path, archive_id) in enumerate(archives_found):
            try:
                # Check timeout periodically
                elapsed_time = time.time() - start_time
                if elapsed_time > self.extraction_timeout:
                    logger.warning(f"Nested extraction timeout reached during processing, stopping after {i} archives")
                    break
                
                logger.info(f"Found nested archive at depth {depth}: {archive_path.name} ({i+1}/{len(archives_found)})")
                
                # Mark this archive as processed
                processed_archives.add(archive_id)
                
                # Create extraction directory next to the archive
                extract_name = f"{archive_path.stem}_extracted"
                extract_dir = archive_path.parent / extract_name
                
                # Skip if already extracted
                if extract_dir.exists():
                    logger.debug(f"Nested archive already extracted: {archive_path.name}")
                    continue
                
                # Check for extraction loop (too many similar extractions)
                similar_extractions = len([d for d in archive_path.parent.iterdir() 
                                         if d.is_dir() and d.name.startswith(archive_path.stem)])
                if similar_extractions >= 3:  # Reduced threshold
                    logger.warning(f"Too many similar extractions detected for {archive_path.name} ({similar_extractions}), skipping to prevent loops")
                    continue
                
                # Determine extraction method
                extraction_method = self._get_extraction_method(archive_path)
                if extraction_method is None:
                    logger.warning(f"Unsupported nested archive format: {archive_path.suffix}")
                    continue
                
                # Extract the nested archive
                extract_dir.mkdir(parents=True, exist_ok=True)
                extraction_method(archive_path, extract_dir)
                
                logger.info(f"Successfully extracted nested archive: {archive_path.name}")
                
                # Update extraction stats
                self.extraction_stats['nested_archives_found'] = self.extraction_stats.get('nested_archives_found', 0) + 1
                
                # Recursively extract any archives found in the newly extracted content
                # Pass the processed_archives set to maintain state across recursive calls
                self._extract_nested_archives(extract_dir, depth + 1, processed_archives, start_time)
                
            except Exception as e:
                logger.error(f"Failed to extract nested archive {archive_path.name}: {e}")
                self.extraction_stats['nested_extraction_failures'] = self.extraction_stats.get('nested_extraction_failures', 0) + 1
                continue

    def cleanup_extracted_artifact(self, artifact_name: str):
        """Clean up extracted artifact directory."""
        patterns = [f"extracted_{artifact_name}", f"cloned_{artifact_name}"]
        for pattern in patterns:
            for item in self.temp_dir.glob(f"{pattern}*"):
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    logger.info(f"Cleaned up: {item}")

    def cleanup_all(self):
        """Clean up all extracted artifacts."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleaned up all extracted artifacts")

    def get_extracted_artifacts(self) -> List[Dict[str, Any]]:
        """Get list of all extracted artifacts."""
        extracted = []
        if not self.temp_dir.exists():
            return extracted

        for item in self.temp_dir.iterdir():
            if item.is_dir() and (item.name.startswith('extracted_') or item.name.startswith('cloned_')):
                try:
                    artifact_info = {
                        "name": item.name,
                        "path": str(item),
                        "created": datetime.fromtimestamp(item.stat().st_ctime).isoformat(),
                        "size": sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    }
                    extracted.append(artifact_info)
                except Exception as e:
                    logger.warning(f"Could not get info for {item}: {e}")

        return extracted


def main():
    """Example usage of the IntegratedArtifactAnalyzer."""
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python integrated_artifact_analyzer.py <artifact_path> [artifact_name] [--force] [--skip-analysis] [--max-depth N]")
        print("Examples:")
        print("  python integrated_artifact_analyzer.py /path/to/artifact.zip")
        print("  python integrated_artifact_analyzer.py /path/to/directory")
        print("  python integrated_artifact_analyzer.py https://github.com/user/repo")
        print("  python integrated_artifact_analyzer.py artifact.zip --max-depth 5")
        sys.exit(1)

    artifact_path = sys.argv[1]
    artifact_name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    force_reextract = '--force' in sys.argv
    skip_analysis = '--skip-analysis' in sys.argv
    
    # Parse max depth parameter
    max_depth = 3  # default
    if '--max-depth' in sys.argv:
        try:
            depth_index = sys.argv.index('--max-depth')
            if depth_index + 1 < len(sys.argv):
                max_depth = int(sys.argv[depth_index + 1])
        except (ValueError, IndexError):
            print("Error: --max-depth requires a numeric value")
            sys.exit(1)

    analyzer = IntegratedArtifactAnalyzer(max_recursion_depth=max_depth)

    try:
        result = analyzer.analyze_artifact(
            artifact_path,
            artifact_name=artifact_name,
            force_reextract=force_reextract,
            skip_analysis=skip_analysis
        )

        if result["success"]:
            print(f"✓ Analysis completed successfully!")
            print(f"  Artifact: {result['artifact_name']}")
            print(f"  Extracted to: {result['extracted_path']}")
            print(f"  Method: {result['extraction_method']}")
            
            # Show extraction statistics including nested archives
            stats = result.get('extraction_stats', {})
            if stats.get('nested_archives_found', 0) > 0:
                print(f"  Nested archives found: {stats['nested_archives_found']}")
                if stats.get('nested_extraction_failures', 0) > 0:
                    print(f"  Nested extraction failures: {stats['nested_extraction_failures']}")

            if result.get("analysis_performed", False):
                print(f"  Files analyzed: {len(result.get('repository_structure', []))}")
                print(f"  Documentation: {len(result.get('documentation_files', []))}")
                print(f"  Code files: {len(result.get('code_files', []))}")
                print(f"  License files: {len(result.get('license_files', []))}")
                print(f"  Build files: {len(result.get('build_files', []))}")
                print(f"  Docker files: {len(result.get('docker_files', []))}")
                print(f"  Data files: {len(result.get('data_files', []))}")
                print(f"  Config files: {len(result.get('config_files', []))}")
                print(f"  Size: {result.get('repo_size_mb', 0)} MB")
                print(f"  Results saved to: ./analysis_outputs/{result['artifact_name']}_analysis.json")
        else:
            print(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
