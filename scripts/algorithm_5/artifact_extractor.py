#!/usr/bin/env python3
"""
Artifact Extractor - Multi-format artifact extraction and processing

This module handles extraction of various artifact formats including:
- ZIP files
- TAR/TAR.GZ/TGZ files  
- Regular directories (git clones)
- Nested archives
"""

import os
import shutil
import tarfile
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import mimetypes
import tempfile
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArtifactExtractor:
    """
    Handles extraction and processing of software artifacts in various formats.
    
    Supported formats:
    - ZIP files (.zip)
    - TAR files (.tar, .tar.gz, .tgz, .tar.bz2, .tar.xz)
    - Regular directories (git repositories)
    - Nested archives
    """
    
    def __init__(self, temp_dir: str = "./temp_extractions", max_file_size: int = 500 * 1024 * 1024):
        """
        Initialize the ArtifactExtractor.
        
        Args:
            temp_dir: Temporary directory for extractions
            max_file_size: Maximum file size to process (default: 500MB)
        """
        self.temp_dir = Path(temp_dir)
        self.max_file_size = max_file_size
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported archive extensions
        self.archive_extensions = {
            '.zip': self._extract_zip,
            '.tar': self._extract_tar,
            '.tar.gz': self._extract_tar,
            '.tgz': self._extract_tar,
            '.tar.bz2': self._extract_tar,
            '.tar.xz': self._extract_tar,
        }
        
        # File type mappings for analysis
        self.file_type_mappings = {
            'code': ['.py', '.java', '.cpp', '.c', '.h', '.js', '.ts', '.go', '.rs', '.rb', '.php', '.cs', '.kt'],
            'documentation': ['.md', '.rst', '.txt', '.doc', '.docx', '.pdf'],
            'config': ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'],
            'data': ['.csv', '.json', '.xml', '.sql', '.db', '.sqlite'],
            'build': ['Makefile', 'CMakeLists.txt', 'build.gradle', 'pom.xml', 'package.json', 'requirements.txt', 'setup.py'],
            'docker': ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml'],
            'license': ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING'],
        }
    
    def extract_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        force_reextract: bool = False
    ) -> Dict[str, Any]:
        """
        Extract an artifact and return extraction information.
        
        Args:
            artifact_path: Path to the artifact (file or directory)
            artifact_name: Optional custom name for the artifact
            force_reextract: Force re-extraction even if already extracted
            
        Returns:
            Dictionary containing extraction results and metadata
        """
        artifact_path = Path(artifact_path)
        
        if not artifact_path.exists():
            return {
                "success": False,
                "error": f"Artifact not found: {artifact_path}",
                "artifact_path": str(artifact_path)
            }
        
        if artifact_name is None:
            artifact_name = artifact_path.stem
        
        logger.info(f"Extracting artifact: {artifact_name}")
        
        # Create extraction directory
        extract_dir = self.temp_dir / f"extracted_{artifact_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = {
            "success": False,
            "artifact_name": artifact_name,
            "artifact_path": str(artifact_path),
            "extracted_path": str(extract_dir),
            "extraction_method": None,
            "metadata": {},
            "error": None,
            "stats": {
                "total_files": 0,
                "total_size": 0,
                "file_types": {},
                "directory_structure": [],
            }
        }
        
        try:
            # Check if it's a directory or file
            if artifact_path.is_dir():
                # Copy directory
                logger.info(f"Copying directory: {artifact_path}")
                shutil.copytree(artifact_path, extract_dir)
                result["extraction_method"] = "directory_copy"
                
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
                logger.info(f"Extracting archive: {artifact_path}")
                extract_dir.mkdir(parents=True, exist_ok=True)
                extraction_method(artifact_path, extract_dir)
                result["extraction_method"] = extraction_method.__name__
            
            else:
                return {
                    **result,
                    "error": f"Invalid artifact type: {artifact_path}"
                }
            
            # Analyze extracted content
            analysis = self._analyze_extracted_content(extract_dir)
            result["metadata"] = analysis["metadata"]
            result["stats"] = analysis["stats"]
            
            result["success"] = True
            logger.info(f"Successfully extracted artifact: {artifact_name}")
            
        except Exception as e:
            error_msg = f"Extraction failed for {artifact_name}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            
            # Clean up on failure
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
        
        return result
    
    def _get_extraction_method(self, file_path: Path):
        """Determine the appropriate extraction method for a file."""
        file_str = str(file_path).lower()
        
        # Check for compound extensions first
        for ext in ['.tar.gz', '.tar.bz2', '.tar.xz']:
            if file_str.endswith(ext):
                return self.archive_extensions[ext]
        
        # Check single extensions
        suffix = file_path.suffix.lower()
        return self.archive_extensions.get(suffix)
    
    def _extract_zip(self, archive_path: Path, extract_dir: Path):
        """Extract ZIP archive."""
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    def _extract_tar(self, archive_path: Path, extract_dir: Path):
        """Extract TAR archive (including compressed variants)."""
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            # Security check for path traversal
            for member in tar_ref.getmembers():
                if os.path.isabs(member.name) or ".." in member.name:
                    logger.warning(f"Skipping unsafe path: {member.name}")
                    continue
            tar_ref.extractall(extract_dir)
    
    def _analyze_extracted_content(self, extract_dir: Path) -> Dict[str, Any]:
        """
        Analyze the extracted content and generate metadata.
        
        Args:
            extract_dir: Directory containing extracted files
            
        Returns:
            Dictionary containing analysis results
        """
        metadata = {
            "repository_structure": [],
            "file_categories": {},
            "programming_languages": set(),
            "has_readme": False,
            "has_license": False,
            "has_dockerfile": False,
            "has_tests": False,
            "build_systems": [],
            "dependencies": [],
        }
        
        stats = {
            "total_files": 0,
            "total_size": 0,
            "file_types": {},
            "directory_structure": [],
            "largest_files": [],
            "code_files": 0,
            "documentation_files": 0,
        }
        
        # Walk through all files
        all_files = []
        for root, dirs, files in os.walk(extract_dir):
            rel_root = Path(root).relative_to(extract_dir)
            
            # Add directory to structure
            if str(rel_root) != '.':
                stats["directory_structure"].append(str(rel_root))
            
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(extract_dir)
                
                try:
                    file_stat = file_path.stat()
                    file_size = file_stat.st_size
                    
                    # Basic file info
                    file_info = {
                        "name": file,
                        "path": str(rel_path),
                        "size": file_size,
                        "extension": file_path.suffix.lower(),
                        "type": self._classify_file_type(file_path),
                    }
                    
                    metadata["repository_structure"].append(file_info)
                    all_files.append((file_path, file_size))
                    
                    # Update stats
                    stats["total_files"] += 1
                    stats["total_size"] += file_size
                    
                    # File type counting
                    file_type = file_info["type"]
                    stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
                    
                    if file_type == "code":
                        stats["code_files"] += 1
                        # Detect programming language
                        lang = self._detect_programming_language(file_path)
                        if lang:
                            metadata["programming_languages"].add(lang)
                    
                    elif file_type == "documentation":
                        stats["documentation_files"] += 1
                    
                    # Special file detection
                    file_lower = file.lower()
                    if file_lower.startswith('readme'):
                        metadata["has_readme"] = True
                    elif file_lower.startswith('license') or file_lower == 'copying':
                        metadata["has_license"] = True
                    elif file_lower == 'dockerfile':
                        metadata["has_dockerfile"] = True
                    elif 'test' in file_lower or file_lower.endswith('_test.py'):
                        metadata["has_tests"] = True
                    
                    # Build system detection
                    if file in ['setup.py', 'pyproject.toml', 'requirements.txt']:
                        metadata["build_systems"].append("Python")
                    elif file in ['package.json', 'yarn.lock']:
                        metadata["build_systems"].append("Node.js")
                    elif file in ['pom.xml', 'build.gradle']:
                        metadata["build_systems"].append("Java")
                    elif file in ['Makefile', 'CMakeLists.txt']:
                        metadata["build_systems"].append("C/C++")
                    
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not analyze file {rel_path}: {e}")
                    continue
        
        # Find largest files
        all_files.sort(key=lambda x: x[1], reverse=True)
        stats["largest_files"] = [
            {"path": str(f[0].relative_to(extract_dir)), "size": f[1]}
            for f in all_files[:10]
        ]
        
        # Convert sets to lists for JSON serialization
        metadata["programming_languages"] = list(metadata["programming_languages"])
        metadata["build_systems"] = list(set(metadata["build_systems"]))
        
        return {"metadata": metadata, "stats": stats}
    
    def _classify_file_type(self, file_path: Path) -> str:
        """Classify a file into a category based on its extension and name."""
        file_name = file_path.name.lower()
        file_ext = file_path.suffix.lower()
        
        # Check by filename first
        for category, patterns in self.file_type_mappings.items():
            if file_name in [p.lower() for p in patterns if not p.startswith('.')]:
                return category
        
        # Check by extension
        for category, extensions in self.file_type_mappings.items():
            if file_ext in [ext.lower() for ext in extensions if ext.startswith('.')]:
                return category
        
        # Default categorization
        if file_ext in ['.exe', '.dll', '.so', '.dylib']:
            return 'binary'
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg']:
            return 'image'
        elif file_ext in ['.mp4', '.avi', '.mov']:
            return 'video'
        
        return 'other'
    
    def _detect_programming_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        ext = file_path.suffix.lower()
        
        language_map = {
            '.py': 'Python',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.cs': 'C#',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'MATLAB',
            '.sh': 'Shell',
            '.sql': 'SQL',
        }
        
        return language_map.get(ext)
    
    def get_extracted_artifacts(self) -> List[Dict[str, Any]]:
        """Get list of all extracted artifacts in temp directory."""
        extracted = []
        if not self.temp_dir.exists():
            return extracted
        
        for item in self.temp_dir.iterdir():
            if item.is_dir() and item.name.startswith('extracted_'):
                try:
                    # Parse artifact name from directory name
                    parts = item.name.split('_')
                    if len(parts) >= 3:
                        artifact_name = '_'.join(parts[1:-2])  # Remove 'extracted_' prefix and timestamp
                        timestamp = '_'.join(parts[-2:])
                        
                        extracted.append({
                            "artifact_name": artifact_name,
                            "extracted_path": str(item),
                            "extraction_time": timestamp,
                            "file_count": len(list(item.rglob('*')))
                        })
                except Exception as e:
                    logger.warning(f"Could not parse extracted directory {item}: {e}")
        
        return extracted
    
    def cleanup_extracted_artifact(self, artifact_name: str):
        """Clean up extracted files for a specific artifact."""
        pattern = f"extracted_{artifact_name}_*"
        removed_count = 0
        
        for item in self.temp_dir.glob(pattern):
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                    removed_count += 1
                    logger.info(f"Cleaned up extracted artifact: {item}")
                except Exception as e:
                    logger.warning(f"Could not remove {item}: {e}")
        
        return removed_count
    
    def cleanup_all(self):
        """Clean up all extracted artifacts."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up all extracted artifacts in {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temp directory: {e}")


def main():
    """Example usage of the ArtifactExtractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Artifact Extractor")
    parser.add_argument("artifact_path", help="Path to artifact file or directory")
    parser.add_argument("--temp-dir", default="./temp_extractions", help="Temporary extraction directory")
    parser.add_argument("--cleanup", action="store_true", help="Clean up after extraction")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = ArtifactExtractor(temp_dir=args.temp_dir)
    
    # Extract artifact
    result = extractor.extract_artifact(args.artifact_path)
    
    if result["success"]:
        print(f"Extraction successful!")
        print(f"Extracted to: {result['extracted_path']}")
        print(f"Total files: {result['stats']['total_files']}")
        print(f"Total size: {result['stats']['total_size']} bytes")
        print(f"File types: {result['stats']['file_types']}")
        
        if result['metadata']['programming_languages']:
            print(f"Programming languages: {result['metadata']['programming_languages']}")
        
        # Clean up if requested
        if args.cleanup:
            extractor.cleanup_all()
            print("Cleaned up extracted files")
    else:
        print(f"Extraction failed: {result['error']}")


if __name__ == "__main__":
    main() 