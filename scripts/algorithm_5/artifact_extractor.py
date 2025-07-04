#!/usr/bin/env python3
"""
Robust Artifact Extractor - Multi-format artifact extraction with error handling

This module handles extraction of various artifact formats including:
- ZIP files (with filename sanitization)
- TAR/TAR.GZ/TGZ files (with robust error handling)
- Regular directories (git clones)
- Nested archives
- Files with invalid characters, encoding issues, and long paths
"""

import logging
import os
import re
import shutil
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustArtifactExtractor:
    """
    Robust artifact extractor for multi-format research artifacts.

    - Extracted directories are preserved by default and not deleted after extraction.
    - If extraction fails, the partially extracted directory is deleted to avoid clutter.
    - If an extracted directory already exists, extraction is skipped (unless force_reextract=True).
    - Use cleanup_extracted_artifact or cleanup_all manually if you want to remove extracted directories.
    """

    def __init__(self, temp_dir: str = "./temp_extractions", max_file_size: int = 500 * 1024 * 1024):
        """
        Initialize the RobustArtifactExtractor.
        
        Args:
            temp_dir: Temporary directory for extractions
            max_file_size: Maximum file size to process (default: 500MB)
        """
        self.temp_dir = Path(temp_dir)
        self.max_file_size = max_file_size
        self.temp_dir.mkdir(parents=True, exist_ok=True)

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

        # File type mappings for analysis
        self.file_type_mappings = {
            'code': ['.py', '.java', '.cpp', '.c', '.h', '.js', '.ts', '.go', '.rs', '.rb', '.php', '.cs', '.kt'],
            'documentation': ['.md', '.rst', '.txt', '.doc', '.docx', '.pdf'],
            'config': ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'],
            'data': ['.csv', '.json', '.xml', '.sql', '.db', '.sqlite'],
            'build': ['Makefile', 'CMakeLists.txt', 'build.gradle', 'pom.xml', 'package.json', 'requirements.txt',
                      'setup.py'],
            'docker': ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml'],
            'license': ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING'],
        }

        # Extraction statistics
        self.extraction_stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'files_renamed': 0,
            'encoding_issues': 0,
            'path_issues': 0,
            'skipped_files': [],
            'renamed_files': []
        }

    def sanitize_filename(self, filename: str, used_names: Set[str] = None) -> str:
        """
        Sanitize filename for Windows compatibility.
        
        Args:
            filename: Original filename
            used_names: Set of already used names to avoid duplicates
            
        Returns:
            Sanitized filename safe for Windows
        """
        if used_names is None:
            used_names = set()

        # Handle encoding issues
        try:
            # Normalize unicode characters
            filename = unicodedata.normalize('NFKD', filename)
            # Convert to ASCII, ignoring problematic characters
            filename = filename.encode('ascii', 'ignore').decode('ascii')
        except (UnicodeError, UnicodeDecodeError):
            # Fallback for severe encoding issues
            filename = re.sub(r'[^\x00-\x7F]+', '_', filename)

        # Replace invalid characters
        sanitized = re.sub(self.invalid_chars, '_', filename)

        # Handle reserved names
        name_part = sanitized.split('.')[0].upper()
        if name_part in self.invalid_names:
            sanitized = f"_{sanitized}"

        # Handle empty or dot-only names
        if not sanitized or sanitized.replace('.', '').strip() == '':
            sanitized = 'unnamed_file'

        # Handle leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')

        # Ensure not empty after stripping
        if not sanitized:
            sanitized = 'unnamed_file'

        # Handle duplicates
        original_sanitized = sanitized
        counter = 1
        while sanitized in used_names:
            name, ext = os.path.splitext(original_sanitized)
            sanitized = f"{name}_{counter}{ext}"
            counter += 1

        # Limit length for Windows (260 char total path limit)
        if len(sanitized) > 200:  # Leave room for directory path
            name, ext = os.path.splitext(sanitized)
            sanitized = f"{name[:200 - len(ext)]}{ext}"

        used_names.add(sanitized)
        return sanitized

    def sanitize_path(self, path: str, used_paths: Set[str] = None) -> str:
        """
        Sanitize full file path.
        
        Args:
            path: Original file path
            used_paths: Set of already used paths
            
        Returns:
            Sanitized path
        """
        if used_paths is None:
            used_paths = set()

        # Split path and sanitize each component
        parts = Path(path).parts
        sanitized_parts = []

        for part in parts:
            if part in ('/', '\\'):
                continue
            sanitized_part = self.sanitize_filename(part)
            sanitized_parts.append(sanitized_part)

        sanitized_path = '/'.join(sanitized_parts)

        # Handle duplicate paths
        original_path = sanitized_path
        counter = 1
        while sanitized_path in used_paths:
            path_obj = Path(original_path)
            sanitized_path = str(path_obj.parent / f"{path_obj.stem}_{counter}{path_obj.suffix}")
            counter += 1

        used_paths.add(sanitized_path)
        return sanitized_path

    def extract_artifact(
            self,
            artifact_path: str,
            artifact_name: Optional[str] = None,
            force_reextract: bool = False
    ) -> Dict[str, Any]:
        """
        Extract an artifact with robust error handling.
        
        Args:
            artifact_path: Path to the artifact (file or directory)
            artifact_name: Optional custom name for the artifact
            force_reextract: Force re-extraction even if already extracted
            
        Returns:
            Dictionary containing extraction results and metadata
        
        Behavior:
            - If extraction succeeds, the extracted directory is kept for reuse/debugging.
            - If extraction fails, the extracted directory is deleted.
            - If the extracted directory already exists, extraction is skipped (unless force_reextract=True).
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

        # Sanitize artifact name
        artifact_name = self.sanitize_filename(artifact_name)
        
        # Create extraction directory (without timestamp for reuse)
        extract_dir = self.temp_dir / f"extracted_{artifact_name}"
        
        # Check if already extracted and skip if not forcing re-extraction
        if extract_dir.exists() and not force_reextract:
            logger.info(f"Artifact already extracted, reusing: {artifact_name}")
            
            # Still analyze the existing content for metadata
            analysis = self._analyze_extracted_content(extract_dir)
            
            return {
                "success": True,
                "artifact_name": artifact_name,
                "artifact_path": str(artifact_path),
                "extracted_path": str(extract_dir),
                "extraction_method": "reused_existing",
                "metadata": analysis["metadata"],
                "stats": analysis["stats"],
                "extraction_stats": {
                    'files_processed': 0,
                    'files_skipped': 0,
                    'files_renamed': 0,
                    'encoding_issues': 0,
                    'path_issues': 0,
                    'skipped_files': [],
                    'renamed_files': [],
                    'reused_existing': True
                }
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
            'renamed_files': []
        }

        result = {
            "success": False,
            "artifact_name": artifact_name,
            "artifact_path": str(artifact_path),
            "extracted_path": str(extract_dir),
            "extraction_method": None,
            "metadata": {},
            "error": None,
            "extraction_stats": {},
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
                # Copy directory with robust handling
                logger.info(f"Copying directory: {artifact_path}")
                self._copy_directory_robust(artifact_path, extract_dir)
                result["extraction_method"] = "directory_copy_robust"

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

            # Log extraction statistics
            result["extraction_stats"] = self.extraction_stats.copy()
            if self.extraction_stats['files_skipped'] > 0:
                logger.warning(f"Skipped {self.extraction_stats['files_skipped']} problematic files")
            if self.extraction_stats['files_renamed'] > 0:
                logger.info(f"Renamed {self.extraction_stats['files_renamed']} files for compatibility")

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
            result["extraction_stats"] = self.extraction_stats.copy()

            # Clean up on failure
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

        return result

    def _copy_directory_robust(self, src_dir: Path, dst_dir: Path):
        """Copy directory with robust error handling."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        used_paths = set()

        for root, dirs, files in os.walk(src_dir):
            try:
                # Calculate relative path
                rel_path = Path(root).relative_to(src_dir)

                # Sanitize directory path
                if str(rel_path) != '.':
                    sanitized_rel_path = self.sanitize_path(str(rel_path), used_paths)
                    dst_subdir = dst_dir / sanitized_rel_path
                    dst_subdir.mkdir(parents=True, exist_ok=True)
                else:
                    dst_subdir = dst_dir

                # Copy files
                for file in files:
                    try:
                        src_file = Path(root) / file
                        sanitized_filename = self.sanitize_filename(file)
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

    def _extract_zip_robust(self, archive_path: Path, extract_dir: Path):
        """Extract ZIP archive with robust error handling."""
        used_paths = set()

        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    try:
                        # Skip directories
                        if member.is_dir():
                            continue

                        # Handle filename encoding
                        try:
                            filename = member.filename
                            # Try to decode with CP437 if it looks like it might be encoded
                            if member.flag_bits & 0x800 == 0:  # No UTF-8 flag
                                try:
                                    filename = member.filename.encode('cp437').decode('utf-8')
                                except (UnicodeDecodeError, UnicodeEncodeError):
                                    pass
                        except Exception:
                            filename = f"file_{self.extraction_stats['files_processed']}"
                            self.extraction_stats['encoding_issues'] += 1

                        # Security check for path traversal
                        if os.path.isabs(filename) or ".." in filename:
                            logger.warning(f"Skipping unsafe path: {filename}")
                            self.extraction_stats['files_skipped'] += 1
                            continue

                        # Sanitize path
                        sanitized_path = self.sanitize_path(filename, used_paths)
                        target_path = extract_dir / sanitized_path

                        if sanitized_path != filename:
                            self.extraction_stats['files_renamed'] += 1
                            self.extraction_stats['renamed_files'].append({
                                'original': filename,
                                'sanitized': sanitized_path
                            })

                        # Create parent directories
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        # Extract file
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
        created_dirs = set()

        try:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                # Process all members (files and directories)
                for member in tar_ref.getmembers():
                    try:
                        # Handle filename encoding
                        try:
                            filename = member.name
                            # Try to decode if it's not valid UTF-8
                            if isinstance(filename, bytes):
                                filename = filename.decode('utf-8', errors='ignore')
                        except Exception:
                            filename = f"item_{self.extraction_stats['files_processed']}"
                            self.extraction_stats['encoding_issues'] += 1

                        # Security check for path traversal
                        if os.path.isabs(filename) or ".." in filename:
                            logger.warning(f"Skipping unsafe path: {filename}")
                            self.extraction_stats['files_skipped'] += 1
                            continue

                        # Sanitize path
                        sanitized_path = self.sanitize_path(filename, used_paths)
                        target_path = extract_dir / sanitized_path

                        if sanitized_path != filename:
                            self.extraction_stats['files_renamed'] += 1
                            self.extraction_stats['renamed_files'].append({
                                'original': filename,
                                'sanitized': sanitized_path
                            })

                        # Handle directories
                        if member.isdir():
                            try:
                                target_path.mkdir(parents=True, exist_ok=True)
                                created_dirs.add(str(target_path))
                            except Exception as e:
                                logger.warning(f"Could not create directory {target_path}: {e}")
                                self.extraction_stats['files_skipped'] += 1
                                self.extraction_stats['skipped_files'].append({
                                    'file': filename,
                                    'error': f"Directory creation failed: {str(e)}"
                                })
                            continue

                        # Handle files
                        if member.isfile():
                            try:
                                # Create parent directories
                                target_path.parent.mkdir(parents=True, exist_ok=True)

                                # Extract file
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

                        # Handle symbolic links (create as regular files with target content if possible)
                        elif member.issym() or member.islnk():
                            try:
                                # Create parent directories
                                target_path.parent.mkdir(parents=True, exist_ok=True)

                                # Create a text file with link information
                                with open(target_path, 'w', encoding='utf-8') as f:
                                    f.write(f"# Symbolic link to: {member.linkname}\n")

                                self.extraction_stats['files_processed'] += 1

                            except Exception as e:
                                self.extraction_stats['files_skipped'] += 1
                                self.extraction_stats['skipped_files'].append({
                                    'file': filename,
                                    'error': f"Symlink handling failed: {str(e)}"
                                })
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
                artifact_info = {
                    "name": item.name,
                    "path": str(item),
                    "created": datetime.fromtimestamp(item.stat().st_ctime).isoformat(),
                    "size": sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                }
                extracted.append(artifact_info)

        return extracted

    def cleanup_extracted_artifact(self, artifact_name: str):
        """Clean up extracted artifact directory."""
        pattern = f"extracted_{artifact_name}_*"
        for item in self.temp_dir.glob(pattern):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                logger.info(f"Cleaned up: {item}")

    def cleanup_all(self):
        """Clean up all extracted artifacts."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleaned up all extracted artifacts")


# For backward compatibility
ArtifactExtractor = RobustArtifactExtractor


def main():
    """Example usage of the RobustArtifactExtractor."""
    extractor = RobustArtifactExtractor()

    # Example extraction
    test_archive = "example.zip"
    if Path(test_archive).exists():
        result = extractor.extract_artifact(test_archive)
        if result["success"]:
            print(f"Extraction successful: {result['extracted_path']}")
            print(f"Files processed: {result['extraction_stats']['files_processed']}")
            print(f"Files skipped: {result['extraction_stats']['files_skipped']}")
            print(f"Files renamed: {result['extraction_stats']['files_renamed']}")
        else:
            print(f"Extraction failed: {result['error']}")


if __name__ == "__main__":
    main()
