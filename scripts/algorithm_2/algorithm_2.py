import json
import logging
import mimetypes
import os
import shutil
import sys
import tempfile

from anytree import Node, RenderTree
from git import Repo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enhanced file extension patterns based on artifact analysis
DOCUMENT_EXTENSIONS = [
    # Markdown and text files
    '.md', '.rst', '.txt', '.rtf',
    # Documentation formats
    '.tex', '.latex', '.org', '.asciidoc', '.adoc',
    # Web documentation
    '.html', '.htm', '.xml'
]

# Expanded code extensions covering more programming languages
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

# Enhanced license file detection patterns
LICENSE_PATTERNS = [
    # Common license file names (case-insensitive)
    'license', 'licence', 'license.txt', 'licence.txt',
    'license.md', 'licence.md', 'copying', 'copying.md',
    'copying.txt', 'copyright', 'copyright.txt', 'copyright.md',
    'legal', 'legal.txt', 'legal.md', 'notice', 'notice.txt',
    'notice.md', 'authors', 'authors.txt', 'authors.md',
    'contributors', 'contributors.txt', 'contributors.md',
    'acknowledgments', 'acknowledgments.txt', 'acknowledgments.md',
    'credits', 'credits.txt', 'credits.md'
]

# Enhanced documentation file patterns
DOCUMENTATION_PATTERNS = [
    # Core documentation
    'readme', 'read_me', 'read-me',
    'contributing', 'contribute', 'contribution',
    'changelog', 'changes', 'history', 'news', 'releases',
    'install', 'installation', 'setup', 'getting-started',
    'getting_started', 'quickstart', 'quick-start',
    'tutorial', 'guide', 'manual', 'documentation', 'docs',
    # Maintenance and development
    'todo', 'roadmap', 'milestones', 'development',
    'architecture', 'design', 'overview', 'summary',
    'faq', 'troubleshooting', 'debugging', 'testing',
    'security', 'code_of_conduct', 'code-of-conduct',
    'maintainers', 'governance', 'support', 'contact',
    # Technical documentation
    'api', 'reference', 'specification', 'spec',
    'requirements', 'dependencies', 'constraints',
    'building', 'build', 'compile', 'deployment',
    'configuration', 'config', 'settings'
]

# Configuration and build files that are important for reproducibility
CONFIG_BUILD_EXTENSIONS = [
    # Build systems
    '.gradle', '.maven', '.sbt', '.ant',
    # Package managers
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    # Docker
    '.dockerfile',
    # CI/CD
    '.travis.yml', '.appveyor.yml', '.gitlab-ci.yml',
    # Python specific
    '.requirements', '.pip',
    # Node.js
    '.npmrc', '.yarnrc',
    # Others
    '.cmake', '.make', '.mk', '.pro', '.qbs'
]

# Important configuration file names (case-insensitive)
CONFIG_FILE_PATTERNS = [
    # Build files
    'makefile', 'makefile.am', 'makefile.in', 'cmakelists.txt',
    'build.gradle', 'pom.xml', 'build.xml', 'setup.py', 'setup.cfg',
    'pyproject.toml', 'requirements.txt', 'pipfile', 'pipfile.lock',
    'package.json', 'package-lock.json', 'yarn.lock', 'composer.json',
    # Docker files
    'dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
    '.dockerignore',
    # CI/CD files
    '.travis.yml', '.github', '.gitlab-ci.yml', 'appveyor.yml',
    'jenkins', 'jenkinsfile',
    # Environment and configuration
    '.env', '.env.example', 'environment.yml', 'conda.yml',
    'tox.ini', 'pytest.ini', '.coveragerc', '.gitignore', '.gitattributes',
    # IDE configuration
    '.vscode', '.idea', '.project', '.classpath'
]

EXCLUDE_DIRS = ['.git', 'node_modules', '__pycache__', '.pytest_cache',
                '.coverage', '.tox', 'venv', 'env', '.env', 'build', 'dist',
                '.DS_Store', 'Thumbs.db']


def is_github_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def clone_repository(repo_url: str, temp_base_dir="./temp_dir_for_git") -> str:
    repo_name = repo_url.rstrip("/").split("/")[-1]
    clone_path = os.path.join(temp_base_dir, repo_name)

    # If already cloned, skip
    if os.path.exists(clone_path):
        logging.info(f"Repository already exists at: {clone_path}, skipping clone.")
        return clone_path

    # Otherwise, clone
    logging.info(f"Cloning repository: {repo_url} → {clone_path}")
    Repo.clone_from(repo_url, clone_path)
    logging.info("Cloned successfully.")
    return clone_path


def generate_file_list(
        root: str,
        exclude_dirs=EXCLUDE_DIRS,
        max_files_per_dir: int = 10,  # Increased from 5
        allowed_extensions=None,
        max_file_size_kb: int = 2048,
        min_files_to_include_anyway: int = 3  # Increased from 2
) -> list:
    """
    Enhanced file collection with better coverage of important files.
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        # Prioritize important files
        important_files = []
        selected = []
        sampled = []

        for filename in sorted(filenames):
            full_path = os.path.join(dirpath, filename)
            ext = os.path.splitext(full_path)[1].lower()
            size_kb = os.path.getsize(full_path) / 1024

            # Skip extremely large files
            if size_kb > max_file_size_kb:
                continue

            # Always include important files regardless of limits
            if _is_important_file(filename):
                important_files.append(full_path)
            elif allowed_extensions and ext in allowed_extensions:
                selected.append(full_path)
            elif len(sampled) < min_files_to_include_anyway:
                sampled.append(full_path)

            if len(selected) >= max_files_per_dir:
                break

        # Combine all files: important files + selected + sampled
        file_paths.extend(important_files + selected[:max_files_per_dir] + sampled)

    return file_paths


def _is_important_file(filename: str) -> bool:
    """Check if a file is considered important (README, LICENSE, etc.)"""
    name_lower = filename.lower()
    base_name = os.path.splitext(name_lower)[0]

    # Check against important patterns
    important_patterns = ['readme', 'license', 'licence', 'contributing',
                          'changelog', 'makefile', 'dockerfile', 'setup.py',
                          'requirements.txt', 'package.json', 'pom.xml']

    return (name_lower in important_patterns or
            base_name in important_patterns or
            any(pattern in name_lower for pattern in ['readme', 'license', 'licence']))


def is_documentation_file(path: str) -> bool:
    """Enhanced documentation file detection."""
    name = os.path.basename(path).lower()
    base_name = os.path.splitext(name)[0]
    ext = os.path.splitext(name)[1]

    # Check file extension
    if ext in DOCUMENT_EXTENSIONS:
        return True

    # Check filename patterns
    for pattern in DOCUMENTATION_PATTERNS:
        if pattern in name or pattern == base_name:
            return True

    # Special cases for common documentation files
    if any(keyword in name for keyword in ['readme', 'doc', 'guide', 'manual', 'help']):
        return True

    return False


def is_code_file(path: str) -> bool:
    """Enhanced code file detection."""
    ext = os.path.splitext(path.lower())[1]
    return ext in CODE_EXTENSIONS


def is_license_file(path: str) -> bool:
    """Enhanced license file detection."""
    name = os.path.basename(path).lower()
    base_name = os.path.splitext(name)[0]

    # Check exact matches and patterns
    if name in LICENSE_PATTERNS or base_name in LICENSE_PATTERNS:
        return True

    # Check for license-related keywords
    license_keywords = ['license', 'licence', 'copying', 'copyright', 'legal',
                        'notice', 'authors', 'contributors', 'credits', 'acknowledgment']

    return any(keyword in name for keyword in license_keywords)


def is_config_build_file(path: str) -> bool:
    """Detect configuration and build files important for reproducibility."""
    name = os.path.basename(path).lower()
    base_name = os.path.splitext(name)[0]
    ext = os.path.splitext(name)[1]

    # Check file extensions
    if ext in CONFIG_BUILD_EXTENSIONS:
        return True

    # Check filename patterns
    if name in CONFIG_FILE_PATTERNS or base_name in CONFIG_FILE_PATTERNS:
        return True

    # Special directory patterns
    if any(pattern in path.lower() for pattern in ['.github', '.gitlab', '.vscode', '.idea']):
        return True

    return False


def read_file_content(path: str) -> list[str]:
    """
    Read the file at `path` and return its contents as a list of non‐empty lines.
    Enhanced with better encoding handling.
    """
    try:
        # Try UTF-8 first, then fall back to other encodings
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(path, 'r', encoding=encoding, errors='ignore') as f:
                    raw = f.read()
                    # Split into lines and filter out empty lines
                    all_lines = raw.splitlines()
                    nonempty = [line for line in all_lines if line.strip() != ""]
                    return nonempty
            except UnicodeDecodeError:
                continue

        # If all encodings fail, try binary mode and decode manually
        with open(path, 'rb') as f:
            raw_bytes = f.read()
            try:
                raw = raw_bytes.decode('utf-8', errors='replace')
                all_lines = raw.splitlines()
                nonempty = [line for line in all_lines if line.strip() != ""]
                return nonempty
            except:
                return [f"[BINARY FILE - COULD NOT READ TEXT CONTENT]"]

    except Exception as e:
        logging.error(f"Failed to read file: {path} — {e}")
        return [f"[ERROR READING FILE]: {e}"]


def generate_tree_lines(file_paths: list, root_dir: str) -> list:
    """
    Generate directory tree structure as a list of strings.
    Enhanced to handle more complex directory structures.
    """
    logging.info("Generating directory tree structure as a list of lines...")
    root_node = Node(os.path.basename(root_dir))
    node_map = {root_dir: root_node}

    # Build a tree of anytree.Node objects
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
            logging.warning(f"Error processing path {path}: {e}")
            continue

    # Generate tree lines
    lines = []
    try:
        for pre, _, node in RenderTree(root_node):
            lines.append(f"{pre}{node.name}")
    except Exception as e:
        logging.error(f"Error generating tree: {e}")
        lines = [f"[ERROR GENERATING TREE]: {e}"]

    return lines


def is_archive_file(path: str) -> bool:
    """Enhanced archive file detection."""
    archive_extensions = ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2',
                          '.tar.xz', '.7z', '.rar', '.gz', '.bz2', '.xz']
    return any(path.lower().endswith(ext) for ext in archive_extensions)


def extract_archive(archive_path: str, extract_to: str) -> str:
    """Enhanced archive extraction with better error handling."""
    try:
        shutil.unpack_archive(archive_path, extract_to)
        logging.info(f"Extracted archive {archive_path} to {extract_to}")
        # Find the top-level directory (if any)
        entries = os.listdir(extract_to)
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_to, entries[0])):
            return os.path.join(extract_to, entries[0])
        return extract_to
    except Exception as e:
        logging.error(f"Failed to extract archive {archive_path}: {e}")
        raise


def analyze_repository(repo_path_or_url: str, temp_base_dir="./temp_dir_for_git") -> dict:
    """Enhanced repository analysis with robust file detection."""
    is_temp = False
    temp_dir = None

    if is_github_url(repo_path_or_url):
        repo_path = clone_repository(repo_path_or_url, temp_base_dir=temp_base_dir)
        is_temp = True
        logging.info(f"Analyzing cloned GitHub repository: {repo_path}")
    elif os.path.isfile(repo_path_or_url) and is_archive_file(repo_path_or_url):
        # Handle local archive file
        temp_dir = tempfile.mkdtemp(prefix="artifact_extract_")
        try:
            repo_path = extract_archive(repo_path_or_url, temp_dir)
            is_temp = True
            logging.info(f"Analyzing extracted archive: {repo_path}")
        except Exception as e:
            logging.error(f"Could not analyze archive {repo_path_or_url}: {e}")
            raise
    elif os.path.isdir(repo_path_or_url):
        repo_path = repo_path_or_url
        logging.info(f"Analyzing local directory: {repo_path}")
    else:
        raise ValueError(f"Input must be a GitHub URL, a directory, or a supported archive file: {repo_path_or_url}")

    # Calculate repository size in MB
    repo_size_mb = 0.0
    if os.path.exists(repo_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(repo_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except (OSError, IOError):
                        # Skip files that can't be accessed
                        continue
        repo_size_mb = round(total_size / (1024 * 1024), 2)
        logging.info(f"Repository size: {repo_size_mb} MB")

    # Enhanced file collection with expanded extensions
    all_extensions = set(DOCUMENT_EXTENSIONS + CODE_EXTENSIONS + CONFIG_BUILD_EXTENSIONS)
    file_paths = generate_file_list(
        repo_path,
        exclude_dirs=EXCLUDE_DIRS,
        max_files_per_dir=10,  # Increased limit
        allowed_extensions=all_extensions,
        max_file_size_kb=2048,
        min_files_to_include_anyway=3
    )

    # Initialize collections
    S = []  # Repository structure
    M, C, L, F = [], [], [], []  # Documentation, code, license, config files

    for path in file_paths:
        try:
            file_info = {
                "name": os.path.basename(path),
                "path": os.path.relpath(path, repo_path),
                "mime_type": mimetypes.guess_type(path)[0],
                "size_kb": round(os.path.getsize(path) / 1024, 2)
            }
            S.append(file_info)

            # Read file content
            content_lines = read_file_content(path)

            # Categorize files with enhanced detection
            if is_license_file(path):
                L.append({"path": os.path.relpath(path, repo_path), "content": content_lines})
            elif is_documentation_file(path):
                M.append({"path": os.path.relpath(path, repo_path), "content": content_lines})
            elif is_code_file(path):
                C.append({"path": os.path.relpath(path, repo_path), "content": content_lines})
            elif is_config_build_file(path):
                # Add config/build files to documentation for now (maintains JSON structure)
                M.append({"path": os.path.relpath(path, repo_path), "content": content_lines})
            else:
                logging.debug(f"[UNCATEGORIZED] {path}")

        except Exception as e:
            logging.warning(f"Error processing file {path}: {e}")
            continue

    # Generate directory tree
    try:
        tree_lines = generate_tree_lines(file_paths, repo_path)
    except Exception as e:
        logging.error(f"Error generating tree structure: {e}")
        tree_lines = [f"[ERROR GENERATING TREE]: {e}"]

    # Log statistics
    logging.info(f"Analysis complete: {len(S)} files, {len(M)} docs, {len(C)} code, {len(L)} license")

    if is_temp:
        logging.info(f"Temporary repo saved at: {repo_path} (not deleted for safety)")

    # Return analysis results with same structure
    return {
        "repository_structure": S,
        "documentation_files": M,  # Now includes config/build files
        "code_files": C,
        "license_files": L,
        "tree_structure": tree_lines,
        "repo_path": repo_path,
        "repo_size_mb": repo_size_mb
    }


def save_analysis_result(result: dict, repo_name: str, output_dir="./algo_outputs/algorithm_2_output"):
    """Enhanced save function with better error handling."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{repo_name}_analysis.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved analysis result to: {json_path}")

        # Log summary statistics
        stats = result
        logging.info(f"Summary: {len(stats['repository_structure'])} total files, "
                     f"{len(stats['documentation_files'])} documentation, "
                     f"{len(stats['code_files'])} code, "
                     f"{len(stats['license_files'])} license files")

    except Exception as e:
        logging.error(f"Error saving analysis result: {e}")
        raise


# Example usage
if __name__ == "__main__":
    # Usage: python algorithm_2.py <repo_url_or_path> [<temp_base_dir>] [<output_dir>]
    repo_url_or_path = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/JackyChok/AI_Code_Detection_Education"
    temp_base_dir = sys.argv[2] if len(sys.argv) > 2 else "../../temp_dir_for_git"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "../../algo_outputs/algorithm_2_output"

    try:
        result = analyze_repository(repo_url_or_path, temp_base_dir=temp_base_dir)
        repo_name = os.path.basename(repo_url_or_path.rstrip("/\\")).replace('.zip', '').replace('.tar.gz', '').replace(
            '.tar', '').replace('.tgz', '').replace('.tar.bz2', '').replace('.tar.xz', '')
        save_analysis_result(result, repo_name, output_dir=output_dir)
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)

# repo_url = "https://github.com/sneh2001patel/ml-image-classifier"
# repo_url = "https://github.com/17YuvrajSehgal/COSC-4P02-PROJECT"
# repo_url = "https://github.com/nntzuekai/Respector"
# repo_url = "https://github.com/SageSELab/MotorEase"
# repo_url = "https://github.com/SageSELab/UI-Bug-Localization-Study"
# repo_url =  "https://github.com/JackyChok/AI_Code_Detection_Education"
# repo_url = "https://github.com/JackyChok/AI_Code_Detection_Education"
# repo_url = "https://github.com/huiAlex/TRIAD"
# repo_url = "https://github.com/sola-st/PyTy"
