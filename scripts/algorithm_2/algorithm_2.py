import json
import logging
import mimetypes
import os

from anytree import Node, RenderTree
from git import Repo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOCUMENT_EXTENSIONS = ['.md', '.rst', '.txt']
CODE_EXTENSIONS = ['.py', '.java', '.cpp', '.js', '.ts']
LICENSE_NAMES = ['license', 'license.txt']
EXCLUDE_DIRS = ['.git', 'node_modules', '__pycache__']


def is_github_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def clone_repository(repo_url: str, temp_base_dir="../../temp_dir_for_git") -> str:
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
        max_files_per_dir: int = 5,
        allowed_extensions=None,
        max_file_size_kb: int = 2048,
        min_files_to_include_anyway: int = 2
) -> list:
    """
    Walk through `root` and collect up to `max_files_per_dir` files per folder
    whose extension ∈ allowed_extensions. If fewer found, sample up to
    min_files_to_include_anyway “extra” files (of any extension) per folder.
    Skips files larger than max_file_size_kb.
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        selected = []
        sampled = []
        for filename in sorted(filenames):
            full_path = os.path.join(dirpath, filename)
            ext = os.path.splitext(full_path)[1].lower()
            size_kb = os.path.getsize(full_path) / 1024

            # Skip extremely large files
            if size_kb > max_file_size_kb:
                continue

            if allowed_extensions and ext in allowed_extensions:
                selected.append(full_path)
            elif len(sampled) < min_files_to_include_anyway:
                sampled.append(full_path)

            if len(selected) >= max_files_per_dir:
                break

        file_paths.extend(selected[:max_files_per_dir] + sampled)
    return file_paths


def is_documentation_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    return (
            any(name.endswith(ext) for ext in DOCUMENT_EXTENSIONS)
            or 'readme' in name
            or 'contributing' in name
    )


def is_code_file(path: str) -> bool:
    return any(path.lower().endswith(ext) for ext in CODE_EXTENSIONS)


def is_license_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    return name in LICENSE_NAMES or 'license' in name


def read_file_content(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to read file: {path} — {e}")
        return f"[ERROR READING FILE]: {e}"


def generate_tree_lines(file_paths: list, root_dir: str) -> list:
    """
    Instead of returning a RenderTree object (which prints lines with \n),
    we build a simple list of strings (one line per node). That way, when we
    write JSON, each line is its own string—no '\n' escapes.
    """
    logging.info("Generating directory tree structure as a list of lines...")
    root_node = Node(os.path.basename(root_dir))
    node_map = {root_dir: root_node}

    # Build a tree of anytree.Node objects
    for path in sorted(file_paths):
        rel_path = os.path.relpath(path, root_dir)
        parts = rel_path.split(os.sep)
        current_path = root_dir
        parent = root_node
        for part in parts:
            current_path = os.path.join(current_path, part)
            if current_path not in node_map:
                node_map[current_path] = Node(part, parent=parent)
            parent = node_map[current_path]

    # Now traverse, but collect each line in a Python list
    lines = []
    for pre, _, node in RenderTree(root_node):
        lines.append(f"{pre}{node.name}")
    return lines


def analyze_repository(repo_path_or_url: str, temp_base_dir="../../temp_dir_for_git") -> dict:
    is_temp = False
    if is_github_url(repo_path_or_url):
        repo_path = clone_repository(repo_path_or_url, temp_base_dir=temp_base_dir)
        is_temp = True
    else:
        repo_path = repo_path_or_url
        logging.info(f"Using local repository path: {repo_path}")

    # 1) Gather all relevant files under repo_path
    file_paths = generate_file_list(
        repo_path,
        exclude_dirs=EXCLUDE_DIRS,
        max_files_per_dir=5,
        allowed_extensions={'.py', '.md', '.txt'},
        max_file_size_kb=2048,
        min_files_to_include_anyway=2
    )

    S = []  # List of file_info dicts
    M, C, L = [], [], []  # Documentation, code, license

    for path in file_paths:
        file_info = {
            "name": os.path.basename(path),
            "path": os.path.relpath(path, repo_path),  # store relative path for portability
            "mime_type": mimetypes.guess_type(path)[0],
            "size_kb": round(os.path.getsize(path) / 1024, 2)
        }
        S.append(file_info)

        content = read_file_content(path)
        if is_documentation_file(path):
            M.append({"path": os.path.relpath(path, repo_path), "content": content})
            logging.info(f"[DOC] {path}")
        elif is_code_file(path):
            C.append({"path": os.path.relpath(path, repo_path), "content": content})
            logging.info(f"[CODE] {path}")
        elif is_license_file(path):
            L.append({"path": os.path.relpath(path, repo_path), "content": content})
            logging.info(f"[LICENSE] {path}")
        else:
            logging.debug(f"[SKIP] {path}")

    # 2) Generate a list of lines representing the directory‐tree
    tree_lines = generate_tree_lines(file_paths, repo_path)

    # If we cloned from GitHub, we might optionally delete it or keep it for caching.
    if is_temp:
        logging.info(f"Temporary repo saved at: {repo_path} (not deleted for safety)")

    # 3) Return one big dictionary containing everything, including `tree_structure` as a list of lines
    return {
        "repository_structure": S,  # list of file_info
        "documentation_files": M,  # list of {path, content}
        "code_files": C,  # list of {path, content}
        "license_files": L,  # list of {path, content}
        "tree_structure": tree_lines  # ← no embedded "\n"—just an array of strings
    }


def save_analysis_result(result: dict, repo_name: str, output_dir="../../data/algorithm_2_output"):
    """
    Write the `result` dict to JSON. Because `tree_structure` is already a list of simple strings,
    the output JSON will not contain any '\\n' escape characters.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{repo_name}_analysis.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved analysis result to: {json_path}")


# Example usage
if __name__ == "__main__":
    repo_url = "https://github.com/sneh2001patel/ml-image-classifier"
    #repo_url = "https://github.com/17YuvrajSehgal/COSC-4P02-PROJECT"
    result = analyze_repository(repo_url)

    repo_name = repo_url.rstrip("/").split("/")[-1]
    save_analysis_result(result, repo_name)
