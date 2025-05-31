import mimetypes
import os
import shutil

from anytree import Node, RenderTree
from git import Repo

DOCUMENT_EXTENSIONS = ['.md', '.rst', '.txt']
CODE_EXTENSIONS = ['.py', '.java', '.cpp', '.js', '.ts']
LICENSE_NAMES = ['license', 'license.txt']
EXCLUDE_DIRS = ['.git', 'node_modules', '__pycache__']


def is_github_url(url):
    return url.startswith("http://") or url.startswith("https://")


def clone_repository(repo_url, temp_base_dir="temp_dir_for_git"):
    repo_name = repo_url.rstrip("/").split("/")[-1]
    clone_path = os.path.join(temp_base_dir, repo_name)

    if os.path.exists(clone_path):
        shutil.rmtree(clone_path, ignore_errors=True)  # force cleanup before re-cloning

    Repo.clone_from(repo_url, clone_path)
    return clone_path


def generate_file_list(root, exclude_dirs=EXCLUDE_DIRS):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_paths.append(full_path)
    return file_paths


def is_documentation_file(path):
    name = os.path.basename(path).lower()
    return any(name.endswith(ext) for ext in DOCUMENT_EXTENSIONS) or 'readme' in name or 'contributing' in name


def is_code_file(path):
    return any(path.endswith(ext) for ext in CODE_EXTENSIONS)


def is_license_file(path):
    name = os.path.basename(path).lower()
    return name in LICENSE_NAMES or 'license' in name


def read_file_content(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"[ERROR READING FILE]: {e}"


def generate_tree_structure(file_paths, root_dir):
    root_node = Node(os.path.basename(root_dir))
    node_map = {root_dir: root_node}
    for path in file_paths:
        rel_path = os.path.relpath(path, root_dir)
        parts = rel_path.split(os.sep)
        current_path = root_dir
        parent = root_node
        for part in parts:
            current_path = os.path.join(current_path, part)
            if current_path not in node_map:
                node_map[current_path] = Node(part, parent=parent)
            parent = node_map[current_path]
    return RenderTree(root_node)


def analyze_repository(repo_path_or_url, temp_base_dir="temp_dir_for_git"):
    is_temp = False
    if is_github_url(repo_path_or_url):
        repo_path = clone_repository(repo_path_or_url, temp_base_dir=temp_base_dir)
        is_temp = True
    else:
        repo_path = repo_path_or_url

    file_paths = generate_file_list(repo_path)

    S = []  # metadata
    M, C, L = [], [], []

    for path in file_paths:
        file_info = {
            "name": os.path.basename(path),
            "path": path,
            "type": mimetypes.guess_type(path)[0],
            "size_kb": round(os.path.getsize(path) / 1024, 2)
        }
        S.append(file_info)

        content = read_file_content(path)
        if is_documentation_file(path):
            M.append({"path": path, "content": content})
        elif is_code_file(path):
            C.append({"path": path, "content": content})
        elif is_license_file(path):
            L.append({"path": path, "content": content})

    tree = generate_tree_structure(file_paths, repo_path)
    for pre, fill, node in tree:
        print(f"{pre}{node.name}")

    if is_temp:
        print(f"Temporary repo saved at: {repo_path} (not deleted for safety)")

    return {
        "repository_structure": S,
        "documentation_files": M,
        "code_files": C,
        "license_files": L
    }


print(analyze_repository("https://github.com/17YuvrajSehgal/COSC-4P02-PROJECT"))
