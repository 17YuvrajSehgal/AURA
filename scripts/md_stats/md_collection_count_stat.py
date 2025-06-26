import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Tuple, List, Dict

# === Logging Setup ===
log_path = Path("../../algo_outputs/logs/md_collection_count_stat.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# File handler
file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def extract_file_extensions_with_count_and_paths(directory: Path) -> Tuple[Dict[str, int], List[Path]]:
    """
    Extract all file extensions, their counts, and paths of non-.md files in the given directory.
    """
    extension_counts = Counter()
    non_md_files = []

    for file in directory.rglob("*"):
        if file.is_file():
            ext = file.suffix
            if ext:
                extension_counts[ext] += 1
                if ext.lower() != ".md":
                    non_md_files.append(file)

    return extension_counts, non_md_files


def count_and_delete_empty_dirs(directory: Path) -> Tuple[int, int]:
    """
    Counts and deletes completely empty directories within the given path.
    """
    if not directory.is_dir():
        raise ValueError(f"The path '{directory}' is not a valid directory.")

    total_dirs = 0
    empty_dirs = 0

    for dir_path in sorted(directory.rglob("*"), reverse=True):
        if dir_path.is_dir():
            total_dirs += 1
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    empty_dirs += 1
                    logger.info(f"Deleted empty directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Could not delete {dir_path}: {e}")

    return empty_dirs, total_dirs


def analyze_md_files(root_dir: Path) -> None:
    """
    Analyze the first-level artifact folders and count all .md files recursively inside each.
    Logs total counts and top 20 folders with most .md files.
    """
    artifact_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    logger.info(f"Total artifact folders found: {len(artifact_dirs)}")

    md_file_stats = defaultdict(int)
    total_md_files = 0

    for artifact_dir in artifact_dirs:
        md_files = list(artifact_dir.rglob("*.md"))
        count = len(md_files)
        md_file_stats[artifact_dir.name] = count
        total_md_files += count

    logger.info(f"Total .md files found: {total_md_files}")

    # Sort and print top 20 artifacts
    top_20 = sorted(md_file_stats.items(), key=lambda x: x[1], reverse=True)[:500]
    logger.info("Top 20 artifacts with the most .md files:")
    for name, count in top_20:
        logger.info(f"{name}: {count} .md files")



def count_node_modules_dirs(directory: Path) -> int:
    """
    Count the number of directories named 'node_modules' in the given directory tree.
    """
    count = sum(1 for d in directory.rglob("node_modules") if d.is_dir())
    logger.info(f"Total 'node_modules' directories found: {count}")
    return count



def delete_named_dirs(directory: Path, dir_names: List[str]) -> int:
    """
    Deletes all directories whose name matches any in the provided list.

    Args:
        directory (Path): Root directory to search.
        dir_names (List[str]): Directory names to delete (e.g., ['node_modules', '__MACOSX']).

    Returns:
        int: Total number of matching directories deleted.
    """
    deleted_count = 0

    for dir_path in directory.rglob("*"):
        if dir_path.is_dir() and dir_path.name in dir_names:
            try:
                shutil.rmtree(dir_path)
                deleted_count += 1
                logger.info(f"Deleted directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {dir_path}: {e}")

    logger.info(f"Total directories deleted ({', '.join(dir_names)}): {deleted_count}")
    return deleted_count


def main():
    directory = Path("../../algo_outputs/md_file_extraction")

    if not directory.exists():
        logger.error(f"The directory {directory} does not exist.")
        return

    # Step 1: Delete 'node_modules' and '__MACOSX' directories
    delete_named_dirs(directory, dir_names=["node_modules", "__MACOSX", "rust_programs", "opt", "gcc-10.1.0", "ext"])

    # Step 2: File extension stats
    extension_counts, non_md_files = extract_file_extensions_with_count_and_paths(directory)
    logger.info("File extensions and their counts in the directory:")
    for ext, count in sorted(extension_counts.items()):
        logger.info(f"{ext}: {count}")

    logger.info("Paths of non-.md files:")
    for file_path in non_md_files:
        logger.info(str(file_path))

    # Step 3: Cleanup empty directories
    empty_count, total_count = count_and_delete_empty_dirs(directory)
    logger.info(f"Total directories scanned: {total_count}")
    logger.info(f"Empty directories deleted: {empty_count}")

    # Step 4: Artifact-level .md analysis
    analyze_md_files(directory)


if __name__ == "__main__":
    main()
