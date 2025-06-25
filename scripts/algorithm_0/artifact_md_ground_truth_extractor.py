import os
import re
import json
import csv
import logging
from collections import defaultdict
from pathlib import Path

# ------------------- CONFIGURATION -------------------
# Use the directory structure as shown in the attached image
ARTIFACTS_ROOT = Path(__file__).parent.parent.parent / 'algo_outputs' / 'md_file_extraction'
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'algo_outputs'
OUTPUT_JSON = OUTPUT_DIR / 'algorithm_0_output' / 'artifact_md_ground_truth.json'
OUTPUT_CSV = OUTPUT_DIR / 'algorithm_0_output' / 'artifact_md_ground_truth.csv'
# Update this path to your latest criteria file as needed
CRITERIA_PATH = OUTPUT_DIR / 'algorithm_1_output' / 'aura_integration_data_20250624_233631.json'

# ------------------- LOGGING SETUP -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("artifact_md_ground_truth")

# ------------------- UTILITY FUNCTIONS -------------------
def extract_sections(md_text):
    """Extract Markdown header sections from text."""
    return re.findall(r'^#+\s+(.*)', md_text, re.MULTILINE)

def score_against_keywords(md_text, keywords):
    """Count how many keywords appear in the markdown text."""
    text = md_text.lower()
    return sum(1 for kw in keywords if kw.lower() in text)

def load_criteria(criteria_path):
    """Load Algorithm 1 criteria from JSON file."""
    if not criteria_path.exists():
        logger.error(f"Criteria file not found: {criteria_path}")
        logger.error("Please run Enhanced Algorithm 1 to generate the criteria file (aura_integration_data_*.json).")
        raise FileNotFoundError(f"Criteria file not found: {criteria_path}")
    with open(criteria_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    criteria = {}
    for crit in data['structured_criteria']:
        all_keywords = set(crit['keywords'])
        for cat in crit['hierarchical_structure'].values():
            all_keywords.update(cat)
        criteria[crit['dimension']] = list(all_keywords)
    return criteria

def ensure_output_dirs():
    """Ensure output directories exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure the algorithm_0_output subdirectory exists
    (OUTPUT_DIR / 'algorithm_0_output').mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {OUTPUT_DIR / 'algorithm_0_output'}")

def process_artifacts(artifacts_root, criteria):
    """Process all artifact folders and score their markdown files."""
    dimensions = list(criteria.keys())
    results = []
    aggregate = defaultdict(int)
    artifact_count = 0
    logger.info(f"Scanning artifacts in: {artifacts_root}")

    for artifact_dir in artifacts_root.iterdir():
        if not artifact_dir.is_dir():
            continue
        artifact_result = {
            'artifact_id': artifact_dir.name,
            'scores': {},
            'files': []
        }
        for root, _, files in os.walk(artifact_dir):
            for file in files:
                if file.lower().endswith('.md'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
                        continue
                    sections = extract_sections(content)
                    file_result = {
                        'path': str(file_path.relative_to(artifacts_root)),
                        'sections': sections,
                        'file_type': file.upper().replace('.MD', '')
                    }
                    artifact_result['files'].append(file_result)
                    for dim in dimensions:
                        score = score_against_keywords(content, criteria[dim])
                        artifact_result['scores'][dim] = artifact_result['scores'].get(dim, 0) + score
        for dim in dimensions:
            if artifact_result['scores'].get(dim, 0) > 0:
                aggregate[dim] += 1
        results.append(artifact_result)
        artifact_count += 1
        if artifact_count % 100 == 0:
            logger.info(f"Processed {artifact_count} artifacts...")
    return results, aggregate, artifact_count, dimensions

def save_json(results, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved per-artifact results to {path}")

def save_csv(results, path, dimensions):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['artifact_id'] + dimensions
        writer.writerow(header)
        for artifact in results:
            row = [artifact['artifact_id']] + [artifact['scores'].get(dim, 0) for dim in dimensions]
            writer.writerow(row)
    logger.info(f"Saved summary CSV to {path}")

def print_aggregate_stats(aggregate, artifact_count, dimensions):
    logger.info("\nAggregate Coverage Statistics:")
    if artifact_count == 0:
        logger.warning("No artifact folders found! Please check your ARTIFACTS_ROOT path and data.")
        return
    for dim in dimensions:
        logger.info(f"Artifacts with {dim}: {aggregate[dim]} / {artifact_count} ({aggregate[dim]/artifact_count:.1%})")

def main():
    ensure_output_dirs()
    try:
        criteria = load_criteria(CRITERIA_PATH)
    except FileNotFoundError:
        return
    results, aggregate, artifact_count, dimensions = process_artifacts(ARTIFACTS_ROOT, criteria)
    save_json(results, OUTPUT_JSON)
    save_csv(results, OUTPUT_CSV, dimensions)
    print_aggregate_stats(aggregate, artifact_count, dimensions)
    logger.info("\nAll done! Review the outputs for your ground truth and statistics.")

if __name__ == '__main__':
    main() 