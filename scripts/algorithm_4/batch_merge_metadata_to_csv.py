import json
import csv
import os
import sys

def extract_repo_name(repo_path):
    return os.path.basename(os.path.normpath(repo_path))

def merge_all_metadata(parent_dir):
    algo2_dir = os.path.join(parent_dir, 'algorithm_2_output')
    algo4_dir = os.path.join(parent_dir, 'algorithm_4_output')
    output_csv_path = os.path.join(parent_dir, 'merged_metadata.csv')

    # Find all *_analysis.json and *_aura_evaluation.json files
    algo2_files = {os.path.splitext(f)[0].replace('_analysis',''): os.path.join(algo2_dir, f)
                   for f in os.listdir(algo2_dir) if f.endswith('_analysis.json')}
    algo4_files = {os.path.splitext(f)[0].replace('_aura_evaluation',''): os.path.join(algo4_dir, f)
                   for f in os.listdir(algo4_dir) if f.endswith('_aura_evaluation.json')}

    # Find common prefixes
    common_keys = sorted(set(algo2_files.keys()) & set(algo4_files.keys()))
    if not common_keys:
        print('No matching analysis/evaluation pairs found.')
        return

    # Prepare header
    header = None
    rows = []
    for key in common_keys:
        with open(algo2_files[key], 'r', encoding='utf-8') as f:
            algo2 = json.load(f)
        with open(algo4_files[key], 'r', encoding='utf-8') as f:
            algo4 = json.load(f)
        repo_name = extract_repo_name(algo2.get('repo_path', key))
        repo_size_mb = algo2.get('repo_size_mb', None)
        dimension_scores = {c['dimension']: c['llm_evaluated_score'] for c in algo4.get('criteria_scores', [])}
        all_dimensions = [c['dimension'] for c in algo4.get('criteria_scores', [])]
        total_weighted_score = algo4.get('total_weighted_score', None)
        acceptance_prediction = algo4.get('acceptance_prediction', None)
        timing = algo4.get('timing', {})
        if header is None:
            header = [
                'repo_name', 'repo_size_mb'
            ] + [f'{dim}_score' for dim in all_dimensions] + [
                'total_weighted_score', 'acceptance_prediction'
            ] + list(timing.keys())
        row = [
            repo_name, repo_size_mb
        ] + [dimension_scores.get(dim, None) for dim in all_dimensions] + [
            total_weighted_score, acceptance_prediction
        ] + [timing.get(k, None) for k in timing.keys()]
        rows.append(row)

    # Write CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f'Merged metadata written to {output_csv_path}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python batch_merge_metadata_to_csv.py <parent_directory>')
        sys.exit(1)
    merge_all_metadata(sys.argv[1])

#python scripts/algorithm_4/batch_merge_metadata_to_csv.py algo_outputs