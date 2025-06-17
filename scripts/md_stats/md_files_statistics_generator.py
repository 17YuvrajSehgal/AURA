import concurrent.futures
import hashlib
import logging
import os
import re
from collections import Counter
from difflib import SequenceMatcher

from tqdm import tqdm

try:
    from readability import Readability
except ImportError:
    Readability = None

# Configure logging to replace the older log file each time
log_path = os.path.abspath('../../algo_outputs/logs/md_stats_logs/md_stats.log')
log_dir = os.path.dirname(log_path)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

EXCLUDED_DIRS = {
    'node_modules',
    '__MACOSX',
    '.github',
    '.gitlab',
    'test',
    'tests',
    'testdata',
    'test_data'
    'benchmarks',
    'benchmark',
    'bench',
    'include',
    'internal',
    'third_party',
    'third-party',
    'legal',
    'googletest',
    'gtest',
    'googlemock',
    'gmock',
    'csv_parser',
    'ini',
    'inih',
    'docker',
    'vendor',
    'external',
    'contrib',
    'examples',
    'example',
    'cjson',
    'cJSON',
    'unity',
    'cmd',
    'util',
    'utils',
    'deps',
    'runtime',
    'apps',
    'rust'
    'config',
    'api',
    'clang',
    'llvm',
    'ext',
    'readable-stream',
    'safe-buffer',
    'semver',
    'strip-ansi',
    'punycode',
    'source-map',
    'github.com',
    'server',
    'pybind11',
    'g3doc',
    'package',
    'acorn',
    'supports-color',
    'color-convert',
    'color-name',
    'etc',
    'taco',
    'experiments',
    'analysis',
    'string_decoder',
    'frontend',
    'bn.js',
    'c',
    'isl',
    'imath',
    'org',
    'ansi-styles',
    'inherits',
    'tablegen',
    'image',
    'templates',
    'afl',
    'experimental',
    'assets',
    'tools'
}


def count_md_files(directory):
    """
    Counts the total number of .md files in a given directory and its subdirectories.
    """
    md_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.md'):
                md_count += 1
    return md_count


def read_file_content(filepath):
    """
    Reads the content of a file, handling encoding errors and file-related issues
    such as missing or inaccessible files.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except (FileNotFoundError, PermissionError) as e:
        # Log a warning and return an empty string (or None)
        logging.warning(f"Cannot open file {filepath} due to error: {e}")
        return ""
    except UnicodeDecodeError:
        # If UTF-8 fails, try latin-1
        try:
            with open(filepath, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logging.warning(f"Cannot open file {filepath} with fallback encoding. Skipping. Error: {e}")
            return ""


def are_files_similar(content1, content2, threshold=0.9):
    """
    Checks if two files are similar based on a given similarity threshold (default=0.9).
    """
    similarity = SequenceMatcher(None, content1, content2).ratio()
    return similarity >= threshold


def extract_headings(content):
    """
    Extracts markdown headings from file content, treating all heading levels (#, ##, ###, etc.) the same.

    Example:
      "# Introduction" -> "introduction"
      "### License"    -> "license"
    """
    headings = []
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            heading_text = stripped_line.lstrip('#').strip().lower()
            if heading_text:
                headings.append(heading_text)
    return headings


def analyze_markdown_features(content, features_counters):
    """
    Updates the features_counters with statistics about:
      - bullet/numbered lists
      - tables
      - images/diagrams
      - hyperlinks (outbound vs. internal)
    """
    lines = content.splitlines()

    # Regex patterns
    bullet_list_pattern = re.compile(r'^\s*[-\*\+]\s')  # e.g., "- item", "* item", "+ item"
    numbered_list_pattern = re.compile(r'^\s*\d+\.\s')  # e.g., "1. item", "2. item"
    table_pattern = re.compile(r'\|[-]+?')  # minimal check for '|---|'

    # For images: ![alt text](path/to/image.png)
    image_pattern = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<file>[^)]+\.(?:png|jpg|jpeg|svg|gif))\)',
                               flags=re.IGNORECASE)

    # For hyperlinks: [text](URL)
    link_pattern = re.compile(r'\[(?P<label>[^\]]+)\]\((?P<target>[^\)]+)\)')

    # Keywords for diagram classification
    diagram_keywords = ('diagram', 'flow', 'architecture', 'screenshot')

    for line in lines:
        # 1) Lists
        if bullet_list_pattern.match(line):
            features_counters['bullet_lists'] += 1
        elif numbered_list_pattern.match(line):
            features_counters['numbered_lists'] += 1

        # 2) Tables
        if table_pattern.search(line):
            features_counters['tables'] += 1

        # 3) Images
        images_found = image_pattern.findall(line)
        for alt, filename in images_found:
            features_counters['images'] += 1

            text_lower = (alt + filename).lower()
            for keyword in diagram_keywords:
                if keyword in text_lower:
                    features_counters[f'{keyword}_images'] += 1

        # 4) Hyperlinks
        links_found = link_pattern.findall(line)
        for (_, target) in links_found:
            # Outbound if starts with http:// or https://
            if target.lower().startswith('http://') or target.lower().startswith('https://'):
                features_counters['outbound_links'] += 1
            else:
                features_counters['internal_links'] += 1


def analyze_licensing_and_attribution(content, license_attrib_counters):
    """
    Updates the license_attrib_counters with statistics about:
      - license details (e.g., MIT, Apache, GPL, etc.)
      - citation instructions (e.g., "How to cite", "DOI", "Zenodo")
      - author/contributor attribution
    """
    LICENSE_KEYWORDS = [
        'mit license', 'apache license', 'gpl', 'lgpl', 'bsd license',
        'cc-by', 'creative commons', 'epl', 'eupl', 'mpl', 'unlicense'
    ]
    CITATION_KEYWORDS = ['how to cite', 'citation', 'doi:', 'zenodo', 'paper url']
    AUTHOR_KEYWORDS = ['author', 'contributors', 'maintainers', 'developer']

    content_lower = content.lower()

    # License keywords
    for lk in LICENSE_KEYWORDS:
        if lk in content_lower:
            license_attrib_counters['license_mentions'][lk] += 1

    # Citation instructions
    for ck in CITATION_KEYWORDS:
        if ck in content_lower:
            license_attrib_counters['citation_mentions'][ck] += 1

    # Author/contributor
    found_author_info = any(ak in content_lower for ak in AUTHOR_KEYWORDS)
    if found_author_info:
        license_attrib_counters['files_with_authors'] += 1


def analyze_installation_instructions(file_path, content, install_counters):
    """
    Extracts stats about installation instructions and environment requirements.
    """
    filename_lower = os.path.basename(file_path).lower()
    content_lower = content.lower()

    # Named "install.md"
    if "install" in filename_lower:
        install_counters['files_named_install'] += 1

    # Headings or text for "installation", "setup", etc.
    installation_headings = re.compile(r'(#+\s*(installation|setup|how to install|install|install requirements))',
                                       re.IGNORECASE)
    if installation_headings.search(content):
        install_counters['files_with_installation_headings'] += 1

    if "installation" in content_lower or "setup" in content_lower:
        install_counters['files_with_installation_text'] += 1

    # Dependencies / environment keywords
    dependencies_keywords = [
        'dependencies', 'requirements', 'pip install', 'conda install', 'npm install', 'npm', 'pip'
    ]
    if any(kw in content_lower for kw in dependencies_keywords):
        install_counters['files_with_dependencies'] += 1

    # Python/OS version checks
    environment_keywords = ['python 3', 'windows', 'linux', 'macos', 'ubuntu', 'os support']
    if any(kw in content_lower for kw in environment_keywords):
        install_counters['files_with_env_requirements'] += 1


def analyze_troubleshooting_faq(content, troubleshooting_counters):
    """
    Updates troubleshooting_counters with statistics about:
      - Common errors or known issues
      - Guidance on how to get help or report bugs
    """
    content_lower = content.lower()

    # Common errors, FAQ, or known issues
    troubleshooting_keywords = [
        'common error', 'common errors', 'known issue', 'known issues',
        'faq', 'frequently asked questions', 'troubleshoot'
    ]
    if any(kw in content_lower for kw in troubleshooting_keywords):
        troubleshooting_counters['files_with_common_issues'] += 1

    # Guidance on how to get help or report bugs
    bug_help_keywords = ['report bug', 'bug tracker', 'issue tracker', 'support', 'help']
    if any(kw in content_lower for kw in bug_help_keywords):
        troubleshooting_counters['files_with_bug_guidance'] += 1


def analyze_usage_and_configuration(content, usage_config_counters):
    """
    Updates usage_config_counters with stats about:
      - Example commands
      - Sample inputs/outputs or interface screenshots
      - Config env variables or config files
      - Advanced customization tips
    """
    content_lower = content.lower()

    # Example commands
    command_keywords = [r'python\s+\w+\.py', r'python3\s+\w+\.py', r'\./\w+\.sh', r'\.\.\/\w+\.sh']
    found_command = False
    for cmd_regex in command_keywords:
        if re.search(cmd_regex, content_lower):
            found_command = True
            break
    if found_command:
        usage_config_counters['files_with_example_commands'] += 1

    # Sample inputs/outputs or interface screenshots
    sample_keywords = ['sample input', 'sample output', 'example usage', 'cli screenshot', 'interface screenshot']
    if any(kw in content_lower for kw in sample_keywords):
        usage_config_counters['files_with_samples'] += 1

    # Config env variables or config files
    config_keywords = ['env var', 'environment variable', 'export ', 'set ', 'config file', 'yaml config',
                       'json config']
    if any(kw in content_lower for kw in config_keywords):
        usage_config_counters['files_with_env_config'] += 1

    # Advanced customization
    customization_keywords = ['advanced usage', 'advanced configuration', 'override default', 'customize']
    if any(kw in content_lower for kw in customization_keywords):
        usage_config_counters['files_with_advanced_customization'] += 1


def analyze_reproducibility_signals(content, reproducibility_counters):
    """
    6. Doc-Driven Reproducibility Signals
      - Reproducibility steps (data setup, environment duplication, seed setting, Binder/Colab/Docker usage)
      - Validation instructions (scripts for testing, expected metrics)
      - References to related work (research paper, data sources, relevant projects)
    """
    content_lower = content.lower()

    # (A) Reproducibility steps
    reproducibility_keywords = [
        'data setup', 'environment duplication', 'random seed', 'set seed', 'reproducibility',
        'docker', 'colab', 'binder', 'container'
    ]
    if any(kw in content_lower for kw in reproducibility_keywords):
        reproducibility_counters['files_with_repro_steps'] += 1

    # (B) Validation instructions
    validation_keywords = [
        'validate results', 'test script', 'accuracy', 'performance metric', 'evaluation script', 'benchmark'
    ]
    if any(kw in content_lower for kw in validation_keywords):
        reproducibility_counters['files_with_validation_steps'] += 1

    # (C) References to related work
    related_work_keywords = [
        'original research paper', 'arxiv', 'data source', 'related project', 'related work', 'reference'
    ]
    if any(kw in content_lower for kw in related_work_keywords):
        reproducibility_counters['files_with_related_references'] += 1


# --- NEW CODE FOR QUALITY & READABILITY BELOW ---

def analyze_quality_and_readability(file_path, content, quality_readability_counters):
    """
    Measures readability only for README files (case-insensitive) with >= 100 words:
      - Flesch-Kincaid, Gunning Fog, Dale-Chall, etc. (via py-readability-metrics)
      - Technical jargon ratio as a simple heuristic
    """
    if not Readability:
        return  # If the library isn't installed or not accessible, skip

    filename_lower = os.path.basename(file_path).lower()
    # Check if it's a README file
    if 'readme' not in filename_lower:
        return

    words = content.split()
    if len(words) < 100:
        return  # readability library needs >= 100 words

    # Use the readability library
    r = Readability(content)
    # Flesch-Kincaid
    try:
        fk = r.flesch_kincaid()
        quality_readability_counters['flesch_kincaid_scores'].append(fk.score)
    except:
        pass

    # Gunning Fog
    try:
        gf = r.gunning_fog()
        quality_readability_counters['gunning_fog_scores'].append(gf.score)
    except:
        pass

    # Dale-Chall
    try:
        dc = r.dale_chall()
        quality_readability_counters['dale_chall_scores'].append(dc.score)
    except:
        pass

    # Simple technical jargon ratio
    TECHNICAL_WORDS = {
        'api', 'library', 'dataset', 'framework', 'dependency', 'gpu',
        'docker', 'virtualenv', 'conda', 'cli', 'shell', 'python',
        'server', 'backend', 'runtime', 'build', 'script', 'repository',
        'compile', 'kernel'
    }
    tech_count = 0
    for w in words:
        # strip punctuation, lowercase
        w_clean = re.sub(r'[^\w]', '', w.lower())
        if w_clean in TECHNICAL_WORDS:
            tech_count += 1

    ratio = tech_count / len(words)
    quality_readability_counters['technical_jargon_ratios'].append(ratio)


def process_artifact(subdir_path):
    """
    Processes all markdown files within a single artifact directory.
    Returns the statistics for this artifact.
    """
    unique_files_counter = Counter()
    heading_total_count = Counter()
    heading_file_count = Counter()

    features_counters = {
        'bullet_lists': 0,
        'numbered_lists': 0,
        'tables': 0,
        'images': 0,
        'diagram_images': 0,
        'flow_images': 0,
        'architecture_images': 0,
        'screenshot_images': 0,
        'outbound_links': 0,
        'internal_links': 0
    }

    license_attrib_counters = {
        'license_mentions': Counter(),
        'citation_mentions': Counter(),
        'files_with_authors': 0
    }

    install_counters = {
        'files_named_install': 0,
        'files_with_installation_headings': 0,
        'files_with_installation_text': 0,
        'files_with_dependencies': 0,
        'files_with_env_requirements': 0
    }

    troubleshooting_counters = {
        'files_with_common_issues': 0,
        'files_with_bug_guidance': 0
    }

    usage_config_counters = {
        'files_with_example_commands': 0,
        'files_with_samples': 0,
        'files_with_env_config': 0,
        'files_with_advanced_customization': 0
    }

    reproducibility_counters = {
        'files_with_repro_steps': 0,
        'files_with_validation_steps': 0,
        'files_with_related_references': 0
    }

    # Dictionary to store readability scores & ratios
    quality_readability_counters = {
        'flesch_kincaid_scores': [],
        'gunning_fog_scores': [],
        'dale_chall_scores': [],
        'technical_jargon_ratios': []
    }

    # Initialize a list to keep track of seen file hashes within this artifact
    seen_hashes = set()
    seen_contents = []  # To store contents for similarity checks within artifact

    # Walk through the artifact directory
    for root, dirs, files in os.walk(subdir_path):
        # Modify dirs in-place to exclude certain directories
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIRS]
        for file in files:
            if file.lower().endswith(('.md', '.markdown')) and not file.lower().startswith("._"):
                file_path = os.path.join(root, file)
                print(file_path)
                file_content = read_file_content(file_path)

                # Skip empty files
                if not file_content:
                    continue

                # Compute a hash of the content to quickly identify exact duplicates
                file_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
                if file_hash in seen_hashes:
                    continue  # Exact duplicate found within the artifact
                else:
                    # Now check for near-duplicates (90% similarity)
                    is_unique = True
                    for sc in seen_contents:
                        if are_files_similar(file_content, sc):
                            is_unique = False
                            break

                    if is_unique:
                        seen_hashes.add(file_hash)
                        seen_contents.append(file_content)
                        unique_files_counter.update([os.path.basename(file_path).lower()])

                        # Extract headings
                        headings = extract_headings(file_content)
                        heading_total_count.update(headings)
                        heading_file_count.update(set(headings))

                        # Analyze general Markdown features
                        analyze_markdown_features(file_content, features_counters)

                        # Analyze licensing and attribution
                        analyze_licensing_and_attribution(file_content, license_attrib_counters)

                        # Analyze installation/technical instructions
                        analyze_installation_instructions(file_path, file_content, install_counters)

                        # Analyze troubleshooting/FAQ
                        analyze_troubleshooting_faq(file_content, troubleshooting_counters)

                        # Analyze usage & configuration
                        analyze_usage_and_configuration(file_content, usage_config_counters)

                        # Analyze reproducibility signals
                        analyze_reproducibility_signals(file_content, reproducibility_counters)

                        # Analyze readability only for README.md files with >=100 words
                        analyze_quality_and_readability(file_path, file_content, quality_readability_counters)

    # Aggregate statistics for this artifact
    return {
        "total_unique_files": sum(unique_files_counter.values()),
        "unique_files_counter": unique_files_counter,
        "heading_total_count": heading_total_count,
        "heading_file_count": heading_file_count,
        "features": features_counters,
        "licensing_attrib": license_attrib_counters,
        "install_stats": install_counters,
        "troubleshooting_stats": troubleshooting_counters,
        "usage_config_stats": usage_config_counters,
        "reproducibility_stats": reproducibility_counters,
        "quality_readability_stats": quality_readability_counters
    }


def generate_statistics(directory):
    """
    Gathers statistics by processing each artifact directory individually.
    Utilizes parallel processing to speed up the computation.
    """
    # List all artifact subdirectories
    artifact_dirs = [
        os.path.join(directory, subdir) for subdir in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, subdir))
    ]

    # Initialize aggregate counters
    aggregate_unique_files_counter = Counter()
    aggregate_heading_total_count = Counter()
    aggregate_heading_file_count = Counter()

    aggregate_features_counters = Counter({
        'bullet_lists': 0,
        'numbered_lists': 0,
        'tables': 0,
        'images': 0,
        'diagram_images': 0,
        'flow_images': 0,
        'architecture_images': 0,
        'screenshot_images': 0,
        'outbound_links': 0,
        'internal_links': 0
    })

    aggregate_license_attrib_counters = {
        'license_mentions': Counter(),
        'citation_mentions': Counter(),
        'files_with_authors': 0
    }

    aggregate_install_counters = Counter({
        'files_named_install': 0,
        'files_with_installation_headings': 0,
        'files_with_installation_text': 0,
        'files_with_dependencies': 0,
        'files_with_env_requirements': 0
    })

    aggregate_troubleshooting_counters = Counter({
        'files_with_common_issues': 0,
        'files_with_bug_guidance': 0
    })

    aggregate_usage_config_counters = Counter({
        'files_with_example_commands': 0,
        'files_with_samples': 0,
        'files_with_env_config': 0,
        'files_with_advanced_customization': 0
    })

    aggregate_reproducibility_counters = Counter({
        'files_with_repro_steps': 0,
        'files_with_validation_steps': 0,
        'files_with_related_references': 0
    })

    aggregate_quality_readability_counters = {
        'flesch_kincaid_scores': [],
        'gunning_fog_scores': [],
        'dale_chall_scores': [],
        'technical_jargon_ratios': []
    }

    # Use ThreadPoolExecutor to process artifacts in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map each artifact directory to the process_artifact function
        future_to_artifact = {executor.submit(process_artifact, artifact_dir): artifact_dir for artifact_dir in
                              artifact_dirs}

        # Initialize progress bar
        with tqdm(total=len(artifact_dirs), desc="Processing Artifacts", unit="artifact") as pbar:
            for future in concurrent.futures.as_completed(future_to_artifact):
                artifact_dir = future_to_artifact[future]
                try:
                    stats = future.result()

                    # Aggregate unique files
                    aggregate_unique_files_counter.update(stats['unique_files_counter'])

                    # Aggregate headings
                    aggregate_heading_total_count.update(stats['heading_total_count'])
                    aggregate_heading_file_count.update(stats['heading_file_count'])

                    # Aggregate features
                    for key, value in stats['features'].items():
                        aggregate_features_counters[key] += value

                    # Aggregate licensing & attribution
                    for key, value in stats['licensing_attrib']['license_mentions'].items():
                        aggregate_license_attrib_counters['license_mentions'][key] += value
                    for key, value in stats['licensing_attrib']['citation_mentions'].items():
                        aggregate_license_attrib_counters['citation_mentions'][key] += value
                    aggregate_license_attrib_counters['files_with_authors'] += stats['licensing_attrib'][
                        'files_with_authors']

                    # Aggregate installation counters
                    for key, value in stats['install_stats'].items():
                        aggregate_install_counters[key] += value

                    # Aggregate troubleshooting counters
                    for key, value in stats['troubleshooting_stats'].items():
                        aggregate_troubleshooting_counters[key] += value

                    # Aggregate usage & config counters
                    for key, value in stats['usage_config_stats'].items():
                        aggregate_usage_config_counters[key] += value

                    # Aggregate reproducibility counters
                    for key, value in stats['reproducibility_stats'].items():
                        aggregate_reproducibility_counters[key] += value

                    # Aggregate quality & readability
                    for key, value in stats['quality_readability_stats'].items():
                        if isinstance(value, list):
                            aggregate_quality_readability_counters[key].extend(value)
                        else:
                            aggregate_quality_readability_counters[key] += value

                except Exception as e:
                    logging.error(f"Error processing artifact {artifact_dir}: {e}")

                # Update the progress bar
                pbar.update(1)

    return {
        "total_unique_files": sum(aggregate_unique_files_counter.values()),
        "unique_files_counter": aggregate_unique_files_counter,
        "heading_total_count": aggregate_heading_total_count,
        "heading_file_count": aggregate_heading_file_count,
        "features": dict(aggregate_features_counters),
        "licensing_attrib": aggregate_license_attrib_counters,
        "install_stats": dict(aggregate_install_counters),
        "troubleshooting_stats": dict(aggregate_troubleshooting_counters),
        "usage_config_stats": dict(aggregate_usage_config_counters),
        "reproducibility_stats": dict(aggregate_reproducibility_counters),
        "quality_readability_stats": aggregate_quality_readability_counters
    }


if __name__ == "__main__":
    relative_path = "../../algo_outputs/md_file_extraction"

    # Normalize and convert to absolute to avoid mixing slashes
    directory_path = os.path.abspath(relative_path)
    directory_path = os.path.normpath(directory_path)

    logging.info(f"Starting processing for directory: {directory_path}")
    try:
        total_md_files = count_md_files(directory_path)
        logging.info(f"Total number of .md files: {total_md_files}")

        stats = generate_statistics(directory_path)

        # --- Print/Log results ---
        logging.info(f"Total number of unique .md files: {stats['total_unique_files']}")
        print(f"Total number of unique .md files: {stats['total_unique_files']}")

        # Markdown features
        logging.info("\nMarkdown Feature Stats (Aggregate across unique files):")
        print("\nMarkdown Feature Stats (Aggregate across unique files):")
        for feature_name, feature_count in stats['features'].items():
            logging.info(f"{feature_name}: {feature_count}")
            print(f"  {feature_name}: {feature_count}")

        # Licensing & attribution
        logging.info("\nLicensing & Attribution Stats:")
        print("\nLicensing & Attribution Stats:")

        # License mentions
        logging.info("  License Mentions:")
        print("  License Mentions:")
        for lic, lic_count in stats['licensing_attrib']['license_mentions'].items():
            logging.info(f"    {lic}: {lic_count}")
            print(f"    {lic}: {lic_count}")

        # Citation mentions
        logging.info("  Citation Mentions:")
        print("  Citation Mentions:")
        for cit, cit_count in stats['licensing_attrib']['citation_mentions'].items():
            logging.info(f"    {cit}: {cit_count}")
            print(f"    {cit}: {cit_count}")

        # Files with authors
        logging.info("  Files with authors/contributors:")
        print("  Files with authors/contributors:")
        fa = stats['licensing_attrib']['files_with_authors']
        logging.info(f"    {fa}")
        print(f"    {fa}")

        # Installation / technical instructions
        install_stats = stats['install_stats']
        logging.info("\nInstallation / Technical Instructions Stats:")
        print("\nInstallation / Technical Instructions Stats:")
        for key, val in install_stats.items():
            logging.info(f"  {key}: {val}")
            print(f"  {key}: {val}")

        # Troubleshooting / FAQ
        troubleshooting_stats = stats['troubleshooting_stats']
        logging.info("\nTroubleshooting / FAQ Stats:")
        print("\nTroubleshooting / FAQ Stats:")
        for key, val in troubleshooting_stats.items():
            logging.info(f"  {key}: {val}")
            print(f"  {key}: {val}")

        # Usage & Config
        usage_config_stats = stats['usage_config_stats']
        logging.info("\nUsage & Configuration Stats:")
        print("\nUsage & Configuration Stats:")
        for key, val in usage_config_stats.items():
            logging.info(f"  {key}: {val}")
            print(f"  {key}: {val}")

        # Reproducibility Signals
        reproducibility_stats = stats['reproducibility_stats']
        logging.info("\nDoc-Driven Reproducibility Signals:")
        print("\nDoc-Driven Reproducibility Signals:")
        for key, val in reproducibility_stats.items():
            logging.info(f"  {key}: {val}")
            print(f"  {key}: {val}")

        # Quality & Readability
        quality_stats = stats['quality_readability_stats']
        fk_scores = quality_stats['flesch_kincaid_scores']
        gf_scores = quality_stats['gunning_fog_scores']
        dc_scores = quality_stats['dale_chall_scores']
        jargon_ratios = quality_stats['technical_jargon_ratios']

        print("\nQuality & Readability Stats (README only, ≥100 words):")
        print("  Flesch-Kincaid Scores:", fk_scores)
        print("  Gunning-Fog Scores:", gf_scores)
        print("  Dale-Chall Scores:", dc_scores)
        print("  Technical-Jargon Ratios:", jargon_ratios)

        # Also log these to md_stats.log
        logging.info("\nQuality & Readability Stats (README only, ≥100 words):")
        logging.info(f"  Flesch-Kincaid Scores: {fk_scores}")
        logging.info(f"  Gunning-Fog Scores: {gf_scores}")
        logging.info(f"  Dale-Chall Scores: {dc_scores}")
        logging.info(f"  Technical-Jargon Ratios: {jargon_ratios}")

        # Top 100 unique file names
        logging.info("\nTop 100 file names:")
        for file_name, count in stats['unique_files_counter'].most_common(100):
            logging.info(f"{file_name}: {count}")

        # Headings by total occurrence
        logging.info("\nTop 100 headings by total occurrences (across unique files):")
        for heading, count in stats['heading_total_count'].most_common(100):
            logging.info(f"{heading}: {count}")

        # Headings by number of unique files containing them
        logging.info("\nTop 100 headings by number of unique files containing them:")
        for heading, count in stats['heading_file_count'].most_common(100):
            logging.info(f"{heading}: {count}")

        print("\nStats have been logged.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred. Check md_stats.log for details.")
