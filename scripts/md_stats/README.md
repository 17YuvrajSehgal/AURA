# Artifact Markdown Documentation Statistics Script

## Overview

This script automatically scans a directory of artifact repositories, recursively collecting detailed statistics and patterns from all Markdown documentation files (`.md`).  
It is designed to help researchers and tool builders **analyze documentation practices**, **detect common structural features**, and **quantify quality, reproducibility, and usability signals** in large sets of scientific artifacts.

---

## Key Features

- **Recursive directory traversal** (supports large code/data archives with nested directories)
- **De-duplication** of identical and near-duplicate Markdown files
- **Aggregate and per-artifact statistics** on:
  - Unique Markdown files
  - Section headings (frequency and coverage)
  - General Markdown features (lists, tables, images, diagrams, links)
  - License mentions and author attribution
  - Installation and environment setup documentation
  - Troubleshooting, FAQ, and user support sections
  - Usage examples, configuration, and advanced customization
  - Reproducibility signals (data, scripts, environment, validation, related work)
  - **Readability and technicality metrics** (Flesch-Kincaid, Gunning-Fog, Dale-Chall, technical jargon ratio)

---

## How It Works

1. **Setup and Logging**
   - Sets up progress logging to both console and a log file.
   - Excludes common third-party and non-relevant directories to focus analysis on artifact-specific documentation.

2. **Markdown File Discovery**
   - Recursively finds all `.md` (and `.markdown`) files in artifact directories.
   - Removes duplicate files (exact match and near-duplicates using content similarity).

3. **Section and Feature Extraction**
   - Extracts all Markdown section headings, normalizes them, and tallies their frequency.
   - Analyzes each file for lists, tables, images, diagrams, outbound/internal hyperlinks, etc.

4. **Special Feature Detection**
   - Searches for license mentions, citation instructions, author/contributor sections.
   - Detects installation/setup instructions, dependency/environment info.
   - Finds troubleshooting/FAQ, usage examples, configuration options, and advanced usage tips.

5. **Reproducibility and Validation Signals**
   - Identifies documentation of reproducibility steps, data setup, validation scripts, environment duplication (Docker, Colab, Binder, etc.).
   - Looks for references to original papers, datasets, and related projects.

6. **Readability and Technicality Metrics**
   - Uses the `readability` Python library to compute Flesch-Kincaid, Gunning-Fog, and Dale-Chall scores for sufficiently long README files.
   - Calculates the ratio of technical jargon as a measure of accessibility.

7. **Parallel Processing**
   - Uses a thread pool to efficiently process large numbers of artifact directories in parallel.

---

## Output

- **Console and Log File Reports:**
  - Total and per-feature counts (lists, tables, images, etc.)
  - Frequency of licenses, citation practices, author attributions
  - Most common section headings and their coverage
  - Installation and usage instruction prevalence
  - Reproducibility and validation feature detection
  - Readability and technicality score distributions
  - Top file names and section headings (for template and prompt mining)

- **Example Output:**
  - "Total number of unique .md files: 1325"
  - "Top section headings: installation (1050), usage (900), license (800), requirements (750), reproducibility (620), citation (420)..."
  - "Flesch-Kincaid Scores: mean=10.5, std=2.1"
  - "Most common license: MIT License (200), Apache License (150)..."
  - "Files with installation text: 1040"
  - "Files with common issues/troubleshooting: 350"

---

## Applications

- **Prompt design:** Identify the most common documentation patterns to improve LLM prompt templates and artifact evaluation rubrics.
- **Documentation best practices:** Find gaps and recommend improvements to artifact authors.
- **Badge/reviewer alignment:** Correlate documentation features with badge/grading outcomes in artifact evaluation.
- **Accessibility research:** Quantify and improve the readability and accessibility of scientific documentation.

---

## Usage

1. Place all artifact directories in a parent folder (each subdir = one artifact).
2. Adjust the `relative_path` in the script to point to your artifact collection.
3. Run the script:
    ```bash
    python your_md_stats_script.py
    ```
4. Review results in the console and the log file (e.g., `algo_outputs/logs/md_stats_logs/md_stats.log`).

> **Note:** Install optional dependencies for readability scores:
> ```bash
> pip install py-readability-metrics tqdm
> ```

---

## Acknowledgments

- Inspired by the needs of artifact evaluation committees and reproducibility researchers.
- Uses `tqdm`, `readability`, and Python’s standard libraries for robust, efficient processing.

---

## Limitations and Extensions

- Section/feature extraction is markdown-based; nonstandard documentation may be missed.
- For full semantic clustering or to extract more sophisticated documentation patterns, consider augmenting with NLP/embedding models.
- Extensible: add more detectors for custom evaluation signals (e.g., Dockerfiles, requirements.txt parsing, test coverage).

---

