#!/usr/bin/env python3
"""
Batch Analyzer - Process multiple artifacts with the IntegratedArtifactAnalyzer

This script demonstrates how to process multiple artifacts in batch mode,
collecting results and generating summary reports.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from integrated_artifact_analyzer import IntegratedArtifactAnalyzer


class BatchArtifactAnalyzer:
    """
    Batch processor for multiple artifacts using the IntegratedArtifactAnalyzer.
    
    Features:
    - Process multiple artifacts from a list or directory
    - Generate summary reports
    - Handle errors gracefully
    - Progress tracking
    """

    def __init__(self, temp_dir: str = "./batch_temp", output_dir: str = "./batch_outputs"):
        """
        Initialize the batch analyzer.
        
        Args:
            temp_dir: Temporary directory for extractions
            output_dir: Directory for saving results
        """
        self.analyzer = IntegratedArtifactAnalyzer(
            temp_dir=temp_dir,
            output_dir=output_dir
        )
        self.results = []
        self.errors = []

    def process_artifacts(self, artifact_list: List[Dict[str, str]],
                          force_reextract: bool = False) -> Dict[str, Any]:
        """
        Process a list of artifacts.
        
        Args:
            artifact_list: List of dictionaries with 'path' and optional 'name' keys
            force_reextract: Force re-extraction of all artifacts
            
        Returns:
            Summary dictionary with results and statistics
        """
        total_artifacts = len(artifact_list)
        successful = 0
        failed = 0

        print(f"Starting batch analysis of {total_artifacts} artifacts...")
        print("=" * 60)

        for i, artifact_info in enumerate(artifact_list, 1):
            artifact_path = artifact_info['path']
            artifact_name = artifact_info.get('name')

            print(f"\n[{i}/{total_artifacts}] Processing: {artifact_path}")

            try:
                result = self.analyzer.analyze_artifact(
                    artifact_path=artifact_path,
                    artifact_name=artifact_name,
                    force_reextract=force_reextract,
                    skip_analysis=False
                )

                if result["success"]:
                    successful += 1
                    self.results.append(result)
                    print(f"✓ Success: {result['artifact_name']}")

                    if result.get("analysis_performed", False):
                        print(f"  Files: {len(result.get('repository_structure', []))}")
                        print(f"  Size: {result.get('repo_size_mb', 0)} MB")
                        print(f"  Code: {len(result.get('code_files', []))}")
                        print(f"  Docs: {len(result.get('documentation_files', []))}")
                else:
                    failed += 1
                    error_info = {
                        'artifact_path': artifact_path,
                        'artifact_name': artifact_name,
                        'error': result.get('error', 'Unknown error')
                    }
                    self.errors.append(error_info)
                    print(f"✗ Failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                failed += 1
                error_info = {
                    'artifact_path': artifact_path,
                    'artifact_name': artifact_name,
                    'error': str(e)
                }
                self.errors.append(error_info)
                print(f"✗ Exception: {e}")

        # Generate summary
        summary = {
            'total_artifacts': total_artifacts,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total_artifacts * 100) if total_artifacts > 0 else 0,
            'processed_at': datetime.now().isoformat(),
            'results': self.results,
            'errors': self.errors
        }

        print("\n" + "=" * 60)
        print("BATCH ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total artifacts: {total_artifacts}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {summary['success_rate']:.1f}%")

        return summary

    def process_directory(self, directory: str, patterns: List[str] = None,
                          force_reextract: bool = False) -> Dict[str, Any]:
        """
        Process all artifacts in a directory.
        
        Args:
            directory: Directory containing artifacts
            patterns: File patterns to match (e.g., ['*.zip', '*.tar.gz'])
            force_reextract: Force re-extraction of all artifacts
            
        Returns:
            Summary dictionary with results and statistics
        """
        if patterns is None:
            patterns = ['*.zip', '*.tar', '*.tar.gz', '*.tgz', '*.tar.bz2', '*.tar.xz']

        directory_path = Path(directory)
        if not directory_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        # Find all matching files
        artifact_files = []
        for pattern in patterns:
            artifact_files.extend(directory_path.glob(pattern))

        # Also include subdirectories
        for item in directory_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                artifact_files.append(item)

        # Convert to artifact list
        artifact_list = []
        for file_path in artifact_files:
            artifact_list.append({
                'path': str(file_path),
                'name': file_path.stem
            })

        print(f"Found {len(artifact_list)} artifacts in {directory}")

        if not artifact_list:
            print("No artifacts found matching the patterns")
            return {
                'total_artifacts': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0,
                'processed_at': datetime.now().isoformat(),
                'results': [],
                'errors': []
            }

        return self.process_artifacts(artifact_list, force_reextract)

    def generate_report(self, summary: Dict[str, Any], output_file: str = None) -> str:
        """
        Generate a detailed report from the batch analysis summary.
        
        Args:
            summary: Summary dictionary from batch processing
            output_file: Optional output file path
            
        Returns:
            Report content as string
        """
        report_lines = []

        # Header
        report_lines.append("BATCH ARTIFACT ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {summary['processed_at']}")
        report_lines.append(f"Total artifacts: {summary['total_artifacts']}")
        report_lines.append(f"Successful: {summary['successful']}")
        report_lines.append(f"Failed: {summary['failed']}")
        report_lines.append(f"Success rate: {summary['success_rate']:.1f}%")
        report_lines.append("")

        # Successful analyses
        if summary['results']:
            report_lines.append("SUCCESSFUL ANALYSES")
            report_lines.append("-" * 30)

            total_files = 0
            total_size = 0
            languages = set()

            for result in summary['results']:
                name = result['artifact_name']
                files = len(result.get('repository_structure', []))
                size = result.get('repo_size_mb', 0)
                code_files = len(result.get('code_files', []))
                docs = len(result.get('documentation_files', []))

                total_files += files
                total_size += size

                report_lines.append(f"• {name}")
                report_lines.append(f"  Files: {files}, Size: {size} MB")
                report_lines.append(f"  Code: {code_files}, Docs: {docs}")
                report_lines.append("")

            # Summary statistics
            report_lines.append("AGGREGATE STATISTICS")
            report_lines.append("-" * 30)
            report_lines.append(f"Total files across all artifacts: {total_files}")
            report_lines.append(f"Total size across all artifacts: {total_size:.2f} MB")
            report_lines.append(f"Average files per artifact: {total_files / len(summary['results']):.1f}")
            report_lines.append(f"Average size per artifact: {total_size / len(summary['results']):.2f} MB")
            report_lines.append("")

        # Errors
        if summary['errors']:
            report_lines.append("FAILED ANALYSES")
            report_lines.append("-" * 30)

            for error in summary['errors']:
                name = error['artifact_name'] or os.path.basename(error['artifact_path'])
                report_lines.append(f"• {name}")
                report_lines.append(f"  Path: {error['artifact_path']}")
                report_lines.append(f"  Error: {error['error']}")
                report_lines.append("")

        # File type distribution
        if summary['results']:
            report_lines.append("FILE TYPE DISTRIBUTION")
            report_lines.append("-" * 30)

            file_type_counts = {}
            for result in summary['results']:
                for file_info in result.get('repository_structure', []):
                    ext = Path(file_info['name']).suffix.lower()
                    file_type_counts[ext] = file_type_counts.get(ext, 0) + 1

            # Sort by count
            sorted_types = sorted(file_type_counts.items(), key=lambda x: x[1], reverse=True)

            for ext, count in sorted_types[:10]:  # Top 10
                ext_display = ext if ext else '(no extension)'
                report_lines.append(f"  {ext_display}: {count}")

            if len(sorted_types) > 10:
                report_lines.append(f"  ... and {len(sorted_types) - 10} more types")

        report_content = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Report saved to: {output_file}")

        return report_content

    def save_summary(self, summary: Dict[str, Any], output_file: str):
        """Save the batch summary to a JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to: {output_file}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Batch Artifact Analyzer')
    parser.add_argument('input', help='Input directory or artifact list file')
    parser.add_argument('--output-dir', default='./batch_outputs',
                        help='Output directory for results')
    parser.add_argument('--temp-dir', default='./batch_temp',
                        help='Temporary directory for extractions')
    parser.add_argument('--force', action='store_true',
                        help='Force re-extraction of all artifacts')
    parser.add_argument('--patterns', nargs='+',
                        default=['*.zip', '*.tar', '*.tar.gz', '*.tgz', '*.tar.bz2', '*.tar.xz'],
                        help='File patterns to match')
    parser.add_argument('--report', default='batch_report.txt',
                        help='Report output file')
    parser.add_argument('--summary', default='batch_summary.json',
                        help='Summary JSON output file')

    args = parser.parse_args()

    # Initialize batch analyzer
    batch_analyzer = BatchArtifactAnalyzer(
        temp_dir=args.temp_dir,
        output_dir=args.output_dir
    )

    try:
        input_path = Path(args.input)

        if input_path.is_dir():
            # Process directory
            summary = batch_analyzer.process_directory(
                directory=str(input_path),
                patterns=args.patterns,
                force_reextract=args.force
            )
        elif input_path.is_file() and input_path.suffix.lower() == '.json':
            # Process artifact list from JSON file
            with open(input_path, 'r', encoding='utf-8') as f:
                artifact_list = json.load(f)

            summary = batch_analyzer.process_artifacts(
                artifact_list=artifact_list,
                force_reextract=args.force
            )
        else:
            raise ValueError(f"Invalid input: {args.input}. Must be a directory or JSON file.")

        # Generate and save report
        report_content = batch_analyzer.generate_report(summary, args.report)
        batch_analyzer.save_summary(summary, args.summary)

        print(f"\nBatch analysis completed!")
        print(f"Report: {args.report}")
        print(f"Summary: {args.summary}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# python scripts/algorithm_5/icse_artifacts_processor.py temp_dir_for_git/icse_artifacts --neo4j-user neo4j --neo4j-password 12345678 --max-workers 2 --output-dir icse_analysis_results


# Analyze any artifact type
# python integrated_artifact_analyzer.py path/to/artifact.zip
# python integrated_artifact_analyzer.py /path/to/directory/
# python integrated_artifact_analyzer.py https://github.com/sneh2001patel/ml-image-classifier

# Batch process multiple artifacts
# python batch_analyzer.py /path/to/artifacts/directory/

# Run tests
# python test_integrated_analyzer.py
