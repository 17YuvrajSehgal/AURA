#!/usr/bin/env python3
"""
Main execution script for Conference-Specific Algorithm 1.

This script provides a command-line interface for running the enhanced
evaluation criteria extraction algorithm with conference-specific features.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conference_specific_algorithm import ConferenceSpecificAlgorithm1
from conference_profiles import ConferenceProfileManager
from utils import setup_logging, validate_input_directory, create_output_directory
from config import Config


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Conference-Specific Algorithm 1: Enhanced Evaluation Criteria Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract criteria for all conferences
  python run_conference_extraction.py --input-dir /path/to/guidelines --output-dir /path/to/output

  # Extract criteria for specific conference
  python run_conference_extraction.py --input-dir /path/to/guidelines --output-dir /path/to/output --conference ICSE

  # List available conferences
  python run_conference_extraction.py --list-conferences

  # Use custom configuration
  python run_conference_extraction.py --input-dir /path/to/guidelines --output-dir /path/to/output --config custom_config.json
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Directory containing conference guideline files (.md or .txt)'
    )
    
    parser.add_argument(
        '--output-dir', '-o', 
        type=str,
        required=True,
        help='Directory to save extraction results'
    )
    
    parser.add_argument(
        '--conference', '-c',
        type=str,
        help='Specific conference to process (e.g., ICSE, SIGMOD, CHI). If not specified, processes all conferences.'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model to use (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--list-conferences',
        action='store_true',
        help='List all available conference profiles and exit'
    )
    
    parser.add_argument(
        '--show-profile',
        type=str,
        help='Show detailed profile for a specific conference and exit'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without saving results'
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_conferences:
        list_conferences()
        return
    
    if args.show_profile:
        show_conference_profile(args.show_profile)
        return
    
    # Validate required arguments
    if not args.input_dir or not args.output_dir:
        parser.error("Both --input-dir and --output-dir are required for extraction")
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_configuration(args.config)
    
    # Run the extraction
    run_extraction(args, config)


def list_conferences():
    """List all available conference profiles."""
    print("\n🎯 Available Conference Profiles:")
    print("=" * 50)
    
    profile_manager = ConferenceProfileManager()
    
    # Group by category
    categories = profile_manager.get_all_categories()
    
    for category in sorted(categories):
        conferences = profile_manager.get_conferences_by_category(category)
        print(f"\n📂 {category.replace('_', ' ').title()}:")
        
        for conf in sorted(conferences):
            profile = profile_manager.get_conference_profile(conf)
            print(f"  • {conf} - {profile.get('full_name', 'Unknown')}")
    
    print(f"\n📊 Total conferences: {len(profile_manager.get_all_conferences())}")
    print("📋 Total categories:", len(categories))


def show_conference_profile(conference_name: str):
    """Show detailed profile for a specific conference."""
    profile_manager = ConferenceProfileManager()
    
    try:
        profile = profile_manager.get_conference_profile(conference_name)
        
        print(f"\n🎯 Conference Profile: {conference_name.upper()}")
        print("=" * 60)
        
        print(f"📋 Full Name: {profile.get('full_name', 'N/A')}")
        print(f"📂 Category: {profile.get('category', 'N/A')}")
        print(f"🎨 Evaluation Style: {profile.get('evaluation_style', 'N/A')}")
        print(f"📊 Quality Threshold: {profile.get('quality_threshold', 'N/A')}")
        
        print(f"\n🎯 Focus Areas:")
        for area in profile.get('focus_areas', []):
            print(f"  • {area}")
        
        print(f"\n⚖️ Emphasis Weights:")
        weights = profile.get('emphasis_weights', {})
        for dimension, weight in sorted(weights.items()):
            print(f"  • {dimension}: {weight:.2f}")
        
        print(f"\n🔑 Domain Keywords:")
        keywords = profile.get('domain_keywords', [])
        for i in range(0, len(keywords), 5):
            print(f"  • {', '.join(keywords[i:i+5])}")
        
        # Show similar conferences
        similar = profile_manager.find_similar_conferences(conference_name, top_n=3)
        if similar:
            print(f"\n🔗 Similar Conferences:")
            for conf in similar:
                print(f"  • {conf}")
                
    except Exception as e:
        print(f"❌ Error showing profile for {conference_name}: {e}")


def load_configuration(config_path: str = None) -> Config:
    """Load configuration from file or use defaults."""
    config = Config()
    
    if config_path:
        try:
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
            print(f"✅ Loaded configuration from: {config_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load config from {config_path}: {e}")
            print("🔄 Using default configuration")
    
    return config


def run_extraction(args, config: Config):
    """Run the conference-specific extraction algorithm."""
    start_time = time.time()
    
    print("\n🚀 Starting Conference-Specific Algorithm 1")
    print("=" * 60)
    
    try:
        # Validate input directory
        print(f"📂 Input Directory: {args.input_dir}")
        valid_files = validate_input_directory(args.input_dir)
        print(f"✅ Found {len(valid_files)} valid guideline files")
        
        # Create output directory
        if not args.dry_run:
            output_dir = create_output_directory(args.output_dir, args.conference)
            print(f"📁 Output Directory: {output_dir}")
        else:
            output_dir = args.output_dir
            print(f"🧪 Dry Run Mode - Output would go to: {output_dir}")
        
        # Initialize algorithm
        print(f"🤖 Initializing algorithm with model: {args.model}")
        algorithm = ConferenceSpecificAlgorithm1(
            model_name=args.model,
            config=config
        )
        
        if args.conference:
            print(f"🎯 Target Conference: {args.conference}")
        else:
            print("🌐 Processing all conferences")
        
        # Run extraction
        print("\n🔄 Running extraction...")
        
        if not args.dry_run:
            results = algorithm.run_conference_specific_extraction(
                input_dir=args.input_dir,
                output_dir=output_dir,
                target_conference=args.conference
            )
        else:
            print("🧪 Dry run completed successfully!")
            results = {"dry_run": True}
        
        # Report results
        print("\n✅ Extraction Complete!")
        print("=" * 40)
        
        if not args.dry_run and results:
            for conf_name, result in results.items():
                if conf_name != "dry_run":
                    print(f"\n📊 {conf_name}:")
                    stats = result.get("processing_stats", {})
                    print(f"  • Documents: {stats.get('total_documents', 'N/A')}")
                    print(f"  • Keywords: {stats.get('total_keywords_extracted', 'N/A')}")
                    print(f"  • Score: {stats.get('total_score', 'N/A'):.2f}")
                    
                    saved_files = result.get("saved_files", {})
                    if saved_files:
                        print(f"  • Files saved: {len(saved_files)}")
        
        duration = time.time() - start_time
        print(f"\n⏱️  Total execution time: {duration:.2f} seconds")
        
        if not args.dry_run:
            print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 