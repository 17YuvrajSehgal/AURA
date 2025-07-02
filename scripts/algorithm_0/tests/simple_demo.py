#!/usr/bin/env python3
"""
Simple Demo: Automated Conference-Specific Algorithm 1
Shows the key features without complex formatting.
"""

import os
import sys
import tempfile

from scripts.algorithm_0 import ConferenceSpecificAlgorithm1
from scripts.algorithm_0.conference_profiles import ConferenceProfileGenerator

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def simple_demo():
    """Simple demonstration of the automated algorithm."""

    print("🚀 AUTOMATED CONFERENCE-SPECIFIC ALGORITHM DEMO")
    print("=" * 50)

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    guidelines_dir = os.path.join(script_dir, "..", "..", "data", "conference_guideline_texts", "processed")

    print(f"📁 Guidelines: {guidelines_dir}")

    # Step 1: Generate profiles automatically
    print("\n🔬 Generating automated profiles...")
    generator = ConferenceProfileGenerator()
    profiles = generator.generate_profiles_from_guidelines(guidelines_dir)
    print(f"✅ Generated {len(profiles)} conference profiles")

    # Show sample profiles
    sample_confs = ['ICSE', 'CHI', 'SIGMOD']
    for conf in sample_confs:
        if conf in profiles:
            profile = profiles[conf]
            top_dim = max(profile['emphasis_weights'], key=profile['emphasis_weights'].get)
            weight = profile['emphasis_weights'][top_dim]
            print(f"   📊 {conf}: {profile['category']} | top={top_dim}({weight:.3f})")

    # Step 2: Initialize algorithm
    print("\n⚙️  Initializing algorithm...")
    algorithm = ConferenceSpecificAlgorithm1()
    available = algorithm.profile_manager.list_available_conferences()
    print(f"✅ Algorithm ready with {len(available)} conferences")

    # Step 3: Test extraction with actual guidelines
    print("\n🎯 Testing conference-specific extraction...")

    # Test one conference using the actual guidelines
    test_conf = 'ICSE'

    with tempfile.TemporaryDirectory() as temp_output_dir:
        # Run extraction using real guidelines
        try:
            result = algorithm.run_conference_specific_extraction(
                input_dir=guidelines_dir,
                output_dir=temp_output_dir,
                target_conference=test_conf
            )
            print(f"✅ {test_conf} extraction completed!")
            print(f"📁 Output saved to: {temp_output_dir}")

            # Show profile details
            profile = algorithm.profile_manager.get_conference_profile(test_conf)
            emphasis = profile['emphasis_weights']
            top_3 = sorted(emphasis.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"📊 {test_conf} emphasis: {', '.join([f'{d}({w:.2f})' for d, w in top_3])}")

            # Show extraction stats
            if test_conf in result:
                stats = result[test_conf]['processing_stats']
                print(f"📈 Extraction stats: {stats.get('total_keywords', 'N/A')} keywords extracted")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🎉 Demo completed successfully!")
    print("\n📋 Key Benefits:")
    print("  ✓ Automated profile generation from actual guidelines")
    print("  ✓ Conference-specific evaluation criteria")
    print("  ✓ Eliminates manual bias")
    print("  ✓ Ready for AURA framework integration")


if __name__ == "__main__":
    simple_demo()
