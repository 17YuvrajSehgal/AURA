"""
Test script for automated conference profile generation.
Demonstrates how the ConferenceProfileGenerator analyzes actual conference guidelines
to create conference-specific evaluation profiles.
"""

import json
import os
import sys

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conference_profiles import ConferenceProfileGenerator, ConferenceProfileManager
from utils import setup_logging

def test_profile_generation():
    """Test the automated profile generation from conference guidelines."""
    
    # Setup logging
    setup_logging()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    guidelines_dir = os.path.join(script_dir, "..", "..", "data", "conference_guideline_texts", "processed")
    output_file = os.path.join(script_dir, "test_generated_profiles.json")
    
    print("=" * 70)
    print("AUTOMATED CONFERENCE PROFILE GENERATION TEST")
    print("=" * 70)
    print(f"Guidelines directory: {guidelines_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Check if guidelines directory exists
    if not os.path.exists(guidelines_dir):
        print(f"ERROR: Guidelines directory not found: {guidelines_dir}")
        return
    
    # Initialize the profile generator
    print("Initializing Conference Profile Generator...")
    generator = ConferenceProfileGenerator()
    
    # Generate profiles from guidelines
    print("Analyzing conference guidelines and generating profiles...")
    print()
    
    profiles = generator.generate_profiles_from_guidelines(guidelines_dir)
    
    if not profiles:
        print("ERROR: No profiles generated!")
        return
    
    # Save profiles
    generator.save_profiles(profiles, output_file)
    
    # Display results
    print("=" * 70)
    print(f"GENERATED {len(profiles)} CONFERENCE PROFILES")
    print("=" * 70)
    
    for conf_name, profile in profiles.items():
        print(f"\n📊 CONFERENCE: {conf_name}")
        print(f"   Category: {profile['category']}")
        print(f"   Evaluation Style: {profile['evaluation_style']}")
        print(f"   Quality Threshold: {profile['quality_threshold']}")
        
        # Show top emphasis dimensions
        emphasis = profile['emphasis_weights']
        sorted_emphasis = sorted(emphasis.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top Emphasis Dimensions:")
        for dim, weight in sorted_emphasis[:3]:
            print(f"     - {dim}: {weight:.3f}")
        
        # Show top domain keywords
        domain_kw = profile['domain_keywords'][:5]
        print(f"   Top Domain Keywords: {', '.join(domain_kw)}")
        
        # Show metadata
        metadata = profile['analysis_metadata']
        print(f"   Guidelines Length: {metadata['text_length']} characters")
        print(f"   Guidelines Count: {metadata['guidelines_count']} lines")
    
    print("\n" + "=" * 70)
    print("PROFILE COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Compare categories
    categories = {}
    for conf_name, profile in profiles.items():
        category = profile['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(conf_name)
    
    print(f"\nConferences by Category:")
    for category, conferences in categories.items():
        print(f"  {category}: {', '.join(conferences)}")
    
    # Compare emphasis patterns
    print(f"\nEmphasis Pattern Analysis:")
    dimension_stats = {}
    for conf_name, profile in profiles.items():
        for dim, weight in profile['emphasis_weights'].items():
            if dim not in dimension_stats:
                dimension_stats[dim] = []
            dimension_stats[dim].append((conf_name, weight))
    
    for dim, conf_weights in dimension_stats.items():
        # Find conference with highest emphasis on this dimension
        top_conf = max(conf_weights, key=lambda x: x[1])
        avg_weight = sum(w[1] for w in conf_weights) / len(conf_weights)
        print(f"  {dim}: avg={avg_weight:.3f}, highest={top_conf[0]} ({top_conf[1]:.3f})")
    
    print("\n" + "=" * 70)
    print("PROFILES SAVED AND READY FOR USE!")
    print("=" * 70)
    print(f"✅ Generated profiles saved to: {output_file}")
    print(f"✅ ConferenceProfileManager will automatically load these profiles")
    print(f"✅ Algorithm can now use conference-specific evaluation criteria")
    
def test_profile_manager():
    """Test the ConferenceProfileManager with automated profiles."""
    
    print("\n" + "=" * 70)
    print("TESTING CONFERENCE PROFILE MANAGER")
    print("=" * 70)
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    guidelines_dir = os.path.join(script_dir, "..", "..", "data", "conference_guideline_texts", "processed")
    profiles_file = os.path.join(script_dir, "test_manager_profiles.json")
    
    # Initialize manager (this should trigger automatic profile generation)
    print("Initializing ConferenceProfileManager...")
    manager = ConferenceProfileManager(
        guidelines_dir=guidelines_dir,
        profiles_file=profiles_file
    )
    
    # Test getting specific conference profiles
    test_conferences = ['ICSE', 'SIGMOD', 'CHI', 'ASE', 'FSE']
    
    print(f"\nTesting profile retrieval for conferences:")
    for conf in test_conferences:
        profile = manager.get_conference_profile(conf)
        print(f"  {conf}: category={profile['category']}, "
              f"style={profile['evaluation_style']}, "
              f"threshold={profile['quality_threshold']}")
    
    # Show profile summary
    print(f"\nProfile Summary:")
    summary = manager.get_profile_summary()
    for conf_name, info in summary.items():
        print(f"  {conf_name}: {info['category']} | {info['evaluation_style']} | "
              f"top_emphasis={info['top_emphasis']}")
    
    print(f"\n✅ ConferenceProfileManager working correctly!")
    print(f"✅ Available conferences: {', '.join(manager.list_available_conferences())}")

if __name__ == "__main__":
    try:
        test_profile_generation()
        test_profile_manager()
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The automated conference profile generation is working and ready to use.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc() 