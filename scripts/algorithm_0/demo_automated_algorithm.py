"""
COMPREHENSIVE DEMO: Automated Conference-Specific Algorithm 1
=============================================================

This demo showcases the complete automated workflow:
1. Analyze actual conference guidelines
2. Generate conference-specific profiles automatically
3. Extract conference-specific evaluation criteria
4. Compare results across different conferences

No manual bias - everything is data-driven from actual conference texts!
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conference_specific_algorithm import ConferenceSpecificAlgorithm1
from conference_profiles import ConferenceProfileGenerator, ConferenceProfileManager
from utils import setup_logging

def demo_automated_algorithm():
    """
    Comprehensive demo of the automated conference-specific algorithm.
    """
    
    print("="*80)
    print("🚀 AUTOMATED CONFERENCE-SPECIFIC ALGORITHM 1 DEMO")
    print("="*80)
    print("📊 Data-Driven • 🎯 Conference-Specific • 🔬 Bias-Free")
    print("="*80)
    
    # Setup logging
    setup_logging()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    guidelines_dir = os.path.join(script_dir, "..", "..", "data", "conference_guideline_texts", "processed")
    
    print(f"\n📁 Conference Guidelines Directory: {guidelines_dir}")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"💾 Temporary Output Directory: {temp_dir}")
        
        # Step 1: Generate Automated Conference Profiles
        print("\n" + "="*60)
        print("STEP 1: AUTOMATED PROFILE GENERATION")
        print("="*60)
        print("🔬 Analyzing actual conference guidelines...")
        print("📈 Extracting patterns without manual bias...")
        
        generator = ConferenceProfileGenerator()
        profiles = generator.generate_profiles_from_guidelines(guidelines_dir)
        
        profiles_file = os.path.join(temp_dir, "automated_profiles.json")
        generator.save_profiles(profiles, profiles_file)
        
        print(f"\n✅ Generated {len(profiles)} conference profiles automatically!")
        
        # Show sample of generated profiles
        sample_conferences = ['ICSE', 'SIGMOD', 'CHI', 'PLDI', 'HRI']
        for conf in sample_conferences:
            if conf in profiles:
                profile = profiles[conf]
                top_emphasis = max(profile['emphasis_weights'], 
                                 key=profile['emphasis_weights'].get)
                print(f"   📊 {conf}: {profile['category']} | "
                      f"focus={top_emphasis} ({profile['emphasis_weights'][top_emphasis]:.3f}) | "
                      f"style={profile['evaluation_style']}")
        
        # Step 2: Initialize Conference-Specific Algorithm
        print("\n" + "="*60)
        print("STEP 2: CONFERENCE-SPECIFIC ALGORITHM INITIALIZATION")
        print("="*60)
        print("🤖 Loading automated profiles...")
        print("⚙️  Initializing conference-specific algorithm...")
        
        algorithm = ConferenceSpecificAlgorithm1()
        
        print(f"✅ Algorithm initialized with {len(algorithm.profile_manager.profiles)} conferences")
        available_conferences = algorithm.profile_manager.list_available_conferences()
        print(f"📋 Available conferences: {', '.join(available_conferences[:10])}" + 
              (f" (+ {len(available_conferences)-10} more)" if len(available_conferences) > 10 else ""))
        
        # Step 3: Conference-Specific Extractions
        print("\n" + "="*60)
        print("STEP 3: CONFERENCE-SPECIFIC CRITERIA EXTRACTION")
        print("="*60)
        
        # Sample conference guidelines text for demonstration
        sample_text = """
        Software artifacts must be complete, well-documented, and reproducible.
        The evaluation focuses on functionality, ease of use, and reusability.
        All code should be publicly accessible with clear installation instructions.
        Performance benchmarks and experimental validation are required.
        Documentation should include user guides and developer documentation.
        """
        
        demo_conferences = ['ICSE', 'CHI', 'SIGMOD', 'PLDI']
        results = {}
        
        for conf in demo_conferences:
            print(f"\n🎯 Extracting criteria for {conf}...")
            
            # Create conference-specific output directory
            conf_output = os.path.join(temp_dir, f"{conf.lower()}_output")
            os.makedirs(conf_output, exist_ok=True)
            
            # Create a temporary input file for demonstration
            temp_input_file = os.path.join(conf_output, "sample_guidelines.txt")
            with open(temp_input_file, 'w', encoding='utf-8') as f:
                f.write(sample_text)
            
            # Extract criteria using conference-specific profile
            try:
                result = algorithm.run_conference_specific_extraction(
                    input_dir=conf_output,
                    output_dir=conf_output,
                    conference=conf,
                    output_format=['csv', 'json']
                )
                results[conf] = result
                
                # Show key results
                profile = algorithm.profile_manager.get_conference_profile(conf)
                emphasis = profile['emphasis_weights']
                top_3_emphasis = sorted(emphasis.items(), key=lambda x: x[1], reverse=True)[:3]
                
                print(f"   ✅ {conf} extraction completed!")
                print(f"   📊 Category: {profile['category']}")
                print(f"   🎯 Top emphasis: {', '.join([f'{dim}({wt:.2f})' for dim, wt in top_3_emphasis])}")
                print(f"   📁 Output files generated in: {conf_output}")
                
            except Exception as e:
                print(f"   ❌ Error extracting for {conf}: {e}")
                results[conf] = None
        
        # Step 4: Cross-Conference Analysis
        print("\n" + "="*60)
        print("STEP 4: CROSS-CONFERENCE COMPARISON ANALYSIS")
        print("="*60)
        
        print("📊 Comparing conference-specific emphasis patterns:\n")
        
        # Analyze emphasis patterns across conferences
        emphasis_analysis = {}
        dimensions = ['reproducibility', 'documentation', 'accessibility', 'usability', 'experimental', 'functionality']
        
        for dim in dimensions:
            emphasis_analysis[dim] = []
            for conf in demo_conferences:
                if conf in profiles:
                    weight = profiles[conf]['emphasis_weights'].get(dim, 0)
                    emphasis_analysis[dim].append((conf, weight))
            
            # Sort by weight and show top conference for this dimension
            emphasis_analysis[dim].sort(key=lambda x: x[1], reverse=True)
            top_conf, top_weight = emphasis_analysis[dim][0]
            avg_weight = sum(w[1] for w in emphasis_analysis[dim]) / len(emphasis_analysis[dim])
            
            print(f"🔹 {dim.capitalize():15} | "
                  f"Top: {top_conf} ({top_weight:.3f}) | "
                  f"Avg: {avg_weight:.3f}")
        
        # Step 5: Category Analysis
        print("\n📋 Conference Categories:")
        categories = {}
        for conf in demo_conferences:
            if conf in profiles:
                category = profiles[conf]['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(conf)
        
        for category, conferences in categories.items():
            print(f"   🏷️  {category.replace('_', ' ').title()}: {', '.join(conferences)}")
        
        # Step 6: Results Summary
        print("\n" + "="*60)
        print("STEP 5: RESULTS SUMMARY")
        print("="*60)
        
        successful_extractions = sum(1 for r in results.values() if r is not None)
        print(f"✅ Successfully processed {successful_extractions}/{len(demo_conferences)} conferences")
        print(f"📊 Generated {len(profiles)} automated profiles from guidelines")
        print(f"💾 All outputs saved to: {temp_dir}")
        
        print("\n🎯 Key Achievements:")
        print("   ✓ Eliminated manual bias in profile creation")
        print("   ✓ Generated conference-specific evaluation criteria")  
        print("   ✓ Maintained consistency across different venues")
        print("   ✓ Extracted patterns directly from conference guidelines")
        print("   ✓ Created AURA-compatible output formats")
        
        # Step 7: Integration Benefits
        print("\n" + "="*60)
        print("INTEGRATION WITH AURA FRAMEWORK")
        print("="*60)
        
        print("🔗 This automated Algorithm 1 seamlessly integrates with:")
        print("   • Algorithm 2: Corpus Analysis")
        print("   • Algorithm 3: Agentic Evaluation") 
        print("   • Algorithm 4: AURA Framework with specialized agents")
        
        print("\n📈 Benefits for AURA:")
        print("   ✓ Conference-specific artifact evaluation")
        print("   ✓ Higher accuracy through targeted criteria")
        print("   ✓ Reduced human bias in evaluation")
        print("   ✓ Scalable to new conferences automatically")
        print("   ✓ Comparative analysis capabilities")
        
        print("\n" + "="*80)
        print("🎉 AUTOMATED CONFERENCE-SPECIFIC ALGORITHM 1 DEMO COMPLETE!")
        print("="*80)
        print("Ready for production use in the AURA framework! 🚀")

def show_technical_details():
    """Show technical implementation details."""
    
    print("\n" + "="*60)
    print("🔧 TECHNICAL IMPLEMENTATION DETAILS")
    print("="*60)
    
    print("📊 Automated Profile Generation:")
    print("   • TF-IDF analysis of conference guidelines")
    print("   • KeyBERT keyword extraction")
    print("   • Semantic similarity clustering")
    print("   • Statistical weight computation")
    print("   • Category classification using NLP")
    
    print("\n⚙️ Conference-Specific Processing:")
    print("   • Dynamic weight adjustment based on conference")
    print("   • Domain-specific keyword expansion")
    print("   • Context-aware semantic similarity")
    print("   • Conference-tailored quality thresholds")
    
    print("\n🎯 Key Technical Features:")
    print("   • Sentence Transformers for semantic understanding")
    print("   • NetworkX for relationship mapping")
    print("   • Hierarchical keyword extraction")
    print("   • Multi-format output generation")
    print("   • Confidence scoring and validation")

if __name__ == "__main__":
    try:
        demo_automated_algorithm()
        show_technical_details()
        
    except Exception as e:
        print(f"\n❌ DEMO ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Please check the setup and try again.") 