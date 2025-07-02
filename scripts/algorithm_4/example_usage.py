#!/usr/bin/env python3
"""
Enhanced Conference-Specific AURA Framework Example Usage
=========================================================

This example demonstrates the enhanced AURA framework with conference-specific
evaluation capabilities. It uses the ICSE profile which emphasizes accessibility
and documentation for software engineering artifacts.

Features demonstrated:
- Conference-specific profile loading
- Targeted accessibility evaluation
- Conference-aware scoring and recommendations
- Enhanced knowledge graph utilization
"""

import os
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_framework import AURAFramework


def create_icse_profile() -> Dict[str, Any]:
    """Create ICSE conference profile with high accessibility emphasis."""
    return {
        "category": "software_engineering",
        "domain_keywords": [
            "artifact documented",
            "comprehensiveness documentation",
            "documentation factor",
            "archival platforms",
            "repository factor",
            "repurposing requires",
            "retrieval doi",
            "provenance setup",
            "software requirements",
            "term availability",
            "involves quality",
            "figshare ensures",
            "including provision",
            "clarifications review",
            "github executable",
            "10 usage",
            "factor evaluates",
            "describing distribution",
            "use docker",
            "zenodo"
        ],
        "emphasis_weights": {
            "accessibility": 0.3620689655172414,
            "usability": 0.25862068965517243,
            "documentation": 0.20689655172413793,
            "functionality": 0.10344827586206896,
            "reproducibility": 0.06896551724137931,
            "experimental": 0.0
        },
        "quality_threshold": 0.8,
        "evaluation_style": "strict",
        "analysis_metadata": {
            "text_length": 2256,
            "guidelines_count": 12,
            "generated_timestamp": "2025-07-01T16:38:04.738282"
        }
    }


def run_conference_specific_evaluation():
    """Run comprehensive conference-specific evaluation example."""

    print("=" * 80)
    print("🎯 CONFERENCE-SPECIFIC AURA FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print("Demonstrating targeted artifact evaluation using conference profiles")
    print("=" * 80)

    # Get the artifact JSON path (using the ML classifier example)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_path = os.path.join(script_dir, "..", "..", "algo_outputs", "algorithm_2_output",
                                 "AI_Code_Detection_Education_analysis.json")

    if not os.path.exists(artifact_path):
        print(f"❌ Artifact JSON not found at {artifact_path}")
        print("Please run Algorithm 2 first to generate the artifact analysis.")
        return

    print(f"📁 Using artifact: {artifact_path}")

    # Test different conference profiles
    conferences = {
        "ICSE": create_icse_profile(),
    }

    results = {}

    for conference_name, profile in conferences.items():
        print(f"\n{'=' * 60}")
        print(f"🔬 EVALUATING FOR {conference_name} CONFERENCE")
        print(f"{'=' * 60}")
        print(f"Category: {profile['category']}")
        print(f"Quality Threshold: {profile['quality_threshold']}")
        print(f"Top Emphasis: {max(profile['emphasis_weights'], key=profile['emphasis_weights'].get)}")
        print(f"Evaluation Style: {profile['evaluation_style']}")

        try:
            # Initialize AURA framework with conference profile
            framework = AURAFramework(
                artifact_json_path=artifact_path,
                neo4j_uri="bolt://localhost:7687",
                conference_profile=profile,
                use_llm=False  # Disable LLM for faster demo
            )

            # Run evaluation
            def progress_callback(message):
                print(f"  📊 {message}")

            result = framework.evaluate_artifact(progress_callback=progress_callback)
            results[conference_name] = result

            # Print summary results
            print(f"\n📈 {conference_name} EVALUATION RESULTS:")
            print(f"   Score: {result.total_weighted_score:.3f}")
            print(f"   Threshold: {result.conference_info['quality_threshold']:.3f}")
            print(f"   Status: {'✅ ACCEPTED' if result.acceptance_prediction else '❌ REJECTED'}")

            # Show top-weighted dimensions
            sorted_criteria = sorted(result.criteria_scores, key=lambda x: x.conference_weight, reverse=True)
            print(f"   Top 3 Weighted Dimensions:")
            for i, criterion in enumerate(sorted_criteria[:3], 1):
                print(
                    f"     {i}. {criterion.dimension}: {criterion.llm_evaluated_score:.3f} (weight: {criterion.conference_weight:.3f})")

            # Save detailed results
            output_dir = os.path.join(script_dir, "..", "..", "algo_outputs", "algorithm_4_output")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{conference_name.lower()}_evaluation_results.json")
            framework.save_results(result, output_file)
            print(f"   📁 Detailed results saved to: {output_file}")

            framework.close()

        except Exception as e:
            print(f"❌ Error evaluating for {conference_name}: {e}")
            results[conference_name] = None

    # Comparative analysis
    print(f"\n{'=' * 80}")
    print("🔍 COMPARATIVE ANALYSIS ACROSS CONFERENCES")
    print(f"{'=' * 80}")

    if all(results.values()):
        # Compare scores
        print("\n📊 Score Comparison:")
        for conf, result in results.items():
            if result:
                status = "✅ PASS" if result.acceptance_prediction else "❌ FAIL"
                print(f"   {conf:8}: {result.total_weighted_score:.3f} {status}")

        # Analyze dimension preferences
        print("\n🎯 Dimension Emphasis Analysis:")
        dimensions = ["accessibility", "usability", "documentation", "functionality", "reproducibility", "experimental"]

        for dim in dimensions:
            print(f"\n   {dim.upper()}:")
            for conf, result in results.items():
                if result:
                    criterion = next((c for c in result.criteria_scores if c.dimension == dim), None)
                    if criterion:
                        emphasis = criterion.conference_weight
                        score = criterion.llm_evaluated_score
                        print(f"     {conf:8}: weight={emphasis:.3f}, score={score:.3f}")

        # Conference-specific recommendations
        print("\n💡 KEY INSIGHTS:")
        for conf, result in results.items():
            if result:
                print(f"\n   {conf}:")
                print(
                    f"     • Threshold: {result.conference_info['quality_threshold']:.3f} ({result.conference_info['evaluation_style']} evaluation)")
                print(f"     • Structure Quality: {result.conference_info['structure_quality']['quality_score']:.3f}")
                if result.recommendations:
                    print(f"     • Top Recommendation: {result.recommendations[0]}")

    print(f"\n{'=' * 80}")
    print("🎉 CONFERENCE-SPECIFIC EVALUATION COMPLETE!")
    print(f"{'=' * 80}")
    print("Key Benefits Demonstrated:")
    print("✓ Conference-specific weight adjustment")
    print("✓ Targeted evaluation criteria")
    print("✓ Venue-appropriate thresholds")
    print("✓ Contextual recommendations")
    print("✓ Enhanced knowledge graph analysis")


def demonstrate_accessibility_focus():
    """Demonstrate detailed accessibility evaluation for ICSE."""

    print("\n" + "=" * 60)
    print("🔐 DETAILED ACCESSIBILITY EVALUATION FOR ICSE")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_path = os.path.join(script_dir, "..", "..", "algo_outputs", "algorithm_2_output",
                                 "AI_Code_Detection_Education_analysis.json")

    if not os.path.exists(artifact_path):
        print(f"❌ Artifact JSON not found at {artifact_path}")
        return

    icse_profile = create_icse_profile()

    print(f"Conference: ICSE (Software Engineering)")
    print(f"Accessibility Weight: {icse_profile['emphasis_weights']['accessibility']:.3f} (36.2%)")
    print(f"Quality Threshold: {icse_profile['quality_threshold']} (strict)")

    try:
        framework = AURAFramework(
            artifact_json_path=artifact_path,
            conference_profile=icse_profile,
            use_llm=False
        )

        # Get accessibility-specific analysis
        kg_agent = framework.kg_agent

        # Check accessibility indicators
        accessibility_indicators = kg_agent.check_accessibility_indicators()
        print(f"\n📋 Accessibility Indicators:")
        for indicator, status in accessibility_indicators.items():
            icon = "✅" if status else "❌"
            print(f"   {icon} {indicator.replace('_', ' ').title()}: {status}")

        # Check conference-specific patterns
        patterns = kg_agent.check_conference_specific_patterns("software_engineering")
        print(f"\n🔧 Software Engineering Patterns:")
        for pattern_type, data in patterns.items():
            icon = "✅" if data['found'] else "❌"
            print(f"   {icon} {pattern_type.replace('_', ' ').title()}: {data['count']} files")

        # Run full evaluation focusing on accessibility
        result = framework.evaluate_artifact()

        # Find accessibility criterion
        accessibility_criterion = next((c for c in result.criteria_scores if c.dimension == "accessibility"), None)

        if accessibility_criterion:
            print(f"\n🎯 ACCESSIBILITY EVALUATION DETAILS:")
            print(f"   Score: {accessibility_criterion.llm_evaluated_score:.3f}")
            print(f"   Conference Weight: {accessibility_criterion.conference_weight:.3f}")
            print(f"   Justification: {accessibility_criterion.justification}")
            if accessibility_criterion.evidence:
                print(f"   Evidence:")
                for evidence in accessibility_criterion.evidence:
                    print(f"     • {evidence}")

        # Overall conference assessment
        print(f"\n📊 ICSE CONFERENCE ASSESSMENT:")
        print(f"   Overall Score: {result.total_weighted_score:.3f}")
        print(f"   Meets ICSE Standards: {'✅ YES' if result.acceptance_prediction else '❌ NO'}")
        print(f"   Repository Quality: {result.conference_info['structure_quality']['quality_score']:.3f}")

        framework.close()

    except Exception as e:
        print(f"❌ Error in accessibility evaluation: {e}")


if __name__ == "__main__":
    try:
        # Run comprehensive conference-specific evaluation
        run_conference_specific_evaluation()

        # Demonstrate detailed accessibility focus
        demonstrate_accessibility_focus()

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ DEMO ERROR: {e}")
        import traceback

        traceback.print_exc()
        print("\n🔧 Please ensure Neo4j is running and try again.")
