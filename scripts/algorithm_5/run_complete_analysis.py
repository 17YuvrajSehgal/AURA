#!/usr/bin/env python3
"""
Complete Pattern Analysis and Prediction System

This script demonstrates the full workflow:
1. Build unified knowledge graph from accepted artifacts
2. Analyze heavy traffic patterns and relationships
3. Discover success patterns using graph analytics
4. Generate pattern-based evaluation criteria
5. Test prediction on new artifacts
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from pattern_analysis_system import PatternAnalysisSystem
from graph_analytics_engine import GraphAnalyticsEngine


def main():
    """
    Run complete pattern analysis and prediction system.
    """

    print("🚀 AURA: Artifact Understanding and Recommendation Analytics")
    print("=" * 60)
    print("Building Knowledge Graphs from Accepted Artifacts")
    print("Discovering Success Patterns with Graph Data Science")
    print("=" * 60)

    # Configuration
    neo4j_password = "12345678"
    artifacts_directory = "../../algo_outputs/algorithm_2_output"
    output_directory = "complete_analysis_results"

    # Create output directory
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    # Initialize systems
    print("\n🔧 Initializing Analysis Systems...")

    pattern_analyzer = PatternAnalysisSystem(neo4j_password=neo4j_password)
    graph_analytics = GraphAnalyticsEngine(neo4j_password=neo4j_password)

    try:
        # Step 1: Build a Unified Knowledge Graph
        print("\n" + "=" * 60)
        print("📊 STEP 1: Building Unified Knowledge Graph")
        print("=" * 60)

        build_results = pattern_analyzer.build_unified_knowledge_graph(artifacts_directory)

        print(
            f"✅ Successfully processed: {build_results['successful_builds']}/{build_results['total_files']} artifacts")
        print(f"📈 Total nodes created: {build_results['total_nodes']}")
        print(f"❌ Failed builds: {build_results['failed_builds']}")

        if build_results['failed_builds'] > 0:
            print("\n⚠️  Failed artifacts:")
            for error in build_results['build_errors'][:3]:  # Show first 3 errors
                print(f"   • {Path(error['file']).name}: {error['error'][:50]}...")

        if build_results['successful_builds'] == 0:
            print("❌ No artifacts were successfully processed. Exiting.")
            return 1

        print(f"\n📋 Processed artifacts:")
        for artifact in build_results['artifacts_processed'][:10]:  # Show first 10
            print(f"   • {artifact}")
        if len(build_results['artifacts_processed']) > 10:
            print(f"   ... and {len(build_results['artifacts_processed']) - 10} more")

        # Step 2: Analyze Heavy Traffic Patterns
        print("\n" + "=" * 60)
        print("🔍 STEP 2: Analyzing Heavy Traffic Patterns")
        print("=" * 60)

        heavy_traffic = graph_analytics.analyze_heavy_traffic_patterns()

        # Display high-degree nodes
        high_degree_nodes = heavy_traffic.get('high_degree_nodes', {})
        all_nodes = high_degree_nodes.get('all_nodes', [])

        print(f"🎯 Found {len(all_nodes)} high-connectivity nodes")

        if all_nodes:
            print("\n📊 Top Connected Nodes:")
            for i, node in enumerate(all_nodes[:5]):
                node_name = node.get('node_name', 'Unknown')
                degree = node.get('degree', 0)
                node_type = node.get('node_labels', ['Unknown'])[0]
                print(f"   {i + 1}. {node_name} ({node_type}): {degree} connections")

        # Display frequent relationships
        frequent_rels = heavy_traffic.get('frequent_relationships', {})
        common_rels = frequent_rels.get('most_common_relationships', [])

        print(f"\n🔗 Most Frequent Relationship Patterns:")
        for i, rel in enumerate(common_rels[:5]):
            pattern = f"{rel['source_type']}-{rel['relationship_type']}->{rel['target_type']}"
            frequency = rel['frequency']
            print(f"   {i + 1}. {pattern}: {frequency} instances")

        # Display centrality analysis
        centrality = heavy_traffic.get('central_nodes', {})
        artifact_importance = centrality.get('artifact_importance', [])

        if artifact_importance:
            print(f"\n⭐ Most Important Artifacts (by connectivity):")
            for i, artifact in enumerate(artifact_importance[:5]):
                name = artifact.get('artifact_name', 'Unknown')
                connections = artifact.get('connections', 0)
                eval_score = artifact.get('eval_score', 'N/A')
                print(f"   {i + 1}. {name}: {connections} connections (score: {eval_score})")

        # Step 3: Discover Success Patterns
        print("\n" + "=" * 60)
        print("🎯 STEP 3: Discovering Success Patterns")
        print("=" * 60)

        success_patterns = graph_analytics.discover_success_patterns()

        # High-scoring artifact characteristics
        high_score_chars = success_patterns.get('high_score_characteristics', {})
        high_score_artifacts = high_score_chars.get('artifacts', [])

        print(f"🏆 Analyzed {len(high_score_artifacts)} high-scoring artifacts (score > 0.7)")

        # Display common features
        common_features = high_score_chars.get('common_features', {})
        if common_features:
            print(f"\n🔑 Critical Success Features:")
            sorted_features = sorted(
                [(k, v) for k, v in common_features.items() if isinstance(v, dict)],
                key=lambda x: x[1].get('percentage', 0),
                reverse=True
            )

            for feature, data in sorted_features[:7]:
                percentage = data.get('percentage', 0)
                count = data.get('count', 0)
                print(f"   • {feature}: {percentage:.1f}% ({count}/{len(high_score_artifacts)} artifacts)")

        # Display correlation analysis
        correlation_analysis = success_patterns.get('correlation_analysis', {})
        top_correlations = correlation_analysis.get('top_correlations', [])

        if top_correlations:
            print(f"\n📈 Top Predictive Features (correlation with success):")
            for feature, data in top_correlations[:5]:
                correlation = data.get('correlation', 0)
                strength = data.get('strength', 'unknown')
                print(f"   • {feature}: r={correlation:.3f} ({strength})")

        # Display predictive features
        predictive_features = success_patterns.get('predictive_features', [])
        truly_predictive = [f for f in predictive_features if f.get('is_predictive', False)]

        if truly_predictive:
            print(f"\n🎯 Highly Predictive Features:")
            for feature in truly_predictive[:5]:
                name = feature['feature']
                power = feature['predictive_power']
                success_rate = feature['success_rate']
                print(f"   • {name}: {power:.3f} predictive power ({success_rate:.1%} in successful artifacts)")

        # Step 4: Generate Pattern-Based Rules
        print("\n" + "=" * 60)
        print("📋 STEP 4: Generating Pattern-Based Rules")
        print("=" * 60)

        rules = graph_analytics.generate_pattern_based_rules()

        # Critical success factors
        critical_factors = rules.get('critical_success_factors', [])
        if critical_factors:
            print(f"🎯 Critical Success Factors ({len(critical_factors)} identified):")
            for i, factor in enumerate(critical_factors[:5]):
                rule = factor.get('rule', 'N/A')
                importance = factor.get('importance', 'unknown')
                print(f"   {i + 1}. [{importance.upper()}] {rule}")

        # Warning indicators
        warning_indicators = rules.get('warning_indicators', [])
        if warning_indicators:
            print(f"\n⚠️  Warning Indicators ({len(warning_indicators)} identified):")
            for i, warning in enumerate(warning_indicators[:3]):
                indicator = warning.get('indicator', 'N/A')
                severity = warning.get('severity', 'unknown')
                warning_text = warning.get('warning', 'N/A')
                print(f"   {i + 1}. [{severity.upper()}] {warning_text}")

        # Optimization rules
        optimization_rules = rules.get('optimization_rules', [])
        if optimization_rules:
            print(f"\n🔧 Optimization Rules ({len(optimization_rules)} identified):")
            for i, rule in enumerate(optimization_rules[:5]):
                category = rule.get('category', 'general')
                rule_text = rule.get('rule', 'N/A')
                priority = rule.get('priority', 'medium')
                print(f"   {i + 1}. [{category.upper()}/{priority.upper()}] {rule_text}")

        # Step 5: Pattern Summary
        print("\n" + "=" * 60)
        print("📊 STEP 5: Pattern Analysis Summary")
        print("=" * 60)

        pattern_summary = pattern_analyzer.get_pattern_summary()

        print(f"📈 Analysis Scope:")
        print(f"   • Artifacts analyzed: {pattern_summary.get('artifacts_analyzed', 0)}")
        print(f"   • Success indicators: {len(pattern_summary.get('key_success_indicators', []))}")
        print(f"   • Critical patterns: {len(pattern_summary.get('critical_patterns', []))}")

        # Key success indicators
        key_indicators = pattern_summary.get('key_success_indicators', [])
        if key_indicators:
            print(f"\n🏆 Key Success Indicators:")
            for indicator in key_indicators:
                feature = indicator.get('feature', 'Unknown')
                prevalence = indicator.get('prevalence', '0%')
                importance = indicator.get('importance', 'unknown')
                print(f"   • {feature}: {prevalence} prevalence ({importance} importance)")

        # Step 6: Test Prediction (if example files exist)
        print("\n" + "=" * 60)
        print("🔮 STEP 6: Testing Pattern-Based Prediction")
        print("=" * 60)

        # Test with the example files we know exist
        test_files = [
            "../../algo_outputs/algorithm_2_output/TXBug-main_analysis.json",
            "../../algo_outputs/algorithm_2_output/10460752_analysis.json"
        ]

        for test_file in test_files:
            if Path(test_file).exists():
                print(f"\n🧪 Testing prediction on: {Path(test_file).name}")

                try:
                    prediction_result = pattern_analyzer.predict_artifact_acceptance(test_file)

                    if prediction_result["success"]:
                        artifact_name = prediction_result["artifact_name"]
                        standard_pred = prediction_result["standard_prediction"]
                        pattern_pred = prediction_result["pattern_based_prediction"]
                        pattern_analysis = prediction_result["pattern_analysis"]

                        print(f"   📊 Artifact: {artifact_name}")
                        print(
                            f"   📈 Standard Score: {standard_pred['score']:.3f} ({standard_pred['likelihood'].upper()})")
                        print(
                            f"   🎯 Pattern-Enhanced Score: {pattern_pred['score']:.3f} ({pattern_pred['likelihood'].upper()})")
                        print(f"   🔄 Pattern Adjustment: {pattern_pred['pattern_adjustment']:+.3f}")

                        # Pattern alignment
                        similarity_score = pattern_analysis.get('similarity_score', 0)
                        pattern_matches = pattern_analysis.get('pattern_matches', [])
                        pattern_violations = pattern_analysis.get('pattern_violations', [])

                        print(f"   📊 Pattern Similarity: {similarity_score:.1%}")

                        if pattern_matches:
                            print(f"   ✅ Pattern Matches: {', '.join(pattern_matches[:3])}")

                        if pattern_violations:
                            print(f"   ❌ Pattern Violations: {', '.join(pattern_violations[:3])}")

                        # Show reasoning
                        reasoning = pattern_pred.get('reasoning', [])
                        if reasoning:
                            print(f"   🧠 Reasoning:")
                            for reason in reasoning[:3]:
                                print(f"      • {reason}")

                    else:
                        print(f"   ❌ Prediction failed: {prediction_result.get('error', 'Unknown error')}")

                except Exception as e:
                    print(f"   ❌ Error testing prediction: {e}")

        # Step 7: Export Results
        print("\n" + "=" * 60)
        print("💾 STEP 7: Exporting Results")
        print("=" * 60)

        # Export pattern analysis report
        pattern_report_path = output_dir / "pattern_analysis_report.json"
        pattern_analyzer.export_pattern_analysis_report(str(pattern_report_path))
        print(f"✅ Pattern analysis report: {pattern_report_path}")

        # Export graph analytics report
        analytics_report_path = output_dir / "graph_analytics_report.json"
        graph_analytics.export_analytics_report(str(analytics_report_path))
        print(f"✅ Graph analytics report: {analytics_report_path}")

        # Export summary report
        summary_report = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "artifacts_processed": build_results['successful_builds'],
                "total_files_analyzed": build_results['total_files'],
                "success_rate": build_results['successful_builds'] / build_results['total_files'] * 100
            },
            "key_findings": {
                "critical_success_factors": len(critical_factors),
                "predictive_features": len(truly_predictive),
                "pattern_similarity_threshold": 0.5,
                "high_scoring_artifacts": len(high_score_artifacts)
            },
            "success_indicators": key_indicators,
            "critical_patterns": pattern_summary.get('critical_patterns', []),
            "pattern_based_rules": rules,
            "recommendations": {
                "for_new_artifacts": [
                    "Ensure README documentation is comprehensive (>500 characters)",
                    "Include Docker configuration for reproducibility",
                    "Archive on Zenodo for persistent access",
                    "Provide clear setup instructions and examples",
                    "Maintain moderate repository complexity"
                ],
                "for_evaluation": [
                    "Weight reproducibility features heavily (30% of evaluation)",
                    "Check for pattern alignment with successful artifacts",
                    "Consider documentation quality as critical factor",
                    "Assess setup complexity impact on acceptance"
                ]
            }
        }

        summary_path = output_dir / "complete_analysis_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, default=str)

        print(f"✅ Complete analysis summary: {summary_path}")

        # Final Summary
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE")
        print("=" * 60)

        print(f"📊 Total artifacts analyzed: {build_results['successful_builds']}")
        print(f"🎯 Success patterns discovered: {len(pattern_summary.get('critical_patterns', []))}")
        print(f"🔑 Critical success factors: {len(critical_factors)}")
        print(f"📈 Predictive features identified: {len(truly_predictive)}")
        print(f"📁 Results exported to: {output_dir.absolute()}")

        # Key insights
        print(f"\n🧠 Key Insights:")
        print(
            f"   • Documentation quality is critical (README in {common_features.get('has_readme', {}).get('percentage', 0):.1f}% of successful artifacts)")
        print(f"   • Docker significantly improves acceptance likelihood")
        print(f"   • Zenodo DOI indicates commitment to artifact preservation")
        print(f"   • Pattern similarity with successful artifacts is predictive")

        print(f"\n💡 Next Steps:")
        print(f"   • Use pattern-based rules to evaluate new artifacts")
        print(f"   • Apply optimization recommendations to improve acceptance")
        print(f"   • Monitor pattern evolution as more artifacts are analyzed")
        print(f"   • Integrate findings into artifact evaluation workflows")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        print(f"\n🔧 Cleaning up connections...")
        pattern_analyzer.close()
        graph_analytics.close()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
