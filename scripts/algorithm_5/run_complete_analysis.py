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

        # Step 2: Analyze README-Centric Patterns
        print("\n" + "=" * 60)
        print("🔍 STEP 2: Analyzing README-Centric Patterns")
        print("=" * 60)

        readme_patterns = graph_analytics.analyze_heavy_traffic_patterns()

        # Display README section patterns
        section_patterns = readme_patterns.get('readme_section_patterns', {})
        total_with_readme = section_patterns.get('total_artifacts_with_readme', 0)
        common_sections = section_patterns.get('common_sections', [])
        universal_sections = section_patterns.get('universal_sections', [])
        critical_sections = section_patterns.get('critical_sections', [])
        readme_archetypes = section_patterns.get('readme_archetypes', [])
        section_centrality = section_patterns.get('section_centrality', {})

        print(f"📚 Artifacts with README: {total_with_readme}")

        # Display README archetypes
        if readme_archetypes:
            print(f"\n🏛️ README Archetypes Found:")
            for archetype in readme_archetypes[:5]:
                name = archetype.get('archetype', 'unknown')
                count = archetype.get('artifact_count', 0)
                avg_sections = archetype.get('avg_sections', 0)
                examples = archetype.get('example_artifacts', [])
                print(f"   • {name}: {count} artifacts (avg {avg_sections:.1f} sections)")
                if examples:
                    print(f"     Examples: {', '.join(examples[:2])}")

        # Display critical sections
        if critical_sections:
            print(f"\n🎯 Critical README Sections (>70% prevalence):")
            for section in critical_sections[:5]:
                heading = section.get('normalized_heading', section.get('heading', 'N/A'))
                prevalence = section.get('prevalence_percentage', 0)
                artifact_count = section.get('artifact_count', 0)
                code_rate = section.get('code_snippet_rate', 0)
                script_rate = section.get('script_reference_rate', 0)
                print(f"   • {heading}: {prevalence:.1f}% ({artifact_count}/{total_with_readme} artifacts)")
                if code_rate > 0 or script_rate > 0:
                    print(f"     Code: {code_rate:.1f}%, Scripts: {script_rate:.1f}%")

        # Display section centrality
        section_rankings = section_centrality.get('section_rankings', [])
        top_critical = section_centrality.get('top_critical_sections', [])

        if top_critical:
            print(f"\n⭐ Most Critical Section Types (by centrality):")
            for section in top_critical[:5]:
                section_type = section.get('section_type', 'unknown')
                importance = section.get('relative_importance', 0)
                actionable = section.get('actionable_content_ratio', 0)
                print(f"   • {section_type}: {importance:.1f}% importance, {actionable:.1f}% actionable content")

        # Display graph motifs
        universal_patterns = readme_patterns.get('universal_artifact_patterns', {})
        readme_motifs = universal_patterns.get('readme_motifs', {})

        if readme_motifs:
            individual_motifs = readme_motifs.get('individual_motifs', {})
            composite_motifs = readme_motifs.get('composite_motifs', [])
            motif_summary = readme_motifs.get('motif_summary', {})

            print(f"\n🔍 README Graph Motifs Discovered:")
            print(f"   📊 Total motifs: {motif_summary.get('total_motifs_found', 0)}")
            print(f"   💪 Strong motifs: {len(motif_summary.get('strong_motifs', []))}")
            print(f"   📈 Average strength: {motif_summary.get('average_motif_strength', 0):.1f}%")

            if individual_motifs:
                print(f"\n🧩 Individual Motifs:")
                for motif_type, motif_data in individual_motifs.items():
                    strength = motif_data.get('motif_strength', 0)
                    artifacts = motif_data.get('successful_artifacts', 0)
                    marker = "🔥" if motif_data.get('is_strong_motif', False) else "📌"
                    print(f"   {marker} {motif_type}: {strength:.1f}% strength ({artifacts} artifacts)")

            if composite_motifs:
                print(f"\n🎯 Composite Motifs (artifacts with multiple patterns):")
                for motif in composite_motifs[:3]:
                    name = motif.get('artifact_name', 'unknown')
                    score = motif.get('score', 0)
                    count = motif.get('motif_count', 0)
                    pattern = motif.get('motif_pattern_string', 'unknown')
                    print(f"   • {name}: {count} motifs, score {score:.2f}")
                    print(f"     Patterns: {pattern}")

        # Display universal patterns
        universal_files = universal_patterns.get('universal_file_types', [])
        common_dirs = universal_patterns.get('common_directory_structures', [])

        if universal_files:
            print(f"\n📁 Universal File Types (>50% of artifacts):")
            for file_type in universal_files[:5]:
                name = file_type.get('file_type', 'N/A')
                prevalence = file_type.get('prevalence_percentage', 0)
                artifact_count = file_type.get('artifact_count', 0)
                total = file_type.get('total_artifacts', 1)
                print(f"   • {name}: {prevalence:.1f}% ({artifact_count}/{total} artifacts)")

        if common_dirs:
            print(f"\n📂 Common Directory Structures (>30% of artifacts):")
            for dir_info in common_dirs[:5]:
                name = dir_info.get('dir_name', 'N/A')
                prevalence = dir_info.get('prevalence_percentage', 0)
                artifact_count = dir_info.get('artifact_count', 0)
                total = dir_info.get('total_artifacts', 1)
                print(f"   • {name}: {prevalence:.1f}% ({artifact_count}/{total} artifacts)")

        # Display README quality analysis
        readme_quality = universal_patterns.get('readme_quality_analysis', {})
        if readme_quality:
            quality_stats = readme_quality.get('quality_statistics', {})
            top_quality = readme_quality.get('top_quality_readmes', [])
            improvement_candidates = readme_quality.get('improvement_candidates', [])

            print(f"\n📊 README Quality Analysis:")
            print(f"   📚 READMEs analyzed: {quality_stats.get('total_readmes_analyzed', 0)}")
            print(f"   📈 Average quality score: {quality_stats.get('avg_quality_score', 0):.1f}/100")
            print(f"   🔗 README-Artifact correlation: {quality_stats.get('readme_artifact_correlation', 0):.3f}")

            # Quality distribution
            quality_dist = quality_stats.get('quality_distribution', {})
            print(f"   📊 Quality Distribution:")
            for grade, count in quality_dist.items():
                if count > 0:
                    print(f"     Grade {grade}: {count} READMEs")

            # Instructional quality
            instr_dist = quality_stats.get('instructional_quality_distribution', {})
            print(f"   🎓 Instructional Quality:")
            for level, count in instr_dist.items():
                if count > 0:
                    print(f"     {level.capitalize()}: {count} READMEs")

            # Top quality READMEs
            if top_quality:
                print(f"\n🌟 Top Quality READMEs (Score ≥80):")
                for readme in top_quality[:5]:
                    name = readme.get('artifact_name', 'unknown')
                    score = readme.get('quality_score', 0)
                    grade = readme.get('quality_grade', 'N/A')
                    instr = readme.get('instructional_quality', 'N/A')
                    print(f"   • {name}: {score:.1f}/100 (Grade {grade}, {instr} instruction)")

            # Improvement candidates
            if improvement_candidates:
                print(f"\n📝 README Improvement Candidates (Score <60):")
                for readme in improvement_candidates[:3]:
                    name = readme.get('artifact_name', 'unknown')
                    score = readme.get('quality_score', 0)
                    grade = readme.get('quality_grade', 'N/A')
                    print(f"   • {name}: {score:.1f}/100 (Grade {grade})")
                    print(f"     💡 Suggested improvements: Add installation, usage, examples sections")

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
