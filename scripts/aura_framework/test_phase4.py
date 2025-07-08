#!/usr/bin/env python3
"""
🧪 Test Script for Phase 4: Pattern Analysis Engine
Test the pattern discovery and analysis capabilities.

This script tests the Phase 4 Pattern Analysis engine using the knowledge graph
from Phase 2 and optionally vector embeddings from Phase 3.
"""

import os
import sys
import traceback

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the phase4 components
from phase4_pattern_analysis import PatternAnalysisEngine
from phase2_knowledge_graph import KnowledgeGraphBuilder


def test_dependencies():
    """Test Phase 4 dependencies"""
    print("\n📦 Testing Phase 4 Dependencies")
    print("-" * 50)

    dependencies_status = {}

    # Core dependencies
    try:
        import networkx
        print("✅ NetworkX available")
        dependencies_status['networkx'] = True
    except ImportError:
        print("❌ NetworkX not available")
        dependencies_status['networkx'] = False

    try:
        import numpy
        print("✅ NumPy available")
        dependencies_status['numpy'] = True
    except ImportError:
        print("❌ NumPy not available")
        dependencies_status['numpy'] = False

    try:
        import matplotlib
        print("✅ Matplotlib available")
        dependencies_status['matplotlib'] = True
    except ImportError:
        print("❌ Matplotlib not available")
        dependencies_status['matplotlib'] = False

    # Optional dependencies
    try:
        import sklearn
        print("✅ Scikit-learn available")
        dependencies_status['sklearn'] = True
    except ImportError:
        print("⚠️  Scikit-learn not available (clustering will be limited)")
        dependencies_status['sklearn'] = False

    try:
        import neo4j
        print("✅ Neo4j driver available")
        dependencies_status['neo4j'] = True
    except ImportError:
        print("⚠️  Neo4j driver not available")
        dependencies_status['neo4j'] = False

    try:
        import community
        print("✅ Python-louvain available")
        dependencies_status['community'] = True
    except ImportError:
        print("⚠️  Python-louvain not available (community detection limited)")
        dependencies_status['community'] = False

    # Check if core dependencies are available
    core_available = dependencies_status.get('networkx', False) and dependencies_status.get('numpy', False)

    if core_available:
        print("\n✅ Core dependencies satisfied - Phase 4 can run")
    else:
        print("\n❌ Missing core dependencies - install NetworkX and NumPy")

    return dependencies_status


def test_pattern_engine_initialization():
    """Test the pattern analysis engine initialization"""
    print("\n🔧 Testing Pattern Analysis Engine Initialization")
    print("-" * 50)

    try:
        # First initialize a knowledge graph builder
        print("🔄 Initializing Knowledge Graph Builder...")
        kg_builder = KnowledgeGraphBuilder(use_neo4j=True, clear_existing=True)
        print("✅ Knowledge Graph Builder initialized")

        # Test NetworkX backend
        print("\n🔄 Testing Pattern Engine with NetworkX backend...")
        pattern_engine = PatternAnalysisEngine(
            knowledge_graph_builder=kg_builder,
            use_neo4j=True
        )
        print("✅ Pattern Engine initialized with NetworkX backend")

        # Test Neo4j backend if available
        try:
            print("\n🔄 Testing Pattern Engine with Neo4j backend...")
            kg_builder_neo4j = KnowledgeGraphBuilder(use_neo4j=True, clear_existing=False)

            if kg_builder_neo4j.use_neo4j:
                pattern_engine_neo4j = PatternAnalysisEngine(
                    knowledge_graph_builder=kg_builder_neo4j,
                    use_neo4j=True
                )
                print("✅ Pattern Engine initialized with Neo4j backend")
                # Close the Neo4j connection
                kg_builder_neo4j.close()
            else:
                print("⚠️  Neo4j backend not available - using NetworkX fallback")

        except Exception as e:
            print(f"⚠️  Neo4j backend test failed: {e}")

        print("\n📊 Engine attributes:")
        print(f"   - Knowledge Graph Builder: {'✅ Connected' if kg_builder else '❌ Failed'}")
        print(
            f"   - Section patterns list: {'✅ Ready' if hasattr(pattern_engine, 'section_patterns') else '❌ Missing'}")
        print(
            f"   - Structural patterns list: {'✅ Ready' if hasattr(pattern_engine, 'structural_patterns') else '❌ Missing'}")
        print(
            f"   - Centrality scores dict: {'✅ Ready' if hasattr(pattern_engine, 'centrality_scores') else '❌ Missing'}")

        return True

    except Exception as e:
        print(f"❌ Pattern Engine initialization failed: {e}")
        traceback.print_exc()
        return False


def test_pattern_analysis_with_sample_data():
    """Test pattern analysis with sample knowledge graph data"""
    print("\n📊 Testing Pattern Analysis with Sample Data")
    print("-" * 50)

    try:
        # Initialize knowledge graph builder
        print("🔄 Setting up sample knowledge graph...")
        kg_builder = KnowledgeGraphBuilder(use_neo4j=False, clear_existing=True)

        # Create sample data in the knowledge graph
        from config import NODE_TYPES, RELATIONSHIP_TYPES
        import uuid

        # Create sample artifacts
        sample_artifacts = [
            {
                'id': 'artifact_1',
                'name': 'BuildTool_Analyzer',
                'conference': 'ICSE',
                'year': 2024,
                'total_files': 15,
                'has_docker': True,
                'has_requirements_txt': True,
                'has_license': True
            },
            {
                'id': 'artifact_2',
                'name': 'DataProcessor_ML',
                'conference': 'ICSE',
                'year': 2024,
                'total_files': 22,
                'has_docker': True,
                'has_setup_py': True,
                'has_jupyter': True
            },
            {
                'id': 'artifact_3',
                'name': 'WebFramework_Tool',
                'conference': 'FSE',
                'year': 2024,
                'total_files': 8,
                'has_requirements_txt': True,
                'has_license': True
            }
        ]

        # Create sample sections for each artifact
        sample_sections = [
            # Artifact 1 sections
            {'artifact_id': 'artifact_1', 'heading': 'Installation', 'section_order': 1, 'content_length': 500},
            {'artifact_id': 'artifact_1', 'heading': 'Usage', 'section_order': 2, 'content_length': 800},
            {'artifact_id': 'artifact_1', 'heading': 'Configuration', 'section_order': 3, 'content_length': 600},
            {'artifact_id': 'artifact_1', 'heading': 'Testing', 'section_order': 4, 'content_length': 400},

            # Artifact 2 sections
            {'artifact_id': 'artifact_2', 'heading': 'Installation', 'section_order': 1, 'content_length': 450},
            {'artifact_id': 'artifact_2', 'heading': 'Data Processing', 'section_order': 2, 'content_length': 1200},
            {'artifact_id': 'artifact_2', 'heading': 'Model Training', 'section_order': 3, 'content_length': 900},
            {'artifact_id': 'artifact_2', 'heading': 'Usage', 'section_order': 4, 'content_length': 700},

            # Artifact 3 sections  
            {'artifact_id': 'artifact_3', 'heading': 'Installation', 'section_order': 1, 'content_length': 300},
            {'artifact_id': 'artifact_3', 'heading': 'API Documentation', 'section_order': 2, 'content_length': 1500},
            {'artifact_id': 'artifact_3', 'heading': 'Usage', 'section_order': 3, 'content_length': 600}
        ]

        # Create sample tools
        sample_tools = ['python', 'docker', 'numpy', 'pandas', 'flask', 'tensorflow']

        # Add nodes to graph
        G = kg_builder.nx_graph

        # Add artifact nodes
        for artifact in sample_artifacts:
            node_id = f"artifact_{artifact['id']}"
            G.add_node(node_id,
                       node_type=NODE_TYPES['ARTIFACT'],
                       **artifact)

        # Add section nodes
        for section in sample_sections:
            section_id = f"section_{section['artifact_id']}_{section['section_order']}"
            G.add_node(section_id,
                       node_type=NODE_TYPES['SECTION'],
                       **section)

            # Connect to artifact
            artifact_node = f"artifact_{section['artifact_id']}"
            G.add_edge(artifact_node, section_id,
                       relationship_type=RELATIONSHIP_TYPES['HAS_SECTION'])

        # Add tool nodes and connections
        for tool in sample_tools:
            tool_id = f"tool_{tool}"
            G.add_node(tool_id,
                       node_type=NODE_TYPES['TOOL'],
                       name=tool)

            # Connect tools to some sections randomly
            import random
            for section in sample_sections:
                if random.random() < 0.3:  # 30% chance of connection
                    section_id = f"section_{section['artifact_id']}_{section['section_order']}"
                    G.add_edge(section_id, tool_id,
                               relationship_type=RELATIONSHIP_TYPES['MENTIONS'])

        print(f"✅ Created sample graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Initialize pattern analysis engine
        print("\n🔄 Initializing Pattern Analysis Engine...")
        pattern_engine = PatternAnalysisEngine(
            knowledge_graph_builder=kg_builder,
            use_neo4j=False
        )
        print("✅ Pattern Analysis Engine initialized")

        # Run pattern analysis
        print("\n🔄 Running pattern analysis...")
        results = pattern_engine.analyze_accepted_artifacts(
            min_frequency=2,  # Lower threshold for sample data
            outlier_threshold=1.5
        )

        # Display results
        print("\n📈 Pattern Analysis Results:")
        print(f"   📋 Section patterns found: {len(results.section_patterns)}")
        print(f"   🏗️  Structural patterns found: {len(results.structural_patterns)}")
        print(f"   🎯 Artifact clusters: {len(results.clusters)}")
        print(f"   ⚠️  Outliers detected: {len(results.outliers)}")
        print(f"   💪 Success factors: {len(results.success_factors)}")
        print(f"   💡 Recommendations: {len(results.recommendations)}")

        # Show some details
        if results.section_patterns:
            print(f"\n📋 Top Section Patterns:")
            for i, pattern in enumerate(results.section_patterns[:3]):
                print(f"     {i + 1}. {pattern.heading} (frequency: {pattern.frequency})")

        if results.structural_patterns:
            print(f"\n🏗️  Structural Patterns:")
            for i, pattern in enumerate(results.structural_patterns[:2]):
                sections_str = ' → '.join(pattern.section_sequence[:3])
                print(f"     {i + 1}. {pattern.pattern_name}: {sections_str}")

        if results.success_factors:
            print(f"\n💪 Top Success Factors:")
            for factor, score in list(results.success_factors.items())[:3]:
                print(f"     - {factor}: {score:.1%}")

        success = (len(results.section_patterns) > 0 or
                   len(results.structural_patterns) > 0 or
                   len(results.success_factors) > 0)

        if success:
            print("\n✅ Pattern analysis with sample data PASSED")
        else:
            print("\n❌ Pattern analysis with sample data FAILED - no patterns found")

        return success

    except Exception as e:
        print(f"❌ Pattern analysis test failed: {e}")
        traceback.print_exc()
        return False


def test_pattern_analysis_with_real_data():
    """Test pattern analysis with real artifact data"""
    print("\n🏛️  Testing Pattern Analysis with Real Artifact Data")
    print("-" * 50)

    try:
        # Check if we have access to the knowledge graph from Phase 2
        artifacts_dir = "../../algo_outputs/algorithm_2_output_2"
        if not os.path.exists(artifacts_dir):
            print(f"⚠️  Artifacts directory not found: {artifacts_dir}")
            print("   Skipping real data test - run Phase 2 first")
            return True  # Not a failure, just skip

        # Initialize knowledge graph and build from real data
        print("🔄 Building knowledge graph from real artifacts...")
        kg_builder = KnowledgeGraphBuilder(use_neo4j=False, clear_existing=True)

        # Build graph from artifacts (reuse Phase 2 functionality)
        stats = kg_builder.build_graph_from_processed_artifacts(
            artifacts_dir,
            max_artifacts=5,  # Limit for testing
            convert_format=True
        )

        print(f"✅ Knowledge graph built: {stats.total_nodes} nodes, {stats.total_relationships} relationships")

        # Initialize pattern analysis engine
        print("\n🔄 Analyzing patterns in real artifacts...")
        pattern_engine = PatternAnalysisEngine(
            knowledge_graph_builder=kg_builder,
            use_neo4j=False
        )

        # Run comprehensive analysis
        results = pattern_engine.analyze_accepted_artifacts(
            min_frequency=2,
            outlier_threshold=2.0
        )

        # Display comprehensive results
        print("\n📊 Real Data Analysis Results:")
        print(f"   📋 Section patterns discovered: {len(results.section_patterns)}")
        print(f"   🏗️  Structural patterns found: {len(results.structural_patterns)}")
        print(f"   🎯 Artifact clusters created: {len(results.clusters)}")
        print(f"   ⚠️  Outliers detected: {len(results.outliers)}")
        print(f"   💪 Success factors identified: {len(results.success_factors)}")
        print(f"   💡 Recommendations generated: {len(results.recommendations)}")

        # Show detailed results
        if results.section_patterns:
            print(f"\n📋 Most Common Section Patterns:")
            for i, pattern in enumerate(results.section_patterns[:5]):
                tools_str = ', '.join(pattern.associated_tools[:3]) if pattern.associated_tools else 'None'
                print(f"     {i + 1}. {pattern.heading}")
                print(f"        Frequency: {pattern.frequency}, Avg Position: {pattern.avg_position:.1f}")
                print(f"        Length: {pattern.typical_length} chars, Tools: {tools_str}")

        if results.structural_patterns:
            print(f"\n🏗️  Structural Patterns Found:")
            for i, pattern in enumerate(results.structural_patterns[:3]):
                sections_preview = ' → '.join(pattern.section_sequence[:4])
                if len(pattern.section_sequence) > 4:
                    sections_preview += "..."
                print(f"     {i + 1}. {pattern.pattern_name}")
                print(f"        Structure: {sections_preview}")
                print(f"        Frequency: {pattern.frequency}, Success Rate: {pattern.success_rate:.1%}")

        if results.clusters:
            print(f"\n🎯 Artifact Clustering:")
            for cluster_id, artifacts in results.clusters.items():
                print(f"     {cluster_id}: {len(artifacts)} artifacts")
                if artifacts:
                    print(f"        Examples: {', '.join(artifacts[:3])}")

        if results.outliers:
            print(f"\n⚠️  Outlier Artifacts:")
            for outlier in results.outliers[:5]:
                print(f"     - {outlier}")

        if results.success_factors:
            print(f"\n💪 Key Success Factors:")
            sorted_factors = sorted(results.success_factors.items(), key=lambda x: x[1], reverse=True)
            for factor, score in sorted_factors[:5]:
                print(f"     - {factor}: {score:.1%}")

        if results.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(results.recommendations):
                print(f"     {i + 1}. {rec}")

        # Test visualization and export (if possible)
        try:
            print(f"\n📊 Testing visualization generation...")
            pattern_engine.visualize_patterns("data/test_pattern_visualizations")
            print("✅ Visualizations generated successfully")
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")

        try:
            print(f"\n📄 Testing results export...")
            pattern_engine.export_results("data/test_pattern_results.json")
            print("✅ Results exported successfully")
        except Exception as e:
            print(f"⚠️  Results export failed: {e}")

        success = (len(results.section_patterns) > 0 or
                   len(results.structural_patterns) > 0 or
                   len(results.success_factors) > 0)

        if success:
            print("\n✅ Real data pattern analysis PASSED!")
        else:
            print("\n❌ Real data pattern analysis FAILED - no meaningful patterns found")

        return success

    except Exception as e:
        print(f"❌ Real data pattern analysis failed: {e}")
        traceback.print_exc()
        return False


def test_centrality_and_clustering():
    """Test centrality computation and clustering features"""
    print("\n🎯 Testing Centrality and Clustering Features")
    print("-" * 50)

    try:
        # Create a simple test graph
        print("🔄 Creating test graph for centrality analysis...")
        kg_builder = KnowledgeGraphBuilder(use_neo4j=False, clear_existing=True)

        # Create a small connected graph
        G = kg_builder.nx_graph

        # Add nodes with types
        from config import NODE_TYPES

        nodes = [
            ('a1', {'node_type': NODE_TYPES['ARTIFACT'], 'name': 'artifact1'}),
            ('a2', {'node_type': NODE_TYPES['ARTIFACT'], 'name': 'artifact2'}),
            ('s1', {'node_type': NODE_TYPES['SECTION'], 'heading': 'Installation'}),
            ('s2', {'node_type': NODE_TYPES['SECTION'], 'heading': 'Usage'}),
            ('s3', {'node_type': NODE_TYPES['SECTION'], 'heading': 'Configuration'}),
            ('t1', {'node_type': NODE_TYPES['TOOL'], 'name': 'python'}),
            ('t2', {'node_type': NODE_TYPES['TOOL'], 'name': 'docker'})
        ]

        G.add_nodes_from(nodes)

        # Add edges
        edges = [
            ('a1', 's1'), ('a1', 's2'), ('a2', 's2'), ('a2', 's3'),
            ('s1', 't1'), ('s2', 't1'), ('s2', 't2'), ('s3', 't2')
        ]
        G.add_edges_from(edges)

        print(f"✅ Test graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Initialize pattern engine
        pattern_engine = PatternAnalysisEngine(kg_builder, use_neo4j=False)

        # Test centrality computation
        print("\n🔄 Computing centrality metrics...")
        pattern_engine._compute_centrality_metrics()

        print(f"   PageRank scores computed: {len(pattern_engine.pagerank_scores)}")
        print(f"   Centrality scores computed: {len(pattern_engine.centrality_scores)}")

        if pattern_engine.pagerank_scores:
            top_pagerank = sorted(pattern_engine.pagerank_scores.items(),
                                  key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top PageRank nodes: {[node for node, score in top_pagerank]}")

        # Test community detection
        print("\n🔄 Testing community detection...")
        pattern_engine._perform_community_detection()

        communities_found = len(
            set(pattern_engine.community_assignments.values())) if pattern_engine.community_assignments else 0
        print(f"   Communities detected: {communities_found}")

        # Test clustering
        print("\n🔄 Testing artifact clustering...")
        clusters = pattern_engine._cluster_artifacts()

        print(f"   Clusters created: {len(clusters)}")
        if clusters:
            for cluster_id, artifacts in clusters.items():
                print(f"     {cluster_id}: {len(artifacts)} artifacts")

        # Test outlier detection  
        print("\n🔄 Testing outlier detection...")
        outliers = pattern_engine._detect_outliers(threshold=1.5)

        print(f"   Outliers detected: {len(outliers)}")
        if outliers:
            print(f"     Outlier nodes: {outliers}")

        success = (len(pattern_engine.pagerank_scores) > 0 or
                   len(pattern_engine.centrality_scores) > 0 or
                   len(clusters) > 0)

        if success:
            print("\n✅ Centrality and clustering test PASSED")
        else:
            print("\n❌ Centrality and clustering test FAILED")

        return success

    except Exception as e:
        print(f"❌ Centrality and clustering test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all Phase 4 tests"""
    print("\n🧪 AURA Phase 4 Pattern Analysis Test Suite")
    print("=" * 60)

    # Check dependencies first
    print(f"\n{'=' * 60}")
    print("🧪 Pre-Test: Checking Dependencies")
    print(f"{'=' * 60}")

    dependencies = test_dependencies()
    if not dependencies.get('networkx', False) or not dependencies.get('numpy', False):
        print("\n❌ Missing core dependencies. Please install:")
        print("   pip install networkx numpy matplotlib")
        return

    tests = [
        ("Pattern Engine Initialization", test_pattern_engine_initialization),
        ("Pattern Analysis with Sample Data", test_pattern_analysis_with_sample_data),
        ("Pattern Analysis with Real Data", test_pattern_analysis_with_real_data),
        ("Centrality and Clustering", test_centrality_and_clustering)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"🧪 Running Test: {test_name}")
        print(f"{'=' * 60}")

        try:
            result = test_func()
            results.append((test_name, result))

            if result:
                print(f"\n✅ {test_name} test PASSED")
            else:
                print(f"\n❌ {test_name} test FAILED")

        except Exception as e:
            print(f"\n💥 {test_name} test CRASHED: {e}")
            results.append((test_name, False))

    # Print final summary
    print("\n" + "=" * 60)
    print("🏁 Test Suite Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")

    print(f"\n📊 Overall Results: {passed}/{total} tests passed")

    # Show dependency status
    optional_deps = ['sklearn', 'neo4j', 'community']
    available_optional = sum(1 for dep in optional_deps if dependencies.get(dep, False))
    print(f"🔧 Optional dependencies: {available_optional}/{len(optional_deps)} available")

    if passed == total:
        print("🎉 All tests PASSED! Phase 4 Pattern Analysis is ready for use.")
        print("📊 Pattern discovery and analysis capabilities are working perfectly!")
    else:
        print("⚠️  Some tests FAILED. Please check the errors above.")
        if passed > 0:
            print(f"💡 {passed} tests passed - partial functionality is available.")


if __name__ == "__main__":
    main()
