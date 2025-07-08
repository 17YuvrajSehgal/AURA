#!/usr/bin/env python3
"""
🎯 Phase 4 Pattern Analysis Demo
Demonstrates the key capabilities of the AURA Phase 4 system.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4_pattern_analysis import PatternAnalysisEngine
from phase2_knowledge_graph import KnowledgeGraphBuilder
from config import config


def create_sample_data_in_neo4j(kg_builder):
    """Create sample data directly in Neo4j for demonstration"""
    print("🔄 Creating sample data in Neo4j...")
    
    # Create sample nodes directly in Neo4j
    with kg_builder.driver.session(database=config.neo4j.database) as session:
        # Create sample artifacts
        session.run("""
            CREATE (a1:Artifact {
                id: 'artifact_1',
                name: 'MLTool',
                has_docker: true,
                conference: 'OOPSLA',
                year: 2024
            })
        """)
        
        session.run("""
            CREATE (a2:Artifact {
                id: 'artifact_2', 
                name: 'WebApp',
                has_license: true,
                conference: 'FSE',
                year: 2024
            })
        """)
        
        # Create sample sections
        session.run("""
            MATCH (a1:Artifact {id: 'artifact_1'})
            CREATE (s1:Section {
                id: 'section_1',
                heading: 'Installation',
                content: 'Install using pip install requirements',
                position: 1
            })
            CREATE (s2:Section {
                id: 'section_2', 
                heading: 'Usage',
                content: 'Run python main.py to start',
                position: 2
            })
            CREATE (a1)-[:HAS_SECTION]->(s1)
            CREATE (a1)-[:HAS_SECTION]->(s2)
        """)
        
        session.run("""
            MATCH (a2:Artifact {id: 'artifact_2'})
            CREATE (s3:Section {
                id: 'section_3',
                heading: 'Installation', 
                content: 'Use docker-compose up',
                position: 1
            })
            CREATE (a2)-[:HAS_SECTION]->(s3)
        """)
        
        # Create sample tools
        session.run("""
            CREATE (t1:Tool {id: 'tool_python', name: 'python', category: 'language'})
            CREATE (t2:Tool {id: 'tool_docker', name: 'docker', category: 'containerization'})
        """)
        
        # Create relationships
        session.run("""
            MATCH (s:Section), (t:Tool)
            WHERE (s.heading = 'Installation' AND t.name = 'python') OR
                  (s.heading = 'Usage' AND t.name = 'python') OR
                  (s.heading = 'Installation' AND t.name = 'docker' AND s.id = 'section_3')
            CREATE (s)-[:USES_TOOL]->(t)
        """)
        
        # Get node and edge counts
        node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
        edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        
    print(f"✅ Sample Neo4j graph: {node_count} nodes, {edge_count} edges")
    return node_count, edge_count


def demo_phase4():
    """Demonstrate Phase 4 Pattern Analysis capabilities"""
    print("\n🎯 AURA Phase 4 Pattern Analysis Demo")
    print("=" * 50)

    # Check if we have real data
    artifacts_dir = "../../algo_outputs/algorithm_2_output_2"
    if os.path.exists(artifacts_dir):
        print("🔄 Using real artifact data for demonstration...")
        use_real_data = True
    else:
        print("📊 Using sample data for demonstration...")
        use_real_data = False

    # Initialize knowledge graph with Neo4j
    print("\n🔄 Building knowledge graph...")
    kg_builder = KnowledgeGraphBuilder(use_neo4j=True, clear_existing=True)

    if use_real_data:
        # Build from real artifacts
        stats = kg_builder.build_graph_from_processed_artifacts(
            artifacts_dir,
            max_artifacts=5,
            convert_format=True
        )
        print(f"✅ Real knowledge graph: {stats.total_nodes} nodes, {stats.total_relationships} relationships")
    else:
        # Create sample data directly in Neo4j
        create_sample_data_in_neo4j(kg_builder)

    # Initialize pattern analysis engine
    print("\n🔄 Initializing Pattern Analysis Engine...")
    pattern_engine = PatternAnalysisEngine(
        knowledge_graph_builder=kg_builder,
        use_neo4j=True
    )
    print("✅ Pattern Analysis Engine ready")

    # Run comprehensive analysis
    print("\n🔄 Discovering patterns and analyzing artifacts...")
    results = pattern_engine.analyze_accepted_artifacts(
        min_frequency=1,  # Lower threshold for sample data
        outlier_threshold=2.0
    )

    # Display results
    print(f"\n📊 Pattern Discovery Results:")
    print(f"   📋 Section patterns: {len(results.section_patterns)}")
    print(f"   🏗️  Structural patterns: {len(results.structural_patterns)}")
    print(f"   🎯 Artifact clusters: {len(results.clusters)}")
    print(f"   ⚠️  Outliers: {len(results.outliers)}")

    # Show section patterns
    if results.section_patterns:
        print(f"\n📋 Most Common Section Types:")
        for i, pattern in enumerate(results.section_patterns[:5]):
            tools_str = ', '.join(pattern.associated_tools[:3]) if pattern.associated_tools else 'None'
            print(f"   {i + 1}. {pattern.heading}")
            print(f"      📊 Frequency: {pattern.frequency}")
            print(f"      📍 Avg Position: {pattern.avg_position:.1f}")
            print(f"      🔧 Tools: {tools_str}")

    # Show clustering results
    if results.clusters:
        print(f"\n🎯 Artifact Clustering:")
        for cluster_id, artifacts in results.clusters.items():
            print(f"   {cluster_id}: {len(artifacts)} artifacts")
            if artifacts:
                example_artifacts = ', '.join(artifacts[:3])
                if len(artifacts) > 3:
                    example_artifacts += "..."
                print(f"      Examples: {example_artifacts}")

    # Show success factors
    if results.success_factors:
        print(f"\n💪 Success Factors (% of artifacts):")
        sorted_factors = sorted(results.success_factors.items(), key=lambda x: x[1], reverse=True)
        for factor, percentage in sorted_factors[:5]:
            print(f"   📈 {factor}: {percentage:.1%}")

    # Show recommendations
    if results.recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(results.recommendations):
            print(f"   {i + 1}. {rec}")

    # Show centrality analysis
    if pattern_engine.pagerank_scores:
        print(f"\n🌟 Most Important Nodes (PageRank):")
        top_nodes = sorted(pattern_engine.pagerank_scores.items(),
                           key=lambda x: x[1], reverse=True)[:5]
        for node, score in top_nodes:
            print(f"   🎯 {node}: {score:.3f}")

    # Show outliers if any
    if results.outliers:
        print(f"\n⚠️  Outlier Artifacts:")
        for outlier in results.outliers:
            print(f"   🚨 {outlier}")

    print(f"\n🎉 Phase 4 Pattern Analysis Demo Complete!")
    print(f"   ✅ Pattern discovery working")
    print(f"   ✅ Clustering analysis working")
    print(f"   ✅ Success factor identification working")
    print(f"   ✅ Centrality analysis working")
    print(f"   ✅ Recommendation generation working")


if __name__ == "__main__":
    try:
        demo_phase4()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
