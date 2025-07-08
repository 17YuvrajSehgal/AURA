#!/usr/bin/env python3
"""
🧪 Test Script for Phase 2: Knowledge Graph Builder
Test the knowledge graph construction with artifact JSON files.

This script tests the Phase 2 Knowledge Graph builder using the user's
artifact analysis JSON files.
"""

import os
import sys
import traceback
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the phase2 components
from phase2_knowledge_graph import KnowledgeGraphBuilder
from artifact_utils import ArtifactJSONProcessor
from config import config


def test_neo4j_connectivity():
    """Test Neo4j connectivity before running main tests"""
    print("\n🔌 Testing Neo4j Connectivity")
    print("-" * 40)

    print(f"📍 Neo4j URI: {config.neo4j.uri}")
    print(f"👤 Username: {config.neo4j.username}")
    print(f"🗄️  Database: {config.neo4j.database}")

    try:
        # Try to import Neo4j libraries
        from neo4j import GraphDatabase, basic_auth
        from py2neo import Graph
        print("✅ Neo4j libraries found")

        # Test basic connection
        print("🔄 Testing Neo4j connection...")
        driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=basic_auth(config.neo4j.username, config.neo4j.password)
        )

        with driver.session(database=config.neo4j.database) as session:
            result = session.run("RETURN 1 as test")
            test_result = result.single()
            print(f"✅ Neo4j connection successful: {test_result['test']}")

        driver.close()
        print("🎉 Neo4j is ready for use!")
        return True

    except ImportError as e:
        print("❌ Neo4j libraries not found")
        print("   Install with: pip install neo4j py2neo")
        return False
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("   Check Neo4j server is running and credentials are correct")
        return False


def test_artifact_processor():
    """Test the artifact JSON processor"""
    print("\n🔧 Testing Artifact JSON Processor")
    print("-" * 40)

    processor = ArtifactJSONProcessor()

    # Test with a single file
    test_file = "../../algo_outputs/algorithm_2_output_2/10460752_analysis.json"

    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False

    try:
        # Read and convert the artifact
        artifact_data = processor.read_artifact_json(test_file)
        print(f"✅ Successfully read artifact: {artifact_data.get('artifact_name', 'unknown')}")

        # Convert to AURA format
        aura_format = processor.convert_to_aura_format(artifact_data)
        print(f"✅ Successfully converted to AURA format")

        # Print some stats
        metadata = aura_format['metadata']
        print(f"   📊 Artifact: {metadata['artifact_name']}")
        print(f"   📊 Conference: {metadata['conference']}")
        print(f"   📊 Year: {metadata['year']}")
        print(f"   📊 Documentation files: {len(aura_format['documentation_files'])}")
        print(f"   📊 Tools: {len(aura_format['tools'])}")
        print(f"   📊 Commands: {len(aura_format['commands'])}")
        print(f"   📊 Entities: {len(aura_format['entities'])}")

        return True

    except Exception as e:
        print(f"❌ Error testing artifact processor: {e}")
        traceback.print_exc()
        return False


def test_knowledge_graph_builder():
    """Test the knowledge graph builder"""
    print("\n📐 Testing Knowledge Graph Builder")
    print("-" * 40)

    try:
        # Initialize builder (attempt Neo4j first)
        print("🔄 Attempting to initialize Knowledge Graph Builder with Neo4j...")
        kg_builder = KnowledgeGraphBuilder(use_neo4j=True, clear_existing=True)

        backend_name = "Neo4j" if kg_builder.use_neo4j else "NetworkX"
        if kg_builder.use_neo4j:
            print("✅ Knowledge Graph Builder initialized with Neo4j")
            print("🎉 Neo4j connection successful!")
        else:
            print("✅ Knowledge Graph Builder initialized with NetworkX (fallback)")
            print("⚠️  Neo4j connection failed - using NetworkX instead")

        # Build graph from artifacts
        artifacts_dir = "../../algo_outputs/algorithm_2_output_2"

        if not os.path.exists(artifacts_dir):
            print(f"❌ Artifacts directory not found: {artifacts_dir}")
            return False

        # List available files
        artifact_files = list(Path(artifacts_dir).glob("*_analysis.json"))
        print(f"📁 Found {len(artifact_files)} artifact files")

        if not artifact_files:
            print("❌ No artifact files found")
            return False

        # Process a subset for testing
        max_artifacts = min(3, len(artifact_files))
        print(f"📊 Processing {max_artifacts} artifacts for testing...")

        stats = kg_builder.build_graph_from_processed_artifacts(
            artifacts_dir,
            max_artifacts=max_artifacts,
            convert_format=True
        )

        # Get and display summary
        summary = kg_builder.get_graph_summary()

        print("\n📈 Knowledge Graph Statistics:")
        print(f"   🗄️  Backend: {summary['graph_backend']}")
        print(f"   📊 Total Nodes: {summary['total_nodes']:,}")
        print(f"   🔗 Total Relationships: {summary['total_relationships']:,}")
        print(f"   🏛️  Artifacts Processed: {summary['artifacts_processed']}")
        print(f"   🎯 Unique Conferences: {summary['unique_conferences']}")
        print(f"   🔧 Unique Tools: {summary['unique_tools']}")
        print(f"   📈 Graph Density: {summary['graph_density']:.4f}")
        print(f"   📊 Average Degree: {summary['avg_degree']:.2f}")

        if summary['node_types']:
            print("\n📋 Node Types Distribution:")
            for node_type, count in summary['node_types'].items():
                print(f"   - {node_type}: {count:,}")

        if summary['relationship_types']:
            print("\n🔗 Relationship Types Distribution:")
            for rel_type, count in summary['relationship_types'].items():
                print(f"   - {rel_type}: {count:,}")

        if summary['conferences']:
            print("\n🎯 Conferences Found:")
            for conf in summary['conferences']:
                print(f"   - {conf}")

        if summary['tools']:
            print("\n🔧 Tools Found (top 10):")
            for tool in list(summary['tools'])[:10]:
                print(f"   - {tool}")
            if len(summary['tools']) > 10:
                print(f"   ... and {len(summary['tools']) - 10} more")

        kg_builder.close()
        print("\n✅ Knowledge Graph Builder test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing knowledge graph builder: {e}")
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full pipeline with all artifacts"""
    print("\n🔄 Testing Full Pipeline")
    print("-" * 40)

    try:
        # Initialize processor
        processor = ArtifactJSONProcessor()

        # Process all artifacts in the directory
        artifacts_dir = "../../algo_outputs/algorithm_2_output_2"

        if not os.path.exists(artifacts_dir):
            print(f"❌ Artifacts directory not found: {artifacts_dir}")
            return False

        # Get artifact files
        artifact_files = list(Path(artifacts_dir).glob("*_analysis.json"))
        print(f"📁 Found {len(artifact_files)} artifact files")

        if not artifact_files:
            print("❌ No artifact files found")
            return False

        # Process first 5 artifacts
        max_artifacts = min(5, len(artifact_files))
        print(f"📊 Processing {max_artifacts} artifacts...")

        # Initialize knowledge graph builder
        print("🔄 Initializing Knowledge Graph Builder for full pipeline test...")
        kg_builder = KnowledgeGraphBuilder(use_neo4j=True, clear_existing=True)

        backend_name = "Neo4j" if kg_builder.use_neo4j else "NetworkX"
        if kg_builder.use_neo4j:
            print("✅ Full pipeline using Neo4j backend")
        else:
            print("⚠️  Full pipeline using NetworkX backend (Neo4j unavailable)")

        # Build the graph
        stats = kg_builder.build_graph_from_processed_artifacts(
            artifacts_dir,
            max_artifacts=max_artifacts,
            convert_format=True
        )

        # Get summary
        summary = kg_builder.get_graph_summary()

        print("\n🎯 Final Pipeline Results:")
        print(f"   ✅ Artifacts processed: {summary['artifacts_processed']}")
        print(f"   📊 Total nodes created: {summary['total_nodes']:,}")
        print(f"   🔗 Total relationships created: {summary['total_relationships']:,}")
        print(f"   🎯 Conferences identified: {summary['unique_conferences']}")
        print(f"   🔧 Tools identified: {summary['unique_tools']}")

        # Validate results
        if summary['total_nodes'] > 0 and summary['total_relationships'] > 0:
            print("\n✅ Full pipeline test PASSED!")
            success = True
        else:
            print("\n❌ Full pipeline test FAILED - No nodes or relationships created")
            success = False

        kg_builder.close()
        return success

    except Exception as e:
        print(f"❌ Error in full pipeline test: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n🧪 AURA Phase 2 Knowledge Graph Builder Test Suite")
    print("=" * 60)

    # Check if the artifacts directory exists
    artifacts_dir = "../../algo_outputs/algorithm_2_output_2"
    if not os.path.exists(artifacts_dir):
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        print("   Please ensure you have the artifact JSON files in the correct location.")
        return

    # First test Neo4j connectivity
    print(f"\n{'=' * 60}")
    print("🧪 Pre-Test: Neo4j Connectivity Check")
    print(f"{'=' * 60}")

    neo4j_available = test_neo4j_connectivity()
    if neo4j_available:
        print("🎉 Neo4j is available - tests will use Neo4j backend")
    else:
        print("⚠️  Neo4j unavailable - tests will use NetworkX backend")

    tests = [
        ("Artifact Processor", test_artifact_processor),
        ("Knowledge Graph Builder", test_knowledge_graph_builder),
        ("Full Pipeline", test_full_pipeline)
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

    # Show final backend status
    backend_status = "Neo4j ✅" if neo4j_available else "NetworkX (fallback) ⚠️"
    print(f"🗄️  Backend Status: {backend_status}")

    if passed == total:
        print("🎉 All tests PASSED! Phase 2 is ready for use.")
        if neo4j_available:
            print("🗄️  Neo4j backend is working perfectly!")
        else:
            print("⚠️  Consider setting up Neo4j for enhanced graph capabilities.")
    else:
        print("⚠️  Some tests FAILED. Please check the errors above.")


if __name__ == "__main__":
    main()
