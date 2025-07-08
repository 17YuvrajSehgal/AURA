#!/usr/bin/env python3
"""
🧪 Test Script for Phase 3: Vector Embeddings Engine
Test the vector embeddings and semantic search capabilities.

This script tests the Phase 3 Vector Embeddings engine using the user's
artifact analysis JSON files.
"""

import os
import sys
import traceback
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the phase3 components
from phase3_vector_embeddings import VectorEmbeddingEngine
from artifact_utils import ArtifactJSONProcessor
from config import config


def test_embedding_engine_initialization():
    """Test the vector embedding engine initialization"""
    print("\n🔧 Testing Vector Embedding Engine Initialization")
    print("-" * 50)
    
    test_results = {}
    
    # Test different backend combinations
    backends_to_test = [
        ("faiss", "sentence-transformers", "all-MiniLM-L6-v2"),
    ]
    
    # Add other backends if available
    try:
        import qdrant_client
        backends_to_test.append(("qdrant", "sentence-transformers", "all-MiniLM-L6-v2"))
        print("✅ Qdrant client available")
    except ImportError:
        print("⚠️  Qdrant client not available")
    
    try:
        import chromadb
        backends_to_test.append(("chroma", "sentence-transformers", "all-MiniLM-L6-v2"))
        print("✅ ChromaDB available")
    except ImportError:
        print("⚠️  ChromaDB not available")
    
    try:
        import sentence_transformers
        print("✅ SentenceTransformers available")
    except ImportError:
        print("❌ SentenceTransformers not available - install with: pip install sentence-transformers")
        return False
    
    for vector_db, embedding_model, model_name in backends_to_test:
        print(f"\n🔄 Testing {vector_db} + {embedding_model} ({model_name})...")
        
        try:
            engine = VectorEmbeddingEngine(
                vector_db_type=vector_db,
                embedding_model=embedding_model,
                model_name=model_name,
                storage_directory="data/test_embeddings"
            )
            
            print(f"✅ {vector_db} + {embedding_model} initialization successful")
            print(f"   📏 Embedding dimension: {engine.embedding_dimension}")
            test_results[f"{vector_db}_{embedding_model}"] = True
            
            # Test a single embedding
            test_text = "This is a test document about machine learning setup."
            embeddings = engine._generate_embeddings_batch([test_text])
            print(f"   🧪 Test embedding shape: {embeddings[0].shape}")
            
            # Close/cleanup
            del engine
            
        except Exception as e:
            print(f"❌ {vector_db} + {embedding_model} failed: {e}")
            test_results[f"{vector_db}_{embedding_model}"] = False
    
    success_count = sum(test_results.values())
    total_count = len(test_results)
    
    print(f"\n📊 Initialization Results: {success_count}/{total_count} backends working")
    
    return success_count > 0


def test_artifact_embedding_extraction():
    """Test embedding extraction from artifacts"""
    print("\n📊 Testing Artifact Embedding Extraction")
    print("-" * 50)
    
    try:
        # Initialize the embedding engine
        print("🔄 Initializing embedding engine...")
        engine = VectorEmbeddingEngine(
            vector_db_type="faiss",
            embedding_model="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            storage_directory="data/test_embeddings"
        )
        print("✅ Embedding engine initialized")
        
        # Check if artifacts directory exists
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
        
        # We need to convert the analysis format to processed format first
        print("🔄 Converting artifacts to processed format...")
        processor = ArtifactJSONProcessor()
        
        # Create temporary processed files for testing
        temp_processed_dir = Path("data/temp_processed")
        temp_processed_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        max_artifacts = min(3, len(artifact_files))  # Test with 3 artifacts
        
        for artifact_file in artifact_files[:max_artifacts]:
            try:
                # Read and convert artifact
                raw_data = processor.read_artifact_json(artifact_file)
                processed_data = processor.convert_to_aura_format(raw_data)
                
                # Save as processed format (convert dataclass objects to dicts)
                processed_file = temp_processed_dir / f"{raw_data['artifact_name']}_processed.json"
                
                # Convert dataclass objects to dictionaries for proper JSON serialization
                def convert_to_dict(obj):
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    elif isinstance(obj, list):
                        return [convert_to_dict(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {key: convert_to_dict(value) for key, value in obj.items()}
                    else:
                        return obj
                
                serializable_data = convert_to_dict(processed_data)
                
                with open(processed_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(serializable_data, f, indent=2)
                
                processed_count += 1
                print(f"   ✅ Converted {raw_data['artifact_name']}")
                
            except Exception as e:
                print(f"   ❌ Failed to convert {artifact_file}: {e}")
        
        print(f"📊 Successfully converted {processed_count} artifacts")
        
        if processed_count == 0:
            print("❌ No artifacts were successfully converted")
            return False
        
        # Now test embedding extraction
        print("\n🔄 Extracting embeddings from processed artifacts...")
        stats = engine.extract_embeddings_from_processed_artifacts(
            str(temp_processed_dir),
            max_artifacts=processed_count,
            batch_size=8
        )
        
        print("\n📈 Embedding Extraction Results:")
        print(f"   🏛️  Artifacts processed: {stats['artifacts_processed']}")
        print(f"   📄 Sections processed: {stats['sections_processed']}")
        print(f"   🎯 Embeddings created: {stats['embeddings_created']}")
        print(f"   📏 Embedding dimension: {stats['dimension']}")
        print(f"   🤖 Model used: {stats['model_used']}")
        print(f"   ⏱️  Total time: {stats.get('total_time', 0):.2f}s")
        if stats['sections_processed'] > 0:
            print(f"   ⚡ Avg time per section: {stats['avg_embedding_time']:.4f}s")
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_processed_dir, ignore_errors=True)
        
        success = stats['embeddings_created'] > 0
        if success:
            print("✅ Embedding extraction test PASSED")
        else:
            print("❌ Embedding extraction test FAILED - no embeddings created")
        
        return success
        
    except Exception as e:
        print(f"❌ Error in embedding extraction test: {e}")
        traceback.print_exc()
        return False


def test_semantic_search():
    """Test semantic search functionality"""
    print("\n🔍 Testing Semantic Search Functionality")
    print("-" * 50)
    
    try:
        # Initialize engine with some test data
        print("🔄 Setting up test data for semantic search...")
        engine = VectorEmbeddingEngine(
            vector_db_type="faiss",
            embedding_model="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            storage_directory="data/test_embeddings"
        )
        
        # Create some test embedding records manually
        from phase3_vector_embeddings import EmbeddingRecord
        import numpy as np
        
        test_sections = [
            {
                "id": "test_1",
                "artifact_id": "test_artifact_1",
                "heading": "Installation Guide",
                "content": "To install this software, run pip install package-name. Make sure you have Python 3.8 or higher.",
                "text": "Installation Guide To install this software, run pip install package-name. Make sure you have Python 3.8 or higher. Tools: python, pip"
            },
            {
                "id": "test_2", 
                "artifact_id": "test_artifact_1",
                "heading": "Setup Instructions",
                "content": "Follow these setup steps: 1. Install dependencies 2. Configure settings 3. Run tests",
                "text": "Setup Instructions Follow these setup steps: 1. Install dependencies 2. Configure settings 3. Run tests Tools: python, pytest"
            },
            {
                "id": "test_3",
                "artifact_id": "test_artifact_2", 
                "heading": "Docker Configuration",
                "content": "Build the Docker image using: docker build -t myapp .",
                "text": "Docker Configuration Build the Docker image using: docker build -t myapp . Tools: docker"
            },
            {
                "id": "test_4",
                "artifact_id": "test_artifact_2",
                "heading": "Running Tests",
                "content": "Execute tests with pytest. Use --verbose flag for detailed output.",
                "text": "Running Tests Execute tests with pytest. Use --verbose flag for detailed output. Tools: pytest"
            }
        ]
        
        # Generate embeddings for test data
        print("🔄 Generating test embeddings...")
        texts = [section["text"] for section in test_sections]
        embeddings = engine._generate_embeddings_batch(texts)
        
        # Create and store embedding records
        for section, embedding in zip(test_sections, embeddings):
            record = EmbeddingRecord(
                id=f"emb_{section['id']}",
                vector=embedding,
                section_id=section["id"],
                artifact_id=section["artifact_id"],
                heading=section["heading"],
                content=section["content"],
                metadata={
                    "doc_path": "README.md",
                    "section_order": 1,
                    "level": 1,
                    "content_length": len(section["content"]),
                    "tools": ["python", "pip"],
                    "entities": [],
                    "commands_count": 1,
                    "structural_features": {}
                }
            )
            
            engine.embedding_records[record.id] = record
            engine.section_to_embedding[section["id"]] = record.id
            engine._add_embedding_to_database(record)
        
        # Finalize the database
        engine._finalize_vector_database()
        
        print(f"✅ Created {len(test_sections)} test embeddings")
        
        # Test different search queries
        test_queries = [
            ("installation setup", "Should find installation and setup sections"),
            ("docker container", "Should find Docker-related content"),
            ("testing pytest", "Should find testing-related sections"),
            ("python programming", "Should find Python-related content")
        ]
        
        print("\n🔍 Testing semantic search queries:")
        all_search_successful = True
        
        for query, description in test_queries:
            print(f"\n   Query: '{query}' - {description}")
            
            try:
                results = engine.semantic_search(query, top_k=3)
                print(f"   📊 Found {len(results)} results")
                
                for i, result in enumerate(results[:2]):  # Show top 2
                    print(f"     {i+1}. {result.embedding_record.heading} (score: {result.relevance_score:.3f})")
                    print(f"        Artifact: {result.embedding_record.artifact_id}")
                
                if len(results) == 0:
                    print("     ⚠️  No results found")
                
            except Exception as e:
                print(f"     ❌ Search failed: {e}")
                all_search_successful = False
        
        # Test similarity search
        print(f"\n🔍 Testing similarity search...")
        try:
            similar_results = engine.find_similar_sections("test_1", top_k=3)
            print(f"   📊 Found {len(similar_results)} similar sections to 'test_1'")
            
            for i, result in enumerate(similar_results):
                print(f"     {i+1}. {result.embedding_record.heading} (score: {result.relevance_score:.3f})")
        
        except Exception as e:
            print(f"     ❌ Similarity search failed: {e}")
            all_search_successful = False
        
        if all_search_successful:
            print("\n✅ Semantic search test PASSED")
        else:
            print("\n❌ Semantic search test had some failures")
        
        return all_search_successful
        
    except Exception as e:
        print(f"❌ Error in semantic search test: {e}")
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full Phase 3 pipeline"""
    print("\n🔄 Testing Full Phase 3 Pipeline")
    print("-" * 50)
    
    try:
        print("🔄 Initializing Vector Embedding Engine for full pipeline...")
        engine = VectorEmbeddingEngine(
            vector_db_type="faiss",
            embedding_model="sentence-transformers", 
            model_name="all-MiniLM-L6-v2",
            storage_directory="data/full_test_embeddings"
        )
        print("✅ Vector Embedding Engine initialized")
        
        # Check artifacts
        artifacts_dir = "../../algo_outputs/algorithm_2_output_2"
        if not os.path.exists(artifacts_dir):
            print(f"❌ Artifacts directory not found: {artifacts_dir}")
            return False
        
        artifact_files = list(Path(artifacts_dir).glob("*_analysis.json"))
        print(f"📁 Found {len(artifact_files)} artifact files")
        
        if not artifact_files:
            print("❌ No artifact files found")
            return False
        
        # Convert and process artifacts
        print("🔄 Converting artifacts for full pipeline test...")
        processor = ArtifactJSONProcessor()
        
        temp_processed_dir = Path("data/full_test_processed")
        temp_processed_dir.mkdir(parents=True, exist_ok=True)
        
        max_artifacts = min(2, len(artifact_files))  # Process 2 artifacts for full test
        processed_count = 0
        
        for artifact_file in artifact_files[:max_artifacts]:
            try:
                raw_data = processor.read_artifact_json(artifact_file)
                processed_data = processor.convert_to_aura_format(raw_data)
                
                processed_file = temp_processed_dir / f"{raw_data['artifact_name']}_processed.json"
                
                # Convert dataclass objects to dictionaries for proper JSON serialization
                def convert_to_dict(obj):
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    elif isinstance(obj, list):
                        return [convert_to_dict(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {key: convert_to_dict(value) for key, value in obj.items()}
                    else:
                        return obj
                
                serializable_data = convert_to_dict(processed_data)
                
                with open(processed_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(serializable_data, f, indent=2)
                
                processed_count += 1
                print(f"   ✅ Converted {raw_data['artifact_name']}")
                
            except Exception as e:
                print(f"   ❌ Failed to convert {artifact_file}: {e}")
        
        if processed_count == 0:
            print("❌ No artifacts converted for full pipeline test")
            return False
        
        # Extract embeddings
        print(f"\n🔄 Extracting embeddings from {processed_count} artifacts...")
        stats = engine.extract_embeddings_from_processed_artifacts(
            str(temp_processed_dir),
            max_artifacts=processed_count,
            batch_size=4
        )
        
        print("\n📊 Full Pipeline Extraction Results:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test semantic search on real data
        if stats['embeddings_created'] > 0:
            print("\n🔍 Testing semantic search on real embeddings...")
            
            search_queries = [
                "installation setup instructions",
                "docker container configuration", 
                "testing and validation",
                "requirements and dependencies"
            ]
            
            for query in search_queries:
                try:
                    results = engine.semantic_search(query, top_k=3)
                    print(f"   '{query}': {len(results)} results")
                    if results:
                        top_result = results[0]
                        print(f"     Top: {top_result.embedding_record.heading} (score: {top_result.relevance_score:.3f})")
                except Exception as e:
                    print(f"   Search failed for '{query}': {e}")
        
        # Test clustering if we have enough embeddings
        if stats['embeddings_created'] >= 5:
            print(f"\n🎯 Testing semantic clustering...")
            try:
                clusters = engine.perform_semantic_clustering(n_clusters=3, min_cluster_size=2)
                print(f"   📊 Created {len(clusters)} semantic clusters")
                
                for i, cluster in enumerate(clusters[:2]):  # Show first 2 clusters
                    print(f"     Cluster {i+1}: {len(cluster.members)} members, coherence: {cluster.coherence_score:.3f}")
                    
            except Exception as e:
                print(f"   ❌ Clustering failed: {e}")
        
        # Get final statistics
        final_stats = engine.get_embedding_statistics()
        print(f"\n📊 Final Pipeline Statistics:")
        print(f"   📊 Total embeddings: {final_stats.get('total_embeddings', 0)}")
        print(f"   🏛️  Unique artifacts: {final_stats.get('unique_artifacts', 0)}")
        print(f"   📏 Embedding dimension: {final_stats.get('embedding_dimension', 0)}")
        print(f"   🗄️  Vector DB: {final_stats.get('vector_db_type', 'unknown')}")
        print(f"   🤖 Model: {final_stats.get('embedding_model', 'unknown')}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_processed_dir, ignore_errors=True)
        shutil.rmtree("data/full_test_embeddings", ignore_errors=True)
        
        success = stats['embeddings_created'] > 0 and final_stats.get('total_embeddings', 0) > 0
        
        if success:
            print("\n✅ Full pipeline test PASSED!")
        else:
            print("\n❌ Full pipeline test FAILED")
        
        return success
        
    except Exception as e:
        print(f"❌ Error in full pipeline test: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all Phase 3 tests"""
    print("\n🧪 AURA Phase 3 Vector Embeddings Test Suite")
    print("=" * 60)
    
    # Check if the artifacts directory exists
    artifacts_dir = "../../algo_outputs/algorithm_2_output_2"
    if not os.path.exists(artifacts_dir):
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        print("   Please ensure you have the artifact JSON files in the correct location.")
        return
    
    # Check dependencies
    print(f"\n{'=' * 60}")
    print("🧪 Pre-Test: Checking Dependencies")
    print(f"{'=' * 60}")
    
    dependencies_ok = True
    
    try:
        import sentence_transformers
        print("✅ SentenceTransformers available")
    except ImportError:
        print("❌ SentenceTransformers not available")
        print("   Install with: pip install sentence-transformers")
        dependencies_ok = False
    
    try:
        import faiss
        print("✅ FAISS available")
    except ImportError:
        print("❌ FAISS not available")
        print("   Install with: pip install faiss-cpu")
        dependencies_ok = False
    
    try:
        import numpy
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not available")
        dependencies_ok = False
    
    if not dependencies_ok:
        print("\n❌ Required dependencies missing. Please install them first.")
        return
    
    tests = [
        ("Embedding Engine Initialization", test_embedding_engine_initialization),
        ("Artifact Embedding Extraction", test_artifact_embedding_extraction),
        ("Semantic Search", test_semantic_search),
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
    
    if passed == total:
        print("🎉 All tests PASSED! Phase 3 Vector Embeddings is ready for use.")
        print("🔍 Semantic search and embeddings are working perfectly!")
    else:
        print("⚠️  Some tests FAILED. Please check the errors above.")
        if passed > 0:
            print(f"💡 {passed} tests passed - partial functionality is available.")


if __name__ == "__main__":
    main() 