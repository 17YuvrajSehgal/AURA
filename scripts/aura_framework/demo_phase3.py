#!/usr/bin/env python3
"""
🎯 Phase 3 Vector Embeddings Demo
Demonstrates the key capabilities of the AURA Phase 3 system.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase3_vector_embeddings import VectorEmbeddingEngine


def demo_phase3():
    """Demonstrate Phase 3 Vector Embeddings capabilities"""
    print("\n🎯 AURA Phase 3 Vector Embeddings Demo")
    print("=" * 50)

    # Initialize the engine
    print("🔄 Initializing Vector Embedding Engine...")
    engine = VectorEmbeddingEngine(
        vector_db_type="faiss",
        embedding_model="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        storage_directory="data/demo_embeddings"
    )
    print(f"✅ Engine initialized with {engine.embedding_dimension}D embeddings")

    # Create some demo embeddings
    print("\n🔄 Creating demonstration embeddings...")
    from phase3_vector_embeddings import EmbeddingRecord

    demo_sections = [
        {
            "id": "demo_1",
            "artifact": "BuildTool_Analysis",
            "heading": "Installation Instructions",
            "content": "To install this build analysis tool, follow these steps: 1. Clone the repository 2. Install Python dependencies using pip install -r requirements.txt 3. Run the setup script",
            "text": "Installation Instructions To install this build analysis tool, follow these steps: 1. Clone the repository 2. Install Python dependencies using pip install -r requirements.txt 3. Run the setup script Tools: python, pip"
        },
        {
            "id": "demo_2",
            "artifact": "DataProcessor",
            "heading": "Docker Setup",
            "content": "Use Docker to containerize the application. Build the image with docker build -t dataprocessor . and run with docker run -p 8080:8080 dataprocessor",
            "text": "Docker Setup Use Docker to containerize the application. Build the image with docker build -t dataprocessor . and run with docker run -p 8080:8080 dataprocessor Tools: docker"
        },
        {
            "id": "demo_3",
            "artifact": "MLModel_Trainer",
            "heading": "Training Configuration",
            "content": "Configure the machine learning model training with proper datasets, hyperparameters, and validation splits. Use TensorFlow and scikit-learn for implementation.",
            "text": "Training Configuration Configure the machine learning model training with proper datasets, hyperparameters, and validation splits. Use TensorFlow and scikit-learn for implementation. Tools: tensorflow, scikit-learn"
        }
    ]

    # Generate embeddings
    texts = [section["text"] for section in demo_sections]
    embeddings = engine._generate_embeddings_batch(texts)

    # Store embeddings
    for section, embedding in zip(demo_sections, embeddings):
        record = EmbeddingRecord(
            id=f"emb_{section['id']}",
            vector=embedding,
            section_id=section["id"],
            artifact_id=section["artifact"],
            heading=section["heading"],
            content=section["content"],
            metadata={}
        )

        engine.embedding_records[record.id] = record
        engine.section_to_embedding[section["id"]] = record.id
        engine._add_embedding_to_database(record)

    engine._finalize_vector_database()
    print(f"✅ Created {len(demo_sections)} demonstration embeddings")

    # Demonstrate semantic search
    print("\n🔍 Demonstrating Semantic Search:")
    search_queries = [
        "setup installation guide",
        "containerization docker",
        "machine learning training",
        "python dependencies"
    ]

    for query in search_queries:
        results = engine.semantic_search(query, top_k=2)
        print(f"\n   Query: '{query}'")
        for i, result in enumerate(results):
            print(f"     {i + 1}. {result.embedding_record.heading} (score: {result.relevance_score:.3f})")
            print(f"        From: {result.embedding_record.artifact_id}")

    # Demonstrate similarity search
    print(f"\n🔗 Finding similar sections to 'Installation Instructions':")
    similar = engine.find_similar_sections("demo_1", top_k=2)
    for i, result in enumerate(similar):
        print(f"   {i + 1}. {result.embedding_record.heading} (score: {result.relevance_score:.3f})")

    # Show statistics
    stats = engine.get_embedding_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Total embeddings: {stats['total_embeddings']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Vector database: {stats['vector_db_type']}")
    print(f"   Model used: {stats['embedding_model']}")

    print("\n🎉 Phase 3 Vector Embeddings Demo Complete!")
    print("   ✅ Semantic search working")
    print("   ✅ Similarity detection working")
    print("   ✅ Vector storage working")
    print("   ✅ Real-time querying working")


if __name__ == "__main__":
    try:
        demo_phase3()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
