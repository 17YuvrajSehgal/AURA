#!/usr/bin/env python3
"""
Test script for the AURA Framework.

This script tests the basic functionality of the AURA framework components
using a real artifact JSON file.

Usage (as import):
    from test_framework import main
    main(artifact_path, skip_evaluation=False)

Usage (as script):
    python test_framework.py
    (prints usage instructions)
"""

import json
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from aura_framework import AURAFramework, CriterionScore, ArtifactEvaluationResult
        print("✓ AURAFramework imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AURAFramework: {e}")
        return False
    
    try:
        from agents.accessibility_evaluation_agent import AccessibilityEvaluationAgent
        print("✓ AccessibilityEvaluationAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AccessibilityEvaluationAgent: {e}")
        return False
    
    try:
        from agents.documentation_evaluation_agent import DocumentationEvaluationAgent
        print("✓ DocumentationEvaluationAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DocumentationEvaluationAgent: {e}")
        return False
    
    try:
        from agents.experimental_evaluation_agent import ExperimentalEvaluationAgent
        print("✓ ExperimentalEvaluationAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ExperimentalEvaluationAgent: {e}")
        return False
    
    try:
        from agents.functionality_evaluation_agent import FunctionalityEvaluationAgent
        print("✓ FunctionalityEvaluationAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FunctionalityEvaluationAgent: {e}")
        return False
    
    try:
        from agents.reproducibility_evaluation_agent import ReproducibilityEvaluationAgent
        print("✓ ReproducibilityEvaluationAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ReproducibilityEvaluationAgent: {e}")
        return False
    
    try:
        from agents.usability_evaluation_agent import UsabilityEvaluationAgent
        print("✓ UsabilityEvaluationAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import UsabilityEvaluationAgent: {e}")
        return False
    
    try:
        from agents.repository_knowledge_graph_agent import RepositoryKnowledgeGraphAgent
        print("✓ RepositoryKnowledgeGraphAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import RepositoryKnowledgeGraphAgent: {e}")
        return False
    
    return True

def test_pydantic_models():
    """Test Pydantic model creation and validation."""
    print("\nTesting Pydantic models...")
    
    try:
        from aura_framework import CriterionScore, ArtifactEvaluationResult
        
        # Test CriterionScore
        criterion = CriterionScore(
            dimension="test",
            raw_score=5.0,
            normalized_weight=0.2,
            llm_evaluated_score=0.8,
            justification="Test justification",
            evidence=["test evidence"]
        )
        print("✓ CriterionScore created successfully")
        
        # Test ArtifactEvaluationResult
        result = ArtifactEvaluationResult(
            criteria_scores=[criterion],
            total_weighted_score=0.8,
            acceptance_prediction=True,
            overall_justification="Test overall justification",
            recommendations=["Test recommendation"]
        )
        print("✓ ArtifactEvaluationResult created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create Pydantic models: {e}")
        return False

def test_framework_initialization(artifact_path):
    """Test framework initialization with the provided artifact file."""
    print(f"\nTesting framework initialization with {artifact_path}...")
    
    try:
        # Verify artifact file exists and is valid JSON
        if not os.path.exists(artifact_path):
            print(f"✗ Artifact file not found: {artifact_path}")
            return False
        
        # Test JSON parsing
        try:
            with open(artifact_path, 'r') as f:
                artifact_data = json.load(f)
            print("✓ Artifact JSON file is valid")
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON in artifact file: {e}")
            return False
        
        # Initialize framework
        from aura_framework import AURAFramework
        framework = AURAFramework(artifact_path)
        print("✓ Framework initialized successfully")
        
        # Clean up
        framework.close()
        
        return True
    except Exception as e:
        print(f"✗ Failed to initialize framework: {e}")
        return False

def test_agent_creation(artifact_path):
    """Test individual agent creation with the provided artifact file."""
    print(f"\nTesting agent creation with {artifact_path}...")
    
    try:
        # Test knowledge graph agent
        from agents.repository_knowledge_graph_agent import RepositoryKnowledgeGraphAgent
        kg_agent = RepositoryKnowledgeGraphAgent(artifact_path, clear_existing=True)
        print("✓ RepositoryKnowledgeGraphAgent created successfully")
        
        # Test evaluation agents
        from agents.accessibility_evaluation_agent import AccessibilityEvaluationAgent
        accessibility_agent = AccessibilityEvaluationAgent(kg_agent)
        print("✓ AccessibilityEvaluationAgent created successfully")
        
        from agents.documentation_evaluation_agent import DocumentationEvaluationAgent
        documentation_agent = DocumentationEvaluationAgent(kg_agent)
        print("✓ DocumentationEvaluationAgent created successfully")
        
        from agents.experimental_evaluation_agent import ExperimentalEvaluationAgent
        experimental_agent = ExperimentalEvaluationAgent(kg_agent)
        print("✓ ExperimentalEvaluationAgent created successfully")
        
        from agents.functionality_evaluation_agent import FunctionalityEvaluationAgent
        functionality_agent = FunctionalityEvaluationAgent(kg_agent)
        print("✓ FunctionalityEvaluationAgent created successfully")
        
        from agents.reproducibility_evaluation_agent import ReproducibilityEvaluationAgent
        reproducibility_agent = ReproducibilityEvaluationAgent(kg_agent)
        print("✓ ReproducibilityEvaluationAgent created successfully")
        
        from agents.usability_evaluation_agent import UsabilityEvaluationAgent
        usability_agent = UsabilityEvaluationAgent(kg_agent)
        print("✓ UsabilityEvaluationAgent created successfully")
        
        # Clean up
        kg_agent.close()
        
        return True
    except Exception as e:
        print(f"✗ Failed to create agents: {e}")
        return False

def test_full_evaluation(artifact_path):
    """Test a complete evaluation run with the provided artifact file."""
    print(f"\nTesting full evaluation with {artifact_path}...")
    
    try:
        from aura_framework import AURAFramework
        
        # Initialize framework
        framework = AURAFramework(artifact_path)
        print("✓ Framework initialized for evaluation")
        
        # Run evaluation
        result = framework.evaluate_artifact()
        print("✓ Evaluation completed successfully")
        
        # Print results summary
        print(f"  - Total weighted score: {result.total_weighted_score:.3f}")
        print(f"  - Acceptance prediction: {result.acceptance_prediction}")
        print(f"  - Number of criteria evaluated: {len(result.criteria_scores)}")
        
        # Clean up
        framework.close()
        
        return True
    except Exception as e:
        print(f"✗ Failed to complete evaluation: {e}")
        return False

def main(artifact_path, skip_evaluation=False):
    """Run all tests with the provided artifact file.
    Args:
        artifact_path (str): Path to the artifact JSON file to use for testing.
        skip_evaluation (bool): If True, skip the full evaluation test.
    Returns:
        int: 0 if all tests pass, 1 otherwise.
    """
    print("AURA Framework Test Suite")
    print("=" * 50)
    print(f"Using artifact file: {artifact_path}")
    print("=" * 50)
    
    # Verify an artifact file exists
    if not os.path.exists(artifact_path):
        print(f"Error: Artifact file not found at {artifact_path}")
        print("Please provide a valid path to an artifact JSON file.")
        return 1
    
    tests = [
        ("Import Tests", lambda: test_imports()),
        ("Pydantic Models", lambda: test_pydantic_models()),
        ("Framework Initialization", lambda: test_framework_initialization(artifact_path)),
        ("Agent Creation", lambda: test_agent_creation(artifact_path))
    ]
    
    # Add full evaluation test unless skipped
    if not skip_evaluation:
        tests.append(("Full Evaluation", lambda: test_full_evaluation(artifact_path)))
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The AURA framework is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    print("\nUsage: Import this module and call main(artifact_path, skip_evaluation=False)")
    print("Example:")
    print("    from test_framework import main")
    print("    main('.../../algo_outputs/algorithm_2_output/ml-image-classifier_analysis.json')")
    sys.exit(0)