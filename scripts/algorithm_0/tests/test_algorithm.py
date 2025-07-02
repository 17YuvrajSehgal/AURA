#!/usr/bin/env python3
"""
Test script for Conference-Specific Algorithm 1.
"""

import os
import sys
import tempfile

from scripts.algorithm_0 import ConferenceProfileManager, ConferenceSpecificAlgorithm1, Config, setup_logging

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_conference_profiles():
    """Test conference profile functionality."""
    print("🧪 Testing Conference Profiles...")

    profile_manager = ConferenceProfileManager()

    # Test basic functionality
    assert len(profile_manager.get_all_conferences()) > 0
    assert len(profile_manager.get_all_categories()) > 0

    # Test specific conference
    icse_profile = profile_manager.get_conference_profile("ICSE")
    assert icse_profile["category"] == "software_engineering"
    assert "reproducibility" in icse_profile["emphasis_weights"]
    assert len(icse_profile["domain_keywords"]) > 0

    print("✅ Conference profiles test passed")


def test_algorithm_initialization():
    """Test algorithm initialization."""
    print("🧪 Testing Algorithm Initialization...")

    try:
        algorithm = ConferenceSpecificAlgorithm1()
        assert algorithm.sentence_model is not None
        assert algorithm.kw_model is not None
        assert algorithm.profile_manager is not None

        print("✅ Algorithm initialization test passed")
    except Exception as e:
        print(f"❌ Algorithm initialization failed: {e}")
        return False

    return True


def test_metadata_extraction():
    """Test conference metadata extraction from filenames."""
    print("🧪 Testing Metadata Extraction...")

    algorithm = ConferenceSpecificAlgorithm1()

    # Create test file paths
    test_files = [
        "/fake/path/13_icse_2025.md",
        "/fake/path/21_sigmod_2024.md",
        "/fake/path/6_chi_2024.md"
    ]

    metadata = algorithm.extract_conference_metadata(test_files)

    assert len(metadata) == 3
    assert metadata["13_icse_2025.md"]["conference_name"] == "ICSE"
    assert metadata["21_sigmod_2024.md"]["conference_name"] == "SIGMOD"
    assert metadata["6_chi_2024.md"]["conference_name"] == "CHI"

    print("✅ Metadata extraction test passed")


def test_full_extraction_with_mock_data():
    """Test full extraction pipeline with mock data."""
    print("🧪 Testing Full Extraction Pipeline...")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # Create mock guideline files
        mock_guidelines = {
            "13_icse_2025.md": """
            # ICSE 2025 Artifact Evaluation Guidelines
            
            Artifacts should be reproducible and well-documented. 
            The code should include comprehensive tests and clear installation instructions.
            Experimental evaluation should demonstrate the claims made in the paper.
            All software artifacts must be publicly accessible and functional.
            Documentation should include README files with setup and usage instructions.
            """,

            "21_sigmod_2024.md": """
            # SIGMOD 2024 Artifact Guidelines
            
            Database systems and algorithms should demonstrate scalability and performance.
            Experimental evaluation must include benchmarks on standard datasets.
            Query processing implementations should be optimized and well-tested.
            Data management artifacts require thorough performance analysis.
            Reproducibility packages should include all necessary data and scripts.
            """
        }

        # Write mock files
        for filename, content in mock_guidelines.items():
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)

        try:
            # Initialize algorithm
            algorithm = ConferenceSpecificAlgorithm1()

            # Test single conference extraction
            results = algorithm.run_conference_specific_extraction(
                input_dir=input_dir,
                output_dir=output_dir,
                target_conference="ICSE"
            )

            # Verify results
            assert "ICSE" in results
            assert "criteria_dataframe" in results["ICSE"]
            assert "conference_profile" in results["ICSE"]
            assert "saved_files" in results["ICSE"]

            # Check criteria dataframe
            df = results["ICSE"]["criteria_dataframe"]
            assert len(df) > 0
            assert "dimension" in df.columns
            assert "keywords" in df.columns
            assert "normalized_weight" in df.columns

            # Verify output files were created
            saved_files = results["ICSE"]["saved_files"]
            assert len(saved_files) > 0

            for file_type, file_path in saved_files.items():
                assert os.path.exists(file_path)
                print(f"  ✓ Created {file_type}: {os.path.basename(file_path)}")

            print("✅ Full extraction pipeline test passed")
            return True

        except Exception as e:
            print(f"❌ Full extraction test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_config_functionality():
    """Test configuration functionality."""
    print("🧪 Testing Configuration...")

    config = Config()

    # Test default values
    assert config.semantic_similarity_threshold > 0
    assert config.keyword_expansion_top_n > 0
    assert len(config.default_dimensions) > 0

    # Test configuration update
    original_threshold = config.semantic_similarity_threshold
    config.update_config(semantic_similarity_threshold=0.9)
    assert config.semantic_similarity_threshold == 0.9
    assert config.semantic_similarity_threshold != original_threshold

    print("✅ Configuration test passed")


def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Conference-Specific Algorithm 1 Tests")
    print("=" * 60)

    # Setup logging for tests
    setup_logging("INFO")

    tests = [
        test_conference_profiles,
        test_algorithm_initialization,
        test_metadata_extraction,
        test_config_functionality,
        test_full_extraction_with_mock_data
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Algorithm is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
