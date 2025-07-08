"""
Demo: Conference-Aware Artifact Evaluation with AURA

This script demonstrates how to use the enhanced AURA evaluation system 
with conference-specific guidelines and criteria.
"""

import logging
from aura_evaluator import AURAEvaluator, quick_evaluate
from conference_guidelines_loader import conference_loader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_available_conferences():
    """Demonstrate listing available conferences"""
    logger.info("=== Available Conference Guidelines ===")
    
    conferences = conference_loader.get_available_conferences()
    logger.info(f"Found {len(conferences)} conference guidelines:")
    
    for conference in conferences[:10]:  # Show first 10
        summary = conference_loader.get_conference_summary(conference)
        logger.info(f"- {conference}: {summary.get('total_criteria', 0)} criteria")
    
    return conferences


def demo_conference_specific_evaluation():
    """Demonstrate conference-specific artifact evaluation"""
    
    # Demo artifact data
    demo_artifact_data = {
        "artifact_name": "sample_ml_artifact",
        "artifact_path": "/path/to/ml_artifact",
        "repo_size_mb": 25.4,
        "extraction_method": "git_clone",
        "success": True,
        "documentation_files": [
            {
                "path": "README.md",
                "content": [
                    "# Machine Learning Artifact",
                    "This artifact implements a novel neural network architecture.",
                    "## Installation",
                    "pip install -r requirements.txt",
                    "## Usage",
                    "python train.py --dataset cifar10 --epochs 100",
                    "## Reproducibility",
                    "All experiments can be reproduced using the provided Docker container.",
                    "## License",
                    "MIT License - see LICENSE file"
                ]
            },
            {
                "path": "LICENSE",
                "content": ["MIT License", "Copyright (c) 2024 Research Team"]
            }
        ],
        "code_files": [
            {
                "path": "train.py",
                "content": [
                    "#!/usr/bin/env python3",
                    "import argparse",
                    "import torch",
                    "import torch.nn as nn",
                    "from dataset import load_data",
                    "def main():",
                    "    parser = argparse.ArgumentParser()",
                    "    parser.add_argument('--dataset', required=True)",
                    "    parser.add_argument('--epochs', type=int, default=100)",
                    "    args = parser.parse_args()",
                    "    # Training code here"
                ]
            },
            {
                "path": "requirements.txt",
                "content": ["torch>=1.9.0", "torchvision>=0.10.0", "numpy>=1.21.0"]
            },
            {
                "path": "Dockerfile",
                "content": [
                    "FROM python:3.9",
                    "WORKDIR /app",
                    "COPY requirements.txt .",
                    "RUN pip install -r requirements.txt",
                    "COPY . .",
                    "CMD ['python', 'train.py']"
                ]
            }
        ],
        "data_files": [
            {
                "name": "cifar10_data.tar.gz",
                "path": "data/cifar10_data.tar.gz",
                "size_kb": 1024,
                "mime_type": "application/gzip"
            }
        ]
    }
    
    # Test different conferences
    test_conferences = ["ICSE", "ASE", "FSE", "ASPLOS"]
    
    logger.info("=== Conference-Specific Evaluations ===")
    
    results = {}
    
    for conference in test_conferences:
        logger.info(f"\n--- Evaluating for {conference} Conference ---")
        
        try:
            # Initialize evaluator with conference-specific settings
            evaluator = AURAEvaluator(
                use_neo4j=False,  # Simplified for demo
                use_rag=True,
                conference_name=conference
            )
            
            # Run evaluation
            report = evaluator.evaluate_artifact_from_data(
                artifact_data=demo_artifact_data,
                dimensions=["accessibility", "documentation", "functionality"],  # Limited for demo
                save_results=False
            )
            
            # Store results
            results[conference] = {
                "overall_rating": report["overall_rating"],
                "dimension_scores": report["dimension_scores"],
                "conference_info": conference_loader.get_conference_summary(conference)
            }
            
            # Print summary
            logger.info(f"Overall Rating for {conference}: {report['overall_rating']:.2f}/5.0")
            for dimension, score in report["dimension_scores"].items():
                logger.info(f"  {dimension.title()}: {score:.2f}/5.0")
            
            evaluator.close()
            
        except Exception as e:
            logger.error(f"Failed to evaluate for {conference}: {e}")
            results[conference] = {"error": str(e)}
    
    return results


def demo_conference_guidelines_injection():
    """Demonstrate how conference guidelines are injected into prompts"""
    
    logger.info("=== Conference Guidelines Injection Demo ===")
    
    conferences_to_test = ["ICSE", "ASE"]
    dimension = "accessibility"
    
    for conference in conferences_to_test:
        logger.info(f"\n--- {conference} Guidelines for {dimension.upper()} ---")
        
        try:
            guidelines_text = conference_loader.format_conference_guidelines_for_prompt(
                conference_name=conference,
                dimension=dimension
            )
            
            logger.info("Guidelines that would be injected into prompt:")
            logger.info("-" * 60)
            logger.info(guidelines_text[:500] + "..." if len(guidelines_text) > 500 else guidelines_text)
            logger.info("-" * 60)
            
        except Exception as e:
            logger.error(f"Failed to get guidelines for {conference}: {e}")


def demo_comparison_across_conferences():
    """Compare how the same artifact scores across different conferences"""
    
    logger.info("=== Cross-Conference Comparison ===")
    
    # Simple artifact for comparison
    simple_artifact = {
        "artifact_name": "simple_python_tool",
        "artifact_path": "/path/to/tool",
        "repo_size_mb": 5.2,
        "documentation_files": [
            {
                "path": "README.md",
                "content": ["# Simple Tool", "A basic Python utility", "## Usage", "python tool.py"]
            }
        ],
        "code_files": [
            {
                "path": "tool.py",
                "content": ["#!/usr/bin/env python3", "print('Hello, World!')"]
            }
        ]
    }
    
    conferences = ["ICSE", "ASE", "FSE"]
    comparison_results = {}
    
    for conference in conferences:
        try:
            # Quick evaluation for comparison
            report = quick_evaluate(
                artifact_json_path="demo_simple_artifact.json",  # Would be created from simple_artifact
                conference_name=conference,
                dimensions=["accessibility", "documentation"],
                use_neo4j=False,
                use_rag=False  # Simplified for demo
            )
            
            comparison_results[conference] = {
                "overall": report.get("overall_rating", 0.0),
                "accessibility": report.get("dimension_scores", {}).get("accessibility", 0.0),
                "documentation": report.get("dimension_scores", {}).get("documentation", 0.0)
            }
            
        except Exception as e:
            logger.warning(f"Skipping {conference} due to error: {e}")
            comparison_results[conference] = {"error": str(e)}
    
    # Display comparison
    logger.info("\nConference Comparison Results:")
    logger.info("Conference | Overall | Accessibility | Documentation")
    logger.info("-" * 55)
    
    for conference, scores in comparison_results.items():
        if "error" not in scores:
            logger.info(f"{conference:10} | {scores['overall']:7.2f} | {scores['accessibility']:13.2f} | {scores['documentation']:13.2f}")
        else:
            logger.info(f"{conference:10} | ERROR   | ERROR         | ERROR")
    
    return comparison_results


def main():
    """Run all conference-aware evaluation demos"""
    
    logger.info("🎯 AURA Conference-Aware Evaluation Demo")
    logger.info("=" * 50)
    
    try:
        # Demo 1: Show available conferences
        conferences = demo_available_conferences()
        
        # Demo 2: Show guidelines injection
        demo_conference_guidelines_injection()
        
        # Demo 3: Conference-specific evaluation
        if conferences:
            evaluation_results = demo_conference_specific_evaluation()
            
            # Demo 4: Cross-conference comparison
            # comparison_results = demo_comparison_across_conferences()
        
        logger.info("\n✅ All demos completed successfully!")
        logger.info("\nKey features demonstrated:")
        logger.info("- Conference guidelines loading and parsing")
        logger.info("- Guidelines injection into evaluation prompts")
        logger.info("- Conference-specific artifact evaluation")
        logger.info("- Cross-conference comparison capabilities")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 