import json
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ReproducibilityEvaluationAgent:
    """
    Evaluates artifact reproducibility based on ICSE 2025 criteria:
    - Reusability: Documented and structured to facilitate reuse and repurposing
    - Setup Instructions: Clarity and completeness for executable artifacts
    - Usage Instructions: Clarity for replicating main results
    - Result Replication: Ability to reproduce the main results of the paper
    """
    
    def __init__(self, kg_agent):
        self.kg_agent = kg_agent
        self.artifact = self._load_artifact()
        
    def _load_artifact(self) -> Dict:
        """Load artifact data from JSON file."""
        with open(self.kg_agent.artifact_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate reproducibility of the artifact.
        
        Returns:
            Dict with score, justification, and evidence
        """
        logger.info("Evaluating artifact reproducibility...")
        
        evidence = []
        score_components = []
        
        # 1. Check for reusability and structure
        reusability_score, reusability_evidence = self._evaluate_reusability()
        score_components.append(reusability_score)
        evidence.extend(reusability_evidence)
        
        # 2. Check for setup instructions clarity
        setup_score, setup_evidence = self._evaluate_setup_clarity()
        score_components.append(setup_score)
        evidence.extend(setup_evidence)
        
        # 3. Check for usage instructions clarity
        usage_score, usage_evidence = self._evaluate_usage_clarity()
        score_components.append(usage_score)
        evidence.extend(usage_evidence)
        
        # 4. Check for result replication capability
        replication_score, replication_evidence = self._evaluate_result_replication()
        score_components.append(replication_score)
        evidence.extend(replication_evidence)
        
        # Calculate overall score (weighted average)
        overall_score = sum(score_components) / len(score_components)
        
        # Generate justification
        justification = self._generate_justification(reusability_score, setup_score, 
                                                   usage_score, replication_score)
        
        return {
            "score": overall_score,
            "justification": justification,
            "evidence": evidence,
            "components": {
                "reusability": reusability_score,
                "setup_clarity": setup_score,
                "usage_clarity": usage_score,
                "result_replication": replication_score
            }
        }
    
    def _evaluate_reusability(self) -> tuple[float, List[str]]:
        """Evaluate reusability and structure of the artifact."""
        evidence = []
        score = 0.0
        
        # Check for modular code structure
        if self._has_modular_structure():
            evidence.append("Modular code structure found")
            score += 0.3
        else:
            evidence.append("Code structure may not be modular")
        
        # Check for configuration management
        if self._has_configuration_management():
            evidence.append("Configuration management found")
            score += 0.2
        else:
            evidence.append("Configuration management could be improved")
        
        # Check for parameterization
        if self._has_parameterization():
            evidence.append("Code is parameterized for reuse")
            score += 0.2
        else:
            evidence.append("Code lacks parameterization")
        
        # Check for documentation quality for reuse
        if self._has_reuse_documentation():
            evidence.append("Documentation supports reuse")
            score += 0.2
        else:
            evidence.append("Documentation may not support reuse")
        
        # Check for clean interfaces
        if self._has_clean_interfaces():
            evidence.append("Clean interfaces found")
            score += 0.1
        else:
            evidence.append("Interfaces could be cleaner")
        
        return min(score, 1.0), evidence
    
    def _evaluate_setup_clarity(self) -> tuple[float, List[str]]:
        """Evaluate clarity and completeness of setup instructions."""
        evidence = []
        score = 0.0
        
        # Check for step-by-step setup instructions
        if self._has_step_by_step_setup():
            evidence.append("Step-by-step setup instructions found")
            score += 0.3
        else:
            evidence.append("Step-by-step setup instructions missing")
        
        # Check for environment setup
        if self._has_environment_setup():
            evidence.append("Environment setup instructions found")
            score += 0.2
        else:
            evidence.append("Environment setup instructions missing")
        
        # Check for dependency installation
        if self._has_dependency_installation():
            evidence.append("Dependency installation instructions found")
            score += 0.2
        else:
            evidence.append("Dependency installation instructions missing")
        
        # Check for troubleshooting section
        if self._has_troubleshooting():
            evidence.append("Troubleshooting section found")
            score += 0.2
        else:
            evidence.append("Troubleshooting section missing")
        
        # Check for verification of setup
        if self._has_setup_verification():
            evidence.append("Setup verification instructions found")
            score += 0.1
        else:
            evidence.append("Setup verification instructions missing")
        
        return min(score, 1.0), evidence
    
    def _evaluate_usage_clarity(self) -> tuple[float, List[str]]:
        """Evaluate clarity of usage instructions."""
        evidence = []
        score = 0.0
        
        # Check for basic usage examples
        if self._has_basic_usage_examples():
            evidence.append("Basic usage examples found")
            score += 0.3
        else:
            evidence.append("Basic usage examples missing")
        
        # Check for detailed commands
        if self._has_detailed_commands():
            evidence.append("Detailed commands provided")
            score += 0.2
        else:
            evidence.append("Detailed commands missing")
        
        # Check for parameter explanations
        if self._has_parameter_explanations():
            evidence.append("Parameter explanations found")
            score += 0.2
        else:
            evidence.append("Parameter explanations missing")
        
        # Check for example outputs
        if self._has_example_outputs():
            evidence.append("Example outputs provided")
            score += 0.2
        else:
            evidence.append("Example outputs missing")
        
        # Check for common use cases
        if self._has_common_use_cases():
            evidence.append("Common use cases documented")
            score += 0.1
        else:
            evidence.append("Common use cases not documented")
        
        return min(score, 1.0), evidence
    
    def _evaluate_result_replication(self) -> tuple[float, List[str]]:
        """Evaluate ability to replicate main results."""
        evidence = []
        score = 0.0
        
        # Check for result replication instructions
        if self._has_replication_instructions():
            evidence.append("Result replication instructions found")
            score += 0.3
        else:
            evidence.append("Result replication instructions missing")
        
        # Check for expected outputs
        if self._has_expected_outputs():
            evidence.append("Expected outputs documented")
            score += 0.2
        else:
            evidence.append("Expected outputs not documented")
        
        # Check for performance benchmarks
        if self._has_performance_benchmarks():
            evidence.append("Performance benchmarks found")
            score += 0.2
        else:
            evidence.append("Performance benchmarks missing")
        
        # Check for validation scripts
        if self._has_validation_scripts():
            evidence.append("Validation scripts found")
            score += 0.2
        else:
            evidence.append("Validation scripts missing")
        
        # Check for comparison with paper results
        if self._has_paper_comparison():
            evidence.append("Comparison with paper results found")
            score += 0.1
        else:
            evidence.append("Comparison with paper results missing")
        
        return min(score, 1.0), evidence
    
    def _has_modular_structure(self) -> bool:
        """Check if artifact has modular code structure."""
        structure = self.artifact.get("repository_structure", [])
        
        # Check for common modular patterns
        has_modules = any("module" in entry.get("path", "").lower() for entry in structure)
        has_packages = any("__init__.py" in entry.get("name", "") for entry in structure)
        has_separate_functions = len(self.artifact.get("code_files", [])) > 3
        
        return has_modules or has_packages or has_separate_functions
    
    def _has_configuration_management(self) -> bool:
        """Check if artifact has configuration management."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["config", "settings", "params", "parameters"]):
                return True
        return False
    
    def _has_parameterization(self) -> bool:
        """Check if code is parameterized for reuse."""
        for code_file in self.artifact.get("code_files", []):
            content = code_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            # Look for parameter patterns
            param_patterns = [
                r'def\s+\w+\s*\([^)]*\)',  # Function definitions
                r'class\s+\w+\s*\([^)]*\)',  # Class definitions
                r'args\.', r'kwargs\.',  # Argument handling
                r'config\[', r'settings\['  # Configuration access
            ]
            if any(re.search(pattern, content) for pattern in param_patterns):
                return True
        return False
    
    def _has_reuse_documentation(self) -> bool:
        """Check if documentation supports reuse."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            reuse_keywords = [
                "reuse", "reusable", "modular", "component", "library", "api",
                "interface", "extend", "customize", "configure"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in reuse_keywords):
                return True
        return False
    
    def _has_clean_interfaces(self) -> bool:
        """Check if artifact has clean interfaces."""
        for code_file in self.artifact.get("code_files", []):
            content = code_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            # Look for clean interface patterns
            interface_patterns = [
                r'def\s+\w+\s*\([^)]*\):\s*"""[^"]*"""',  # Documented functions
                r'class\s+\w+\s*\([^)]*\):\s*"""[^"]*"""',  # Documented classes
                r'@property', r'@staticmethod', r'@classmethod'  # Decorators
            ]
            if any(re.search(pattern, content) for pattern in interface_patterns):
                return True
        return False
    
    def _has_step_by_step_setup(self) -> bool:
        """Check if artifact has step-by-step setup instructions."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            # Look for numbered or bulleted steps
            step_patterns = [
                r'\d+\.\s+',  # Numbered steps
                r'[-*]\s+',   # Bullet points
                r'step\s+\d+',  # "Step X" format
                r'first\s+', r'second\s+', r'third\s+'  # Ordinal words
            ]
            content_lower = content.lower()
            if any(re.search(pattern, content_lower) for pattern in step_patterns):
                return True
        return False
    
    def _has_environment_setup(self) -> bool:
        """Check if artifact has environment setup instructions."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            env_keywords = [
                "environment", "virtual", "venv", "conda", "docker", "container",
                "python", "java", "node", "npm", "pip", "install"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in env_keywords):
                return True
        return False
    
    def _has_dependency_installation(self) -> bool:
        """Check if artifact has dependency installation instructions."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            dep_keywords = [
                "install", "dependency", "requirement", "package", "library",
                "pip install", "npm install", "maven", "gradle"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in dep_keywords):
                return True
        return False
    
    def _has_troubleshooting(self) -> bool:
        """Check if artifact has troubleshooting section."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            trouble_keywords = [
                "troubleshoot", "troubleshooting", "faq", "common issues",
                "problems", "errors", "debug", "fix"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in trouble_keywords):
                return True
        return False
    
    def _has_setup_verification(self) -> bool:
        """Check if artifact has setup verification instructions."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            verify_keywords = [
                "verify", "test", "check", "validate", "confirm", "run test",
                "hello world", "example", "demo"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in verify_keywords):
                return True
        return False
    
    def _has_basic_usage_examples(self) -> bool:
        """Check if artifact has basic usage examples."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            example_keywords = [
                "example", "usage", "how to", "basic", "simple", "quick start",
                "getting started", "demo", "tutorial"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in example_keywords):
                return True
        return False
    
    def _has_detailed_commands(self) -> bool:
        """Check if artifact has detailed commands."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            # Look for command patterns
            command_patterns = [
                r'python\s+\w+\.py', r'java\s+\w+', r'node\s+\w+',
                r'\./[\w-]+', r'bash\s+[\w-]+', r'curl\s+'
            ]
            if any(re.search(pattern, content) for pattern in command_patterns):
                return True
        return False
    
    def _has_parameter_explanations(self) -> bool:
        """Check if artifact has parameter explanations."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            param_keywords = [
                "parameter", "argument", "option", "flag", "--", "-",
                "input", "output", "config", "setting"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in param_keywords):
                return True
        return False
    
    def _has_example_outputs(self) -> bool:
        """Check if artifact has example outputs."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            output_keywords = [
                "output", "result", "example", "sample", "expected",
                "should see", "will produce", "generates"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in output_keywords):
                return True
        return False
    
    def _has_common_use_cases(self) -> bool:
        """Check if artifact has common use cases documented."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            use_case_keywords = [
                "use case", "scenario", "example", "case study", "application",
                "workflow", "pipeline", "process"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in use_case_keywords):
                return True
        return False
    
    def _has_replication_instructions(self) -> bool:
        """Check if artifact has result replication instructions."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            replication_keywords = [
                "reproduce", "replicate", "results", "experiments", "evaluation",
                "benchmark", "performance", "metrics", "paper results"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in replication_keywords):
                return True
        return False
    
    def _has_expected_outputs(self) -> bool:
        """Check if artifact has expected outputs documented."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            expected_keywords = [
                "expected", "should", "will", "output", "result", "accuracy",
                "precision", "recall", "f1", "performance"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in expected_keywords):
                return True
        return False
    
    def _has_performance_benchmarks(self) -> bool:
        """Check if artifact has performance benchmarks."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["benchmark", "performance", "speed", "timing", "eval"]):
                return True
        return False
    
    def _has_validation_scripts(self) -> bool:
        """Check if artifact has validation scripts."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["validate", "validation", "verify", "test", "check"]):
                return True
        return False
    
    def _has_paper_comparison(self) -> bool:
        """Check if artifact has comparison with paper results."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            comparison_keywords = [
                "paper", "publication", "table", "figure", "comparison",
                "baseline", "state-of-the-art", "sota", "previous work"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in comparison_keywords):
                return True
        return False
    
    def _generate_justification(self, reusability_score: float, setup_score: float,
                              usage_score: float, replication_score: float) -> str:
        """Generate justification based on component scores."""
        if reusability_score >= 0.8 and setup_score >= 0.8 and usage_score >= 0.8:
            return "Excellent reproducibility: artifact is highly reusable with clear setup and usage instructions."
        elif reusability_score >= 0.6 and setup_score >= 0.6 and usage_score >= 0.6:
            return "Good reproducibility: artifact is reusable with adequate instructions but could be clearer."
        elif reusability_score >= 0.4 and setup_score >= 0.4 and usage_score >= 0.4:
            return "Fair reproducibility: basic reproducibility elements present but significant improvements needed."
        else:
            return "Poor reproducibility: artifact lacks essential reproducibility elements and clear instructions."
