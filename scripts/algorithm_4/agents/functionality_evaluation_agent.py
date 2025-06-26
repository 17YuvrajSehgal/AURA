import json
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FunctionalityEvaluationAgent:
    """
    Evaluates artifact functionality based on ICSE 2025 criteria:
    - Functionality: Documented, consistent, complete, and exercisable
    - Executability: Can be executed successfully
    - Verification Evidence: Evidence of verification and validation
    - Executable Artifacts: Installation packages and Docker/VM images
    """
    
    def __init__(self, kg_agent, llm_evaluator=None):
        self.kg_agent = kg_agent
        self.llm_evaluator = llm_evaluator
        self.artifact = self._load_artifact()
        
    def _load_artifact(self) -> Dict:
        """Load artifact data from JSON file."""
        with open(self.kg_agent.artifact_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate functionality of the artifact (rule-based + LLM augmentation).
        Returns:
            Dict with score, justification, and evidence
        """
        logger.info("Evaluating artifact functionality...")
        evidence = []
        score_components = []
        
        # 1. Check for executability and main entry points
        executability_score, executability_evidence = self._evaluate_executability()
        score_components.append(executability_score)
        evidence.extend(executability_evidence)
        
        # 2. Check for consistency and completeness
        consistency_score, consistency_evidence = self._evaluate_consistency_completeness()
        score_components.append(consistency_score)
        evidence.extend(consistency_evidence)
        
        # 3. Check for verification and validation evidence
        verification_score, verification_evidence = self._evaluate_verification_evidence()
        score_components.append(verification_score)
        evidence.extend(verification_evidence)
        
        # 4. Check for executable artifact preparation
        preparation_score, preparation_evidence = self._evaluate_executable_preparation()
        score_components.append(preparation_score)
        evidence.extend(preparation_evidence)
        
        # Calculate overall score (weighted average)
        overall_score = sum(score_components) / len(score_components)
        
        # Generate justification
        justification = self._generate_justification(executability_score, consistency_score, 
                                                   verification_score, preparation_score)
        
        result = {
            "score": overall_score,
            "justification": justification,
            "evidence": evidence,
            "components": {
                "executability": executability_score,
                "consistency_completeness": consistency_score,
                "verification_evidence": verification_score,
                "executable_preparation": preparation_score
            }
        }
        
        # LLM augmentation
        if self.llm_evaluator:
            context = self._get_functionality_context()
            llm_data = self.llm_evaluator.evaluate_dimension(
                "functionality",
                result["score"],
                result["justification"],
                result["evidence"],
                context
            )
            result["score"] = max(result["score"], llm_data.get("revised_score", result["score"]))
            result["justification"] = llm_data.get("revised_justification", result["justification"])
            if llm_data.get("additional_evidence"):
                result["evidence"].extend(llm_data["additional_evidence"])
        
        return result
    
    def _evaluate_executability(self) -> tuple[float, List[str]]:
        """Evaluate if artifact can be executed successfully."""
        evidence = []
        score = 0.0
        
        # Check for main entry points
        main_files = []
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if filename in ["main.py", "app.py", "run.py", "start.py", "index.py"]:
                main_files.append(entry)
        
        if main_files:
            evidence.append(f"Found {len(main_files)} main entry point(s)")
            score += 0.3
        else:
            evidence.append("No clear main entry point found")
        
        # Check for executable scripts
        executable_files = []
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if (filename.endswith(".sh") or filename.endswith(".bat") or 
                filename.endswith(".py") or filename.endswith(".js")):
                executable_files.append(entry)
        
        if executable_files:
            evidence.append(f"Found {len(executable_files)} executable file(s)")
            score += 0.2
        
        # Check for proper imports and dependencies
        if self._has_proper_imports():
            evidence.append("Proper imports and dependencies found")
            score += 0.2
        
        # Check for configuration files
        if self._has_configuration_files():
            evidence.append("Configuration files found")
            score += 0.1
        
        # Check for error handling
        if self._has_error_handling():
            evidence.append("Error handling found in code")
            score += 0.2
        
        return min(score, 1.0), evidence
    
    def _evaluate_consistency_completeness(self) -> tuple[float, List[str]]:
        """Evaluate consistency and completeness of the artifact."""
        evidence = []
        score = 0.0
        
        # Check for complete file structure
        structure = self.artifact.get("repository_structure", [])
        if len(structure) >= 10:  # Reasonable number of files
            evidence.append("Complete file structure found")
            score += 0.2
        else:
            evidence.append("File structure may be incomplete")
        
        # Check for consistent naming conventions
        if self._has_consistent_naming():
            evidence.append("Consistent naming conventions found")
            score += 0.2
        else:
            evidence.append("Inconsistent naming conventions")
        
        # Check for complete documentation
        doc_files = self.artifact.get("documentation_files", [])
        if len(doc_files) >= 2:
            evidence.append("Complete documentation found")
            score += 0.2
        else:
            evidence.append("Documentation may be incomplete")
        
        # Check for code completeness
        code_files = self.artifact.get("code_files", [])
        if len(code_files) >= 3:
            evidence.append("Sufficient code files found")
            score += 0.2
        else:
            evidence.append("Code files may be insufficient")
        
        # Check for logical organization
        if self._has_logical_organization():
            evidence.append("Logical file organization found")
            score += 0.2
        else:
            evidence.append("File organization could be improved")
        
        return min(score, 1.0), evidence
    
    def _evaluate_verification_evidence(self) -> tuple[float, List[str]]:
        """Evaluate verification and validation evidence."""
        evidence = []
        score = 0.0
        
        # Check for test files
        test_files = []
        for entry in self.artifact.get("repository_structure", []):
            if self._is_test_file(entry):
                test_files.append(entry)
        
        if test_files:
            evidence.append(f"Found {len(test_files)} test file(s)")
            score += 0.3
        else:
            evidence.append("No test files found")
        
        # Check for validation scripts
        validation_files = []
        for entry in self.artifact.get("repository_structure", []):
            if self._is_validation_file(entry):
                validation_files.append(entry)
        
        if validation_files:
            evidence.append(f"Found {len(validation_files)} validation file(s)")
            score += 0.2
        
        # Check for verification documentation
        if self._has_verification_documentation():
            evidence.append("Verification documentation found")
            score += 0.2
        
        # Check for quality assurance measures
        if self._has_quality_assurance():
            evidence.append("Quality assurance measures found")
            score += 0.2
        
        # Check for performance benchmarks
        if self._has_performance_benchmarks():
            evidence.append("Performance benchmarks found")
            score += 0.1
        
        return min(score, 1.0), evidence
    
    def _evaluate_executable_preparation(self) -> tuple[float, List[str]]:
        """Evaluate executable artifact preparation."""
        evidence = []
        score = 0.0
        
        # Check for Docker support
        if self._has_docker_support():
            evidence.append("Docker support found")
            score += 0.3
        else:
            evidence.append("No Docker support found")
        
        # Check for installation packages
        if self._has_installation_packages():
            evidence.append("Installation packages found")
            score += 0.2
        
        # Check for virtual environment setup
        if self._has_virtual_environment():
            evidence.append("Virtual environment setup found")
            score += 0.2
        
        # Check for build scripts
        if self._has_build_scripts():
            evidence.append("Build scripts found")
            score += 0.2
        
        # Check for deployment instructions
        if self._has_deployment_instructions():
            evidence.append("Deployment instructions found")
            score += 0.1
        
        return min(score, 1.0), evidence
    
    def _has_proper_imports(self) -> bool:
        """Check if code has proper imports."""
        for code_file in self.artifact.get("code_files", []):
            content = code_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            # Look for import statements
            if re.search(r'^import\s+', content, re.MULTILINE) or re.search(r'^from\s+.*\s+import', content, re.MULTILINE):
                return True
        return False
    
    def _has_configuration_files(self) -> bool:
        """Check if artifact has configuration files."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["config", "settings", "env", "ini", "yaml", "yml", "json"]):
                return True
        return False
    
    def _has_error_handling(self) -> bool:
        """Check if code has error handling."""
        for code_file in self.artifact.get("code_files", []):
            content = code_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            # Look for error handling patterns
            error_patterns = [
                r'try\s*:', r'except\s*:', r'catch\s*\(', r'finally\s*:',
                r'raise\s+', r'throw\s+', r'error', r'exception'
            ]
            if any(re.search(pattern, content, re.IGNORECASE) for pattern in error_patterns):
                return True
        return False
    
    def _has_consistent_naming(self) -> bool:
        """Check if artifact has consistent naming conventions."""
        filenames = [entry.get("name", "") for entry in self.artifact.get("repository_structure", [])]
        
        # Check for consistent case usage
        has_snake_case = any('_' in name for name in filenames)
        has_kebab_case = any('-' in name for name in filenames)
        has_camel_case = any(re.search(r'[a-z][A-Z]', name) for name in filenames)
        
        # If multiple naming conventions are used, it's inconsistent
        conventions = sum([has_snake_case, has_kebab_case, has_camel_case])
        return conventions <= 2  # Allow up to 2 conventions
    
    def _has_logical_organization(self) -> bool:
        """Check if artifact has logical file organization."""
        structure = self.artifact.get("repository_structure", [])
        
        # Check for common organizational patterns
        has_src = any("src" in entry.get("path", "").lower() for entry in structure)
        has_docs = any("doc" in entry.get("path", "").lower() for entry in structure)
        has_tests = any("test" in entry.get("path", "").lower() for entry in structure)
        
        # At least 2 organizational patterns should be present
        patterns = sum([has_src, has_docs, has_tests])
        return patterns >= 2
    
    def _is_test_file(self, entry: Dict) -> bool:
        """Check if file is a test file."""
        filename = entry.get("name", "").lower()
        path = entry.get("path", "").lower()
        
        return (filename.startswith("test") or 
                filename.endswith("_test.py") or 
                "test" in filename or
                "test" in path)
    
    def _is_validation_file(self, entry: Dict) -> bool:
        """Check if file is a validation file."""
        filename = entry.get("name", "").lower()
        return any(keyword in filename for keyword in ["validate", "validation", "verify", "verification"])
    
    def _has_verification_documentation(self) -> bool:
        """Check if artifact has verification documentation."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            verification_keywords = [
                "verify", "verification", "validate", "validation", "test", "testing",
                "quality", "assurance", "correctness", "reliability"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in verification_keywords):
                return True
        return False
    
    def _has_quality_assurance(self) -> bool:
        """Check if artifact has quality assurance measures."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["lint", "flake8", "pylint", "eslint", "checkstyle"]):
                return True
        return False
    
    def _has_performance_benchmarks(self) -> bool:
        """Check if artifact has performance benchmarks."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["benchmark", "performance", "speed", "timing"]):
                return True
        return False
    
    def _has_docker_support(self) -> bool:
        """Check if artifact has Docker support."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if filename in ["dockerfile", "docker-compose.yml", "docker-compose.yaml"]:
                return True
        return False
    
    def _has_installation_packages(self) -> bool:
        """Check if artifact has installation packages."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["setup.py", "install", "package", "dist", "build"]):
                return True
        return False
    
    def _has_virtual_environment(self) -> bool:
        """Check if artifact has virtual environment setup."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["venv", "env", "virtualenv", "conda", "environment.yml"]):
                return True
        return False
    
    def _has_build_scripts(self) -> bool:
        """Check if artifact has build scripts."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["build", "make", "compile", "install.sh", "setup.sh"]):
                return True
        return False
    
    def _has_deployment_instructions(self) -> bool:
        """Check if artifact has deployment instructions."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            deployment_keywords = [
                "deploy", "deployment", "production", "server", "hosting",
                "install", "setup", "configuration"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in deployment_keywords):
                return True
        return False
    
    def _generate_justification(self, executability_score: float, consistency_score: float,
                              verification_score: float, preparation_score: float) -> str:
        """Generate justification based on component scores."""
        if executability_score >= 0.8 and consistency_score >= 0.8 and verification_score >= 0.8:
            return "Excellent functionality: artifact is executable, consistent, complete, and well-verified."
        elif executability_score >= 0.6 and consistency_score >= 0.6 and verification_score >= 0.6:
            return "Good functionality: artifact is mostly executable and consistent but verification could be improved."
        elif executability_score >= 0.4 and consistency_score >= 0.4 and verification_score >= 0.4:
            return "Fair functionality: basic functionality present but significant improvements needed in consistency and verification."
        else:
            return "Poor functionality: artifact lacks essential functional elements and verification evidence."
    
    def _get_functionality_context(self):
        # Use README and main code files as context
        context = []
        for doc_file in self.artifact.get("documentation_files", []):
            if "readme" in doc_file.get("path", "").lower():
                content = doc_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                context.append(content)
        for code_file in self.artifact.get("code_files", []):
            if code_file.get("path", "").lower().endswith(("main.py", "app.py")):
                content = code_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                context.append(content)
        return "\n\n".join(context)
