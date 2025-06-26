import json
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ExperimentalEvaluationAgent:
    """
    Evaluates artifact experimental aspects based on ICSE 2025 criteria:
    - Experimental Setup: Hardware and software requirements
    - Data Availability: Access to datasets and experimental data
    - Validation Evidence: Evidence of verification and validation
    - Non-executable Artifacts: Proper packaging of data and documents
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
        Evaluate experimental aspects of the artifact (rule-based + LLM augmentation).
        Returns:
            Dict with score, justification, and evidence
        """
        logger.info("Evaluating artifact experimental aspects...")
        evidence = []
        score_components = []
        # 1. Check for experimental setup and requirements
        setup_score, setup_evidence = self._evaluate_experimental_setup()
        score_components.append(setup_score)
        evidence.extend(setup_evidence)
        # 2. Check for data availability and datasets
        data_score, data_evidence = self._evaluate_data_availability()
        score_components.append(data_score)
        evidence.extend(data_evidence)
        # 3. Check for validation and verification evidence
        validation_score, validation_evidence = self._evaluate_validation_evidence()
        score_components.append(validation_score)
        evidence.extend(validation_evidence)
        # 4. Check for non-executable artifacts packaging
        packaging_score, packaging_evidence = self._evaluate_artifact_packaging()
        score_components.append(packaging_score)
        evidence.extend(packaging_evidence)
        # Calculate overall score (weighted average)
        overall_score = sum(score_components) / len(score_components)
        # Generate justification
        justification = self._generate_justification(setup_score, data_score, validation_score, packaging_score)
        result = {
            "score": overall_score,
            "justification": justification,
            "evidence": evidence,
            "components": {
                "experimental_setup": setup_score,
                "data_availability": data_score,
                "validation_evidence": validation_score,
                "artifact_packaging": packaging_score
            }
        }
        # LLM augmentation
        if self.llm_evaluator:
            context = self._get_experimental_context()
            llm_data = self.llm_evaluator.evaluate_dimension(
                "experimental",
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
    
    def _get_experimental_context(self):
        # Use README and any setup/experiment docs as context
        context = []
        for doc_file in self.artifact.get("documentation_files", []):
            path = doc_file.get("path", "").lower()
            if any(k in path for k in ["readme", "experiment", "setup", "requirements"]):
                content = doc_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                context.append(content)
        return "\n\n".join(context)
    
    def _evaluate_experimental_setup(self) -> tuple[float, List[str]]:
        """Evaluate experimental setup and hardware/software requirements."""
        evidence = []
        score = 0.0
        
        # Check for hardware requirements documentation
        hardware_found = False
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            if self._has_hardware_requirements(content):
                evidence.append("Hardware requirements documented")
                score += 0.3
                hardware_found = True
                break
        
        if not hardware_found:
            evidence.append("Hardware requirements not documented")
        
        # Check for software requirements
        software_found = False
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            if self._has_software_requirements(content):
                evidence.append("Software requirements documented")
                score += 0.3
                software_found = True
                break
        
        if not software_found:
            evidence.append("Software requirements not documented")
        
        # Check for environment configuration files
        config_files = []
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["config", "env", "environment", "settings"]):
                config_files.append(entry)
        
        if config_files:
            evidence.append(f"Found {len(config_files)} configuration file(s)")
            score += 0.2
        
        # Check for dependency management
        if self._has_dependency_management():
            evidence.append("Dependency management found")
            score += 0.2
        
        return min(score, 1.0), evidence
    
    def _evaluate_data_availability(self) -> tuple[float, List[str]]:
        """Evaluate data availability and dataset access."""
        evidence = []
        score = 0.0
        
        # Check for dataset files
        dataset_files = []
        for entry in self.artifact.get("repository_structure", []):
            if self._is_dataset_file(entry):
                dataset_files.append(entry)
        
        if dataset_files:
            evidence.append(f"Found {len(dataset_files)} dataset file(s)")
            score += 0.4
        
        # Check for data documentation
        data_doc_found = False
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            if self._has_data_documentation(content):
                evidence.append("Data documentation found")
                score += 0.3
                data_doc_found = True
                break
        
        # Check for external data references
        if self._has_external_data_references():
            evidence.append("External data references found")
            score += 0.2
        
        # Check for data preprocessing scripts
        if self._has_data_preprocessing():
            evidence.append("Data preprocessing scripts found")
            score += 0.1
        
        if not dataset_files and not data_doc_found:
            evidence.append("No datasets or data documentation found")
        
        return min(score, 1.0), evidence
    
    def _evaluate_validation_evidence(self) -> tuple[float, List[str]]:
        """Evaluate validation and verification evidence."""
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
        
        # Check for evaluation scripts
        eval_files = []
        for entry in self.artifact.get("repository_structure", []):
            if self._is_evaluation_file(entry):
                eval_files.append(entry)
        
        if eval_files:
            evidence.append(f"Found {len(eval_files)} evaluation file(s)")
            score += 0.3
        
        # Check for results or metrics
        if self._has_results_documentation():
            evidence.append("Results/metrics documentation found")
            score += 0.2
        
        # Check for validation documentation
        if self._has_validation_documentation():
            evidence.append("Validation documentation found")
            score += 0.2
        
        if not test_files and not eval_files:
            evidence.append("No test or evaluation files found")
        
        return min(score, 1.0), evidence
    
    def _evaluate_artifact_packaging(self) -> tuple[float, List[str]]:
        """Evaluate non-executable artifact packaging."""
        evidence = []
        score = 0.0
        
        # Check for proper file organization
        if self._has_proper_organization():
            evidence.append("Artifact has proper file organization")
            score += 0.3
        
        # Check for common tool accessibility
        if self._has_common_tool_accessibility():
            evidence.append("Artifact uses common tool formats")
            score += 0.3
        
        # Check for documentation completeness
        if self._has_complete_documentation():
            evidence.append("Artifact has complete documentation")
            score += 0.2
        
        # Check for versioning information
        if self._has_versioning_info():
            evidence.append("Versioning information found")
            score += 0.2
        
        return min(score, 1.0), evidence
    
    def _has_hardware_requirements(self, content: str) -> bool:
        """Check if content contains hardware requirements."""
        hardware_keywords = [
            "hardware", "cpu", "gpu", "memory", "ram", "storage", "disk space",
            "processor", "cores", "threads", "minimum requirements"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in hardware_keywords)
    
    def _has_software_requirements(self, content: str) -> bool:
        """Check if content contains software requirements."""
        software_keywords = [
            "software", "operating system", "os", "python", "java", "c++", "dependencies",
            "libraries", "packages", "frameworks", "version", "requirements"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in software_keywords)
    
    def _has_dependency_management(self) -> bool:
        """Check if artifact has dependency management."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if filename in ["requirements.txt", "package.json", "pom.xml", "build.gradle", 
                           "cargo.toml", "go.mod", "environment.yml"]:
                return True
        return False
    
    def _is_dataset_file(self, entry: Dict) -> bool:
        """Check if file is a dataset."""
        filename = entry.get("name", "").lower()
        path = entry.get("path", "").lower()
        
        # Check file extensions
        data_extensions = [".csv", ".json", ".tsv", ".xlsx", ".xls", ".parquet", ".h5", ".hdf5"]
        if any(filename.endswith(ext) for ext in data_extensions):
            return True
        
        # Check path for data indicators
        data_indicators = ["/data/", "/dataset/", "/datasets/", "data/", "dataset/"]
        if any(indicator in path for indicator in data_indicators):
            return True
        
        return False
    
    def _has_data_documentation(self, content: str) -> bool:
        """Check if content contains data documentation."""
        data_keywords = [
            "dataset", "data", "format", "schema", "columns", "fields", "attributes",
            "data description", "data format", "data structure"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in data_keywords)
    
    def _has_external_data_references(self) -> bool:
        """Check if artifact references external data sources."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            # Look for URLs or data source references
            if re.search(r'https?://', content) or "dataset" in content.lower():
                return True
        return False
    
    def _has_data_preprocessing(self) -> bool:
        """Check if artifact has data preprocessing scripts."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["preprocess", "clean", "transform", "prepare"]):
                return True
        return False
    
    def _is_test_file(self, entry: Dict) -> bool:
        """Check if file is a test file."""
        filename = entry.get("name", "").lower()
        return (filename.startswith("test") or 
                filename.endswith("_test.py") or 
                "test" in filename)
    
    def _is_evaluation_file(self, entry: Dict) -> bool:
        """Check if file is an evaluation file."""
        filename = entry.get("name", "").lower()
        return any(keyword in filename for keyword in ["eval", "evaluate", "benchmark", "metrics", "performance"])
    
    def _has_results_documentation(self) -> bool:
        """Check if artifact has results documentation."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            results_keywords = [
                "results", "performance", "metrics", "accuracy", "precision", "recall",
                "f1", "benchmark", "evaluation", "comparison"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in results_keywords):
                return True
        return False
    
    def _has_validation_documentation(self) -> bool:
        """Check if artifact has validation documentation."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            validation_keywords = [
                "validation", "verification", "testing", "quality assurance", "qa",
                "correctness", "reliability"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in validation_keywords):
                return True
        return False
    
    def _has_proper_organization(self) -> bool:
        """Check if artifact has proper file organization."""
        structure = self.artifact.get("repository_structure", [])
        if len(structure) < 5:  # Too few files
            return False
        
        # Check for common organizational patterns
        has_docs = any("readme" in entry.get("name", "").lower() for entry in structure)
        has_code = any(entry.get("name", "").endswith((".py", ".java", ".cpp", ".js")) for entry in structure)
        has_data = any(self._is_dataset_file(entry) for entry in structure)
        
        return has_docs and has_code
    
    def _has_common_tool_accessibility(self) -> bool:
        """Check if artifact uses common tool formats."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            common_formats = [
                ".txt", ".md", ".pdf", ".csv", ".json", ".xml", ".yaml", ".yml",
                ".zip", ".tar.gz", ".py", ".ipynb"
            ]
            if any(filename.endswith(fmt) for fmt in common_formats):
                return True
        return False
    
    def _has_complete_documentation(self) -> bool:
        """Check if artifact has complete documentation."""
        doc_files = self.artifact.get("documentation_files", [])
        if len(doc_files) < 2:  # Need at least README and one other doc
            return False
        
        has_readme = any("readme" in doc.get("path", "").lower() for doc in doc_files)
        has_other_docs = len(doc_files) > 1
        
        return has_readme and has_other_docs
    
    def _has_versioning_info(self) -> bool:
        """Check if artifact has versioning information."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if filename in ["version", "version.txt", "changelog", "changelog.md"]:
                return True
        
        # Check in documentation
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            if "version" in content.lower() or "changelog" in content.lower():
                return True
        
        return False
    
    def _generate_justification(self, setup_score: float, data_score: float, 
                              validation_score: float, packaging_score: float) -> str:
        """Generate justification based on component scores."""
        if setup_score >= 0.8 and data_score >= 0.8 and validation_score >= 0.8:
            return "Excellent experimental setup: comprehensive requirements, data availability, and validation evidence."
        elif setup_score >= 0.6 and data_score >= 0.6 and validation_score >= 0.6:
            return "Good experimental setup: adequate requirements and data but validation could be improved."
        elif setup_score >= 0.4 and data_score >= 0.4 and validation_score >= 0.4:
            return "Fair experimental setup: basic experimental elements present but significant gaps remain."
        else:
            return "Poor experimental setup: missing essential experimental elements and validation."
