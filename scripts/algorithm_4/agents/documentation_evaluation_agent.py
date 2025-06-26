import json
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DocumentationEvaluationAgent:
    """
    Evaluates artifact documentation based on ICSE 2025 criteria:
    - Documentation: Quality and comprehensiveness of documentation
    - README: Purpose, provenance, setup, and usage details
    - Setup Instructions: Clarity and completeness for executable artifacts
    - Usage Instructions: Clarity for replicating main results
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
        Evaluate documentation quality of the artifact (rule-based + LLM augmentation).
        Returns:
            Dict with score, justification, and evidence
        """
        logger.info("Evaluating artifact documentation...")
        evidence = []
        score_components = []
        # 1. Check for README file and its quality
        readme_score, readme_evidence = self._evaluate_readme()
        score_components.append(readme_score)
        evidence.extend(readme_evidence)
        # 2. Check for setup instructions
        setup_score, setup_evidence = self._evaluate_setup_instructions()
        score_components.append(setup_score)
        evidence.extend(setup_evidence)
        # 3. Check for usage instructions
        usage_score, usage_evidence = self._evaluate_usage_instructions()
        score_components.append(usage_score)
        evidence.extend(usage_evidence)
        # 4. Check for comprehensive documentation
        comprehensive_score, comprehensive_evidence = self._evaluate_comprehensive_documentation()
        score_components.append(comprehensive_score)
        evidence.extend(comprehensive_evidence)
        # Calculate overall score (weighted average)
        overall_score = sum(score_components) / len(score_components)
        # Generate justification
        justification = self._generate_justification(readme_score, setup_score, usage_score, comprehensive_score)
        result = {
            "score": overall_score,
            "justification": justification,
            "evidence": evidence,
            "components": {
                "readme": readme_score,
                "setup_instructions": setup_score,
                "usage_instructions": usage_score,
                "comprehensive_documentation": comprehensive_score
            }
        }
        # LLM augmentation
        if self.llm_evaluator:
            readme_content = self._get_readme_content()
            llm_data = self.llm_evaluator.evaluate_dimension(
                "documentation",
                result["score"],
                result["justification"],
                result["evidence"],
                readme_content
            )
            result["score"] = max(result["score"], llm_data.get("revised_score", result["score"]))
            result["justification"] = llm_data.get("revised_justification", result["justification"])
            if llm_data.get("additional_evidence"):
                result["evidence"].extend(llm_data["additional_evidence"])
        return result
    
    def _get_readme_content(self):
        for doc_file in self.artifact.get("documentation_files", []):
            if "readme" in doc_file.get("path", "").lower():
                content = doc_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                return content
        return ""
    
    def _evaluate_readme(self) -> tuple[float, List[str]]:
        """Evaluate README file quality and completeness."""
        evidence = []
        score = 0.0
        
        readme_files = []
        for doc_file in self.artifact.get("documentation_files", []):
            if "readme" in doc_file.get("path", "").lower():
                readme_files.append(doc_file)
        
        if not readme_files:
            evidence.append("No README file found")
            return 0.0, evidence
        
        evidence.append(f"Found {len(readme_files)} README file(s)")
        score += 0.2
        
        # Analyze the main README file
        main_readme = readme_files[0]
        content = main_readme.get("content", "")
        if isinstance(content, list):
            content = "\n".join(content)
        
        # Check for required sections
        required_sections = [
            ("purpose", ["purpose", "description", "about", "overview"]),
            ("provenance", ["provenance", "citation", "reference", "paper"]),
            ("setup", ["setup", "install", "installation", "requirements"]),
            ("usage", ["usage", "how to use", "examples", "commands"])
        ]
        
        for section_name, keywords in required_sections:
            if self._has_section(content, keywords):
                evidence.append(f"README contains {section_name} section")
                score += 0.2
            else:
                evidence.append(f"README missing {section_name} section")
        
        # Check for code blocks and examples
        if self._has_code_blocks(content):
            evidence.append("README contains code examples")
            score += 0.1
        
        # Check for proper formatting
        if self._has_proper_formatting(content):
            evidence.append("README has good formatting")
            score += 0.1
        
        return min(score, 1.0), evidence
    
    def _evaluate_setup_instructions(self) -> tuple[float, List[str]]:
        """Evaluate setup instructions clarity and completeness."""
        evidence = []
        score = 0.0
        
        # Check for setup instructions in README
        setup_found = False
        for doc_file in self.artifact.get("documentation_files", []):
            if "readme" in doc_file.get("path", "").lower():
                content = doc_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                
                if self._has_setup_instructions(content):
                    evidence.append("Setup instructions found in README")
                    score += 0.4
                    setup_found = True
                    break
        
        # Check for dedicated setup files
        setup_files = []
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["setup", "install", "requirements", "dockerfile"]):
                setup_files.append(entry)
        
        if setup_files:
            evidence.append(f"Found {len(setup_files)} setup-related file(s)")
            score += 0.3
        
        # Check for Docker support
        if self._has_docker_support():
            evidence.append("Docker support found")
            score += 0.2
        
        # Check for requirements/dependencies
        if self._has_dependencies_listed():
            evidence.append("Dependencies/requirements listed")
            score += 0.1
        
        if not setup_found and not setup_files:
            evidence.append("No setup instructions found")
        
        return min(score, 1.0), evidence
    
    def _evaluate_usage_instructions(self) -> tuple[float, List[str]]:
        """Evaluate usage instructions for replicating results."""
        evidence = []
        score = 0.0
        
        # Check for usage instructions in README
        for doc_file in self.artifact.get("documentation_files", []):
            if "readme" in doc_file.get("path", "").lower():
                content = doc_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                
                if self._has_usage_instructions(content):
                    evidence.append("Usage instructions found in README")
                    score += 0.3
                    break
        
        # Check for example scripts or notebooks
        example_files = []
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["example", "demo", "tutorial", "notebook"]):
                example_files.append(entry)
        
        if example_files:
            evidence.append(f"Found {len(example_files)} example/demo file(s)")
            score += 0.3
        
        # Check for command-line interface or main scripts
        if self._has_executable_interface():
            evidence.append("Executable interface found")
            score += 0.2
        
        # Check for result replication instructions
        if self._has_result_replication_instructions():
            evidence.append("Result replication instructions found")
            score += 0.2
        
        return min(score, 1.0), evidence
    
    def _evaluate_comprehensive_documentation(self) -> tuple[float, List[str]]:
        """Evaluate overall documentation comprehensiveness."""
        evidence = []
        score = 0.0
        
        # Count documentation files
        doc_files = self.artifact.get("documentation_files", [])
        if doc_files:
            evidence.append(f"Found {len(doc_files)} documentation file(s)")
            score += 0.2
        
        # Check for API documentation
        if self._has_api_documentation():
            evidence.append("API documentation found")
            score += 0.2
        
        # Check for architecture/design documentation
        if self._has_architecture_documentation():
            evidence.append("Architecture/design documentation found")
            score += 0.2
        
        # Check for troubleshooting/FAQ
        if self._has_troubleshooting_section():
            evidence.append("Troubleshooting/FAQ section found")
            score += 0.2
        
        # Check for contribution guidelines
        if self._has_contribution_guidelines():
            evidence.append("Contribution guidelines found")
            score += 0.2
        
        return min(score, 1.0), evidence
    
    def _has_section(self, content: str, keywords: List[str]) -> bool:
        """Check if content has a section with given keywords."""
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in keywords)
    
    def _has_code_blocks(self, content: str) -> bool:
        """Check if content contains code blocks."""
        # Look for markdown code blocks or indented code
        return "```" in content or re.search(r'^\s+[a-zA-Z]', content, re.MULTILINE)
    
    def _has_proper_formatting(self, content: str) -> bool:
        """Check if content has proper markdown formatting."""
        # Check for headers, lists, links
        has_headers = re.search(r'^#{1,6}\s+', content, re.MULTILINE)
        has_lists = re.search(r'^[\s]*[-*+]\s+', content, re.MULTILINE)
        has_links = re.search(r'\[.*\]\(.*\)', content)
        
        return bool(has_headers or has_lists or has_links)
    
    def _has_setup_instructions(self, content: str) -> bool:
        """Check if content contains setup instructions."""
        setup_keywords = [
            "setup", "install", "installation", "requirements", "dependencies",
            "prerequisites", "environment", "configuration"
        ]
        return self._has_section(content, setup_keywords)
    
    def _has_usage_instructions(self, content: str) -> bool:
        """Check if content contains usage instructions."""
        usage_keywords = [
            "usage", "how to use", "examples", "commands", "run", "execute",
            "quick start", "getting started"
        ]
        return self._has_section(content, usage_keywords)
    
    def _has_docker_support(self) -> bool:
        """Check if artifact has Docker support."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if filename in ["dockerfile", "docker-compose.yml", "docker-compose.yaml"]:
                return True
        return False
    
    def _has_dependencies_listed(self) -> bool:
        """Check if dependencies are listed."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if filename in ["requirements.txt", "package.json", "pom.xml", "build.gradle", "cargo.toml"]:
                return True
        return False
    
    def _has_executable_interface(self) -> bool:
        """Check if artifact has executable interface."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if filename in ["main.py", "app.py", "run.py", "cli.py"] or filename.endswith(".sh"):
                return True
        return False
    
    def _has_result_replication_instructions(self) -> bool:
        """Check if there are instructions for replicating results."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            replication_keywords = [
                "reproduce", "replicate", "results", "experiments", "evaluation",
                "benchmark", "performance", "metrics"
            ]
            if self._has_section(content, replication_keywords):
                return True
        return False
    
    def _has_api_documentation(self) -> bool:
        """Check if API documentation exists."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if "api" in filename or "docs" in filename:
                return True
        return False
    
    def _has_architecture_documentation(self) -> bool:
        """Check if architecture/design documentation exists."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            arch_keywords = ["architecture", "design", "structure", "overview", "system"]
            if self._has_section(content, arch_keywords):
                return True
        return False
    
    def _has_troubleshooting_section(self) -> bool:
        """Check if troubleshooting/FAQ section exists."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            trouble_keywords = ["troubleshooting", "faq", "issues", "problems", "debug"]
            if self._has_section(content, trouble_keywords):
                return True
        return False
    
    def _has_contribution_guidelines(self) -> bool:
        """Check if contribution guidelines exist."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if "contributing" in filename or "contribute" in filename:
                return True
        return False
    
    def _generate_justification(self, readme_score: float, setup_score: float, 
                              usage_score: float, comprehensive_score: float) -> str:
        """Generate justification based on component scores."""
        if readme_score >= 0.8 and setup_score >= 0.8 and usage_score >= 0.8:
            return "Excellent documentation: comprehensive README with clear setup and usage instructions."
        elif readme_score >= 0.6 and setup_score >= 0.6 and usage_score >= 0.6:
            return "Good documentation: adequate README and instructions but room for improvement."
        elif readme_score >= 0.4 and setup_score >= 0.4 and usage_score >= 0.4:
            return "Fair documentation: basic documentation present but significant gaps remain."
        else:
            return "Poor documentation: missing essential documentation elements."
