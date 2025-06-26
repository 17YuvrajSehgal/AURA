import json
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class UsabilityEvaluationAgent:
    """
    Evaluates artifact usability based on ICSE 2025 criteria:
    - User Experience: Ease of use and user-friendly design
    - Iterative Review Process: Authors' responsiveness to reviewer requests
    - User Interface: Quality of user interfaces and interactions
    - Error Handling: Graceful error handling and user feedback
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
        Evaluate usability of the artifact (rule-based + LLM augmentation).
        Returns:
            Dict with score, justification, and evidence
        """
        logger.info("Evaluating artifact usability...")
        evidence = []
        score_components = []
        # 1. Check for user experience and ease of use
        ux_score, ux_evidence = self._evaluate_user_experience()
        score_components.append(ux_score)
        evidence.extend(ux_evidence)
        # 2. Check for user interface quality
        ui_score, ui_evidence = self._evaluate_user_interface()
        score_components.append(ui_score)
        evidence.extend(ui_evidence)
        # 3. Check for error handling and feedback
        error_score, error_evidence = self._evaluate_error_handling()
        score_components.append(error_score)
        evidence.extend(error_evidence)
        # 4. Check for iterative review process support
        review_score, review_evidence = self._evaluate_review_process()
        score_components.append(review_score)
        evidence.extend(review_evidence)
        # Calculate overall score (weighted average)
        overall_score = sum(score_components) / len(score_components)
        # Generate justification
        justification = self._generate_justification(ux_score, ui_score, error_score, review_score)
        result = {
            "score": overall_score,
            "justification": justification,
            "evidence": evidence,
            "components": {
                "user_experience": ux_score,
                "user_interface": ui_score,
                "error_handling": error_score,
                "review_process": review_score
            }
        }
        # LLM augmentation
        if self.llm_evaluator:
            context = self._get_usability_context()
            llm_data = self.llm_evaluator.evaluate_dimension(
                "usability",
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
    
    def _get_usability_context(self):
        # Use README and any UI/UX docs as context
        context = []
        for doc_file in self.artifact.get("documentation_files", []):
            path = doc_file.get("path", "").lower()
            if any(k in path for k in ["readme", "ui", "ux", "usage", "interface"]):
                content = doc_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                context.append(content)
        return "\n\n".join(context)
    
    def _evaluate_user_experience(self) -> tuple[float, List[str]]:
        """Evaluate user experience and ease of use."""
        evidence = []
        score = 0.0
        
        # Check for intuitive design
        if self._has_intuitive_design():
            evidence.append("Intuitive design elements found")
            score += 0.3
        else:
            evidence.append("Design may not be intuitive")
        
        # Check for progressive disclosure
        if self._has_progressive_disclosure():
            evidence.append("Progressive disclosure implemented")
            score += 0.2
        else:
            evidence.append("Progressive disclosure not implemented")
        
        # Check for user guidance
        if self._has_user_guidance():
            evidence.append("User guidance provided")
            score += 0.2
        else:
            evidence.append("User guidance could be improved")
        
        # Check for consistency
        if self._has_consistency():
            evidence.append("Consistent design patterns found")
            score += 0.2
        else:
            evidence.append("Design consistency could be improved")
        
        # Check for accessibility features
        if self._has_accessibility_features():
            evidence.append("Accessibility features found")
            score += 0.1
        else:
            evidence.append("Accessibility features could be added")
        
        return min(score, 1.0), evidence
    
    def _evaluate_user_interface(self) -> tuple[float, List[str]]:
        """Evaluate user interface quality."""
        evidence = []
        score = 0.0
        
        # Check for web interface
        if self._has_web_interface():
            evidence.append("Web interface found")
            score += 0.3
        else:
            evidence.append("No web interface found")
        
        # Check for command-line interface
        if self._has_cli_interface():
            evidence.append("Command-line interface found")
            score += 0.2
        else:
            evidence.append("Command-line interface could be improved")
        
        # Check for GUI components
        if self._has_gui_components():
            evidence.append("GUI components found")
            score += 0.2
        else:
            evidence.append("No GUI components found")
        
        # Check for interactive elements
        if self._has_interactive_elements():
            evidence.append("Interactive elements found")
            score += 0.2
        else:
            evidence.append("Interactive elements could be added")
        
        # Check for responsive design
        if self._has_responsive_design():
            evidence.append("Responsive design implemented")
            score += 0.1
        else:
            evidence.append("Responsive design could be improved")
        
        return min(score, 1.0), evidence
    
    def _evaluate_error_handling(self) -> tuple[float, List[str]]:
        """Evaluate error handling and user feedback."""
        evidence = []
        score = 0.0
        
        # Check for graceful error handling
        if self._has_graceful_error_handling():
            evidence.append("Graceful error handling found")
            score += 0.3
        else:
            evidence.append("Error handling could be improved")
        
        # Check for user-friendly error messages
        if self._has_user_friendly_errors():
            evidence.append("User-friendly error messages found")
            score += 0.3
        else:
            evidence.append("Error messages could be more user-friendly")
        
        # Check for validation feedback
        if self._has_validation_feedback():
            evidence.append("Validation feedback provided")
            score += 0.2
        else:
            evidence.append("Validation feedback could be improved")
        
        # Check for recovery mechanisms
        if self._has_recovery_mechanisms():
            evidence.append("Recovery mechanisms found")
            score += 0.2
        else:
            evidence.append("Recovery mechanisms could be added")
        
        return min(score, 1.0), evidence
    
    def _evaluate_review_process(self) -> tuple[float, List[str]]:
        """Evaluate iterative review process support."""
        evidence = []
        score = 0.0
        
        # Check for version control
        if self._has_version_control():
            evidence.append("Version control found")
            score += 0.3
        else:
            evidence.append("Version control could be improved")
        
        # Check for change tracking
        if self._has_change_tracking():
            evidence.append("Change tracking implemented")
            score += 0.2
        else:
            evidence.append("Change tracking could be improved")
        
        # Check for feedback mechanisms
        if self._has_feedback_mechanisms():
            evidence.append("Feedback mechanisms found")
            score += 0.2
        else:
            evidence.append("Feedback mechanisms could be added")
        
        # Check for documentation updates
        if self._has_documentation_updates():
            evidence.append("Documentation update process found")
            score += 0.2
        else:
            evidence.append("Documentation update process could be improved")
        
        # Check for issue tracking
        if self._has_issue_tracking():
            evidence.append("Issue tracking found")
            score += 0.1
        else:
            evidence.append("Issue tracking could be added")
        
        return min(score, 1.0), evidence
    
    def _has_intuitive_design(self) -> bool:
        """Check if artifact has intuitive design."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            intuitive_keywords = [
                "intuitive", "easy to use", "user-friendly", "simple", "straightforward",
                "clear", "obvious", "self-explanatory", "logical"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in intuitive_keywords):
                return True
        return False
    
    def _has_progressive_disclosure(self) -> bool:
        """Check if artifact implements progressive disclosure."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            progressive_keywords = [
                "basic", "advanced", "beginner", "expert", "simple", "complex",
                "step by step", "progressive", "tutorial", "guide"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in progressive_keywords):
                return True
        return False
    
    def _has_user_guidance(self) -> bool:
        """Check if artifact provides user guidance."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            guidance_keywords = [
                "guide", "tutorial", "help", "assistance", "support", "walkthrough",
                "how to", "instructions", "manual", "guidebook"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in guidance_keywords):
                return True
        return False
    
    def _has_consistency(self) -> bool:
        """Check if artifact has consistent design patterns."""
        # Check for consistent naming conventions
        filenames = [entry.get("name", "") for entry in self.artifact.get("repository_structure", [])]
        
        # Check for consistent case usage
        has_snake_case = any('_' in name for name in filenames)
        has_kebab_case = any('-' in name for name in filenames)
        has_camel_case = any(re.search(r'[a-z][A-Z]', name) for name in filenames)
        
        # If multiple naming conventions are used, it's inconsistent
        conventions = sum([has_snake_case, has_kebab_case, has_camel_case])
        return conventions <= 2  # Allow up to 2 conventions
    
    def _has_accessibility_features(self) -> bool:
        """Check if artifact has accessibility features."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            accessibility_keywords = [
                "accessibility", "accessible", "screen reader", "keyboard navigation",
                "high contrast", "font size", "color blind", "disability"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in accessibility_keywords):
                return True
        return False
    
    def _has_web_interface(self) -> bool:
        """Check if artifact has web interface."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["html", "css", "js", "web", "app", "server"]):
                return True
        return False
    
    def _has_cli_interface(self) -> bool:
        """Check if artifact has command-line interface."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["cli", "command", "shell", "script", "main.py"]):
                return True
        return False
    
    def _has_gui_components(self) -> bool:
        """Check if artifact has GUI components."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["gui", "tkinter", "pyqt", "wx", "gtk", "ui"]):
                return True
        return False
    
    def _has_interactive_elements(self) -> bool:
        """Check if artifact has interactive elements."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["interactive", "notebook", "jupyter", "demo", "playground"]):
                return True
        return False
    
    def _has_responsive_design(self) -> bool:
        """Check if artifact has responsive design."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["responsive", "mobile", "adaptive", "flexible"]):
                return True
        return False
    
    def _has_graceful_error_handling(self) -> bool:
        """Check if artifact has graceful error handling."""
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
    
    def _has_user_friendly_errors(self) -> bool:
        """Check if artifact has user-friendly error messages."""
        for code_file in self.artifact.get("code_files", []):
            content = code_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            # Look for user-friendly error patterns
            friendly_patterns = [
                r'print\s*\([^)]*error[^)]*\)', r'logging\.[^)]*error[^)]*\)',
                r'raise\s+[^)]*Error[^)]*\)', r'return\s+[^)]*error[^)]*\)'
            ]
            if any(re.search(pattern, content, re.IGNORECASE) for pattern in friendly_patterns):
                return True
        return False
    
    def _has_validation_feedback(self) -> bool:
        """Check if artifact provides validation feedback."""
        for code_file in self.artifact.get("code_files", []):
            content = code_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            validation_keywords = [
                "validate", "validation", "check", "verify", "confirm",
                "feedback", "message", "notification", "alert"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in validation_keywords):
                return True
        return False
    
    def _has_recovery_mechanisms(self) -> bool:
        """Check if artifact has recovery mechanisms."""
        for code_file in self.artifact.get("code_files", []):
            content = code_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            recovery_keywords = [
                "recover", "restore", "backup", "retry", "fallback",
                "alternative", "default", "safe", "secure"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in recovery_keywords):
                return True
        return False
    
    def _has_version_control(self) -> bool:
        """Check if artifact has version control."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if filename in [".git", ".gitignore", ".gitattributes"]:
                return True
        return False
    
    def _has_change_tracking(self) -> bool:
        """Check if artifact has change tracking."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["changelog", "version", "history", "log"]):
                return True
        return False
    
    def _has_feedback_mechanisms(self) -> bool:
        """Check if artifact has feedback mechanisms."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["feedback", "issue", "bug", "report", "contact"]):
                return True
        return False
    
    def _has_documentation_updates(self) -> bool:
        """Check if artifact has documentation update process."""
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            update_keywords = [
                "update", "version", "changelog", "history", "revision",
                "latest", "current", "new", "improved"
            ]
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in update_keywords):
                return True
        return False
    
    def _has_issue_tracking(self) -> bool:
        """Check if artifact has issue tracking."""
        for entry in self.artifact.get("repository_structure", []):
            filename = entry.get("name", "").lower()
            if any(keyword in filename for keyword in ["issue", "bug", "tracker", "todo", "fixme"]):
                return True
        return False
    
    def _generate_justification(self, ux_score: float, ui_score: float, 
                              error_score: float, review_score: float) -> str:
        """Generate justification based on component scores."""
        if ux_score >= 0.8 and ui_score >= 0.8 and error_score >= 0.8:
            return "Excellent usability: artifact provides great user experience with quality interfaces and error handling."
        elif ux_score >= 0.6 and ui_score >= 0.6 and error_score >= 0.6:
            return "Good usability: artifact has adequate user experience but could be improved in some areas."
        elif ux_score >= 0.4 and ui_score >= 0.4 and error_score >= 0.4:
            return "Fair usability: basic usability elements present but significant improvements needed."
        else:
            return "Poor usability: artifact lacks essential usability elements and user-friendly design."
