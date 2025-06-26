import json
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class AccessibilityEvaluationAgent:
    """
    Evaluates artifact accessibility based on ICSE 2025 criteria:
    - Availability: Publicly accessible archival repository with DOI/link
    - Archival Repository: Suitable repository (Zenodo, FigShare) vs non-archival (GitHub)
    - License: LICENSE file with distribution rights and open-source/open data license
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
        Evaluate accessibility of the artifact.
        
        Returns:
            Dict with score, justification, and evidence
        """
        logger.info("Evaluating artifact accessibility...")
        
        evidence = []
        score_components = []
        
        # 1. Check for public accessibility and repository link
        availability_score, availability_evidence = self._evaluate_availability()
        score_components.append(availability_score)
        evidence.extend(availability_evidence)
        
        # 2. Check for archival repository (Zenodo, FigShare, etc.)
        archival_score, archival_evidence = self._evaluate_archival_repository()
        score_components.append(archival_score)
        evidence.extend(archival_evidence)
        
        # 3. Check for LICENSE file and open licensing
        license_score, license_evidence = self._evaluate_licensing()
        score_components.append(license_score)
        evidence.extend(license_evidence)
        
        # Calculate overall score (weighted average)
        overall_score = sum(score_components) / len(score_components)
        
        # Generate justification
        justification = self._generate_justification(availability_score, archival_score, license_score)
        
        return {
            "score": overall_score,
            "justification": justification,
            "evidence": evidence,
            "components": {
                "availability": availability_score,
                "archival_repository": archival_score,
                "licensing": license_score
            }
        }
    
    def _evaluate_availability(self) -> tuple[float, List[str]]:
        """Evaluate if artifact is publicly accessible with proper repository link."""
        evidence = []
        score = 0.0
        
        # Check for repository information
        repo_name = self.artifact.get("repository_name", "")
        repo_url = self.artifact.get("repository_url", "")
        
        if repo_name:
            evidence.append(f"Repository name found: {repo_name}")
            score += 0.3
        
        if repo_url:
            evidence.append(f"Repository URL found: {repo_url}")
            score += 0.4
            
            # Check if URL is accessible
            if self._is_valid_url(repo_url):
                evidence.append("Repository URL appears valid")
                score += 0.3
            else:
                evidence.append("Repository URL may not be accessible")
        else:
            evidence.append("No repository URL found")
        
        return min(score, 1.0), evidence
    
    def _evaluate_archival_repository(self) -> tuple[float, List[str]]:
        """Evaluate if artifact is in suitable archival repository."""
        evidence = []
        score = 0.0
        
        repo_url = self.artifact.get("repository_url", "").lower()
        
        # Check for archival repositories
        archival_repos = ["zenodo", "figshare", "dataverse", "dryad", "osf.io"]
        non_archival_repos = ["github.com", "gitlab.com", "bitbucket.org"]
        
        if any(repo in repo_url for repo in archival_repos):
            evidence.append("Artifact is in archival repository (Zenodo/FigShare/etc.)")
            score = 1.0
        elif any(repo in repo_url for repo in non_archival_repos):
            evidence.append("Artifact is in non-archival repository (GitHub/GitLab/etc.)")
            score = 0.5
        else:
            evidence.append("Repository type could not be determined")
            score = 0.3
        
        # Check for DOI
        if self._has_doi():
            evidence.append("DOI found - indicates archival status")
            score = min(score + 0.2, 1.0)
        else:
            evidence.append("No DOI found")
        
        return score, evidence
    
    def _evaluate_licensing(self) -> tuple[float, List[str]]:
        """Evaluate licensing compliance."""
        evidence = []
        score = 0.0
        
        # Check for LICENSE file existence
        license_files = self.artifact.get("license_files", [])
        if license_files:
            evidence.append(f"Found {len(license_files)} license file(s)")
            score += 0.4
            
            # Check license content
            for license_file in license_files:
                content = license_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                
                if self._is_open_source_license(content):
                    evidence.append("Open source license detected")
                    score += 0.4
                    break
            else:
                evidence.append("License type could not be determined")
                score += 0.2
        else:
            evidence.append("No LICENSE file found")
        
        # Check for license information in README
        if self._has_license_in_readme():
            evidence.append("License information found in README")
            score = min(score + 0.2, 1.0)
        
        return min(score, 1.0), evidence
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _has_doi(self) -> bool:
        """Check if artifact has a DOI."""
        # Check in repository information
        repo_url = self.artifact.get("repository_url", "")
        if "doi.org" in repo_url or "10." in repo_url:
            return True
        
        # Check in documentation files
        for doc_file in self.artifact.get("documentation_files", []):
            content = doc_file.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            if "doi.org" in content or "10." in content:
                return True
        
        return False
    
    def _is_open_source_license(self, content: str) -> bool:
        """Check if content contains open source license."""
        open_source_keywords = [
            "mit license", "apache license", "gpl", "bsd license", 
            "creative commons", "cc-by", "cc0", "public domain",
            "mozilla public license", "eclipse public license"
        ]
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in open_source_keywords)
    
    def _has_license_in_readme(self) -> bool:
        """Check if README contains license information."""
        for doc_file in self.artifact.get("documentation_files", []):
            if "readme" in doc_file.get("path", "").lower():
                content = doc_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                if "license" in content.lower():
                    return True
        return False
    
    def _generate_justification(self, availability_score: float, archival_score: float, license_score: float) -> str:
        """Generate justification based on component scores."""
        if availability_score >= 0.8 and archival_score >= 0.8 and license_score >= 0.8:
            return "Excellent accessibility: artifact is publicly available in archival repository with proper licensing."
        elif availability_score >= 0.6 and archival_score >= 0.6 and license_score >= 0.6:
            return "Good accessibility: artifact is accessible but may need improvements in archival status or licensing."
        elif availability_score >= 0.4 and archival_score >= 0.4 and license_score >= 0.4:
            return "Fair accessibility: artifact has basic accessibility but significant improvements needed."
        else:
            return "Poor accessibility: artifact lacks proper public access, archival status, or licensing."
