import json
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class AccessibilityEvaluationAgent:
    """
    Conference-aware accessibility evaluator that uses conference-specific profiles
    to evaluate artifact accessibility based on venue-specific criteria and emphasis.
    
    Evaluates based on conference guidelines:
    - Availability: Publicly accessible archival repository with DOI/link
    - Archival Repository: Suitable repository (Zenodo, FigShare) vs non-archival (GitHub)
    - License: LICENSE file with distribution rights and open-source/open data license
    """
    
    def __init__(self, kg_agent, llm_evaluator=None, conference_profile=None):
        self.kg_agent = kg_agent
        self.llm_evaluator = llm_evaluator
        self.conference_profile = conference_profile or self._get_default_profile()
        self.artifact = self._load_artifact()
        
        # Conference-specific configuration
        self.quality_threshold = self.conference_profile.get('quality_threshold', 0.8)
        self.evaluation_style = self.conference_profile.get('evaluation_style', 'standard')
        self.domain_keywords = self.conference_profile.get('domain_keywords', [])
        
    def _load_artifact(self) -> Dict:
        """Load artifact data from JSON file."""
        with open(self.kg_agent.artifact_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _get_default_profile(self) -> Dict:
        """Default profile for general conference evaluation."""
        return {
            "category": "general",
            "emphasis_weights": {
                "accessibility": 0.3,
                "documentation": 0.2,
                "reproducibility": 0.2,
                "usability": 0.15,
                "functionality": 0.1,
                "experimental": 0.05
            },
            "quality_threshold": 0.7,
            "evaluation_style": "standard",
            "domain_keywords": []
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Conference-aware evaluation of artifact accessibility.
        Returns:
            Dict with score, justification, and evidence
        """
        logger.info(f"Evaluating artifact accessibility for {self.conference_profile.get('category', 'general')} conference...")
        evidence = []
        score_components = []
        
        # 1. Check for public accessibility and repository link
        availability_score, availability_evidence = self._evaluate_availability()
        score_components.append(availability_score)
        evidence.extend(availability_evidence)
        
        # 2. Check for archival repository (weighted by conference emphasis)
        archival_score, archival_evidence = self._evaluate_archival_repository()
        score_components.append(archival_score)
        evidence.extend(archival_evidence)
        
        # 3. Check for LICENSE file and open licensing
        license_score, license_evidence = self._evaluate_licensing()
        score_components.append(license_score)
        evidence.extend(license_evidence)
        
        # 4. Conference-specific accessibility checks
        conference_score, conference_evidence = self._evaluate_conference_specific_accessibility()
        score_components.append(conference_score)
        evidence.extend(conference_evidence)
        
        # Calculate overall score with conference-specific weighting
        overall_score = self._calculate_conference_weighted_score(
            availability_score, archival_score, license_score, conference_score
        )
        
        # Apply conference-specific quality threshold
        meets_threshold = overall_score >= self.quality_threshold
        
        # Generate justification
        justification = self._generate_conference_aware_justification(
            availability_score, archival_score, license_score, conference_score, meets_threshold
        )
        
        result = {
            "score": overall_score,
            "justification": justification,
            "evidence": evidence,
            "meets_conference_threshold": meets_threshold,
            "conference_profile": {
                "category": self.conference_profile.get('category'),
                "quality_threshold": self.quality_threshold,
                "evaluation_style": self.evaluation_style,
                "accessibility_emphasis": self.conference_profile.get('emphasis_weights', {}).get('accessibility', 0.3)
            },
            "components": {
                "availability": availability_score,
                "archival_repository": archival_score,
                "licensing": license_score,
                "conference_specific": conference_score
            }
        }
        
        # LLM augmentation with conference context
        if self.llm_evaluator:
            readme_content = self._get_readme_content()
            conference_context = self._build_conference_context()
            
            llm_data = self.llm_evaluator.evaluate_dimension(
                "accessibility",
                result["score"],
                result["justification"],
                result["evidence"],
                readme_content,
                conference_context=conference_context
            )
            
            # Use conference-specific score combination strategy
            if self.evaluation_style == "strict":
                # For strict conferences, use minimum of rule-based and LLM
                result["score"] = min(result["score"], llm_data.get("revised_score", result["score"]))
            else:
                # For standard conferences, use maximum
                result["score"] = max(result["score"], llm_data.get("revised_score", result["score"]))
            
            result["justification"] = llm_data.get("revised_justification", result["justification"])
            if llm_data.get("additional_evidence"):
                result["evidence"].extend(llm_data["additional_evidence"])
            result["llm_justification"] = llm_data.get("revised_justification", "")
            result["llm_evidence"] = llm_data.get("additional_evidence", [])
        
        return result
    
    def _evaluate_availability(self) -> tuple[float, List[str]]:
        """Evaluate if artifact is publicly accessible with proper repository link."""
        evidence = []
        score = 0.0
        
        # Use knowledge graph to check repository information
        repo_accessible = self.kg_agent.run_cypher("""
            MATCH (r:Repository) 
            RETURN r.name as repo_name, r.url as repo_url
        """)
        
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
        
        # Conference-specific availability checks
        if self.conference_profile.get('category') == 'software_engineering':
            # Software engineering conferences often require GitHub/GitLab
            if any(platform in repo_url.lower() for platform in ['github.com', 'gitlab.com']):
                evidence.append("Repository on recognized software development platform")
                score = min(score + 0.1, 1.0)
        
        return min(score, 1.0), evidence
    
    def _evaluate_archival_repository(self) -> tuple[float, List[str]]:
        """Conference-aware evaluation of archival repository requirements."""
        evidence = []
        score = 0.0
        
        repo_url = self.artifact.get("repository_url", "").lower()
        
        # Define archival repositories with conference-specific preferences
        archival_repos = ["zenodo", "figshare", "dataverse", "dryad", "osf.io"]
        development_repos = ["github.com", "gitlab.com", "bitbucket.org"]
        
        # Conference-specific archival requirements
        category = self.conference_profile.get('category', 'general')
        
        if any(repo in repo_url for repo in archival_repos):
            evidence.append("Artifact is in archival repository (Zenodo/FigShare/etc.)")
            # Higher score for conferences that emphasize long-term preservation
            if category in ['data_systems', 'experimental']:
                score = 1.0
            else:
                score = 0.9
                
        elif any(repo in repo_url for repo in development_repos):
            evidence.append("Artifact is in development repository (GitHub/GitLab/etc.)")
            # Software engineering conferences may accept GitHub more readily
            if category == 'software_engineering':
                score = 0.7
            else:
                score = 0.5
        else:
            evidence.append("Repository type could not be determined")
            score = 0.3
        
        # Check for DOI - critical for some conference types
        if self._has_doi():
            evidence.append("DOI found - indicates archival status")
            doi_bonus = 0.3 if category in ['data_systems', 'experimental'] else 0.2
            score = min(score + doi_bonus, 1.0)
        else:
            evidence.append("No DOI found")
            # Penalize more for conferences that emphasize reproducibility
            if category in ['data_systems', 'experimental']:
                score *= 0.8
        
        # Check for conference-specific archival keywords
        if self._check_conference_archival_keywords():
            evidence.append("Conference-specific archival indicators found")
            score = min(score + 0.1, 1.0)
        
        return score, evidence
    
    def _evaluate_licensing(self) -> tuple[float, List[str]]:
        """Conference-aware licensing evaluation."""
        evidence = []
        score = 0.0
        
        # Use knowledge graph to check for license files
        license_results = self.kg_agent.run_cypher("""
            MATCH (f:File {type: 'license'}) 
            RETURN f.name as filename, f.path as filepath
        """)
        
        # Check for LICENSE file existence
        license_files = self.artifact.get("license_files", [])
        if license_files or license_results:
            evidence.append(f"Found {len(license_files)} license file(s)")
            score += 0.4
            
            # Check license content
            for license_file in license_files:
                content = license_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                
                license_type = self._identify_license_type(content)
                if license_type:
                    evidence.append(f"Identified license: {license_type}")
                    
                    # Conference-specific license preferences
                    if self._is_conference_preferred_license(license_type):
                        evidence.append("License meets conference preferences")
                        score += 0.5
                    elif self._is_open_source_license(content):
                        evidence.append("Open source license detected")
                        score += 0.4
                    break
            else:
                evidence.append("License type could not be determined")
                score += 0.2
        else:
            evidence.append("No LICENSE file found")
            # Heavy penalty for conferences that emphasize accessibility
            if self.conference_profile.get('emphasis_weights', {}).get('accessibility', 0) > 0.3:
                score = 0.0
        
        # Check for license information in README
        if self._has_license_in_readme():
            evidence.append("License information found in README")
            score = min(score + 0.2, 1.0)
        
        return min(score, 1.0), evidence
    
    def _evaluate_conference_specific_accessibility(self) -> tuple[float, List[str]]:
        """Evaluate conference-specific accessibility requirements."""
        evidence = []
        score = 0.0
        
        category = self.conference_profile.get('category', 'general')
        domain_keywords = self.domain_keywords
        
        # Check for conference-specific accessibility patterns
        if category == 'software_engineering':
            # Check for GitHub pages, CI/CD indicators
            if self._has_github_pages():
                evidence.append("GitHub Pages detected for online access")
                score += 0.2
            
            if self._has_ci_cd_indicators():
                evidence.append("CI/CD indicators found (accessibility through automation)")
                score += 0.2
                
        elif category == 'data_systems':
            # Check for data accessibility indicators
            if self._has_data_access_patterns():
                evidence.append("Data access patterns detected")
                score += 0.3
                
        elif category == 'hci':
            # Check for user interface accessibility
            if self._has_ui_accessibility_indicators():
                evidence.append("UI accessibility indicators found")
                score += 0.3
        
        # Check for domain-specific keywords in documentation
        keyword_matches = self._check_domain_keywords_in_docs()
        if keyword_matches:
            evidence.append(f"Found {len(keyword_matches)} conference-specific accessibility keywords")
            score += min(len(keyword_matches) * 0.1, 0.4)
        
        # Check for conference-specific file patterns
        if self._check_conference_file_patterns():
            evidence.append("Conference-specific file patterns detected")
            score += 0.2
        
        return min(score, 1.0), evidence
    
    def _calculate_conference_weighted_score(self, availability: float, archival: float, 
                                           license: float, conference_specific: float) -> float:
        """Calculate score using conference-specific weighting."""
        # Base weights
        weights = {
            'availability': 0.3,
            'archival': 0.3,
            'license': 0.25,
            'conference_specific': 0.15
        }
        
        # Adjust weights based on conference emphasis
        accessibility_emphasis = self.conference_profile.get('emphasis_weights', {}).get('accessibility', 0.3)
        
        if accessibility_emphasis > 0.35:  # High emphasis conferences
            weights['archival'] += 0.1
            weights['license'] += 0.1
            weights['availability'] -= 0.1
            weights['conference_specific'] -= 0.1
        elif accessibility_emphasis < 0.2:  # Low emphasis conferences
            weights['availability'] += 0.1
            weights['conference_specific'] += 0.1
            weights['archival'] -= 0.1
            weights['license'] -= 0.1
        
        return (availability * weights['availability'] + 
                archival * weights['archival'] + 
                license * weights['license'] + 
                conference_specific * weights['conference_specific'])
    
    def _generate_conference_aware_justification(self, availability_score: float, archival_score: float, 
                                               license_score: float, conference_score: float, 
                                               meets_threshold: bool) -> str:
        """Generate justification based on conference-specific evaluation."""
        conference_name = self.conference_profile.get('category', 'general').replace('_', ' ').title()
        threshold = self.quality_threshold
        
        justification = f"Conference-specific accessibility evaluation for {conference_name} venue (threshold: {threshold:.2f}). "
        
        if meets_threshold:
            justification += f"MEETS THRESHOLD: Artifact satisfies {conference_name} accessibility requirements. "
        else:
            justification += f"BELOW THRESHOLD: Artifact needs improvement to meet {conference_name} standards. "
        
        # Detailed component analysis
        if availability_score >= 0.8:
            justification += "Strong public availability. "
        elif availability_score < 0.5:
            justification += "Poor public accessibility - critical issue. "
        
        if archival_score >= 0.8:
            justification += "Excellent archival repository usage. "
        elif archival_score < 0.5:
            justification += "Inadequate archival status for long-term accessibility. "
        
        if license_score >= 0.8:
            justification += "Proper licensing for open access. "
        elif license_score < 0.5:
            justification += "Licensing issues may restrict accessibility. "
        
        if conference_score >= 0.5:
            justification += f"Meets {conference_name}-specific accessibility patterns."
        else:
            justification += f"Missing {conference_name}-specific accessibility features."
        
        return justification
    
    def _build_conference_context(self) -> str:
        """Build conference context for LLM evaluation."""
        context = f"Conference Category: {self.conference_profile.get('category', 'general')}\n"
        context += f"Quality Threshold: {self.quality_threshold}\n"
        context += f"Evaluation Style: {self.evaluation_style}\n"
        context += f"Accessibility Emphasis: {self.conference_profile.get('emphasis_weights', {}).get('accessibility', 0.3):.2f}\n"
        
        if self.domain_keywords:
            context += f"Domain Keywords: {', '.join(self.domain_keywords[:10])}\n"
        
        return context
    
    # Helper methods for conference-specific checks
    def _check_conference_archival_keywords(self) -> bool:
        """Check for conference-specific archival keywords in documentation."""
        for keyword in ['zenodo', 'figshare', 'archival', 'doi', 'persistent']:
            if self.kg_agent.readme_has_section(keyword):
                return True
        return False
    
    def _identify_license_type(self, content: str) -> Optional[str]:
        """Identify specific license type from content."""
        license_patterns = {
            'MIT': ['mit license'],
            'Apache-2.0': ['apache license', 'apache 2.0'],
            'GPL': ['gnu general public license', 'gpl'],
            'BSD': ['bsd license'],
            'CC-BY': ['creative commons', 'cc-by'],
            'CC0': ['cc0', 'public domain']
        }
        
        content_lower = content.lower()
        for license_type, patterns in license_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                return license_type
        return None
    
    def _is_conference_preferred_license(self, license_type: str) -> bool:
        """Check if license type is preferred by the conference category."""
        category = self.conference_profile.get('category', 'general')
        
        preferred_licenses = {
            'software_engineering': ['MIT', 'Apache-2.0', 'BSD'],
            'data_systems': ['CC-BY', 'CC0', 'MIT'],
            'hci': ['MIT', 'CC-BY'],
            'general': ['MIT', 'Apache-2.0', 'CC-BY']
        }
        
        return license_type in preferred_licenses.get(category, [])
    
    def _has_github_pages(self) -> bool:
        """Check for GitHub Pages indicators."""
        return self.kg_agent.file_exists("_config.yml") or self.kg_agent.file_exists("index.html")
    
    def _has_ci_cd_indicators(self) -> bool:
        """Check for CI/CD indicators."""
        ci_files = [".github/workflows", ".gitlab-ci.yml", ".travis.yml", "Jenkinsfile"]
        return any(self.kg_agent.file_exists(f) for f in ci_files)
    
    def _has_data_access_patterns(self) -> bool:
        """Check for data access patterns."""
        return (self.kg_agent.dataset_files_exist() and 
                self.kg_agent.readme_has_section("data"))
    
    def _has_ui_accessibility_indicators(self) -> bool:
        """Check for UI accessibility indicators."""
        return self.kg_agent.readme_has_section("accessibility") or self.kg_agent.readme_has_section("a11y")
    
    def _check_domain_keywords_in_docs(self) -> List[str]:
        """Check for domain keywords in documentation."""
        matches = []
        readme_content = self._get_readme_content().lower()
        
        for keyword in self.domain_keywords:
            if keyword.lower() in readme_content:
                matches.append(keyword)
        
        return matches
    
    def _check_conference_file_patterns(self) -> bool:
        """Check for conference-specific file patterns."""
        category = self.conference_profile.get('category', 'general')
        
        if category == 'software_engineering':
            return (self.kg_agent.file_exists("Dockerfile") or 
                   self.kg_agent.file_exists("requirements.txt") or
                   self.kg_agent.file_exists("setup.py"))
        elif category == 'data_systems':
            return (self.kg_agent.file_exists("data") or 
                   self.kg_agent.file_exists("datasets"))
        
        return False
    
    # Existing helper methods
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

    def _get_readme_content(self):
        for doc_file in self.artifact.get("documentation_files", []):
            if "readme" in doc_file.get("path", "").lower():
                content = doc_file.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(content)
                return content
        return ""
