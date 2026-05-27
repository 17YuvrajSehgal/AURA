"""
Conference Guidelines Loader for AURA Evaluation System

This module loads and manages conference-specific evaluation criteria and guidelines
to make evaluations conference-aware and aligned with submission requirements.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)


class ConferenceGuidelinesLoader:
    """Loads and manages conference-specific evaluation guidelines"""
    
    def __init__(self, guidelines_dir: str = "processed"):
        """
        Initialize the conference guidelines loader
        
        Args:
            guidelines_dir: Directory containing processed conference guidelines
        """
        # Handle path resolution - look for guidelines directory relative to project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent  # Go up two levels to reach AURA root
        
        if not Path(guidelines_dir).is_absolute():
            # Try relative to project root first
            self.guidelines_dir = project_root / "data" / "conference_guideline_texts" / guidelines_dir
            if not self.guidelines_dir.exists():
                # Fallback to relative to script directory
                self.guidelines_dir = script_dir / guidelines_dir
        else:
            self.guidelines_dir = Path(guidelines_dir)
        self.conference_guidelines: Dict[str, Dict[str, Any]] = {}
        self.conference_mapping: Dict[str, str] = {}
        self._load_all_guidelines()
    
    def _load_all_guidelines(self):
        """Load all available conference guidelines"""
        if not self.guidelines_dir.exists():
            logger.warning(f"Guidelines directory not found: {self.guidelines_dir}")
            return
        
        for guideline_file in self.guidelines_dir.glob("*.md"):
            try:
                conference_info = self._parse_guideline_file(guideline_file)
                if conference_info:
                    conference_key = conference_info["conference_key"]
                    self.conference_guidelines[conference_key] = conference_info
                    self.conference_mapping[conference_info["conference_name"].lower()] = conference_key
                    
                    # Also add alternative mappings
                    name_parts = conference_info["conference_name"].lower().split()
                    for part in name_parts:
                        if len(part) > 2:  # Only meaningful parts
                            self.conference_mapping[part] = conference_key
                    
                    logger.info(f"Loaded guidelines for {conference_info['conference_name']}")
                    
            except Exception as e:
                logger.error(f"Failed to load guidelines from {guideline_file}: {e}")
    
    def _parse_guideline_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a single conference guideline file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract conference information from filename
            filename = file_path.stem
            conference_info = self._parse_conference_name(filename)
            
            # Parse guidelines content
            guidelines = self._parse_guidelines_content(content)
            
            return {
                "conference_key": filename,
                "conference_name": conference_info["name"],
                "conference_year": conference_info["year"],
                "original_filename": filename,
                "file_path": str(file_path),
                "guidelines": guidelines,
                "raw_content": content,
                "criteria_count": len(guidelines),
                "accessibility_criteria": self._extract_dimension_criteria(guidelines, "accessibility"),
                "documentation_criteria": self._extract_dimension_criteria(guidelines, "documentation"),
                "functionality_criteria": self._extract_dimension_criteria(guidelines, "functionality"),
                "reproducibility_criteria": self._extract_dimension_criteria(guidelines, "reproducibility"),
                "experimental_criteria": self._extract_dimension_criteria(guidelines, "experimental"),
                "usability_criteria": self._extract_dimension_criteria(guidelines, "usability")
            }
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _parse_conference_name(self, filename: str) -> Dict[str, str]:
        """Extract conference name and year from filename"""
        # Pattern: number_conference_year (e.g., "13_icse_2025")
        parts = filename.split('_')
        
        if len(parts) >= 3:
            conference_name = parts[1].upper()
            year = parts[2]
        else:
            conference_name = filename.upper()
            year = "2024"  # Default year
        
        return {
            "name": conference_name,
            "year": year
        }
    
    def _parse_guidelines_content(self, content: str) -> List[Dict[str, str]]:
        """Parse the guidelines content into structured criteria"""
        guidelines = []
        
        # Split content by numbered items
        pattern = r'(\d+)\.\s*\*\*([^*]+)\*\*:\s*(.+?)(?=\n\d+\.\s*\*\*|\n*$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for number, title, description in matches:
            guidelines.append({
                "number": int(number),
                "title": title.strip(),
                "description": description.strip(),
                "dimension": self._classify_criterion_dimension(title, description)
            })
        
        return guidelines
    
    def _classify_criterion_dimension(self, title: str, description: str) -> str:
        """Classify a criterion into evaluation dimensions"""
        title_lower = title.lower()
        description_lower = description.lower()
        
        # Accessibility indicators
        if any(keyword in title_lower for keyword in ['availability', 'access', 'repository', 'doi', 'archival']):
            return 'accessibility'
        
        # Documentation indicators  
        if any(keyword in title_lower for keyword in ['documentation', 'readme', 'license', 'instruction']):
            return 'documentation'
        
        # Functionality indicators
        if any(keyword in title_lower for keyword in ['functionality', 'executable', 'exercisable', 'complete']):
            return 'functionality'
        
        # Reproducibility indicators
        if any(keyword in title_lower for keyword in ['reproducib', 'replicat', 'result', 'experiment']):
            return 'reproducibility'
        
        # Experimental indicators
        if any(keyword in title_lower for keyword in ['evaluat', 'result', 'data', 'experiment', 'benchmark']):
            return 'experimental'
        
        # Usability indicators (reusability, setup, etc.)
        if any(keyword in title_lower for keyword in ['reusab', 'setup', 'installation', 'usage', 'user']):
            return 'usability'
        
        # Default to general if no clear classification
        return 'general'
    
    def _extract_dimension_criteria(self, guidelines: List[Dict[str, str]], dimension: str) -> List[Dict[str, str]]:
        """Extract criteria relevant to a specific evaluation dimension"""
        return [criterion for criterion in guidelines if criterion['dimension'] == dimension]
    
    def get_conference_guidelines(self, conference_name: str) -> Optional[Dict[str, Any]]:
        """Get guidelines for a specific conference"""
        # Try exact match first
        conference_key = conference_name.lower()
        if conference_key in self.conference_mapping:
            return self.conference_guidelines[self.conference_mapping[conference_key]]
        
        # Try partial match
        for mapped_name, key in self.conference_mapping.items():
            if conference_key in mapped_name or mapped_name in conference_key:
                return self.conference_guidelines[key]
        
        logger.warning(f"No guidelines found for conference: {conference_name}")
        return None
    
    def get_available_conferences(self) -> List[str]:
        """Get list of available conference names"""
        return [info["conference_name"] for info in self.conference_guidelines.values()]
    
    def get_conference_criteria_for_dimension(self, conference_name: str, dimension: str) -> List[Dict[str, str]]:
        """Get criteria for a specific conference and dimension"""
        guidelines = self.get_conference_guidelines(conference_name)
        if not guidelines:
            return []
        
        dimension_key = f"{dimension}_criteria"
        return guidelines.get(dimension_key, [])
    
    def format_conference_guidelines_for_prompt(self, conference_name: str, dimension: str = None) -> str:
        """Format conference guidelines for injection into prompts"""
        guidelines = self.get_conference_guidelines(conference_name)
        if not guidelines:
            return f"No specific guidelines found for {conference_name}. Using general evaluation criteria."
        
        conference_info = f"**{guidelines['conference_name']} {guidelines['conference_year']}**"
        
        if dimension:
            # Get dimension-specific criteria
            criteria = self.get_conference_criteria_for_dimension(conference_name, dimension)
            if criteria:
                formatted_criteria = []
                for criterion in criteria:
                    formatted_criteria.append(f"- **{criterion['title']}**: {criterion['description']}")
                
                return f"{conference_info} has the following {dimension.upper()} requirements:\n\n" + "\n".join(formatted_criteria)
            else:
                # If no dimension-specific criteria, use general guidelines
                return f"{conference_info} general artifact evaluation criteria:\n\n" + self._format_all_guidelines(guidelines['guidelines'])
        else:
            # Return all guidelines
            return f"{conference_info} artifact evaluation criteria:\n\n" + self._format_all_guidelines(guidelines['guidelines'])
    
    def _format_all_guidelines(self, guidelines: List[Dict[str, str]]) -> str:
        """Format all guidelines into a readable string"""
        formatted = []
        for criterion in guidelines:
            formatted.append(f"{criterion['number']}. **{criterion['title']}**: {criterion['description']}")
        return "\n".join(formatted)
    
    def get_conference_summary(self, conference_name: str) -> Dict[str, Any]:
        """Get a summary of conference requirements"""
        guidelines = self.get_conference_guidelines(conference_name)
        if not guidelines:
            return {}
        
        return {
            "conference_name": guidelines["conference_name"],
            "conference_year": guidelines["conference_year"],
            "total_criteria": guidelines["criteria_count"],
            "dimensions": {
                "accessibility": len(guidelines["accessibility_criteria"]),
                "documentation": len(guidelines["documentation_criteria"]),
                "functionality": len(guidelines["functionality_criteria"]),
                "reproducibility": len(guidelines["reproducibility_criteria"]),
                "experimental": len(guidelines["experimental_criteria"]),
                "usability": len(guidelines["usability_criteria"])
            },
            "key_requirements": [criterion['title'] for criterion in guidelines['guidelines'][:5]]  # Top 5
        }


# Global instance
conference_loader = ConferenceGuidelinesLoader() 