"""
Automated Conference Profile Generator for AURA Framework.
Generates conference-specific evaluation profiles by analyzing actual conference guidelines.
"""

import json
import logging
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

try:
    # Try relative imports first (when used as a package)
    from .utils import setup_logging
except ImportError:
    # Fall back to direct imports (when used as standalone)
    from utils import setup_logging

logger = logging.getLogger(__name__)


class ConferenceProfileGenerator:
    """
    Automatically generates conference-specific evaluation profiles by analyzing 
    conference guideline texts using NLP techniques.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the profile generator with NLP models."""
        self.sentence_model = SentenceTransformer(model_name)
        self.kw_model = KeyBERT(self.sentence_model)
        
        # Domain classification keywords for categorizing conferences
        self.domain_classification_keywords = {
            'software_engineering': [
                'software', 'code', 'programming', 'development', 'testing', 'debugging',
                'version control', 'build', 'compilation', 'deployment', 'maintenance',
                'requirements', 'design', 'architecture', 'engineering', 'methodology'
            ],
            'data_systems': [
                'database', 'query', 'storage', 'data processing', 'indexing', 'transaction',
                'performance', 'scalability', 'distributed', 'parallel', 'optimization',
                'throughput', 'latency', 'system', 'framework', 'platform'
            ],
            'human_computer_interaction': [
                'user', 'interface', 'interaction', 'usability', 'accessibility', 'design',
                'user experience', 'user study', 'evaluation', 'human factors', 'ergonomics',
                'prototype', 'visualization', 'mobile', 'web', 'application'
            ],
            'machine_learning': [
                'learning', 'algorithm', 'model', 'training', 'prediction', 'classification',
                'regression', 'neural', 'network', 'deep', 'feature', 'dataset', 'accuracy',
                'performance', 'optimization', 'validation', 'evaluation'
            ],
            'systems_architecture': [
                'architecture', 'system', 'performance', 'scalability', 'distributed',
                'parallel', 'concurrent', 'memory', 'processor', 'hardware', 'benchmark',
                'evaluation', 'optimization', 'efficiency', 'resource'
            ],
            'security': [
                'security', 'privacy', 'encryption', 'authentication', 'authorization',
                'vulnerability', 'attack', 'defense', 'cryptography', 'protocol',
                'threat', 'risk', 'compliance', 'audit', 'forensics'
            ]
        }
        
        # Evaluation dimension keywords
        self.dimension_keywords = {
            'reproducibility': [
                'reproduce', 'replication', 'reproducible', 'replicate', 'repeat',
                'validation', 'verify', 'confirm', 'independent', 'results'
            ],
            'documentation': [
                'documentation', 'readme', 'guide', 'manual', 'tutorial', 'instruction',
                'description', 'explanation', 'comment', 'annotation', 'help'
            ],
            'accessibility': [
                'accessible', 'available', 'public', 'open', 'free', 'download',
                'repository', 'archive', 'access', 'obtain', 'retrieve'
            ],
            'usability': [
                'usable', 'easy', 'simple', 'user-friendly', 'intuitive', 'straightforward',
                'install', 'setup', 'configure', 'run', 'execute', 'use'
            ],
            'experimental': [
                'experiment', 'evaluation', 'benchmark', 'test', 'analysis', 'study',
                'empirical', 'statistical', 'measurement', 'metric', 'comparison'
            ],
            'functionality': [
                'functional', 'function', 'work', 'operate', 'perform', 'execute',
                'implement', 'feature', 'capability', 'behavior', 'correct'
            ]
        }
        
        logger.info("Conference Profile Generator initialized")
    
    def generate_profiles_from_guidelines(self, guidelines_dir: str) -> Dict[str, Dict]:
        """
        Generate conference profiles by analyzing guideline texts.
        
        Args:
            guidelines_dir: Directory containing conference guideline files
            
        Returns:
            Dictionary of conference profiles
        """
        logger.info(f"Generating conference profiles from {guidelines_dir}")
        
        # Load conference guideline files
        conference_texts = self._load_conference_guidelines(guidelines_dir)
        
        if not conference_texts:
            logger.warning("No conference guideline files found")
            return {}
        
        profiles = {}
        
        for conf_name, text_content in conference_texts.items():
            logger.info(f"Analyzing {conf_name}...")
            
            # Extract domain keywords and classify conference
            domain_keywords = self._extract_domain_keywords(text_content)
            category = self._classify_conference_domain(text_content, domain_keywords)
            
            # Analyze dimension emphasis
            emphasis_weights = self._analyze_dimension_emphasis(text_content)
            
            # Extract quality thresholds and evaluation style
            quality_threshold, evaluation_style = self._analyze_evaluation_approach(text_content)
            
            # Create profile
            profile = {
                'category': category,
                'domain_keywords': domain_keywords,
                'emphasis_weights': emphasis_weights,
                'quality_threshold': quality_threshold,
                'evaluation_style': evaluation_style,
                'analysis_metadata': {
                    'text_length': len(text_content),
                    'guidelines_count': len(text_content.split('\n')),
                    'generated_timestamp': None  # Will be set when saved
                }
            }
            
            profiles[conf_name] = profile
            logger.info(f"Generated profile for {conf_name}: category={category}, "
                       f"top_emphasis={max(emphasis_weights, key=emphasis_weights.get)}")
        
        logger.info(f"Generated {len(profiles)} conference profiles")
        return profiles
    
    def _load_conference_guidelines(self, guidelines_dir: str) -> Dict[str, str]:
        """Load conference guideline files and extract conference names."""
        conference_texts = {}
        
        for filename in os.listdir(guidelines_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(guidelines_dir, filename)
                
                # Extract conference name from filename
                match = re.match(r'\d+_([a-zA-Z_]+)_\d{4}\.md', filename)
                if match:
                    conf_name = match.group(1).upper()
                else:
                    conf_name = filename.replace('.md', '').upper()
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        conference_texts[conf_name] = content
                        logger.debug(f"Loaded {conf_name}: {len(content)} characters")
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
        
        return conference_texts
    
    def _extract_domain_keywords(self, text_content: str, top_n: int = 20) -> List[str]:
        """Extract domain-specific keywords from conference guidelines."""
        try:
            # Use KeyBERT to extract domain keywords
            keywords = self.kw_model.extract_keywords(
                text_content,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                use_mmr=True,
                diversity=0.5
            )
            
            domain_keywords = [kw[0] for kw in keywords]
            logger.debug(f"Extracted domain keywords: {domain_keywords[:10]}")
            return domain_keywords
            
        except Exception as e:
            logger.error(f"Error extracting domain keywords: {e}")
            return []
    
    def _classify_conference_domain(self, text_content: str, domain_keywords: List[str]) -> str:
        """Classify conference into domain category based on content analysis."""
        text_lower = text_content.lower()
        domain_scores = {}
        
        # Score each domain category
        for domain, keywords in self.domain_classification_keywords.items():
            score = 0
            
            # Count keyword occurrences in text
            for keyword in keywords:
                score += text_lower.count(keyword.lower())
            
            # Bonus for domain keywords that match classification keywords
            for domain_kw in domain_keywords:
                for class_kw in keywords:
                    if class_kw.lower() in domain_kw.lower() or domain_kw.lower() in class_kw.lower():
                        score += 2
            
            domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            logger.debug(f"Domain classification scores: {domain_scores}")
            return best_domain
        
        return 'general'
    
    def _analyze_dimension_emphasis(self, text_content: str) -> Dict[str, float]:
        """Analyze emphasis on different evaluation dimensions."""
        text_lower = text_content.lower()
        dimension_scores = {}
        
        # Count occurrences of dimension-related keywords
        for dimension, keywords in self.dimension_keywords.items():
            score = 0
            
            for keyword in keywords:
                # Count exact matches
                score += text_lower.count(keyword.lower())
                
                # Count variations (word boundaries)
                import re
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                score += len(re.findall(pattern, text_lower))
            
            dimension_scores[dimension] = score
        
        # Normalize scores to weights (sum to 1.0)
        total_score = sum(dimension_scores.values())
        if total_score > 0:
            emphasis_weights = {dim: score / total_score for dim, score in dimension_scores.items()}
        else:
            # Default equal weights if no keywords found
            emphasis_weights = {dim: 1.0 / len(dimension_scores) for dim in dimension_scores}
        
        logger.debug(f"Dimension emphasis weights: {emphasis_weights}")
        return emphasis_weights
    
    def _analyze_evaluation_approach(self, text_content: str) -> Tuple[float, str]:
        """Analyze evaluation approach and quality thresholds."""
        text_lower = text_content.lower()
        
        # Quality threshold indicators
        strict_indicators = ['rigorous', 'comprehensive', 'thorough', 'detailed', 'complete']
        moderate_indicators = ['adequate', 'sufficient', 'appropriate', 'reasonable']
        lenient_indicators = ['minimal', 'basic', 'simple', 'straightforward']
        
        strict_count = sum(text_lower.count(word) for word in strict_indicators)
        moderate_count = sum(text_lower.count(word) for word in moderate_indicators)
        lenient_count = sum(text_lower.count(word) for word in lenient_indicators)
        
        # Determine quality threshold
        if strict_count > moderate_count and strict_count > lenient_count:
            quality_threshold = 0.8
            evaluation_style = 'strict'
        elif lenient_count > moderate_count and lenient_count > strict_count:
            quality_threshold = 0.5
            evaluation_style = 'lenient'
        else:
            quality_threshold = 0.7
            evaluation_style = 'moderate'
        
        logger.debug(f"Evaluation approach: threshold={quality_threshold}, style={evaluation_style}")
        return quality_threshold, evaluation_style
    
    def save_profiles(self, profiles: Dict[str, Dict], output_file: str):
        """Save generated profiles to JSON file."""
        # Add timestamp to metadata
        from datetime import datetime
        for profile in profiles.values():
            profile['analysis_metadata']['generated_timestamp'] = datetime.now().isoformat()
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(profiles, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(profiles)} profiles to {output_file}")
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")
    
    def load_profiles(self, input_file: str) -> Dict[str, Dict]:
        """Load profiles from JSON file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
            logger.info(f"Loaded {len(profiles)} profiles from {input_file}")
            return profiles
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            return {}


class ConferenceProfileManager:
    """
    Manager class for conference profiles with automatic generation capability.
    """
    
    def __init__(self, guidelines_dir: str = None, profiles_file: str = None):
        """
        Initialize the profile manager.
        
        Args:
            guidelines_dir: Directory containing conference guidelines
            profiles_file: File to save/load generated profiles
        """
        self.guidelines_dir = guidelines_dir
        self.profiles_file = profiles_file or "conference_profiles.json"
        self.profiles = {}
        self.generator = ConferenceProfileGenerator()
        
        # Try to load existing profiles
        if os.path.exists(self.profiles_file):
            self.profiles = self.generator.load_profiles(self.profiles_file)
            logger.info(f"Loaded existing profiles: {list(self.profiles.keys())}")
        elif guidelines_dir and os.path.exists(guidelines_dir):
            # Generate profiles from guidelines
            logger.info("No existing profiles found, generating from guidelines...")
            self.generate_and_save_profiles()
        else:
            logger.warning("No profiles found and no guidelines directory specified")
    
    def generate_and_save_profiles(self):
        """Generate profiles from guidelines and save them."""
        if not self.guidelines_dir:
            raise ValueError("Guidelines directory not specified")
        
        self.profiles = self.generator.generate_profiles_from_guidelines(self.guidelines_dir)
        self.generator.save_profiles(self.profiles, self.profiles_file)
    
    def get_conference_profile(self, conference_name: str) -> Dict[str, Any]:
        """
        Get profile for a specific conference.
        
        Args:
            conference_name: Name of the conference
            
        Returns:
            Conference profile dictionary
        """
        conf_name = conference_name.upper()
        
        if conf_name in self.profiles:
            return self.profiles[conf_name]
        
        # If not found, try fuzzy matching
        for name in self.profiles.keys():
            if conf_name in name or name in conf_name:
                logger.info(f"Using fuzzy match: {name} for {conf_name}")
                return self.profiles[name]
        
        # Return default profile if not found
        logger.warning(f"Conference {conf_name} not found, using default profile")
        return self._get_default_profile()
    
    def _get_default_profile(self) -> Dict[str, Any]:
        """Get default conference profile."""
        return {
            'category': 'general',
            'domain_keywords': [
                'artifact', 'evaluation', 'research', 'software', 'data',
                'experiment', 'analysis', 'implementation', 'methodology'
            ],
            'emphasis_weights': {
                'reproducibility': 0.25,
                'documentation': 0.20,
                'accessibility': 0.15,
                'usability': 0.15,
                'experimental': 0.15,
                'functionality': 0.10
            },
            'quality_threshold': 0.7,
            'evaluation_style': 'moderate',
            'analysis_metadata': {
                'text_length': 0,
                'guidelines_count': 0,
                'generated_timestamp': None
            }
        }
    
    def list_available_conferences(self) -> List[str]:
        """List all available conference profiles."""
        return list(self.profiles.keys())
    
    def get_profile_summary(self) -> Dict[str, Dict]:
        """Get summary of all profiles."""
        summary = {}
        for conf_name, profile in self.profiles.items():
            summary[conf_name] = {
                'category': profile.get('category', 'unknown'),
                'top_emphasis': max(profile.get('emphasis_weights', {}), 
                                  key=profile.get('emphasis_weights', {}).get, 
                                  default='unknown'),
                'evaluation_style': profile.get('evaluation_style', 'unknown'),
                'domain_keywords_count': len(profile.get('domain_keywords', []))
            }
        return summary 