"""
Configuration settings for Conference-Specific Algorithm 1
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    """Configuration class for conference-specific algorithm parameters."""
    
    # Model settings
    sentence_model_name: str = 'all-MiniLM-L6-v2'
    keybert_model_name: str = 'all-MiniLM-L6-v2'
    
    # Extraction parameters
    semantic_similarity_threshold: float = 0.7
    keyword_expansion_top_n: int = 8
    confidence_threshold: float = 0.6
    min_keyword_frequency: int = 2
    max_keywords_per_dimension: int = 50
    severity_weight_decay: float = 0.8
    
    # TF-IDF parameters
    tfidf_min_df: int = 1
    tfidf_max_df: float = 0.95
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_max_features: int = 10000
    
    # Clustering parameters
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 2
    
    # Confidence scoring weights
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'source_coverage': 0.2,
        'keyword_consensus': 0.3,
        'semantic_coherence': 0.2,
        'frequency_stability': 0.2,
        'cross_validation_score': 0.1
    })
    
    # Default dimensions
    default_dimensions: List[str] = field(default_factory=lambda: [
        'reproducibility',
        'documentation', 
        'accessibility',
        'usability',
        'experimental',
        'functionality'
    ])
    
    # Reliability thresholds
    high_reliability_threshold: float = 0.8
    medium_reliability_threshold: float = 0.6
    
    # Output settings
    output_formats: List[str] = field(default_factory=lambda: [
        'enhanced_csv',
        'aura_integration',
        'conference_analysis',
        'cross_conference_comparison'
    ])
    
    def get_reliability_flag(self, confidence_score: float) -> str:
        """Get reliability flag based on confidence score."""
        if confidence_score >= self.high_reliability_threshold:
            return 'high'
        elif confidence_score >= self.medium_reliability_threshold:
            return 'medium'
        else:
            return 'low'
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


# Default configuration instance
DEFAULT_CONFIG = Config() 