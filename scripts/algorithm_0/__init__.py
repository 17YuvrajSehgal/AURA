"""
Algorithm 0: Conference-Specific Evaluation Criteria Extraction

This package provides enhanced evaluation criteria extraction tailored to specific conferences,
enabling more accurate artifact evaluation based on conference-specific requirements.
"""

__version__ = "1.0.0"
__author__ = "AURA Framework Team"

from .conference_specific_algorithm import ConferenceSpecificAlgorithm1
from .conference_profiles import ConferenceProfileManager
from .utils import NumpyEncoder, setup_logging
from .config import Config

__all__ = [
    'ConferenceSpecificAlgorithm1',
    'ConferenceProfileManager', 
    'NumpyEncoder',
    'setup_logging',
    'Config'
] 