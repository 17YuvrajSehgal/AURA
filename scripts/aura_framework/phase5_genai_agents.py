"""
🧠 Phase 5: Evaluation Engine with GenAI Agents
Goal: Evaluate new artifacts using AI-powered agents per dimension.

Features:
- ReproducibilityAgent: Looks for scripts, dataset usage, environments
- DocumentationAgent: Examines README structure, clarity, organization  
- AccessibilityAgent: Checks public repo, open license, data sharing
- UsabilityAgent: Evaluates installation ease, setup guidance
- ExperimentalAgent: Seeks experiments, results, benchmarks
- FunctionalityAgent: Verifies testing, modular code, model execution
- Multi-agent orchestration and consensus building
"""

import json
import logging
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

# LLM imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Local imports
from config import config, NODE_TYPES, RELATIONSHIP_TYPES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationDimension(Enum):
    """Evaluation dimensions for artifacts"""
    REPRODUCIBILITY = "reproducibility"
    DOCUMENTATION = "documentation"
    ACCESSIBILITY = "accessibility"
    USABILITY = "usability"
    EXPERIMENTAL = "experimental"
    FUNCTIONALITY = "functionality"


@dataclass
class EvaluationResult:
    """Result from a single evaluation agent"""
    dimension: EvaluationDimension
    score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    justification: str
    evidence: List[str]
    suggestions: List[str]
    agent_version: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MultiAgentResult:
    """Combined result from multiple evaluation agents"""
    artifact_id: str
    individual_results: Dict[EvaluationDimension, EvaluationResult]
    weighted_score: float
    consensus_score: float
    confidence_score: float
    final_recommendation: str
    improvement_suggestions: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ArtifactContext:
    """Context information for artifact evaluation"""
    artifact_id: str
    metadata: Dict[str, Any]
    documentation_content: str
    graph_context: Dict[str, Any]
    vector_context: List[Dict[str, Any]]
    pattern_context: Dict[str, Any]


class BaseEvaluationAgent(ABC):
    """Base class for all evaluation agents"""
    
    def __init__(self, 
                 dimension: EvaluationDimension,
                 llm_provider: str = "openai",
                 model_name: str = "gpt-4-turbo",
                 temperature: float = 0.1):
        self.dimension = dimension
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.agent_version = "1.0"
        
        # Initialize LLM client
        self._initialize_llm_client()
        
        logger.info(f"Initialized {self.__class__.__name__} with {llm_provider}")

    def _initialize_llm_client(self):
        """Initialize the LLM client"""
        if self.llm_provider == "openai":
            if not OPENAI_AVAILABLE or not config.llm.api_key:
                raise ValueError("OpenAI not available or API key not provided")
            self.llm_client = openai
            self.llm_client.api_key = config.llm.api_key
            
        elif self.llm_provider == "anthropic":
            if not ANTHROPIC_AVAILABLE or not config.llm.api_key:
                raise ValueError("Anthropic not available or API key not provided")
            self.llm_client = Anthropic(api_key=config.llm.api_key)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    @abstractmethod
    def get_evaluation_prompt(self, context: ArtifactContext) -> str:
        """Get the evaluation prompt for this agent"""
        pass

    @abstractmethod
    def parse_llm_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Parse LLM response into score, justification, evidence, and suggestions"""
        pass

    async def evaluate(self, context: ArtifactContext) -> EvaluationResult:
        """Evaluate an artifact and return results"""
        try:
            # Get evaluation prompt
            prompt = self.get_evaluation_prompt(context)
            
            # Call LLM
            llm_response = await self._call_llm(prompt)
            
            # Parse response
            score, justification, evidence, suggestions = self.parse_llm_response(llm_response)
            
            # Calculate confidence based on evidence quality
            confidence = self._calculate_confidence(evidence, context)
            
            return EvaluationResult(
                dimension=self.dimension,
                score=score,
                confidence=confidence,
                justification=justification,
                evidence=evidence,
                suggestions=suggestions,
                agent_version=self.agent_version
            )
            
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            return self._create_error_result(str(e))

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt"""
        if self.llm_provider == "openai":
            response = await self._call_openai(prompt)
        elif self.llm_provider == "anthropic":
            response = await self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        return response

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=config.llm.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        try:
            response = await self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=config.llm.max_tokens,
                temperature=self.temperature,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _get_system_prompt(self) -> str:
        """Get system prompt for this agent"""
        return f"""You are an expert artifact evaluator specializing in {self.dimension.value}. 
        Your job is to evaluate research artifacts based on specific criteria and provide:
        1. A score from 0.0 to 1.0
        2. Clear justification for the score
        3. Specific evidence from the artifact
        4. Actionable improvement suggestions
        
        Be thorough, objective, and constructive in your evaluation."""

    def _calculate_confidence(self, evidence: List[str], context: ArtifactContext) -> float:
        """Calculate confidence score based on evidence quality"""
        if not evidence:
            return 0.1
        
        # Base confidence on evidence quantity and quality
        confidence = min(len(evidence) / 5.0, 1.0)  # Up to 5 pieces of evidence
        
        # Boost confidence if evidence is specific and detailed
        specific_evidence = sum(1 for e in evidence if len(e) > 50)
        confidence += specific_evidence * 0.1
        
        return min(confidence, 1.0)

    def _create_error_result(self, error_message: str) -> EvaluationResult:
        """Create error result when evaluation fails"""
        return EvaluationResult(
            dimension=self.dimension,
            score=0.0,
            confidence=0.0,
            justification=f"Evaluation failed: {error_message}",
            evidence=[],
            suggestions=["Fix evaluation error and retry"],
            agent_version=self.agent_version
        )


class ReproducibilityAgent(BaseEvaluationAgent):
    """Agent for evaluating reproducibility aspects"""
    
    def __init__(self, **kwargs):
        super().__init__(EvaluationDimension.REPRODUCIBILITY, **kwargs)

    def get_evaluation_prompt(self, context: ArtifactContext) -> str:
        return f"""
        Evaluate the REPRODUCIBILITY of this research artifact.
        
        Artifact: {context.artifact_id}
        
        Documentation Content:
        {context.documentation_content[:2000]}
        
        Metadata:
        - Has Docker: {context.metadata.get('has_docker', False)}
        - Has Requirements: {context.metadata.get('has_requirements_txt', False)}
        - Has Setup Script: {context.metadata.get('has_setup_py', False)}
        - Repository Size: {context.metadata.get('repo_size_mb', 0)} MB
        
        Graph Context (related tools and commands):
        {json.dumps(context.graph_context, indent=2)[:1000]}
        
        Evaluation Criteria:
        1. Environment Setup: Are dependencies clearly specified?
        2. Execution Scripts: Are there clear scripts to run experiments?
        3. Data Availability: Is required data accessible or provided?
        4. Configuration: Are parameters and settings documented?
        5. Version Control: Are specific versions of tools/libraries specified?
        
        Please provide:
        1. SCORE (0.0-1.0): Overall reproducibility score
        2. JUSTIFICATION: Detailed explanation of the score
        3. EVIDENCE: Specific examples from the artifact (list format)
        4. SUGGESTIONS: Concrete improvements to enhance reproducibility (list format)
        
        Format your response as:
        SCORE: X.X
        JUSTIFICATION: [detailed explanation]
        EVIDENCE:
        - [evidence item 1]
        - [evidence item 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]
        """

    def parse_llm_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Parse LLM response for reproducibility evaluation"""
        lines = response.strip().split('\n')
        
        score = 0.0
        justification = ""
        evidence = []
        suggestions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except:
                    score = 0.0
                    
            elif line.startswith('JUSTIFICATION:'):
                justification = line.split(':', 1)[1].strip()
                current_section = 'justification'
                
            elif line.startswith('EVIDENCE:'):
                current_section = 'evidence'
                
            elif line.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
                
            elif line.startswith('- ') and current_section == 'evidence':
                evidence.append(line[2:])
                
            elif line.startswith('- ') and current_section == 'suggestions':
                suggestions.append(line[2:])
                
            elif current_section == 'justification' and line:
                justification += " " + line
        
        return score, justification.strip(), evidence, suggestions


class DocumentationAgent(BaseEvaluationAgent):
    """Agent for evaluating documentation quality"""
    
    def __init__(self, **kwargs):
        super().__init__(EvaluationDimension.DOCUMENTATION, **kwargs)

    def get_evaluation_prompt(self, context: ArtifactContext) -> str:
        return f"""
        Evaluate the DOCUMENTATION quality of this research artifact.
        
        Artifact: {context.artifact_id}
        
        Documentation Content:
        {context.documentation_content[:3000]}
        
        Pattern Context (common successful patterns):
        {json.dumps(context.pattern_context, indent=2)[:1000]}
        
        Evaluation Criteria:
        1. Structure: Is the documentation well-organized with clear sections?
        2. Completeness: Are all necessary components documented?
        3. Clarity: Is the language clear and easy to understand?
        4. Examples: Are there sufficient examples and use cases?
        5. Navigation: Is it easy to find information?
        
        Essential Sections to Look For:
        - Introduction/Overview
        - Installation/Setup
        - Usage/Examples
        - API Documentation (if applicable)
        - Contributing Guidelines
        - License Information
        
        Please provide:
        1. SCORE (0.0-1.0): Overall documentation quality score
        2. JUSTIFICATION: Detailed explanation of the score
        3. EVIDENCE: Specific examples from the documentation (list format)
        4. SUGGESTIONS: Concrete improvements to enhance documentation (list format)
        
        Format your response as:
        SCORE: X.X
        JUSTIFICATION: [detailed explanation]
        EVIDENCE:
        - [evidence item 1]
        - [evidence item 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]
        """

    def parse_llm_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Parse LLM response for documentation evaluation"""
        return self._parse_standard_response(response)
    
    def _parse_standard_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Standard response parser for most agents"""
        lines = response.strip().split('\n')
        
        score = 0.0
        justification = ""
        evidence = []
        suggestions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except:
                    score = 0.0
                    
            elif line.startswith('JUSTIFICATION:'):
                justification = line.split(':', 1)[1].strip()
                current_section = 'justification'
                
            elif line.startswith('EVIDENCE:'):
                current_section = 'evidence'
                
            elif line.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
                
            elif line.startswith('- ') and current_section == 'evidence':
                evidence.append(line[2:])
                
            elif line.startswith('- ') and current_section == 'suggestions':
                suggestions.append(line[2:])
                
            elif current_section == 'justification' and line:
                justification += " " + line
        
        return score, justification.strip(), evidence, suggestions


class AccessibilityAgent(BaseEvaluationAgent):
    """Agent for evaluating accessibility aspects"""
    
    def __init__(self, **kwargs):
        super().__init__(EvaluationDimension.ACCESSIBILITY, **kwargs)

    def get_evaluation_prompt(self, context: ArtifactContext) -> str:
        return f"""
        Evaluate the ACCESSIBILITY of this research artifact.
        
        Artifact: {context.artifact_id}
        
        Metadata:
        - Repository URL: {context.metadata.get('repository_url', 'Not specified')}
        - DOI: {context.metadata.get('doi', 'Not specified')}
        - License: {context.metadata.get('license_type', 'Not specified')}
        - Has License File: {context.metadata.get('has_license', False)}
        
        Documentation Content:
        {context.documentation_content[:2000]}
        
        Evaluation Criteria:
        1. Public Availability: Is the artifact publicly accessible?
        2. Archival Repository: Is it stored in a persistent repository?
        3. Licensing: Is there an appropriate open-source license?
        4. Data Sharing: Are datasets and data files accessible?
        5. Persistent Identifiers: Are DOIs or other persistent IDs provided?
        
        Please provide:
        1. SCORE (0.0-1.0): Overall accessibility score
        2. JUSTIFICATION: Detailed explanation of the score
        3. EVIDENCE: Specific examples from the artifact (list format)
        4. SUGGESTIONS: Concrete improvements to enhance accessibility (list format)
        
        Format your response as:
        SCORE: X.X
        JUSTIFICATION: [detailed explanation]
        EVIDENCE:
        - [evidence item 1]
        - [evidence item 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]
        """

    def parse_llm_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Parse LLM response for accessibility evaluation"""
        return self._parse_standard_response(response)

    def _parse_standard_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Standard response parser"""
        lines = response.strip().split('\n')
        
        score = 0.0
        justification = ""
        evidence = []
        suggestions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except:
                    score = 0.0
                    
            elif line.startswith('JUSTIFICATION:'):
                justification = line.split(':', 1)[1].strip()
                current_section = 'justification'
                
            elif line.startswith('EVIDENCE:'):
                current_section = 'evidence'
                
            elif line.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
                
            elif line.startswith('- ') and current_section == 'evidence':
                evidence.append(line[2:])
                
            elif line.startswith('- ') and current_section == 'suggestions':
                suggestions.append(line[2:])
                
            elif current_section == 'justification' and line:
                justification += " " + line
        
        return score, justification.strip(), evidence, suggestions


class UsabilityAgent(BaseEvaluationAgent):
    """Agent for evaluating usability aspects"""
    
    def __init__(self, **kwargs):
        super().__init__(EvaluationDimension.USABILITY, **kwargs)

    def get_evaluation_prompt(self, context: ArtifactContext) -> str:
        return f"""
        Evaluate the USABILITY of this research artifact.
        
        Artifact: {context.artifact_id}
        
        Documentation Content:
        {context.documentation_content[:2500]}
        
        Vector Context (similar successful artifacts):
        {json.dumps(context.vector_context, indent=2)[:800]}
        
        Evaluation Criteria:
        1. Installation Ease: How easy is it to install and set up?
        2. User Guidance: Are there clear instructions for users?
        3. Error Handling: Is error handling and troubleshooting covered?
        4. User Interface: Is the interface (CLI/GUI/API) user-friendly?
        5. Learning Curve: How easy is it for new users to get started?
        
        Please provide:
        1. SCORE (0.0-1.0): Overall usability score
        2. JUSTIFICATION: Detailed explanation of the score
        3. EVIDENCE: Specific examples from the artifact (list format)
        4. SUGGESTIONS: Concrete improvements to enhance usability (list format)
        
        Format your response as:
        SCORE: X.X
        JUSTIFICATION: [detailed explanation]
        EVIDENCE:
        - [evidence item 1]
        - [evidence item 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]
        """

    def parse_llm_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Parse LLM response for usability evaluation"""
        return self._parse_standard_response(response)

    def _parse_standard_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Standard response parser"""
        lines = response.strip().split('\n')
        
        score = 0.0
        justification = ""
        evidence = []
        suggestions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except:
                    score = 0.0
                    
            elif line.startswith('JUSTIFICATION:'):
                justification = line.split(':', 1)[1].strip()
                current_section = 'justification'
                
            elif line.startswith('EVIDENCE:'):
                current_section = 'evidence'
                
            elif line.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
                
            elif line.startswith('- ') and current_section == 'evidence':
                evidence.append(line[2:])
                
            elif line.startswith('- ') and current_section == 'suggestions':
                suggestions.append(line[2:])
                
            elif current_section == 'justification' and line:
                justification += " " + line
        
        return score, justification.strip(), evidence, suggestions


class ExperimentalAgent(BaseEvaluationAgent):
    """Agent for evaluating experimental aspects"""
    
    def __init__(self, **kwargs):
        super().__init__(EvaluationDimension.EXPERIMENTAL, **kwargs)

    def get_evaluation_prompt(self, context: ArtifactContext) -> str:
        return f"""
        Evaluate the EXPERIMENTAL aspects of this research artifact.
        
        Artifact: {context.artifact_id}
        
        Documentation Content:
        {context.documentation_content[:2500]}
        
        Metadata:
        - Total Files: {context.metadata.get('total_files', 0)}
        - Code Files: {context.metadata.get('code_files', 0)}
        - Data Files: {context.metadata.get('data_files', 0)}
        
        Evaluation Criteria:
        1. Experimental Setup: Are experiments clearly defined and documented?
        2. Data Requirements: Are data requirements and sources specified?
        3. Validation Evidence: Is there evidence of validation and testing?
        4. Results Reproduction: Can experimental results be reproduced?
        5. Benchmarking: Are appropriate baselines and comparisons provided?
        
        Please provide:
        1. SCORE (0.0-1.0): Overall experimental quality score
        2. JUSTIFICATION: Detailed explanation of the score
        3. EVIDENCE: Specific examples from the artifact (list format)
        4. SUGGESTIONS: Concrete improvements to enhance experimental rigor (list format)
        
        Format your response as:
        SCORE: X.X
        JUSTIFICATION: [detailed explanation]
        EVIDENCE:
        - [evidence item 1]
        - [evidence item 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]
        """

    def parse_llm_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Parse LLM response for experimental evaluation"""
        return self._parse_standard_response(response)

    def _parse_standard_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Standard response parser"""
        lines = response.strip().split('\n')
        
        score = 0.0
        justification = ""
        evidence = []
        suggestions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except:
                    score = 0.0
                    
            elif line.startswith('JUSTIFICATION:'):
                justification = line.split(':', 1)[1].strip()
                current_section = 'justification'
                
            elif line.startswith('EVIDENCE:'):
                current_section = 'evidence'
                
            elif line.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
                
            elif line.startswith('- ') and current_section == 'evidence':
                evidence.append(line[2:])
                
            elif line.startswith('- ') and current_section == 'suggestions':
                suggestions.append(line[2:])
                
            elif current_section == 'justification' and line:
                justification += " " + line
        
        return score, justification.strip(), evidence, suggestions


class FunctionalityAgent(BaseEvaluationAgent):
    """Agent for evaluating functionality aspects"""
    
    def __init__(self, **kwargs):
        super().__init__(EvaluationDimension.FUNCTIONALITY, **kwargs)

    def get_evaluation_prompt(self, context: ArtifactContext) -> str:
        return f"""
        Evaluate the FUNCTIONALITY of this research artifact.
        
        Artifact: {context.artifact_id}
        
        Documentation Content:
        {context.documentation_content[:2500]}
        
        Metadata:
        - Has Jupyter: {context.metadata.get('has_jupyter', False)}
        - Has Docker: {context.metadata.get('has_docker', False)}
        - Code Files: {context.metadata.get('code_files', 0)}
        
        Graph Context (commands and tools):
        {json.dumps(context.graph_context, indent=2)[:1000]}
        
        Evaluation Criteria:
        1. Core Functionality: Is the main functionality clearly implemented?
        2. Code Quality: Is the code well-structured and modular?
        3. Testing: Are there tests to verify functionality?
        4. Executability: Can the artifact be executed successfully?
        5. Completeness: Are all claimed features implemented?
        
        Please provide:
        1. SCORE (0.0-1.0): Overall functionality score
        2. JUSTIFICATION: Detailed explanation of the score
        3. EVIDENCE: Specific examples from the artifact (list format)
        4. SUGGESTIONS: Concrete improvements to enhance functionality (list format)
        
        Format your response as:
        SCORE: X.X
        JUSTIFICATION: [detailed explanation]
        EVIDENCE:
        - [evidence item 1]
        - [evidence item 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]
        """

    def parse_llm_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Parse LLM response for functionality evaluation"""
        return self._parse_standard_response(response)

    def _parse_standard_response(self, response: str) -> Tuple[float, str, List[str], List[str]]:
        """Standard response parser"""
        lines = response.strip().split('\n')
        
        score = 0.0
        justification = ""
        evidence = []
        suggestions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except:
                    score = 0.0
                    
            elif line.startswith('JUSTIFICATION:'):
                justification = line.split(':', 1)[1].strip()
                current_section = 'justification'
                
            elif line.startswith('EVIDENCE:'):
                current_section = 'evidence'
                
            elif line.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
                
            elif line.startswith('- ') and current_section == 'evidence':
                evidence.append(line[2:])
                
            elif line.startswith('- ') and current_section == 'suggestions':
                suggestions.append(line[2:])
                
            elif current_section == 'justification' and line:
                justification += " " + line
        
        return score, justification.strip(), evidence, suggestions


class MultiAgentEvaluationOrchestrator:
    """Orchestrates multiple evaluation agents and combines results"""
    
    def __init__(self, 
                 kg_builder=None,
                 vector_engine=None,
                 pattern_engine=None,
                 evaluation_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the multi-agent evaluation orchestrator
        
        Args:
            kg_builder: Knowledge graph builder for context
            vector_engine: Vector embedding engine for context
            pattern_engine: Pattern analysis engine for context
            evaluation_weights: Weights for different evaluation dimensions
        """
        self.kg_builder = kg_builder
        self.vector_engine = vector_engine
        self.pattern_engine = pattern_engine
        
        # Default evaluation weights
        self.evaluation_weights = evaluation_weights or config.default_evaluation_weights
        
        # Initialize agents
        self.agents = {
            EvaluationDimension.REPRODUCIBILITY: ReproducibilityAgent(),
            EvaluationDimension.DOCUMENTATION: DocumentationAgent(),
            EvaluationDimension.ACCESSIBILITY: AccessibilityAgent(),
            EvaluationDimension.USABILITY: UsabilityAgent(),
            EvaluationDimension.EXPERIMENTAL: ExperimentalAgent(),
            EvaluationDimension.FUNCTIONALITY: FunctionalityAgent()
        }
        
        logger.info("Multi-Agent Evaluation Orchestrator initialized with 6 agents")

    async def evaluate_artifact(self, 
                               artifact_id: str,
                               artifact_data: Dict[str, Any],
                               target_conference: Optional[str] = None) -> MultiAgentResult:
        """
        Perform comprehensive multi-agent evaluation of an artifact
        
        Args:
            artifact_id: Unique identifier for the artifact
            artifact_data: Processed artifact data
            target_conference: Target conference for evaluation (optional)
            
        Returns:
            Combined evaluation results from all agents
        """
        logger.info(f"Starting multi-agent evaluation for artifact: {artifact_id}")
        
        # Prepare context for agents
        context = await self._prepare_artifact_context(artifact_id, artifact_data)
        
        # Run all agents in parallel
        evaluation_tasks = []
        for dimension, agent in self.agents.items():
            task = agent.evaluate(context)
            evaluation_tasks.append((dimension, task))
        
        # Collect results
        individual_results = {}
        
        for dimension, task in evaluation_tasks:
            try:
                result = await task
                individual_results[dimension] = result
                logger.info(f"Completed {dimension.value} evaluation: {result.score:.3f}")
            except Exception as e:
                logger.error(f"Failed {dimension.value} evaluation: {e}")
                # Create error result
                individual_results[dimension] = EvaluationResult(
                    dimension=dimension,
                    score=0.0,
                    confidence=0.0,
                    justification=f"Evaluation failed: {str(e)}",
                    evidence=[],
                    suggestions=["Fix evaluation error and retry"],
                    agent_version="1.0"
                )
        
        # Combine results
        combined_result = self._combine_agent_results(
            artifact_id, individual_results, target_conference
        )
        
        logger.info(f"Multi-agent evaluation completed for {artifact_id}: {combined_result.weighted_score:.3f}")
        return combined_result

    async def _prepare_artifact_context(self, 
                                      artifact_id: str, 
                                      artifact_data: Dict[str, Any]) -> ArtifactContext:
        """Prepare comprehensive context for agent evaluation"""
        
        # Extract documentation content
        documentation_content = ""
        for doc_file in artifact_data.get('documentation_files', []):
            for section in doc_file.get('sections', []):
                documentation_content += f"\n# {section['heading']}\n{section['content']}\n"
        
        # Get graph context (tools, commands, relationships)
        graph_context = self._get_graph_context(artifact_id)
        
        # Get vector context (similar artifacts)
        vector_context = await self._get_vector_context(artifact_id)
        
        # Get pattern context (successful patterns)
        pattern_context = self._get_pattern_context()
        
        return ArtifactContext(
            artifact_id=artifact_id,
            metadata=artifact_data.get('metadata', {}),
            documentation_content=documentation_content[:5000],  # Limit size
            graph_context=graph_context,
            vector_context=vector_context,
            pattern_context=pattern_context
        )

    def _get_graph_context(self, artifact_id: str) -> Dict[str, Any]:
        """Get relevant graph context for the artifact"""
        context = {
            'tools': [],
            'commands': [],
            'related_artifacts': []
        }
        
        if not self.kg_builder:
            return context
        
        try:
            # Get tools and commands from knowledge graph
            if hasattr(self.kg_builder, 'nx_graph'):
                G = self.kg_builder.nx_graph
                
                for node, data in G.nodes(data=True):
                    if data.get('artifact_id') == artifact_id:
                        node_type = data.get('node_type')
                        
                        if node_type == NODE_TYPES['TOOL']:
                            context['tools'].append(data.get('name', ''))
                        elif node_type == NODE_TYPES['COMMAND']:
                            context['commands'].append(data.get('command', ''))
                
        except Exception as e:
            logger.warning(f"Error getting graph context: {e}")
        
        return context

    async def _get_vector_context(self, artifact_id: str) -> List[Dict[str, Any]]:
        """Get similar artifacts for context"""
        context = []
        
        if not self.vector_engine:
            return context
        
        try:
            # Find similar artifacts
            search_results = self.vector_engine.semantic_search(
                query=f"artifact {artifact_id}",
                top_k=3,
                exclude_same_artifact=True
            )
            
            for result in search_results:
                context.append({
                    'artifact_id': result.embedding_record.artifact_id,
                    'heading': result.embedding_record.heading,
                    'relevance_score': result.relevance_score
                })
                
        except Exception as e:
            logger.warning(f"Error getting vector context: {e}")
        
        return context

    def _get_pattern_context(self) -> Dict[str, Any]:
        """Get successful pattern context"""
        context = {
            'common_sections': [],
            'successful_tools': [],
            'structural_patterns': []
        }
        
        if not self.pattern_engine:
            return context
        
        try:
            # Get common section patterns
            if hasattr(self.pattern_engine, 'section_patterns'):
                for pattern in self.pattern_engine.section_patterns[:5]:
                    context['common_sections'].append({
                        'heading': pattern.heading,
                        'frequency': pattern.frequency,
                        'success_correlation': pattern.success_correlation
                    })
            
            # Get successful tools
            if hasattr(self.pattern_engine, 'structural_patterns'):
                for pattern in self.pattern_engine.structural_patterns[:3]:
                    context['successful_tools'].extend(pattern.common_tools)
                    context['structural_patterns'].append({
                        'name': pattern.pattern_name,
                        'sequence': pattern.section_sequence[:5],
                        'success_rate': pattern.success_rate
                    })
        
        except Exception as e:
            logger.warning(f"Error getting pattern context: {e}")
        
        return context

    def _combine_agent_results(self, 
                             artifact_id: str,
                             individual_results: Dict[EvaluationDimension, EvaluationResult],
                             target_conference: Optional[str] = None) -> MultiAgentResult:
        """Combine individual agent results into final assessment"""
        
        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, result in individual_results.items():
            weight = self.evaluation_weights.get(dimension.value, 0.0)
            weighted_score += result.score * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        # Calculate consensus score (agreement between agents)
        scores = [result.score for result in individual_results.values()]
        if scores:
            consensus_score = 1.0 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.0
            consensus_score = max(0.0, min(1.0, consensus_score))
        else:
            consensus_score = 0.0
        
        # Calculate overall confidence
        confidences = [result.confidence for result in individual_results.values()]
        confidence_score = np.mean(confidences) if confidences else 0.0
        
        # Generate final recommendation
        final_recommendation = self._generate_final_recommendation(
            weighted_score, consensus_score, confidence_score, target_conference
        )
        
        # Collect improvement suggestions
        improvement_suggestions = []
        for result in individual_results.values():
            improvement_suggestions.extend(result.suggestions[:2])  # Top 2 per agent
        
        return MultiAgentResult(
            artifact_id=artifact_id,
            individual_results=individual_results,
            weighted_score=weighted_score,
            consensus_score=consensus_score,
            confidence_score=confidence_score,
            final_recommendation=final_recommendation,
            improvement_suggestions=improvement_suggestions[:10]  # Limit to 10
        )

    def _generate_final_recommendation(self, 
                                     weighted_score: float,
                                     consensus_score: float,
                                     confidence_score: float,
                                     target_conference: Optional[str] = None) -> str:
        """Generate final recommendation based on scores"""
        
        if weighted_score >= 0.8 and consensus_score >= 0.7 and confidence_score >= 0.7:
            recommendation = "ACCEPT - High quality artifact with strong evaluation consensus"
        elif weighted_score >= 0.6 and consensus_score >= 0.5:
            recommendation = "CONDITIONAL ACCEPT - Good artifact with room for improvement"
        elif weighted_score >= 0.4:
            recommendation = "MAJOR REVISION - Significant improvements needed"
        else:
            recommendation = "REJECT - Multiple critical issues identified"
        
        if target_conference:
            recommendation += f" for {target_conference}"
        
        return recommendation

    def get_evaluation_summary(self, result: MultiAgentResult) -> Dict[str, Any]:
        """Get comprehensive evaluation summary"""
        summary = {
            'artifact_id': result.artifact_id,
            'overall_scores': {
                'weighted_score': result.weighted_score,
                'consensus_score': result.consensus_score,
                'confidence_score': result.confidence_score
            },
            'final_recommendation': result.final_recommendation,
            'dimension_scores': {
                dimension.value: {
                    'score': result.score,
                    'confidence': result.confidence,
                    'justification': result.justification[:200] + "..." if len(result.justification) > 200 else result.justification
                }
                for dimension, result in result.individual_results.items()
            },
            'top_strengths': [],
            'top_weaknesses': [],
            'priority_improvements': result.improvement_suggestions[:5],
            'evaluation_timestamp': result.timestamp
        }
        
        # Identify strengths and weaknesses
        for dimension, eval_result in result.individual_results.items():
            if eval_result.score >= 0.7:
                summary['top_strengths'].append(f"{dimension.value}: {eval_result.score:.2f}")
            elif eval_result.score <= 0.4:
                summary['top_weaknesses'].append(f"{dimension.value}: {eval_result.score:.2f}")
        
        return summary


async def main():
    """Example usage of the Multi-Agent Evaluation System"""
    
    # Initialize orchestrator
    orchestrator = MultiAgentEvaluationOrchestrator()
    
    # Example artifact data
    artifact_data = {
        'metadata': {
            'artifact_name': 'example-ml-tool',
            'has_docker': True,
            'has_requirements_txt': True,
            'has_license': True,
            'repository_url': 'https://github.com/example/ml-tool',
            'total_files': 25,
            'code_files': 15
        },
        'documentation_files': [
            {
                'path': 'README.md',
                'sections': [
                    {
                        'heading': 'Installation',
                        'content': 'To install this tool, run: pip install -r requirements.txt'
                    },
                    {
                        'heading': 'Usage',
                        'content': 'To use this tool: python main.py --input data.csv --output results.json'
                    }
                ]
            }
        ]
    }
    
    # Perform evaluation
    result = await orchestrator.evaluate_artifact(
        artifact_id='example-ml-tool',
        artifact_data=artifact_data,
        target_conference='ICSE'
    )
    
    # Print results
    print("\n🧠 Phase 5: GenAI Multi-Agent Evaluation Results")
    print("=" * 60)
    print(f"🎯 Artifact: {result.artifact_id}")
    print(f"📊 Weighted Score: {result.weighted_score:.3f}")
    print(f"🤝 Consensus Score: {result.consensus_score:.3f}")
    print(f"🎪 Confidence Score: {result.confidence_score:.3f}")
    print(f"💡 Recommendation: {result.final_recommendation}")
    
    print(f"\n📋 Individual Agent Scores:")
    for dimension, eval_result in result.individual_results.items():
        print(f"  - {dimension.value.title()}: {eval_result.score:.3f} (confidence: {eval_result.confidence:.3f})")
    
    print(f"\n💪 Top Improvement Suggestions:")
    for i, suggestion in enumerate(result.improvement_suggestions[:5]):
        print(f"  {i+1}. {suggestion}")
    
    # Get detailed summary
    summary = orchestrator.get_evaluation_summary(result)
    print(f"\n✅ Strengths: {', '.join(summary['top_strengths'])}")
    print(f"⚠️  Weaknesses: {', '.join(summary['top_weaknesses'])}")


if __name__ == "__main__":
    asyncio.run(main()) 