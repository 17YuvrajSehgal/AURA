"""
LangChain Chains for Research Artifact Evaluation

This module implements LangChain-based evaluation chains for each evaluation dimension,
integrating with the RAG retrieval system to provide contextual information for assessment.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

try:
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from config import config, PROMPT_TEMPLATES, AuraConfig, DIMENSION_WEIGHTS, ACCEPTANCE_THRESHOLDS
from rag_retrieval import RAGRetriever
from knowledge_graph_builder import KnowledgeGraphBuilder
from conference_guidelines_loader import conference_loader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Represents the result of an evaluation dimension"""
    dimension: str
    overall_rating: float
    detailed_assessment: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    summary: str
    raw_output: str


class EvaluationOutputParser:
    """Custom output parser for evaluation results"""
    
    def __init__(self, dimension: str):
        self.dimension = dimension
    
    def parse(self, text: str) -> EvaluationResult:
        """Parse the LLM output into structured evaluation result"""
        
        # Initialize default values
        overall_rating = 0.0
        detailed_assessment = {}
        strengths = []
        weaknesses = []
        recommendations = []
        summary = ""
        
        try:
            import re
            
            # Extract overall rating
            if "Overall Rating:" in text:
                rating_lines = [line for line in text.split('\n') if 'Overall Rating:' in line]
                if rating_lines:
                    rating_line = rating_lines[0]
                    rating_str = rating_line.split('Overall Rating:')[1].strip()
                    # Extract number from rating string (e.g., "[1-5]" or "3/5" or "3")
                    rating_match = re.search(r'(\d+(?:\.\d+)?)', rating_str)
                    if rating_match:
                        overall_rating = min(5.0, max(1.0, float(rating_match.group(1))))  # Clamp between 1-5
            
            # Extract detailed ratings
            rating_pattern = r'\*\*\d+\.\s*(.+?)\s*\(Rating:\s*(\d+(?:\.\d+)?)/5\)\*\*'
            ratings = re.findall(rating_pattern, text)
            for criterion, score in ratings:
                try:
                    detailed_assessment[criterion.strip()] = min(5.0, max(1.0, float(score)))  # Clamp between 1-5
                except ValueError:
                    logger.warning(f"Could not parse score '{score}' for criterion '{criterion}'")
            
            # Extract strengths
            if "### Strengths:" in text:
                strengths_section = text.split("### Strengths:")[1].split("###")[0]
                strengths = [line.strip().lstrip('- ') for line in strengths_section.split('\n') 
                            if line.strip() and line.strip().startswith('-')]
            
            # Extract weaknesses
            if "### Weaknesses:" in text:
                weaknesses_section = text.split("### Weaknesses:")[1].split("###")[0]
                weaknesses = [line.strip().lstrip('- ') for line in weaknesses_section.split('\n') 
                             if line.strip() and line.strip().startswith('-')]
            
            # Extract recommendations
            if "### Recommendations:" in text:
                recommendations_section = text.split("### Recommendations:")[1].split("###")[0]
                recommendations = [line.strip().lstrip('- ') for line in recommendations_section.split('\n') 
                                  if line.strip() and line.strip().startswith('-')]
            
            # Extract summary
            if "### Summary:" in text:
                summary_section = text.split("### Summary:")[1]
                # Clean up summary (remove extra whitespace, handle multi-line)
                summary_lines = [line.strip() for line in summary_section.split('\n') if line.strip()]
                if summary_lines:
                    summary = ' '.join(summary_lines[:3])  # Take first 3 non-empty lines
                
        except Exception as e:
            logger.warning(f"Error parsing evaluation output for {self.dimension}: {e}")
            logger.debug(f"Raw text that failed to parse: {text[:500]}...")
        
        return EvaluationResult(
            dimension=self.dimension,
            overall_rating=overall_rating,
            detailed_assessment=detailed_assessment,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            summary=summary,
            raw_output=text
        )


class PromptTemplateLoader:
    """Loads and manages prompt templates for evaluation dimensions"""
    
    def __init__(self, templates_dir: str = "templates"):
        # Handle both absolute and relative paths
        if not Path(templates_dir).is_absolute():
            # If relative path, make it relative to this script's directory
            script_dir = Path(__file__).parent
            self.templates_dir = script_dir / templates_dir
        else:
            self.templates_dir = Path(templates_dir)
        
        self.templates: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates from the templates directory"""
        for dimension, template_file in PROMPT_TEMPLATES.items():
            template_path = self.templates_dir / template_file
            
            if template_path.exists():
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        self.templates[dimension] = f.read()
                    logger.info(f"Loaded template for {dimension}")
                except Exception as e:
                    logger.error(f"Failed to load template for {dimension}: {e}")
                    self.templates[dimension] = self._get_default_template(dimension)
            else:
                logger.warning(f"Template file not found for {dimension}: {template_path}")
                self.templates[dimension] = self._get_default_template(dimension)
    
    def _get_default_template(self, dimension: str) -> str:
        """Provide a default template if the file is not found"""
        return f"""
        You are evaluating the {dimension.upper()} of a research artifact.
        
        Artifact Information: {{artifact_info}}
        Context: {{context}}
        
        Please provide a detailed evaluation of the {dimension} aspects of this artifact.
        Rate the overall {dimension} on a scale of 1-5 and provide specific feedback.
        
        ## {dimension.title()} Evaluation
        
        ### Overall Rating: [1-5]
        
        ### Assessment:
        [Your detailed assessment here]
        
        ### Strengths:
        - [List strengths]
        
        ### Weaknesses:
        - [List weaknesses]
        
        ### Recommendations:
        - [Provide recommendations]
        
        ### Summary:
        [Brief summary]
        """
    
    def get_template(self, dimension: str) -> str:
        """Get the template for a specific dimension"""
        return self.templates.get(dimension, self._get_default_template(dimension))


class EvaluationChain:
    """Individual evaluation chain for a specific dimension"""
    
    def __init__(self, dimension: str, template_loader: PromptTemplateLoader, 
                 llm: ChatOpenAI, rag_retriever: Optional[RAGRetriever] = None,
                 conference_name: Optional[str] = None):
        self.dimension = dimension
        self.template_loader = template_loader
        self.conference_name = conference_name
        self.llm = llm
        self.rag_retriever = rag_retriever
        
        # Create the prompt template
        template_content = template_loader.get_template(dimension)
        self.prompt = PromptTemplate.from_template(template_content)
        
        # Create the chain
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt
        )
        
        # Create output parser separately
        self.output_parser = EvaluationOutputParser(dimension)
        
        logger.info(f"Created evaluation chain for {dimension}")
    
    def evaluate(self, artifact_data: Dict[str, Any], 
                 additional_context: Dict[str, Any] = None) -> EvaluationResult:
        """Run the evaluation for this dimension"""
        
        # Prepare context data
        context_data = self._prepare_context(artifact_data, additional_context)
        
        # Get relevant context from RAG if available
        if self.rag_retriever:
            rag_context = self._get_rag_context(artifact_data, context_data)
            context_data.update(rag_context)
        
        try:
            # Run the chain
            raw_output = self.chain.run(**context_data)
            
            # Parse the output
            result = self.output_parser.parse(raw_output)
            logger.info(f"Completed {self.dimension} evaluation")
            return result
        
        except Exception as e:
            logger.error(f"Error in {self.dimension} evaluation: {e}")
            # Return a default result in case of error
            return EvaluationResult(
                dimension=self.dimension,
                overall_rating=0.0,
                detailed_assessment={},
                strengths=[],
                weaknesses=[f"Evaluation failed: {str(e)}"],
                recommendations=["Re-run evaluation after fixing issues"],
                summary=f"Evaluation failed due to error: {str(e)}",
                raw_output=""
            )
    
    def _prepare_context(self, artifact_data: Dict[str, Any], 
                        additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare context data for the evaluation prompt"""
        
        context = {
            "artifact_info": self._format_artifact_info(artifact_data),
            "context": "No additional context provided"
        }
        
        # Add dimension-specific context
        if self.dimension == "accessibility":
            context.update({
                "files_info": self._format_files_info(artifact_data),
                "dependencies_info": self._format_dependencies_info(artifact_data)
            })
        
        elif self.dimension == "documentation":
            context.update({
                "documentation_files": self._format_documentation_files(artifact_data),
                "code_structure": self._format_code_structure(artifact_data)
            })
        
        elif self.dimension == "experimental":
            context.update({
                "experimental_components": self._format_experimental_components(artifact_data),
                "datasets_info": self._format_datasets_info(artifact_data),
                "scripts_info": self._format_scripts_info(artifact_data)
            })
        
        elif self.dimension == "functionality":
            context.update({
                "code_components": self._format_code_components(artifact_data),
                "dependencies_info": self._format_dependencies_info(artifact_data),
                "execution_results": self._format_execution_results(artifact_data)
            })
        
        elif self.dimension == "reproducibility":
            context.update({
                "setup_instructions": self._format_setup_instructions(artifact_data),
                "dependencies_info": self._format_dependencies_info(artifact_data),
                "data_config": self._format_data_config(artifact_data)
            })
        
        elif self.dimension == "usability":
            context.update({
                "interface_info": self._format_interface_info(artifact_data),
                "documentation_info": self._format_documentation_info(artifact_data),
                "examples_info": self._format_examples_info(artifact_data)
            })
        
        # Add additional context if provided
        if additional_context:
            context["context"] = self._format_additional_context(additional_context)
        
        # Add conference-specific guidelines
        context["conference_guidelines"] = self._format_conference_guidelines()
        
        return context
    
    def _get_rag_context(self, artifact_data: Dict[str, Any], 
                        context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant context from RAG retrieval system"""
        
        if not self.rag_retriever:
            return {}
        
        try:
            # Create a query based on the dimension
            query = f"Information about {self.dimension} aspects of the artifact"
            
            # Get relevant context
            retrieval_results = self.rag_retriever.retrieve_for_section(
                section_type=self.dimension,
                query=query,
                top_k=5
            )
            
            # Format the retrieved context
            if retrieval_results:
                rag_context = "\n\n".join([
                    f"Source: {result.source_path}\n{result.content}"
                    for result in retrieval_results
                ])
                return {"rag_context": rag_context}
        
        except Exception as e:
            logger.warning(f"Failed to get RAG context for {self.dimension}: {e}")
        
        return {}
    
    def _format_artifact_info(self, artifact_data: Dict[str, Any]) -> str:
        """Format basic artifact information"""
        info_parts = []
        
        if "artifact_name" in artifact_data:
            info_parts.append(f"Name: {artifact_data['artifact_name']}")
        
        if "artifact_path" in artifact_data:
            info_parts.append(f"Path: {artifact_data['artifact_path']}")
        
        if "repo_size_mb" in artifact_data:
            info_parts.append(f"Size: {artifact_data['repo_size_mb']} MB")
        
        if "extraction_method" in artifact_data:
            info_parts.append(f"Extraction: {artifact_data['extraction_method']}")
        
        return "\n".join(info_parts) if info_parts else "No artifact information available"
    
    def _format_files_info(self, artifact_data: Dict[str, Any]) -> str:
        """Format file structure information"""
        files_info = []
        
        for file_type in ["documentation_files", "code_files", "docker_files", "data_files"]:
            if file_type in artifact_data and artifact_data[file_type]:
                files_info.append(f"{file_type}: {len(artifact_data[file_type])} files")
        
        return "\n".join(files_info) if files_info else "No file information available"
    
    def _format_dependencies_info(self, artifact_data: Dict[str, Any]) -> str:
        """Format dependency information"""
        # This would extract dependency info from various files
        deps = []
        
        # Check for Python requirements
        for file_info in artifact_data.get("code_files", []):
            if "requirements" in file_info.get("path", "").lower():
                deps.append("Python requirements file found")
        
        # Check for Docker files
        if artifact_data.get("docker_files"):
            deps.append("Docker configuration available")
        
        return "\n".join(deps) if deps else "No dependency information found"
    
    def _format_documentation_files(self, artifact_data: Dict[str, Any]) -> str:
        """Format documentation files information"""
        doc_files = artifact_data.get("documentation_files", [])
        if not doc_files:
            return "No documentation files found"
        
        doc_info = []
        for doc_file in doc_files:
            path = doc_file.get("path", "")
            content_lines = len(doc_file.get("content", []))
            doc_info.append(f"- {path} ({content_lines} lines)")
        
        return "\n".join(doc_info)
    
    def _format_code_structure(self, artifact_data: Dict[str, Any]) -> str:
        """Format code structure information"""
        tree_structure = artifact_data.get("tree_structure", [])
        if tree_structure:
            return "\n".join(tree_structure[:20])  # Limit to first 20 lines
        return "No code structure information available"
    
    def _format_experimental_components(self, artifact_data: Dict[str, Any]) -> str:
        """Format experimental components information"""
        # Look for experimental scripts, test files, etc.
        experimental = []
        
        code_files = artifact_data.get("code_files", [])
        for file_info in code_files:
            path = file_info.get("path", "").lower()
            if any(keyword in path for keyword in ["test", "experiment", "eval", "benchmark"]):
                experimental.append(f"- {file_info.get('path', '')}")
        
        return "\n".join(experimental) if experimental else "No experimental components found"
    
    def _format_datasets_info(self, artifact_data: Dict[str, Any]) -> str:
        """Format dataset information"""
        data_files = artifact_data.get("data_files", [])
        if not data_files:
            return "No datasets found"
        
        data_info = []
        for data_file in data_files:
            name = data_file.get("name", "")
            size = data_file.get("size_kb", 0)
            data_info.append(f"- {name} ({size} KB)")
        
        return "\n".join(data_info)
    
    def _format_scripts_info(self, artifact_data: Dict[str, Any]) -> str:
        """Format scripts information"""
        scripts = []
        
        code_files = artifact_data.get("code_files", [])
        for file_info in code_files:
            path = file_info.get("path", "")
            if path.endswith((".sh", ".py", ".r")):
                scripts.append(f"- {path}")
        
        return "\n".join(scripts) if scripts else "No scripts found"
    
    def _format_code_components(self, artifact_data: Dict[str, Any]) -> str:
        """Format code components information"""
        code_files = artifact_data.get("code_files", [])
        if not code_files:
            return "No code files found"
        
        components = []
        for file_info in code_files:
            path = file_info.get("path", "")
            content_lines = len(file_info.get("content", []))
            components.append(f"- {path} ({content_lines} lines)")
        
        return "\n".join(components)
    
    def _format_execution_results(self, artifact_data: Dict[str, Any]) -> str:
        """Format execution results if available"""
        # This would contain results from actually running the artifact
        return "No execution results available (static analysis only)"
    
    def _format_setup_instructions(self, artifact_data: Dict[str, Any]) -> str:
        """Format setup instructions"""
        # Look for README or setup files
        doc_files = artifact_data.get("documentation_files", [])
        for doc_file in doc_files:
            if "readme" in doc_file.get("path", "").lower():
                content = doc_file.get("content", [])
                # Look for setup sections
                setup_lines = []
                in_setup = False
                for line in content:
                    if any(keyword in line.lower() for keyword in ["setup", "install", "requirement"]):
                        in_setup = True
                    if in_setup and line.strip():
                        setup_lines.append(line)
                    if in_setup and len(setup_lines) > 10:  # Limit output
                        break
                
                if setup_lines:
                    return "\n".join(setup_lines)
        
        return "No setup instructions found"
    
    def _format_data_config(self, artifact_data: Dict[str, Any]) -> str:
        """Format data configuration information"""
        config_info = []
        
        # Look for configuration files
        code_files = artifact_data.get("code_files", [])
        for file_info in code_files:
            path = file_info.get("path", "").lower()
            if any(keyword in path for keyword in ["config", "setting", "param"]):
                config_info.append(f"- {file_info.get('path', '')}")
        
        return "\n".join(config_info) if config_info else "No configuration files found"
    
    def _format_interface_info(self, artifact_data: Dict[str, Any]) -> str:
        """Format user interface information"""
        # Look for main entry points, CLIs, etc.
        interfaces = []
        
        code_files = artifact_data.get("code_files", [])
        for file_info in code_files:
            path = file_info.get("path", "")
            if any(keyword in path.lower() for keyword in ["main", "cli", "app", "run"]):
                interfaces.append(f"- {path}")
        
        return "\n".join(interfaces) if interfaces else "No clear interface files found"
    
    def _format_documentation_info(self, artifact_data: Dict[str, Any]) -> str:
        """Format documentation information for usability"""
        doc_files = artifact_data.get("documentation_files", [])
        if not doc_files:
            return "No documentation available"
        
        return f"{len(doc_files)} documentation files found"
    
    def _format_examples_info(self, artifact_data: Dict[str, Any]) -> str:
        """Format examples information"""
        examples = []
        
        # Look for example files or directories
        all_files = (artifact_data.get("code_files", []) + 
                    artifact_data.get("documentation_files", []))
        
        for file_info in all_files:
            path = file_info.get("path", "").lower()
            if any(keyword in path for keyword in ["example", "demo", "tutorial", "sample"]):
                examples.append(f"- {file_info.get('path', '')}")
        
        return "\n".join(examples) if examples else "No examples found"
    
    def _format_additional_context(self, additional_context: Dict[str, Any]) -> str:
        """Format additional context information"""
        context_parts = []
        for key, value in additional_context.items():
            if isinstance(value, (str, int, float)):
                context_parts.append(f"{key}: {value}")
            elif isinstance(value, (list, dict)):
                context_parts.append(f"{key}: {json.dumps(value, indent=2)}")
        
        return "\n".join(context_parts)
    
    def _format_conference_guidelines(self) -> str:
        """Format conference-specific guidelines for this dimension"""
        if not self.conference_name:
            return "No specific conference guidelines provided. Using general evaluation criteria."
        
        try:
            guidelines_text = conference_loader.format_conference_guidelines_for_prompt(
                conference_name=self.conference_name,
                dimension=self.dimension
            )
            return guidelines_text
        except Exception as e:
            logger.warning(f"Failed to load conference guidelines for {self.conference_name}: {e}")
            return f"Failed to load guidelines for {self.conference_name}. Using general evaluation criteria."


class ArtifactEvaluationOrchestrator:
    """Main orchestrator for running all evaluation dimensions"""
    
    def __init__(self, knowledge_graph_builder: Optional[KnowledgeGraphBuilder] = None,
                 use_rag: bool = True, conference_name: Optional[str] = None):
        
        self.conference_name = conference_name
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        
        # Initialize template loader
        self.template_loader = PromptTemplateLoader(config.aura.template_dir)
        
        # Initialize RAG retriever if knowledge graph is provided
        self.rag_retriever = None
        if use_rag and knowledge_graph_builder:
            try:
                knowledge_graph_builder.enrich_section_links_with_similarity(threshold=0.6)

                self.rag_retriever = RAGRetriever(knowledge_graph_builder)
                logger.info("RAG retriever initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG retriever: {e}")

        # Create evaluation chains for each dimension
        self.evaluation_chains = {}
        aura_config = AuraConfig()
        
        for dimension in aura_config.sections:
            self.evaluation_chains[dimension] = EvaluationChain(
                dimension=dimension,
                template_loader=self.template_loader,
                llm=self.llm,
                rag_retriever=self.rag_retriever,
                conference_name=self.conference_name
            )
        
        logger.info(f"Initialized orchestrator with {len(self.evaluation_chains)} evaluation dimensions")
    
    def evaluate_artifact(self, artifact_data: Dict[str, Any], 
                         dimensions: Optional[List[str]] = None,
                         additional_context: Dict[str, Any] = None) -> Dict[str, EvaluationResult]:
        """
        Run evaluation for specified dimensions or all dimensions
        
        Args:
            artifact_data: The artifact data to evaluate
            dimensions: List of dimensions to evaluate (None for all)
            additional_context: Additional context to provide to evaluators
            
        Returns:
            Dictionary mapping dimension names to evaluation results
        """
        
        if dimensions is None:
            dimensions = list(self.evaluation_chains.keys())
        
        logger.info(f"Starting evaluation for dimensions: {dimensions}")
        
        results = {}
        
        for dimension in dimensions:
            if dimension not in self.evaluation_chains:
                logger.warning(f"Unknown dimension: {dimension}")
                continue
            
            logger.info(f"Evaluating {dimension}...")
            
            try:
                result = self.evaluation_chains[dimension].evaluate(
                    artifact_data=artifact_data,
                    additional_context=additional_context
                )
                results[dimension] = result
                logger.info(f"Completed {dimension} evaluation (Rating: {result.overall_rating}/5)")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {dimension}: {e}")
                results[dimension] = EvaluationResult(
                    dimension=dimension,
                    overall_rating=0.0,
                    detailed_assessment={},
                    strengths=[],
                    weaknesses=[f"Evaluation failed: {str(e)}"],
                    recommendations=["Re-run evaluation after fixing issues"],
                    summary=f"Evaluation failed: {str(e)}",
                    raw_output=""
                )
        
        logger.info("Completed all evaluations")
        return results
    
    def generate_comprehensive_report(self, evaluation_results: Dict[str, EvaluationResult],
                                    artifact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report with weighted scoring"""
        
        # Calculate simple average (unweighted)
        overall_scores = [result.overall_rating for result in evaluation_results.values()]
        simple_average_rating = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        
        # Calculate weighted scores
        weighted_scores = self._calculate_weighted_scores(evaluation_results)
        weighted_overall_rating = weighted_scores["weighted_overall_score"]
        dimension_percentages = weighted_scores["dimension_percentages"]
        dimension_weighted_scores = weighted_scores["dimension_weighted_scores"]
        
        # Collect all strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        all_recommendations = []
        
        for result in evaluation_results.values():
            all_strengths.extend(result.strengths)
            all_weaknesses.extend(result.weaknesses)
            all_recommendations.extend(result.recommendations)
        
        # Create comprehensive report
        report = {
            "artifact_info": {
                "name": artifact_data.get("artifact_name", "Unknown"),
                "path": artifact_data.get("artifact_path", ""),
                "size_mb": artifact_data.get("repo_size_mb", 0),
                "extraction_method": artifact_data.get("extraction_method", "")
            },
            "overall_rating": weighted_overall_rating,  # Weighted overall score
            "simple_average_rating": simple_average_rating,  # Unweighted average for comparison
            "dimension_scores": {
                dimension: result.overall_rating 
                for dimension, result in evaluation_results.items()
            },
            "weighted_scoring": {
                "weighted_overall_score": weighted_overall_rating,
                "weighted_overall_percentage": weighted_overall_rating * 20,  # Convert 5-point scale to percentage
                "dimension_percentages": dimension_percentages,
                "dimension_weighted_scores": dimension_weighted_scores,
                "dimension_weights": DIMENSION_WEIGHTS,
                "acceptance_probability": self._calculate_acceptance_probability(weighted_overall_rating)
            },
            "detailed_evaluations": {
                dimension: {
                    "rating": result.overall_rating,
                    "detailed_assessment": result.detailed_assessment,
                    "strengths": result.strengths,
                    "weaknesses": result.weaknesses,
                    "recommendations": result.recommendations,
                    "summary": result.summary
                }
                for dimension, result in evaluation_results.items()
            },
            "summary": {
                "total_strengths": len(set(all_strengths)),
                "total_weaknesses": len(set(all_weaknesses)),
                "total_recommendations": len(set(all_recommendations)),
                "dimensions_evaluated": len(evaluation_results),
                "highest_scoring_dimension": max(evaluation_results.items(), 
                                               key=lambda x: x[1].overall_rating)[0] if evaluation_results else None,
                "lowest_scoring_dimension": min(evaluation_results.items(), 
                                              key=lambda x: x[1].overall_rating)[0] if evaluation_results else None
            }
        }
        
        return report
    
    def _calculate_weighted_scores(self, evaluation_results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """Calculate weighted scores based on dimension importance"""
        
        # Initialize results
        dimension_percentages = {}
        dimension_weighted_scores = {}
        total_weighted_score = 0.0
        total_weight_used = 0.0
        
        # Calculate weighted scores for each dimension
        for dimension, result in evaluation_results.items():
            if dimension in DIMENSION_WEIGHTS:
                weight = DIMENSION_WEIGHTS[dimension]
                raw_score = result.overall_rating  # Score out of 5
                
                # Convert to percentage (5-point scale to 100%)
                dimension_percentage = (raw_score / 5.0) * 100
                dimension_percentages[dimension] = dimension_percentage
                
                # Calculate weighted contribution
                weighted_contribution = (raw_score / 5.0) * weight
                dimension_weighted_scores[dimension] = weighted_contribution
                
                total_weighted_score += weighted_contribution
                total_weight_used += weight
                
                logger.debug(f"{dimension}: {raw_score:.2f}/5 ({dimension_percentage:.1f}%) × {weight:.3f} = {weighted_contribution:.3f}")
            else:
                logger.warning(f"No weight defined for dimension: {dimension}")
                # Default weight for unknown dimensions
                dimension_percentages[dimension] = (result.overall_rating / 5.0) * 100
                dimension_weighted_scores[dimension] = 0.0
        
        # Normalize if weights don't sum to 1.0
        if total_weight_used > 0 and abs(total_weight_used - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total_weight_used:.3f}, normalizing...")
            total_weighted_score = total_weighted_score / total_weight_used
        
        # Convert back to 5-point scale for consistency
        weighted_overall_score = total_weighted_score * 5.0
        
        return {
            "weighted_overall_score": weighted_overall_score,
            "dimension_percentages": dimension_percentages,
            "dimension_weighted_scores": dimension_weighted_scores,
            "total_weight_used": total_weight_used
        }
    
    def _calculate_acceptance_probability(self, weighted_score: float) -> Dict[str, Any]:
        """Calculate acceptance probability based on weighted score"""
        
        # Convert 5-point scale to percentage
        score_percentage = (weighted_score / 5.0)
        
        # Determine acceptance category
        if score_percentage >= ACCEPTANCE_THRESHOLDS["excellent"]:
            category = "excellent"
            probability_text = "Very High Chance"
            probability_range = "85-100%"
        elif score_percentage >= ACCEPTANCE_THRESHOLDS["good"]:
            category = "good"
            probability_text = "Good Chance"
            probability_range = "70-85%"
        elif score_percentage >= ACCEPTANCE_THRESHOLDS["acceptable"]:
            category = "acceptable"
            probability_text = "Moderate Chance"
            probability_range = "55-70%"
        elif score_percentage >= ACCEPTANCE_THRESHOLDS["needs_improvement"]:
            category = "needs_improvement"
            probability_text = "Low Chance"
            probability_range = "40-55%"
        else:
            category = "poor"
            probability_text = "Very Low Chance"
            probability_range = "0-40%"
        
        return {
            "category": category,
            "probability_text": probability_text,
            "probability_range": probability_range,
            "score_percentage": score_percentage * 100,
            "weighted_score": weighted_score
        }
    
    def close(self):
        """Clean up resources"""
        if self.rag_retriever:
            self.rag_retriever.close()
        logger.info("Evaluation orchestrator closed")


# Convenience function for quick evaluation
def evaluate_artifact(artifact_data: Dict[str, Any], 
                     knowledge_graph_builder: Optional[KnowledgeGraphBuilder] = None,
                     dimensions: Optional[List[str]] = None,
                     use_rag: bool = True) -> Dict[str, Any]:
    """
    Convenience function to quickly evaluate an artifact
    
    Args:
        artifact_data: The artifact data to evaluate
        knowledge_graph_builder: Optional knowledge graph builder for RAG
        dimensions: List of dimensions to evaluate (None for all)
        use_rag: Whether to use RAG for context retrieval
        
    Returns:
        Comprehensive evaluation report
    """
    
    orchestrator = ArtifactEvaluationOrchestrator(
        knowledge_graph_builder=knowledge_graph_builder,
        use_rag=use_rag
    )
    
    try:
        evaluation_results = orchestrator.evaluate_artifact(
            artifact_data=artifact_data,
            dimensions=dimensions
        )
        
        report = orchestrator.generate_comprehensive_report(
            evaluation_results=evaluation_results,
            artifact_data=artifact_data
        )
        
        return report
    
    finally:
        orchestrator.close()
