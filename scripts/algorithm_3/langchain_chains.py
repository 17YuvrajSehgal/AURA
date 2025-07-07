"""
LangChain Chains for README Documentation Generator

This module implements LangChain chains that orchestrate the generation of different
README sections using prompt templates and RAG-retrieved context.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

from config import config, PROMPT_TEMPLATES, SECTION_PRIORITIES
from rag_retrieval import RAGRetriever, RetrievalResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class READMEChainOrchestrator:
    """
    Orchestrates the generation of README sections using LangChain chains
    """
    
    def __init__(self, rag_retriever: RAGRetriever):
        self.rag_retriever = rag_retriever
        self.llm = ChatOpenAI(
            model_name=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            api_key=config.llm.api_key
        )
        
        # Initialize section chains
        self.section_chains = {}
        self.prompt_templates = {}
        
        # Load prompt templates
        self._load_prompt_templates()
        
        # Create chains for each section
        self._create_section_chains()
        
        logger.info("README Chain Orchestrator initialized")
    
    def _load_prompt_templates(self):
        """Load prompt templates from files"""
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        template_dir = script_dir / "templates"
        
        for section_type, template_filename in PROMPT_TEMPLATES.items():
            full_path = template_dir / template_filename
            
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        template_content = f.read()
                    
                    # Create PromptTemplate
                    prompt_template = PromptTemplate.from_template(template_content)
                    self.prompt_templates[section_type] = prompt_template
                    
                    logger.info(f"Loaded template for {section_type}")
                    
                except Exception as e:
                    logger.error(f"Failed to load template {template_filename}: {e}")
            else:
                logger.warning(f"Template file not found: {full_path}")
    
    def _create_section_chains(self):
        """Create LangChain chains for each section type"""
        for section_type, prompt_template in self.prompt_templates.items():
            # Create a chain for this section
            # Fix lambda closure issue by using default parameter
            chain = (
                RunnablePassthrough.assign(
                    context=lambda x, st=section_type: self._get_section_context(st)
                )
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            self.section_chains[section_type] = chain
            logger.info(f"Created chain for {section_type}")
    
    def _get_section_context(self, section_type: str) -> str:
        """Get context for a specific section using RAG retrieval"""
        try:
            # Get comprehensive context from RAG retriever
            context = self.rag_retriever.get_section_context(section_type)
            
            # Format context for prompt
            formatted_context = self._format_context_for_prompt(context, section_type)
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Failed to get context for {section_type}: {e}")
            return ""
    
    def _format_context_for_prompt(self, context: Dict[str, Any], section_type: str) -> str:
        """Format context dictionary for inclusion in prompts"""
        formatted_parts = []
        
        # Artifact information
        if context.get('artifact_info'):
            artifact = context['artifact_info']
            formatted_parts.append(f"**Artifact Information:**")
            formatted_parts.append(f"- ID: {artifact.get('id', 'N/A')}")
            formatted_parts.append(f"- Size: {artifact.get('size_mb', 0)} MB")
            formatted_parts.append(f"- Extraction Method: {artifact.get('extraction_method', 'N/A')}")
            formatted_parts.append(f"- Description: {artifact.get('description', 'N/A')}")
            formatted_parts.append("")
        
        # Dependencies (for setup section)
        if context.get('dependencies') and section_type in ['setup', 'provenance']:
            formatted_parts.append("**Dependencies:**")
            for dep in context['dependencies'][:10]:  # Limit to top 10
                formatted_parts.append(f"- {dep.get('name', 'N/A')} ({dep.get('type', 'N/A')})")
            formatted_parts.append("")
        
        # Commands (for usage and setup sections)
        if context.get('commands') and section_type in ['usage', 'setup']:
            formatted_parts.append("**Commands:**")
            for cmd in context['commands'][:10]:  # Limit to top 10
                formatted_parts.append(f"- {cmd.get('command', 'N/A')}")
            formatted_parts.append("")
        
        # Files (for structure section)
        if context.get('files') and section_type in ['structure', 'provenance']:
            formatted_parts.append("**Files:**")
            file_types = {}
            for file in context['files']:
                file_type = file.get('type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            for file_type, count in file_types.items():
                formatted_parts.append(f"- {file_type}: {count} files")
            formatted_parts.append("")
        
        # Structure (for structure section)
        if context.get('structure') and section_type == 'structure':
            formatted_parts.append("**Directory Structure:**")
            for struct in context['structure']:
                content = struct.get('content', '')
                if content:
                    # Limit structure content to avoid overwhelming the prompt
                    lines = content.split('\n')[:20]
                    formatted_parts.append('\n'.join(lines))
                    if len(content.split('\n')) > 20:
                        formatted_parts.append("... (truncated)")
                    formatted_parts.append("")
        
        # Outputs (for outputs section)
        if context.get('outputs') and section_type in ['outputs', 'usage']:
            formatted_parts.append("**Expected Outputs:**")
            for output in context['outputs'][:10]:  # Limit to top 10
                formatted_parts.append(f"- {output.get('name', 'N/A')} ({output.get('type', 'N/A')})")
                if output.get('description'):
                    formatted_parts.append(f"  Description: {output['description']}")
            formatted_parts.append("")
        
        return '\n'.join(formatted_parts)
    
    def generate_section(self, section_type: str, additional_context: Dict[str, Any] = None) -> str:
        """
        Generate a specific README section
        
        Args:
            section_type: Type of section to generate
            additional_context: Additional context to include
            
        Returns:
            Generated section content
        """
        if section_type not in self.section_chains:
            logger.error(f"No chain available for section type: {section_type}")
            return f"# {section_type.replace('_', ' ').title()}\n\n(Section generation not available)"
        
        try:
            # Prepare input for the chain
            chain_input = {
                'section_type': section_type,
                'artifact_id': additional_context.get('artifact_id', 'unknown') if additional_context else 'unknown',
                'artifact_type': additional_context.get('artifact_type', 'research_artifact') if additional_context else 'research_artifact',
                'description': additional_context.get('description', '') if additional_context else '',
                'research_domain': additional_context.get('research_domain', '') if additional_context else ''
            }
            
            # Add section-specific context
            if additional_context:
                chain_input.update(additional_context)
            
            # Generate section with cost tracking
            with get_openai_callback() as cb:
                result = self.section_chains[section_type].invoke(chain_input)
                
                logger.info(f"Generated {section_type} section. Cost: ${cb.total_cost:.4f}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to generate {section_type} section: {e}")
            return f"# {section_type.replace('_', ' ').title()}\n\n(Error generating section: {str(e)})"
    
    def generate_full_readme(self, artifact_id: str, additional_context: Dict[str, Any] = None) -> str:
        """
        Generate a complete README document
        
        Args:
            artifact_id: ID of the artifact
            additional_context: Additional context for generation
            
        Returns:
            Complete README content
        """
        logger.info(f"Generating full README for artifact: {artifact_id}")
        
        # Prepare base context
        base_context = {
            'artifact_id': artifact_id,
            'artifact_type': 'research_artifact',
            'description': f'Research artifact {artifact_id}',
            'research_domain': 'computer_science'
        }
        
        if additional_context:
            base_context.update(additional_context)
        
        # Generate sections in priority order
        sections = []
        total_cost = 0.0
        
        # Sort sections by priority
        sorted_sections = sorted(
            config.readme.sections,
            key=lambda x: SECTION_PRIORITIES.get(x, 999)
        )
        
        for section_type in sorted_sections:
            if section_type in self.section_chains:
                logger.info(f"Generating section: {section_type}")
                
                try:
                    section_content = self.generate_section(section_type, base_context)
                    sections.append(section_content)
                    
                except Exception as e:
                    logger.error(f"Failed to generate {section_type}: {e}")
                    sections.append(f"# {section_type.replace('_', ' ').title()}\n\n(Error generating section)")
            else:
                logger.warning(f"No chain available for {section_type}")
        
        # Combine all sections
        full_readme = '\n\n'.join(sections)
        
        # Add metadata comment
        metadata = f"<!-- Generated by README Documentation Generator for {artifact_id} -->\n\n"
        full_readme = metadata + full_readme
        
        logger.info(f"Full README generated for {artifact_id}")
        
        return full_readme
    
    def generate_parallel_sections(self, section_types: List[str], additional_context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate multiple sections in parallel for efficiency
        
        Args:
            section_types: List of section types to generate
            additional_context: Additional context for generation
            
        Returns:
            Dictionary mapping section types to generated content
        """
        logger.info(f"Generating {len(section_types)} sections in parallel")
        
        # Prepare base context
        base_context = {
            'artifact_id': additional_context.get('artifact_id', 'unknown') if additional_context else 'unknown',
            'artifact_type': 'research_artifact',
            'description': 'Research artifact',
            'research_domain': 'computer_science'
        }
        
        if additional_context:
            base_context.update(additional_context)
        
        # Create parallel runnable for available sections
        parallel_chains = {}
        for section_type in section_types:
            if section_type in self.section_chains:
                parallel_chains[section_type] = self.section_chains[section_type]
        
        if not parallel_chains:
            logger.warning("No chains available for parallel generation")
            return {}
        
        try:
            # Create parallel runnable
            parallel_runnable = RunnableParallel(parallel_chains)
            
            # Execute parallel generation
            with get_openai_callback() as cb:
                results = parallel_runnable.invoke(base_context)
                
                logger.info(f"Generated {len(results)} sections in parallel. Cost: ${cb.total_cost:.4f}")
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to generate sections in parallel: {e}")
            return {}
    
    def customize_section_generation(self, section_type: str, custom_prompt: str) -> bool:
        """
        Customize the prompt for a specific section type
        
        Args:
            section_type: Section type to customize
            custom_prompt: Custom prompt template
            
        Returns:
            True if customization was successful
        """
        try:
            # Create new prompt template
            custom_template = PromptTemplate.from_template(custom_prompt)
            
            # Update the template
            self.prompt_templates[section_type] = custom_template
            
            # Recreate the chain
            chain = (
                RunnablePassthrough.assign(
                    retrieved_context=lambda x: self._get_section_context(x['section_type'])
                )
                | custom_template
                | self.llm
                | StrOutputParser()
            )
            
            self.section_chains[section_type] = chain
            
            logger.info(f"Customized chain for {section_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to customize {section_type}: {e}")
            return False
    
    def get_available_sections(self) -> List[str]:
        """Get list of available section types"""
        return list(self.section_chains.keys())
    
    def get_section_statistics(self) -> Dict[str, Any]:
        """Get statistics about available sections and templates"""
        return {
            'available_sections': len(self.section_chains),
            'loaded_templates': len(self.prompt_templates),
            'section_types': list(self.section_chains.keys()),
            'missing_templates': [
                section for section in config.readme.sections 
                if section not in self.prompt_templates
            ]
        }
    
    def close(self):
        """Clean up resources"""
        self.rag_retriever.close()
        logger.info("README Chain Orchestrator closed") 