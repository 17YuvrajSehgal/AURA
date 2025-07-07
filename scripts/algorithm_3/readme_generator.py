"""
Main README Documentation Generator

This module orchestrates the complete README generation pipeline by combining:
- Knowledge Graph Builder (extracts structure from artifacts)
- RAG Retrieval (retrieves relevant context)
- LangChain Chains (generates sections using LLMs)
"""

import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import os

from knowledge_graph_builder import KnowledgeGraphBuilder
from rag_retrieval import RAGRetriever
from langchain_chains import READMEChainOrchestrator
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class READMEGenerator:
    """
    Main README Documentation Generator
    
    Orchestrates the complete pipeline for generating comprehensive README
    documentation from research artifact JSON files.
    """
    
    def __init__(self, use_neo4j: bool = True):
        """
        Initialize the README generator
        
        Args:
            use_neo4j: Whether to use Neo4j for knowledge graph storage
        """
        self.use_neo4j = use_neo4j
        self.kg_builder = None
        self.rag_retriever = None
        self.chain_orchestrator = None
        
        # Create output directory
        self.output_dir = Path(config.readme.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("README Generator initialized")
    
    def generate_readme_from_artifact(self, artifact_json_path: str, 
                                    output_path: Optional[str] = None,
                                    sections: Optional[List[str]] = None,
                                    additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate README from artifact JSON file
        
        Args:
            artifact_json_path: Path to the artifact JSON file
            output_path: Optional path to save the generated README
            sections: Optional list of sections to generate (default: all)
            additional_context: Additional context for generation
            
        Returns:
            Generated README content
        """
        logger.info(f"Starting README generation for: {artifact_json_path}")
        
        try:
            # Step 1: Build Knowledge Graph
            logger.info("Step 1: Building Knowledge Graph")
            self.kg_builder = KnowledgeGraphBuilder(use_neo4j=self.use_neo4j)
            graph_stats = self.kg_builder.build_from_artifact_json(artifact_json_path)
            
            logger.info(f"Knowledge graph built: {graph_stats}")
            
            # Step 2: Initialize RAG Retriever
            logger.info("Step 2: Initializing RAG Retriever")
            self.rag_retriever = RAGRetriever(self.kg_builder)
            
            # Step 3: Initialize LangChain Orchestrator
            logger.info("Step 3: Initializing LangChain Orchestrator")
            self.chain_orchestrator = READMEChainOrchestrator(self.rag_retriever)
            
            # Step 4: Generate README
            logger.info("Step 4: Generating README")
            
            # Get artifact ID from the JSON
            with open(artifact_json_path, 'r', encoding='utf-8') as f:
                artifact_data = json.load(f)
            
            artifact_id = artifact_data.get('artifact_name', 'unknown')
            
            # Prepare context
            context = {
                'artifact_id': artifact_id,
                'artifact_type': 'research_artifact',
                'description': f'Research artifact {artifact_id}',
                'research_domain': 'computer_science'
            }
            
            if additional_context:
                context.update(additional_context)
            
            # Generate README
            if sections:
                # Generate specific sections
                readme_content = self._generate_partial_readme(sections, context)
            else:
                # Generate complete README
                readme_content = self.chain_orchestrator.generate_full_readme(
                    artifact_id, context
                )
            
            # Step 5: Save README if output path provided
            if output_path:
                self._save_readme(readme_content, output_path, artifact_id)
            
            logger.info("README generation completed successfully")
            
            return readme_content
            
        except Exception as e:
            logger.error(f"README generation failed: {e}")
            raise
        
        finally:
            # Clean up resources
            self._cleanup()
    
    def generate_readme_batch(self, artifact_json_paths: List[str], 
                             output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate READMEs for multiple artifacts
        
        Args:
            artifact_json_paths: List of paths to artifact JSON files
            output_dir: Directory to save generated READMEs
            
        Returns:
            Dictionary mapping artifact IDs to generated README content
        """
        logger.info(f"Starting batch README generation for {len(artifact_json_paths)} artifacts")
        
        results = {}
        
        for i, artifact_path in enumerate(artifact_json_paths):
            logger.info(f"Processing artifact {i+1}/{len(artifact_json_paths)}: {artifact_path}")
            
            try:
                # Determine output path
                if output_dir:
                    artifact_name = Path(artifact_path).stem
                    output_path = Path(output_dir) / f"{artifact_name}_README.md"
                else:
                    output_path = None
                
                # Generate README
                readme_content = self.generate_readme_from_artifact(
                    artifact_path, 
                    str(output_path) if output_path else None
                )
                
                # Extract artifact ID for results
                with open(artifact_path, 'r', encoding='utf-8') as f:
                    artifact_data = json.load(f)
                artifact_id = artifact_data.get('artifact_name', f'artifact_{i}')
                
                results[artifact_id] = readme_content
                
            except Exception as e:
                logger.error(f"Failed to process {artifact_path}: {e}")
                results[f"error_{i}"] = f"Error: {str(e)}"
        
        logger.info(f"Batch generation completed. Processed {len(results)} artifacts")
        
        return results
    
    def _generate_partial_readme(self, sections: List[str], context: Dict[str, Any]) -> str:
        """Generate README with only specified sections"""
        logger.info(f"Generating partial README with sections: {sections}")
        
        # Generate sections in parallel if possible
        if len(sections) > 1:
            section_contents = self.chain_orchestrator.generate_parallel_sections(
                sections, context
            )
        else:
            section_contents = {
                sections[0]: self.chain_orchestrator.generate_section(
                    sections[0], context
                )
            }
        
        # Combine sections
        readme_parts = []
        for section_type in sections:
            if section_type in section_contents:
                readme_parts.append(section_contents[section_type])
            else:
                logger.warning(f"Section {section_type} not generated")
        
        # Add metadata comment
        artifact_id = context.get('artifact_id', 'unknown')
        metadata = f"<!-- Generated by README Documentation Generator for {artifact_id} -->\n\n"
        
        return metadata + '\n\n'.join(readme_parts)
    
    def _save_readme(self, content: str, output_path: str, artifact_id: str):
        """Save README content to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"README saved to: {output_file}")
            
            # Also save metadata
            metadata_path = output_file.with_suffix('.meta.json')
            metadata = {
                'artifact_id': artifact_id,
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'sections_generated': config.readme.sections,
                'word_count': len(content.split()),
                'character_count': len(content)
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save README: {e}")
            raise
    
    def _cleanup(self):
        """Clean up resources"""
        if self.chain_orchestrator:
            self.chain_orchestrator.close()
        if self.rag_retriever:
            self.rag_retriever.close()
        if self.kg_builder:
            self.kg_builder.close()
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generation process"""
        stats = {
            'kg_builder_stats': None,
            'chain_orchestrator_stats': None,
            'available_sections': []
        }
        
        if self.kg_builder:
            stats['kg_builder_stats'] = self.kg_builder.get_graph_stats()
        
        if self.chain_orchestrator:
            stats['chain_orchestrator_stats'] = self.chain_orchestrator.get_section_statistics()
            stats['available_sections'] = self.chain_orchestrator.get_available_sections()
        
        return stats
    
    def customize_section_prompt(self, section_type: str, custom_prompt: str) -> bool:
        """
        Customize the prompt for a specific section
        
        Args:
            section_type: Section type to customize
            custom_prompt: Custom prompt template
            
        Returns:
            True if customization was successful
        """
        if not self.chain_orchestrator:
            logger.error("Chain orchestrator not initialized")
            return False
        
        return self.chain_orchestrator.customize_section_generation(
            section_type, custom_prompt
        )
    
    def preview_section_context(self, artifact_json_path: str, section_type: str) -> Dict[str, Any]:
        """
        Preview the context that would be used for a specific section
        
        Args:
            artifact_json_path: Path to the artifact JSON file
            section_type: Section type to preview
            
        Returns:
            Context dictionary for the section
        """
        logger.info(f"Previewing context for section: {section_type}")
        
        try:
            # Build temporary knowledge graph
            temp_kg_builder = KnowledgeGraphBuilder(use_neo4j=self.use_neo4j)
            temp_kg_builder.build_from_artifact_json(artifact_json_path)
            
            # Initialize temporary RAG retriever
            temp_rag_retriever = RAGRetriever(temp_kg_builder)
            
            # Get context
            context = temp_rag_retriever.get_section_context(section_type)
            
            # Clean up
            temp_rag_retriever.close()
            temp_kg_builder.close()
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to preview context: {e}")
            return {}


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate README documentation from artifact JSON')
    parser.add_argument('artifact_json', help='Path to artifact JSON file')
    parser.add_argument('-o', '--output', help='Output README file path')
    parser.add_argument('-s', '--sections', nargs='+', 
                       help='Specific sections to generate',
                       choices=config.readme.sections)
    parser.add_argument('--neo4j', action='store_true', 
                       help='Use Neo4j for knowledge graph storage')
    parser.add_argument('--batch', nargs='+', 
                       help='Process multiple artifact JSON files')
    parser.add_argument('--preview', choices=config.readme.sections,
                       help='Preview context for a specific section')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = READMEGenerator(True)
    
    try:
        if args.preview:
            # Preview context
            context = generator.preview_section_context(args.artifact_json, args.preview)
            print(json.dumps(context, indent=2))
            
        elif args.batch:
            # Batch processing
            results = generator.generate_readme_batch(args.batch, args.output)
            print(f"Processed {len(results)} artifacts")
            
        else:
            # Single artifact processing
            readme_content = generator.generate_readme_from_artifact(
                args.artifact_json,
                args.output,
                args.sections
            )
            
            if not args.output:
                print(readme_content)
            else:
                print(f"README generated and saved to: {args.output}")
                
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 