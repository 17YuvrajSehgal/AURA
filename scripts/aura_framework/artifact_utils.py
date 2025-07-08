"""
🔧 Artifact Utilities
Utility functions to read and process artifact JSON files in various formats.

This module handles the conversion of artifact JSON files from the analysis format
to the expected format for the AURA framework processing.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedSection:
    """A processed documentation section"""
    heading: str
    content: str
    doc_path: str
    section_order: int
    level: int
    commands: List[str]
    tools: List[str]
    entities: List[str]
    images: List[str]
    references: List[str]
    structural_features: Dict[str, int]


@dataclass
class ProcessedDocumentationFile:
    """A processed documentation file"""
    path: str
    file_type: str
    total_length: int
    readability_score: float
    sections: List[ProcessedSection]


@dataclass
class ProcessedArtifactMetadata:
    """Processed artifact metadata"""
    artifact_name: str
    repo_path: str
    repo_size_mb: float
    conference: str
    year: int
    extraction_method: str
    success: bool
    analysis_performed: bool
    total_files: int
    code_files: int
    doc_files: int
    data_files: int
    has_docker: bool
    has_requirements_txt: bool
    has_setup_py: bool
    has_makefile: bool
    has_jupyter: bool
    has_license: bool
    license_type: str
    repository_url: str
    doi: str


class ArtifactJSONProcessor:
    """Processor for artifact JSON files"""
    
    def __init__(self):
        # Tool detection patterns
        self.tool_patterns = {
            'docker': [r'docker', r'dockerfile', r'docker-compose'],
            'python': [r'python', r'\.py\b', r'pip', r'conda', r'virtualenv'],
            'jupyter': [r'jupyter', r'\.ipynb', r'notebook'],
            'r': [r'\br\b', r'\.r\b', r'rstudio', r'cran'],
            'matlab': [r'matlab', r'\.m\b', r'\.mat\b'],
            'java': [r'java', r'\.java\b', r'maven', r'gradle'],
            'javascript': [r'javascript', r'\.js\b', r'node', r'npm'],
            'c_cpp': [r'\bc\+\+', r'\bc\b', r'\.cpp\b', r'\.c\b', r'gcc', r'cmake'],
            'sql': [r'sql', r'\.sql\b', r'database', r'postgresql', r'mysql'],
            'shell': [r'bash', r'\.sh\b', r'shell', r'terminal']
        }
        
        # Command patterns
        self.command_patterns = [
            r'pip install[^\n]*',
            r'conda install[^\n]*',
            r'python [^\n]*\.py[^\n]*',
            r'java -[^\n]*',
            r'docker run[^\n]*',
            r'docker build[^\n]*',
            r'npm install[^\n]*',
            r'make[^\s]*[^\n]*',
            r'cmake[^\n]*',
            r'./[^\s]*[^\n]*',
            r'Rscript[^\n]*',
            r'git clone[^\n]*'
        ]
        
        # Entity patterns
        self.entity_patterns = {
            'datasets': [r'dataset', r'data set', r'corpus', r'benchmark', r'collection'],
            'models': [r'model', r'neural network', r'classifier', r'regression', r'algorithm'],
            'frameworks': [r'tensorflow', r'pytorch', r'keras', r'scikit-learn', r'pandas'],
            'apis': [r'api', r'rest', r'endpoint', r'service', r'interface'],
            'databases': [r'database', r'db', r'mongodb', r'postgresql', r'mysql', r'sqlite']
        }

    def read_artifact_json(self, file_path: str) -> Dict[str, Any]:
        """Read and parse artifact JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise

    def convert_to_aura_format(self, artifact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert artifact JSON to AURA framework format"""
        try:
            # Extract metadata
            metadata = self._extract_metadata(artifact_data)
            
            # Process documentation files
            documentation_files = self._process_documentation_files(
                artifact_data.get('documentation_files', [])
            )
            
            # Extract tools from various sources
            tools = self._extract_tools(artifact_data, documentation_files)
            
            # Extract commands
            commands = self._extract_commands(documentation_files)
            
            # Extract entities
            entities = self._extract_entities(documentation_files)
            
            # Create AURA format
            aura_format = {
                'metadata': metadata.__dict__,
                'documentation_files': [doc.__dict__ for doc in documentation_files],
                'commands': commands,
                'tools': list(tools),
                'entities': list(entities),
                'processing_timestamp': artifact_data.get('processing_timestamp', '')
            }
            
            logger.info(f"Converted artifact {metadata.artifact_name} to AURA format")
            return aura_format
            
        except Exception as e:
            logger.error(f"Failed to convert artifact: {e}")
            raise

    def _extract_metadata(self, artifact_data: Dict[str, Any]) -> ProcessedArtifactMetadata:
        """Extract and process artifact metadata"""
        # Determine conference and year from various sources
        conference = "ICSE"  # Default
        year = 2024  # Default
        
        # Try to extract from artifact name or path
        artifact_name = artifact_data.get('artifact_name', 'unknown')
        if 'icse' in artifact_name.lower():
            conference = "ICSE"
        elif 'fse' in artifact_name.lower():
            conference = "FSE"
        elif 'ase' in artifact_name.lower():
            conference = "ASE"
        
        # Extract year from various fields
        for field in ['artifact_name', 'artifact_path', 'repo_path']:
            if field in artifact_data:
                year_match = re.search(r'20\d{2}', str(artifact_data[field]))
                if year_match:
                    year = int(year_match.group())
                    break
        
        return ProcessedArtifactMetadata(
            artifact_name=artifact_name,
            repo_path=artifact_data.get('repo_path', artifact_data.get('artifact_path', '')),
            repo_size_mb=artifact_data.get('repo_size_mb', 0.0),
            conference=conference,
            year=year,
            extraction_method=artifact_data.get('extraction_method', 'unknown'),
            success=artifact_data.get('success', True),
            analysis_performed=artifact_data.get('analysis_performed', True),
            total_files=len(artifact_data.get('tree_structure', [])),
            code_files=len(artifact_data.get('code_files', [])),
            doc_files=len(artifact_data.get('documentation_files', [])),
            data_files=len(artifact_data.get('data_files', [])),
            has_docker=len(artifact_data.get('docker_files', [])) > 0,
            has_requirements_txt=any('requirements.txt' in str(f) for f in artifact_data.get('code_files', [])),
            has_setup_py=any('setup.py' in str(f) for f in artifact_data.get('code_files', [])),
            has_makefile=any('makefile' in str(f).lower() for f in artifact_data.get('build_files', [])),
            has_jupyter=any('.ipynb' in str(f) for f in artifact_data.get('code_files', [])),
            has_license=len(artifact_data.get('license_files', [])) > 0,
            license_type=self._extract_license_type(artifact_data.get('license_files', [])),
            repository_url='',
            doi=''
        )

    def _extract_license_type(self, license_files: List[Dict]) -> str:
        """Extract license type from license files"""
        if not license_files:
            return ''
        
        for license_file in license_files:
            content = ' '.join(license_file.get('content', []))
            if 'MIT' in content:
                return 'MIT'
            elif 'Apache' in content:
                return 'Apache'
            elif 'GPL' in content:
                return 'GPL'
            elif 'BSD' in content:
                return 'BSD'
        
        return 'Other'

    def _process_documentation_files(self, doc_files: List[Dict]) -> List[ProcessedDocumentationFile]:
        """Process documentation files into structured format"""
        processed_docs = []
        
        for doc_file in doc_files:
            path = doc_file.get('path', '')
            content_lines = doc_file.get('content', [])
            
            # Join content lines
            full_content = '\n'.join(content_lines)
            
            # Extract sections
            sections = self._extract_sections(full_content, path)
            
            processed_doc = ProcessedDocumentationFile(
                path=path,
                file_type=self._get_file_type(path),
                total_length=len(full_content),
                readability_score=self._calculate_readability_score(full_content),
                sections=sections
            )
            
            processed_docs.append(processed_doc)
        
        return processed_docs

    def _extract_sections(self, content: str, doc_path: str) -> List[ProcessedSection]:
        """Extract sections from documentation content"""
        sections = []
        
        # Split by markdown headers
        lines = content.split('\n')
        current_section = None
        current_content = []
        section_order = 0
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            
            if header_match:
                # Save previous section
                if current_section:
                    section_content = '\n'.join(current_content).strip()
                    if section_content:  # Only add non-empty sections
                        processed_section = self._create_processed_section(
                            current_section, section_content, doc_path, section_order
                        )
                        sections.append(processed_section)
                        section_order += 1
                
                # Start new section
                current_section = {
                    'heading': header_match.group(2).strip(),
                    'level': len(header_match.group(1))
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_section:
            section_content = '\n'.join(current_content).strip()
            if section_content:
                processed_section = self._create_processed_section(
                    current_section, section_content, doc_path, section_order
                )
                sections.append(processed_section)
        
        # If no sections found, create a default section
        if not sections and content.strip():
            processed_section = self._create_processed_section(
                {'heading': 'Main Content', 'level': 1},
                content, doc_path, 0
            )
            sections.append(processed_section)
        
        return sections

    def _create_processed_section(self, section_info: Dict, content: str, 
                                doc_path: str, section_order: int) -> ProcessedSection:
        """Create a processed section with extracted features"""
        # Extract commands
        commands = self._extract_commands_from_text(content)
        
        # Extract tools
        tools = self._extract_tools_from_text(content)
        
        # Extract entities
        entities = self._extract_entities_from_text(content)
        
        # Extract images and references
        images = re.findall(r'!\[.*?\]\((.*?)\)', content)
        references = re.findall(r'\[.*?\]\((.*?)\)', content)
        
        # Calculate structural features
        structural_features = {
            'bullet_points': len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE)),
            'code_blocks': len(re.findall(r'```.*?```', content, re.DOTALL)),
            'numbered_lists': len(re.findall(r'^\s*\d+\.\s', content, re.MULTILINE)),
            'tables': len(re.findall(r'\|.*?\|', content)),
            'links': len(references),
            'images': len(images)
        }
        
        return ProcessedSection(
            heading=section_info['heading'],
            content=content,
            doc_path=doc_path,
            section_order=section_order,
            level=section_info['level'],
            commands=commands,
            tools=tools,
            entities=entities,
            images=images,
            references=references,
            structural_features=structural_features
        )

    def _extract_commands_from_text(self, text: str) -> List[str]:
        """Extract commands from text"""
        commands = []
        for pattern in self.command_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            commands.extend(matches)
        return list(set(commands))  # Remove duplicates

    def _extract_tools_from_text(self, text: str) -> List[str]:
        """Extract tools from text"""
        tools = []
        for tool, patterns in self.tool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    tools.append(tool)
                    break
        return list(set(tools))

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entities from text"""
        entities = []
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    entities.append(entity_type)
                    break
        return list(set(entities))

    def _extract_tools(self, artifact_data: Dict, documentation_files: List) -> Set[str]:
        """Extract tools from various sources"""
        tools = set()
        
        # From docker files
        if artifact_data.get('docker_files'):
            tools.add('docker')
        
        # From code files
        code_files = artifact_data.get('code_files', [])
        for code_file in code_files:
            path = code_file.get('path', '')
            if '.py' in path:
                tools.add('python')
            elif '.R' in path:
                tools.add('r')
            elif '.js' in path:
                tools.add('javascript')
            elif '.java' in path:
                tools.add('java')
            elif '.sh' in path:
                tools.add('shell')
        
        # From documentation
        for doc_file in documentation_files:
            for section in doc_file.sections:
                tools.update(section.tools)
        
        return tools

    def _extract_commands(self, documentation_files: List) -> List[str]:
        """Extract all commands from documentation"""
        commands = []
        for doc_file in documentation_files:
            for section in doc_file.sections:
                commands.extend(section.commands)
        return list(set(commands))

    def _extract_entities(self, documentation_files: List) -> Set[str]:
        """Extract all entities from documentation"""
        entities = set()
        for doc_file in documentation_files:
            for section in doc_file.sections:
                entities.update(section.entities)
        return entities

    def _get_file_type(self, path: str) -> str:
        """Determine file type from path"""
        if path.endswith('.md'):
            return 'markdown'
        elif path.endswith('.txt'):
            return 'text'
        elif path.endswith('.rst'):
            return 'restructuredtext'
        else:
            return 'unknown'

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate a simple readability score"""
        if not text:
            return 0.0
        
        # Simple metrics
        words = len(text.split())
        sentences = len(re.findall(r'[.!?]+', text))
        
        if sentences == 0:
            return 0.5
        
        avg_words_per_sentence = words / sentences
        
        # Simple readability score (inverse of average words per sentence, normalized)
        readability = max(0.0, min(1.0, 1.0 - (avg_words_per_sentence - 10) / 20))
        
        return readability

    def process_artifacts_directory(self, directory_path: str, 
                                  output_directory: str = None) -> Dict[str, Any]:
        """Process all artifact JSON files in a directory"""
        directory = Path(directory_path)
        artifact_files = list(directory.glob("*_analysis.json"))
        
        if output_directory:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'total_artifacts': len(artifact_files),
            'processed_artifacts': 0,
            'failed_artifacts': 0,
            'converted_files': []
        }
        
        for artifact_file in artifact_files:
            try:
                # Read original artifact
                artifact_data = self.read_artifact_json(str(artifact_file))
                
                # Convert to AURA format
                aura_format = self.convert_to_aura_format(artifact_data)
                
                # Save converted file if output directory specified
                if output_directory:
                    output_file = output_dir / f"{artifact_file.stem}_processed.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(aura_format, f, indent=2, default=str)
                    results['converted_files'].append(str(output_file))
                
                results['processed_artifacts'] += 1
                logger.info(f"Processed {artifact_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {artifact_file.name}: {e}")
                results['failed_artifacts'] += 1
        
        logger.info(f"Processing complete: {results['processed_artifacts']}/{results['total_artifacts']} successful")
        return results


def main():
    """Example usage of the ArtifactJSONProcessor"""
    processor = ArtifactJSONProcessor()
    
    # Process artifacts directory
    results = processor.process_artifacts_directory(
        "algo_outputs/algorithm_2_output_2",
        "scripts/aura_framework/processed_artifacts"
    )
    
    print(f"Processed {results['processed_artifacts']} artifacts")
    print(f"Failed {results['failed_artifacts']} artifacts")
    print(f"Converted files: {len(results['converted_files'])}")


if __name__ == "__main__":
    main() 