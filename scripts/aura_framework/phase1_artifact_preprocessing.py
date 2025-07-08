"""
🧱 Phase 1: Artifact Preprocessing Pipeline
Goal: Convert 500+ JSON artifacts into a rich, analyzable structure.

Features:
- Load artifact metadata (name, repo_path, size_mb, conference, etc.)
- Extract documentation data with section breakdown
- Identify commands (pip install, python train.py)
- Detect tools (Docker, Python, Jupyter)
- Extract entities (datasets, models)
- Analyze structural features (bullets, lists, tables, images)
"""

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a documentation section with metadata"""
    heading: str
    content: str
    doc_path: str
    section_order: int
    level: int  # Header level (1-6)
    commands: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    structural_features: Dict[str, int] = field(default_factory=dict)


@dataclass
class DocumentationFile:
    """Represents a processed documentation file"""
    path: str
    file_type: str
    sections: List[Section]
    total_length: int
    language: str = "en"
    readability_score: float = 0.0


@dataclass
class ArtifactMetadata:
    """Comprehensive artifact metadata"""
    artifact_name: str
    repo_path: str
    repo_size_mb: float
    conference: str
    year: int
    extraction_method: str
    success: bool
    analysis_performed: bool
    
    # File counts
    total_files: int = 0
    code_files: int = 0
    doc_files: int = 0
    data_files: int = 0
    
    # Tool detection
    has_docker: bool = False
    has_requirements_txt: bool = False
    has_setup_py: bool = False
    has_makefile: bool = False
    has_jupyter: bool = False
    
    # License and repository info
    has_license: bool = False
    license_type: str = ""
    repository_url: str = ""
    doi: str = ""


@dataclass
class ProcessedArtifact:
    """Complete processed artifact with all extracted information"""
    metadata: ArtifactMetadata
    documentation_files: List[DocumentationFile]
    commands: List[str]
    tools: Set[str]
    entities: Set[str]
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ArtifactPreprocessor:
    """Advanced artifact preprocessing pipeline"""
    
    def __init__(self, 
                 data_directory: str = "data/acm_bib_to_json_data",
                 output_directory: str = "data/processed_artifacts",
                 max_workers: int = 4):
        self.data_directory = Path(data_directory)
        self.output_directory = Path(output_directory)
        self.max_workers = max_workers
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
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
            r'./[^\s]*[^\n]*'
        ]
        
        # Entity patterns
        self.entity_patterns = {
            'datasets': [r'dataset', r'data set', r'corpus', r'benchmark', r'collection'],
            'models': [r'model', r'neural network', r'classifier', r'regression', r'algorithm'],
            'frameworks': [r'tensorflow', r'pytorch', r'keras', r'scikit-learn', r'pandas'],
            'apis': [r'api', r'rest', r'endpoint', r'service', r'interface'],
            'databases': [r'database', r'db', r'mongodb', r'postgresql', r'mysql', r'sqlite']
        }
        
        logger.info(f"Artifact Preprocessor initialized - Data: {self.data_directory}, Output: {self.output_directory}")

    def process_artifacts_batch(self, 
                               artifact_files: List[str], 
                               max_artifacts: Optional[int] = None) -> Dict[str, Any]:
        """
        Process multiple artifacts in parallel
        
        Args:
            artifact_files: List of JSON artifact file paths
            max_artifacts: Maximum number of artifacts to process
            
        Returns:
            Processing statistics and results
        """
        logger.info(f"Starting batch processing of {len(artifact_files)} artifacts")
        
        if max_artifacts:
            artifact_files = artifact_files[:max_artifacts]
        
        results = {
            'total_artifacts': len(artifact_files),
            'processed_successfully': 0,
            'failed_processing': 0,
            'total_sections': 0,
            'total_commands': 0,
            'tools_detected': set(),
            'conferences': set(),
            'processing_time': 0,
            'errors': []
        }
        
        start_time = datetime.now()
        
        # Process artifacts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_artifact, file_path): file_path 
                for file_path in artifact_files
            }
            
            # Process completed tasks
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                             total=len(artifact_files), 
                             desc="Processing artifacts"):
                file_path = future_to_file[future]
                
                try:
                    processed_artifact = future.result()
                    
                    if processed_artifact:
                        # Update statistics
                        results['processed_successfully'] += 1
                        results['total_sections'] += sum(len(doc.sections) for doc in processed_artifact.documentation_files)
                        results['total_commands'] += len(processed_artifact.commands)
                        results['tools_detected'].update(processed_artifact.tools)
                        results['conferences'].add(processed_artifact.metadata.conference)
                        
                        # Save processed artifact
                        self._save_processed_artifact(processed_artifact)
                        
                    else:
                        results['failed_processing'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results['failed_processing'] += 1
                    results['errors'].append(f"{file_path}: {str(e)}")
        
        # Calculate processing time
        end_time = datetime.now()
        results['processing_time'] = (end_time - start_time).total_seconds()
        
        # Convert sets to lists for JSON serialization
        results['tools_detected'] = list(results['tools_detected'])
        results['conferences'] = list(results['conferences'])
        
        logger.info(f"Batch processing completed: {results['processed_successfully']}/{results['total_artifacts']} successful")
        
        return results

    def process_single_artifact(self, artifact_file_path: str) -> Optional[ProcessedArtifact]:
        """
        Process a single artifact JSON file
        
        Args:
            artifact_file_path: Path to the artifact JSON file
            
        Returns:
            ProcessedArtifact object or None if processing failed
        """
        try:
            # Load artifact JSON
            with open(artifact_file_path, 'r', encoding='utf-8') as f:
                artifact_data = json.load(f)
            
            # Extract metadata
            metadata = self._extract_artifact_metadata(artifact_data)
            
            # Process documentation files
            documentation_files = self._process_documentation_files(
                artifact_data.get('documentation_files', [])
            )
            
            # Extract commands across all documentation
            commands = self._extract_all_commands(documentation_files)
            
            # Detect tools
            tools = self._detect_tools(artifact_data, documentation_files)
            
            # Extract entities
            entities = self._extract_entities(documentation_files)
            
            # Create processed artifact
            processed_artifact = ProcessedArtifact(
                metadata=metadata,
                documentation_files=documentation_files,
                commands=commands,
                tools=tools,
                entities=entities
            )
            
            logger.debug(f"Successfully processed artifact: {metadata.artifact_name}")
            return processed_artifact
            
        except Exception as e:
            logger.error(f"Failed to process artifact {artifact_file_path}: {e}")
            return None

    def _extract_artifact_metadata(self, artifact_data: Dict[str, Any]) -> ArtifactMetadata:
        """Extract comprehensive artifact metadata"""
        
        # Basic metadata
        metadata = ArtifactMetadata(
            artifact_name=artifact_data.get('artifact_name', 'unknown'),
            repo_path=artifact_data.get('artifact_path', ''),
            repo_size_mb=artifact_data.get('repo_size_mb', 0.0),
            conference=self._extract_conference_info(artifact_data),
            year=self._extract_year_info(artifact_data),
            extraction_method=artifact_data.get('extraction_method', ''),
            success=artifact_data.get('success', False),
            analysis_performed=artifact_data.get('analysis_performed', False)
        )
        
        # File counts
        metadata.total_files = len(artifact_data.get('repository_structure', []))
        metadata.code_files = len(artifact_data.get('code_files', []))
        metadata.doc_files = len(artifact_data.get('documentation_files', []))
        metadata.data_files = len(artifact_data.get('data_files', []))
        
        # Tool detection from file structure
        metadata.has_docker = self._has_file_type(artifact_data, ['dockerfile', 'docker-compose'])
        metadata.has_requirements_txt = self._has_file_type(artifact_data, ['requirements.txt'])
        metadata.has_setup_py = self._has_file_type(artifact_data, ['setup.py'])
        metadata.has_makefile = self._has_file_type(artifact_data, ['makefile', 'cmake'])
        metadata.has_jupyter = self._has_file_type(artifact_data, ['.ipynb'])
        
        # License detection
        license_files = artifact_data.get('license_files', [])
        metadata.has_license = len(license_files) > 0
        if license_files:
            metadata.license_type = self._detect_license_type(license_files[0])
        
        # Repository info
        metadata.repository_url = artifact_data.get('repository_url', '')
        metadata.doi = artifact_data.get('doi', '')
        
        return metadata

    def _process_documentation_files(self, doc_files: List[Dict]) -> List[DocumentationFile]:
        """Process all documentation files and extract sections"""
        documentation_files = []
        
        for doc_file in doc_files:
            try:
                # Get file info
                file_path = doc_file.get('path', '')
                file_type = self._determine_file_type(file_path)
                content = doc_file.get('content', [])
                
                if isinstance(content, list):
                    content_text = '\n'.join(content)
                else:
                    content_text = str(content)
                
                # Extract sections
                sections = self._extract_sections(content_text, file_path)
                
                # Calculate readability
                readability_score = self._calculate_readability(content_text)
                
                doc_file_obj = DocumentationFile(
                    path=file_path,
                    file_type=file_type,
                    sections=sections,
                    total_length=len(content_text),
                    readability_score=readability_score
                )
                
                documentation_files.append(doc_file_obj)
                
            except Exception as e:
                logger.warning(f"Failed to process documentation file {doc_file.get('path', 'unknown')}: {e}")
        
        return documentation_files

    def _extract_sections(self, content: str, doc_path: str) -> List[Section]:
        """Extract sections from markdown/text content"""
        sections = []
        
        # Find markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = None
        section_order = 0
        
        for i, line in enumerate(lines):
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                heading = header_match.group(2).strip()
                
                current_section = Section(
                    heading=heading,
                    content="",
                    doc_path=doc_path,
                    section_order=section_order,
                    level=level
                )
                section_order += 1
                
            elif current_section:
                # Add content to current section
                current_section.content += line + '\n'
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        # If no sections found, create one section with all content
        if not sections:
            sections.append(Section(
                heading="Main Content",
                content=content,
                doc_path=doc_path,
                section_order=0,
                level=1
            ))
        
        # Process each section for commands, tools, entities
        for section in sections:
            self._analyze_section_content(section)
        
        return sections

    def _analyze_section_content(self, section: Section):
        """Analyze section content for commands, tools, entities, and structural features"""
        content = section.content.lower()
        
        # Extract commands
        for pattern in self.command_patterns:
            matches = re.findall(pattern, section.content, re.IGNORECASE)
            section.commands.extend(matches)
        
        # Detect tools
        for tool, patterns in self.tool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    section.tools.append(tool)
        
        # Extract entities
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    section.entities.append(entity_type)
        
        # Find images
        image_pattern = r'!\[.*?\]\((.*?)\)|<img[^>]+src=["\']([^"\']+)["\']'
        image_matches = re.findall(image_pattern, section.content)
        section.images = [match[0] or match[1] for match in image_matches]
        
        # Find references (URLs)
        url_pattern = r'https?://[^\s\)>\]]*'
        section.references = re.findall(url_pattern, section.content)
        
        # Analyze structural features
        section.structural_features = {
            'bullet_points': len(re.findall(r'^\s*[-*+]', section.content, re.MULTILINE)),
            'numbered_lists': len(re.findall(r'^\s*\d+\.', section.content, re.MULTILINE)),
            'code_blocks': len(re.findall(r'```', section.content)),
            'inline_code': len(re.findall(r'`[^`]+`', section.content)),
            'tables': len(re.findall(r'\|.*\|', section.content)),
            'bold_text': len(re.findall(r'\*\*[^*]+\*\*', section.content)),
            'italic_text': len(re.findall(r'\*[^*]+\*', section.content)),
            'links': len(section.references),
            'images': len(section.images)
        }

    def _extract_all_commands(self, documentation_files: List[DocumentationFile]) -> List[str]:
        """Extract all unique commands from documentation"""
        all_commands = set()
        
        for doc_file in documentation_files:
            for section in doc_file.sections:
                all_commands.update(section.commands)
        
        return list(all_commands)

    def _detect_tools(self, artifact_data: Dict, documentation_files: List[DocumentationFile]) -> Set[str]:
        """Detect tools from file structure and documentation"""
        tools = set()
        
        # Check file extensions and names
        all_files = []
        all_files.extend(artifact_data.get('code_files', []))
        all_files.extend(artifact_data.get('build_files', []))
        all_files.extend(artifact_data.get('docker_files', []))
        
        for file_info in all_files:
            file_path = file_info.get('path', '').lower()
            
            # Detect by file extension
            if file_path.endswith('.py'):
                tools.add('python')
            elif file_path.endswith('.r'):
                tools.add('r')
            elif file_path.endswith('.java'):
                tools.add('java')
            elif file_path.endswith('.js'):
                tools.add('javascript')
            elif file_path.endswith(('.cpp', '.c', '.h')):
                tools.add('c_cpp')
            elif file_path.endswith('.ipynb'):
                tools.add('jupyter')
            elif 'dockerfile' in file_path:
                tools.add('docker')
            elif file_path.endswith('.sql'):
                tools.add('sql')
            elif file_path.endswith('.sh'):
                tools.add('shell')
        
        # Check documentation content
        for doc_file in documentation_files:
            for section in doc_file.sections:
                tools.update(section.tools)
        
        return tools

    def _extract_entities(self, documentation_files: List[DocumentationFile]) -> Set[str]:
        """Extract entities from documentation"""
        entities = set()
        
        for doc_file in documentation_files:
            for section in doc_file.sections:
                entities.update(section.entities)
        
        return entities

    def _extract_conference_info(self, artifact_data: Dict) -> str:
        """Extract conference information from artifact data"""
        # Try different fields that might contain conference info
        conference_fields = ['conference', 'venue', 'publication']
        
        for field in conference_fields:
            if field in artifact_data:
                return str(artifact_data[field])
        
        # Try to extract from artifact name or path
        artifact_name = artifact_data.get('artifact_name', '').lower()
        path = artifact_data.get('artifact_path', '').lower()
        
        # Common conference patterns
        conferences = ['icse', 'fse', 'ase', 'pldi', 'oopsla', 'sigmod', 'vldb', 'issta']
        
        for conf in conferences:
            if conf in artifact_name or conf in path:
                return conf.upper()
        
        return 'unknown'

    def _extract_year_info(self, artifact_data: Dict) -> int:
        """Extract year information"""
        # Try direct year field
        if 'year' in artifact_data:
            return int(artifact_data['year'])
        
        # Try to extract from path or name
        artifact_name = str(artifact_data.get('artifact_name', ''))
        path = str(artifact_data.get('artifact_path', ''))
        
        # Look for 4-digit years
        year_pattern = r'20\d{2}'
        
        for text in [artifact_name, path]:
            year_match = re.search(year_pattern, text)
            if year_match:
                return int(year_match.group())
        
        return 2024  # Default to current year

    def _has_file_type(self, artifact_data: Dict, file_patterns: List[str]) -> bool:
        """Check if artifact has files matching patterns"""
        all_files = []
        
        # Check all file lists
        for file_list_key in ['repository_structure', 'code_files', 'build_files', 'docker_files']:
            all_files.extend(artifact_data.get(file_list_key, []))
        
        for file_info in all_files:
            file_path = file_info.get('path', '').lower()
            file_name = file_info.get('name', '').lower()
            
            for pattern in file_patterns:
                if pattern.lower() in file_path or pattern.lower() in file_name:
                    return True
        
        return False

    def _detect_license_type(self, license_file: Dict) -> str:
        """Detect license type from license file content"""
        content = license_file.get('content', [])
        if isinstance(content, list):
            content_text = ' '.join(content).lower()
        else:
            content_text = str(content).lower()
        
        # Common license patterns
        if 'mit' in content_text:
            return 'MIT'
        elif 'apache' in content_text:
            return 'Apache'
        elif 'gpl' in content_text:
            return 'GPL'
        elif 'bsd' in content_text:
            return 'BSD'
        elif 'creative commons' in content_text or 'cc by' in content_text:
            return 'Creative Commons'
        else:
            return 'Other'

    def _determine_file_type(self, file_path: str) -> str:
        """Determine file type from path"""
        path_lower = file_path.lower()
        
        if path_lower.endswith('.md'):
            return 'markdown'
        elif path_lower.endswith('.rst'):
            return 'restructuredtext'
        elif path_lower.endswith(('.txt', '.text')):
            return 'plaintext'
        elif 'readme' in path_lower:
            return 'readme'
        elif 'license' in path_lower:
            return 'license'
        elif 'changelog' in path_lower:
            return 'changelog'
        else:
            return 'documentation'

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score for text"""
        if not text.strip():
            return 0.0
        
        try:
            # Simple readability metric based on sentence and word length
            sentences = len(re.split(r'[.!?]+', text))
            words = len(text.split())
            
            if sentences == 0:
                return 0.0
            
            avg_words_per_sentence = words / sentences
            
            # Normalize to 0-1 scale (assuming 15 words per sentence is optimal)
            readability = max(0, 1 - abs(avg_words_per_sentence - 15) / 15)
            
            return readability
            
        except:
            return 0.5  # Default neutral score

    def _save_processed_artifact(self, processed_artifact: ProcessedArtifact):
        """Save processed artifact to disk"""
        artifact_id = processed_artifact.metadata.artifact_name
        output_file = self.output_directory / f"{artifact_id}_processed.json"
        
        # Convert to serializable format
        artifact_dict = {
            'metadata': processed_artifact.metadata.__dict__,
            'documentation_files': [
                {
                    'path': doc.path,
                    'file_type': doc.file_type,
                    'total_length': doc.total_length,
                    'readability_score': doc.readability_score,
                    'sections': [
                        {
                            'heading': section.heading,
                            'content': section.content,
                            'doc_path': section.doc_path,
                            'section_order': section.section_order,
                            'level': section.level,
                            'commands': section.commands,
                            'tools': section.tools,
                            'entities': section.entities,
                            'images': section.images,
                            'references': section.references,
                            'structural_features': section.structural_features
                        }
                        for section in doc.sections
                    ]
                }
                for doc in processed_artifact.documentation_files
            ],
            'commands': processed_artifact.commands,
            'tools': list(processed_artifact.tools),
            'entities': list(processed_artifact.entities),
            'processing_timestamp': processed_artifact.processing_timestamp
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(artifact_dict, f, indent=2, ensure_ascii=False)

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processed artifacts"""
        processed_files = list(self.output_directory.glob("*_processed.json"))
        
        summary = {
            'total_processed_artifacts': len(processed_files),
            'conferences': set(),
            'tools': set(),
            'total_sections': 0,
            'total_commands': 0,
            'avg_readability': 0.0
        }
        
        total_readability = 0
        doc_count = 0
        
        for file_path in processed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    artifact_data = json.load(f)
                
                summary['conferences'].add(artifact_data['metadata']['conference'])
                summary['tools'].update(artifact_data['tools'])
                summary['total_sections'] += sum(len(doc['sections']) for doc in artifact_data['documentation_files'])
                summary['total_commands'] += len(artifact_data['commands'])
                
                for doc in artifact_data['documentation_files']:
                    total_readability += doc['readability_score']
                    doc_count += 1
                    
            except Exception as e:
                logger.warning(f"Error reading processed file {file_path}: {e}")
        
        # Calculate averages
        if doc_count > 0:
            summary['avg_readability'] = total_readability / doc_count
        
        # Convert sets to lists
        summary['conferences'] = list(summary['conferences'])
        summary['tools'] = list(summary['tools'])
        
        return summary


def main():
    """Example usage of the artifact preprocessor"""
    # Initialize preprocessor
    preprocessor = ArtifactPreprocessor(
        data_directory="data/acm_bib_to_json_data",
        output_directory="data/processed_artifacts"
    )
    
    # Find artifact files
    artifact_files = list(Path("data/sample_artifacts").glob("*.json"))
    
    if not artifact_files:
        logger.warning("No artifact files found. Please check the data directory.")
        return
    
    # Process artifacts
    results = preprocessor.process_artifacts_batch(artifact_files, max_artifacts=10)
    
    # Print results
    print("\n🧱 Phase 1: Artifact Preprocessing Results")
    print("=" * 50)
    print(f"✅ Processed: {results['processed_successfully']}/{results['total_artifacts']}")
    print(f"📄 Total sections: {results['total_sections']}")
    print(f"⚡ Commands found: {results['total_commands']}")
    print(f"🔧 Tools detected: {', '.join(results['tools_detected'])}")
    print(f"🏛️ Conferences: {', '.join(results['conferences'])}")
    print(f"⏱️ Processing time: {results['processing_time']:.2f} seconds")
    
    if results['errors']:
        print(f"\n❌ Errors: {len(results['errors'])}")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")


if __name__ == "__main__":
    main() 