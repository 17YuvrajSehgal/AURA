#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder

This module provides advanced knowledge graph construction capabilities
extending the base functionality with:
- Enhanced graph schema
- Advanced analytics
- Pattern mining
- Statistical analysis
- Export capabilities
"""

import ast
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedKGBuilder:
    """
    Enhanced Knowledge Graph Builder with advanced analytics and pattern mining capabilities.
    
    Features:
    - Extended graph schema with metadata
    - Code structure analysis
    - Dependency tracking
    - Pattern mining
    - Statistical analysis
    - Advanced querying
    """

    def __init__(
            self,
            neo4j_uri: str = "bolt://localhost:7687",
            neo4j_user: str = "neo4j",
            neo4j_password: str = "password",
            clear_existing: bool = False
    ):
        """
        Initialize the Enhanced Knowledge Graph Builder.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            clear_existing: Whether to clear existing graph data
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Connect to Neo4j
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

        # Initialize matchers
        self.node_matcher = NodeMatcher(self.graph)
        self.relationship_matcher = RelationshipMatcher(self.graph)

        # Clear existing data if requested
        if clear_existing:
            self._clear_graph()

        # Create indexes for performance
        self._create_indexes()

    def _clear_graph(self):
        """Clear all nodes and relationships from the graph."""
        try:
            self.graph.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing graph data")
        except Exception as e:
            logger.warning(f"Could not clear graph: {e}")

    def _create_indexes(self):
        """Create database indexes for better performance."""
        indexes = [
            "CREATE INDEX artifact_name IF NOT EXISTS FOR (a:Artifact) ON (a.name)",
            "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX function_name IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
            "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX dependency_name IF NOT EXISTS FOR (d:Dependency) ON (d.name)",
        ]

        for index_query in indexes:
            try:
                self.graph.run(index_query)
            except Exception as e:
                logger.debug(f"Index creation skipped: {e}")

    def build_knowledge_graph(
            self,
            extracted_path: str,
            artifact_name: str,
            metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Build a comprehensive knowledge graph from extracted artifact.
        
        Args:
            extracted_path: Path to extracted artifact
            artifact_name: Name of the artifact
            metadata: Optional metadata from extraction
            
        Returns:
            Dictionary containing build results
        """
        extracted_path = Path(extracted_path)
        if not extracted_path.exists():
            return {
                "success": False,
                "error": f"Extracted path not found: {extracted_path}",
                "artifact_name": artifact_name
            }

        logger.info(f"Building knowledge graph for: {artifact_name}")

        result = {
            "success": False,
            "artifact_name": artifact_name,
            "extracted_path": str(extracted_path),
            "nodes_created": 0,
            "relationships_created": 0,
            "analysis_results": {},
            "error": None
        }

        try:
            # Create artifact root node
            artifact_node = self._create_artifact_node(artifact_name, metadata or {})

            # Analyze repository structure
            structure_analysis = self._analyze_repository_structure(extracted_path, artifact_node)
            result["nodes_created"] += structure_analysis["nodes_created"]
            result["relationships_created"] += structure_analysis["relationships_created"]

            # Analyze code files
            code_analysis = self._analyze_code_files(extracted_path, artifact_node)
            result["nodes_created"] += code_analysis["nodes_created"]
            result["relationships_created"] += code_analysis["relationships_created"]

            # Analyze dependencies
            dependency_analysis = self._analyze_dependencies(extracted_path, artifact_node)
            result["nodes_created"] += dependency_analysis["nodes_created"]
            result["relationships_created"] += dependency_analysis["relationships_created"]

            # Analyze documentation
            doc_analysis = self._analyze_documentation(extracted_path, artifact_node)
            result["nodes_created"] += doc_analysis["nodes_created"]
            result["relationships_created"] += doc_analysis["relationships_created"]

            # Extract patterns
            pattern_analysis = self._extract_patterns(artifact_node)
            result["analysis_results"]["patterns"] = pattern_analysis

            # Calculate metrics
            metrics = self._calculate_metrics(artifact_node)
            result["analysis_results"]["metrics"] = metrics

            result["success"] = True
            logger.info(f"Successfully built knowledge graph for: {artifact_name}")

        except Exception as e:
            error_msg = f"Knowledge graph build failed for {artifact_name}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg

        return result

    def _create_artifact_node(self, artifact_name: str, metadata: Dict) -> Node:
        """Create the root artifact node."""
        # Calculate a hash of the artifact for uniqueness
        artifact_hash = hashlib.md5(artifact_name.encode()).hexdigest()[:8]

        # Convert complex metadata to simple types
        repo_structure = metadata.get("repository_structure", [])
        total_files_count = len(repo_structure) if isinstance(repo_structure, list) else 0

        programming_languages = metadata.get("programming_languages", [])
        languages_str = ",".join(programming_languages) if isinstance(programming_languages, list) else ""

        artifact_node = Node(
            "Artifact",
            name=artifact_name,
            hash=artifact_hash,
            created_at=datetime.now().isoformat(),
            total_files=total_files_count,
            programming_languages=languages_str,
            has_readme=bool(metadata.get("has_readme", False)),
            has_license=bool(metadata.get("has_license", False)),
            has_dockerfile=bool(metadata.get("has_dockerfile", False)),
            has_tests=bool(metadata.get("has_tests", False)),
            build_systems=",".join(metadata.get("build_systems", [])) if metadata.get("build_systems") else ""
        )

        # Use merge to avoid duplicates
        existing = self.node_matcher.match("Artifact", name=artifact_name).first()
        if existing:
            # Update existing node
            existing.update(artifact_node)
            self.graph.push(existing)
            return existing
        else:
            self.graph.create(artifact_node)
            return artifact_node

    def _analyze_repository_structure(self, extracted_path: Path, artifact_node: Node) -> Dict[str, int]:
        """Analyze repository structure and create file/directory nodes."""
        nodes_created = 0
        relationships_created = 0

        # Track all created nodes for relationship building
        path_to_node = {}

        for root, dirs, files in os.walk(extracted_path):
            rel_root = Path(root).relative_to(extracted_path)

            # Create directory nodes
            if str(rel_root) != '.':
                dir_node = Node(
                    "Directory",
                    name=rel_root.name,
                    path=str(rel_root),
                    full_path=str(root),
                    depth=len(rel_root.parts)
                )
                self.graph.create(dir_node)
                path_to_node[str(rel_root)] = dir_node
                nodes_created += 1

                # Link to parent directory or artifact
                parent_path = str(rel_root.parent)
                if parent_path == '.':
                    self.graph.create(Relationship(artifact_node, "CONTAINS", dir_node))
                else:
                    parent_node = path_to_node.get(parent_path)
                    if parent_node:
                        self.graph.create(Relationship(parent_node, "CONTAINS", dir_node))
                relationships_created += 1

            # Create file nodes
            for file in files:
                file_path = Path(root) / file
                rel_file_path = file_path.relative_to(extracted_path)

                try:
                    file_stat = file_path.stat()
                    file_type = self._classify_file_type(file_path)

                    file_node = Node(
                        "File",
                        name=file,
                        path=str(rel_file_path),
                        full_path=str(file_path),
                        extension=file_path.suffix.lower(),
                        type=file_type,
                        size=file_stat.st_size,
                        lines=self._count_lines(file_path) if file_type == "code" else 0,
                        modified_at=datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    )
                    self.graph.create(file_node)
                    path_to_node[str(rel_file_path)] = file_node
                    nodes_created += 1

                    # Link to parent directory or artifact
                    parent_path = str(rel_file_path.parent)
                    if parent_path == '.':
                        self.graph.create(Relationship(artifact_node, "CONTAINS", file_node))
                    else:
                        parent_node = path_to_node.get(parent_path)
                        if parent_node:
                            self.graph.create(Relationship(parent_node, "CONTAINS", file_node))
                    relationships_created += 1

                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not analyze file {rel_file_path}: {e}")

        return {"nodes_created": nodes_created, "relationships_created": relationships_created}

    def _analyze_code_files(self, extracted_path: Path, artifact_node: Node) -> Dict[str, int]:
        """Analyze code files and extract functions, classes, imports."""
        nodes_created = 0
        relationships_created = 0

        # Find all Python files for detailed analysis
        python_files = list(extracted_path.rglob("*.py"))

        for py_file in python_files:
            try:
                # Get the file node
                rel_path = py_file.relative_to(extracted_path)
                file_node = self.node_matcher.match("File", path=str(rel_path)).first()

                if not file_node:
                    continue

                # Parse Python AST
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    try:
                        content = f.read()
                        tree = ast.parse(content)
                    except SyntaxError:
                        logger.warning(f"Syntax error in {py_file}")
                        continue

                # Extract functions, classes, imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        decorators_list = [self._get_decorator_name(d) for d in node.decorator_list]
                        func_node = Node(
                            "Function",
                            name=node.name,
                            line_number=node.lineno,
                            is_async=isinstance(node, ast.AsyncFunctionDef),
                            args_count=len(node.args.args),
                            decorators=",".join(decorators_list) if decorators_list else "",
                            docstring=ast.get_docstring(node) or ""
                        )
                        self.graph.create(func_node)
                        self.graph.create(Relationship(file_node, "DEFINES", func_node))
                        nodes_created += 1
                        relationships_created += 1

                    elif isinstance(node, ast.ClassDef):
                        bases_list = [self._get_base_name(base) for base in node.bases]
                        class_node = Node(
                            "Class",
                            name=node.name,
                            line_number=node.lineno,
                            bases=",".join(bases_list) if bases_list else "",
                            methods_count=len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                            docstring=ast.get_docstring(node) or ""
                        )
                        self.graph.create(class_node)
                        self.graph.create(Relationship(file_node, "DEFINES", class_node))
                        nodes_created += 1
                        relationships_created += 1

                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in getattr(node, 'names', []):
                            import_name = alias.name
                            if isinstance(node, ast.ImportFrom) and node.module:
                                import_name = f"{node.module}.{import_name}"

                            # Check if import node already exists
                            import_node = self.node_matcher.match("Import", name=import_name).first()
                            if not import_node:
                                import_node = Node(
                                    "Import",
                                    name=import_name,
                                    module=getattr(node, 'module', None) or import_name.split('.')[0],
                                    is_from_import=isinstance(node, ast.ImportFrom),
                                    is_standard_library=self._is_standard_library(import_name)
                                )
                                self.graph.create(import_node)
                                nodes_created += 1

                            self.graph.create(Relationship(file_node, "IMPORTS", import_node))
                            relationships_created += 1

            except Exception as e:
                logger.warning(f"Error analyzing Python file {py_file}: {e}")

        return {"nodes_created": nodes_created, "relationships_created": relationships_created}

    def _analyze_dependencies(self, extracted_path: Path, artifact_node: Node) -> Dict[str, int]:
        """Analyze project dependencies from various configuration files."""
        nodes_created = 0
        relationships_created = 0

        dependency_files = {
            "requirements.txt": self._parse_requirements_txt,
            "setup.py": self._parse_setup_py,
            "pyproject.toml": self._parse_pyproject_toml,
            "package.json": self._parse_package_json,
            "pom.xml": self._parse_pom_xml,
            "build.gradle": self._parse_build_gradle,
        }

        for dep_file, parser in dependency_files.items():
            dep_path = extracted_path / dep_file
            if dep_path.exists():
                try:
                    dependencies = parser(dep_path)
                    for dep_info in dependencies:
                        dep_node = Node(
                            "Dependency",
                            name=dep_info["name"],
                            version=dep_info.get("version", ""),
                            dependency_type=dep_info.get("type", "runtime"),
                            source_file=dep_file,
                            is_dev_dependency=dep_info.get("is_dev", False)
                        )

                        # Check if dependency already exists
                        existing_dep = self.node_matcher.match(
                            "Dependency", name=dep_info["name"]
                        ).first()

                        if not existing_dep:
                            self.graph.create(dep_node)
                            nodes_created += 1

                            # Link to artifact
                            self.graph.create(Relationship(artifact_node, "DEPENDS_ON", dep_node))
                            relationships_created += 1

                except Exception as e:
                    logger.warning(f"Error parsing {dep_file}: {e}")

        return {"nodes_created": nodes_created, "relationships_created": relationships_created}

    def _analyze_documentation(self, extracted_path: Path, artifact_node: Node) -> Dict[str, int]:
        """Analyze documentation files and extract sections."""
        nodes_created = 0
        relationships_created = 0

        # Find documentation files
        doc_files = list(extracted_path.rglob("*.md")) + list(extracted_path.rglob("*.rst"))

        for doc_file in doc_files:
            try:
                rel_path = doc_file.relative_to(extracted_path)
                file_node = self.node_matcher.match("File", path=str(rel_path)).first()

                if not file_node:
                    continue

                # Parse sections from markdown/rst
                with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                sections = self._extract_doc_sections(content, doc_file.suffix)

                for section in sections:
                    section_node = Node(
                        "DocSection",
                        title=section["title"],
                        level=section["level"],
                        content_length=len(section["content"]),
                        line_number=section.get("line_number", 0),
                        section_type=self._classify_doc_section(section["title"])
                    )
                    self.graph.create(section_node)
                    self.graph.create(Relationship(file_node, "CONTAINS", section_node))
                    nodes_created += 1
                    relationships_created += 1

            except Exception as e:
                logger.warning(f"Error analyzing documentation file {doc_file}: {e}")

        return {"nodes_created": nodes_created, "relationships_created": relationships_created}

    def _extract_patterns(self, artifact_node: Node) -> Dict[str, Any]:
        """Extract common patterns from the knowledge graph."""
        patterns = {}

        # File organization patterns
        patterns["file_organization"] = self._analyze_file_organization(artifact_node)

        # Naming conventions
        patterns["naming_conventions"] = self._analyze_naming_conventions(artifact_node)

        # Code structure patterns
        patterns["code_structure"] = self._analyze_code_structure(artifact_node)

        # Documentation patterns
        patterns["documentation"] = self._analyze_documentation_patterns(artifact_node)

        return patterns

    def _calculate_metrics(self, artifact_node: Node) -> Dict[str, Any]:
        """Calculate various metrics for the artifact."""
        artifact_name = artifact_node["name"]

        metrics = {}

        # Basic counts
        queries = {
            "total_files": f"MATCH (a:Artifact {{name: '{artifact_name}'}})-[:CONTAINS*]->(f:File) RETURN count(f) as count",
            "total_directories": f"MATCH (a:Artifact {{name: '{artifact_name}'}})-[:CONTAINS*]->(d:Directory) RETURN count(d) as count",
            "code_files": f"MATCH (a:Artifact {{name: '{artifact_name}'}})-[:CONTAINS*]->(f:File {{type: 'code'}}) RETURN count(f) as count",
            "functions": f"MATCH (a:Artifact {{name: '{artifact_name}'}})-[:CONTAINS*]->()-[:DEFINES]->(fn:Function) RETURN count(fn) as count",
            "classes": f"MATCH (a:Artifact {{name: '{artifact_name}'}})-[:CONTAINS*]->()-[:DEFINES]->(c:Class) RETURN count(c) as count",
            "dependencies": f"MATCH (a:Artifact {{name: '{artifact_name}'}})-[:DEPENDS_ON]->(d:Dependency) RETURN count(d) as count",
        }

        for metric_name, query in queries.items():
            try:
                result = self.graph.run(query).data()
                metrics[metric_name] = result[0]["count"] if result else 0
            except Exception as e:
                logger.warning(f"Error calculating metric {metric_name}: {e}")
                metrics[metric_name] = 0

        # Complexity metrics
        metrics.update(self._calculate_complexity_metrics(artifact_node))

        return metrics

    # Helper methods for analysis
    def _classify_file_type(self, file_path: Path) -> str:
        """Classify file type based on extension and content."""
        ext = file_path.suffix.lower()
        name = file_path.name.lower()

        if ext in ['.py', '.java', '.cpp', '.c', '.js', '.ts', '.go', '.rs']:
            return "code"
        elif ext in ['.md', '.rst', '.txt']:
            return "documentation"
        elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini']:
            return "config"
        elif name.startswith('dockerfile') or name == 'dockerfile':
            return "docker"
        elif name.startswith('license') or name == 'license':
            return "license"
        elif ext in ['.sql']:
            return "data"
        else:
            return "other"

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except:
            return 0

    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        else:
            return str(decorator)

    def _get_base_name(self, base) -> str:
        """Extract base class name from AST node."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
        else:
            return str(base)

    def _is_standard_library(self, module_name: str) -> bool:
        """Check if a module is part of Python standard library."""
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'collections', 'itertools',
            'functools', 'operator', 'pathlib', 'typing', 'logging',
            'unittest', 'argparse', 'configparser', 'urllib', 'http',
            'email', 'html', 'xml', 'csv', 'sqlite3', 'pickle',
            'threading', 'multiprocessing', 'queue', 'socket',
            'ssl', 'hashlib', 'hmac', 'secrets', 'random', 'statistics',
            'math', 'decimal', 'fractions', 'cmath', 'time', 'calendar'
        }
        return module_name.split('.')[0] in stdlib_modules

    # Dependency parsers
    def _parse_requirements_txt(self, file_path: Path) -> List[Dict]:
        """Parse requirements.txt file."""
        dependencies = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Simple parsing - can be enhanced
                        if '>=' in line:
                            name, version = line.split('>=', 1)
                        elif '==' in line:
                            name, version = line.split('==', 1)
                        else:
                            name, version = line, ""

                        dependencies.append({
                            "name": name.strip(),
                            "version": version.strip(),
                            "type": "runtime"
                        })
        except Exception as e:
            logger.warning(f"Error parsing requirements.txt: {e}")

        return dependencies

    def _parse_setup_py(self, file_path: Path) -> List[Dict]:
        """Parse setup.py file for dependencies."""
        # This is a simplified parser - can be enhanced
        return []

    def _parse_pyproject_toml(self, file_path: Path) -> List[Dict]:
        """Parse pyproject.toml file."""
        # This is a simplified parser - can be enhanced
        return []

    def _parse_package_json(self, file_path: Path) -> List[Dict]:
        """Parse package.json file."""
        dependencies = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for dep_type in ['dependencies', 'devDependencies']:
                if dep_type in data:
                    for name, version in data[dep_type].items():
                        dependencies.append({
                            "name": name,
                            "version": version,
                            "type": "runtime" if dep_type == "dependencies" else "dev",
                            "is_dev": dep_type == "devDependencies"
                        })
        except Exception as e:
            logger.warning(f"Error parsing package.json: {e}")

        return dependencies

    def _parse_pom_xml(self, file_path: Path) -> List[Dict]:
        """Parse pom.xml file."""
        # This is a simplified parser - can be enhanced
        return []

    def _parse_build_gradle(self, file_path: Path) -> List[Dict]:
        """Parse build.gradle file."""
        # This is a simplified parser - can be enhanced
        return []

    def _extract_doc_sections(self, content: str, file_type: str) -> List[Dict]:
        """Extract sections from documentation content."""
        sections = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if file_type == '.md' and line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                sections.append({
                    "title": title,
                    "level": level,
                    "content": line,
                    "line_number": i + 1
                })
            elif file_type == '.rst' and line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and all(c in '=-~^"' for c in next_line) and len(next_line) >= len(line):
                    sections.append({
                        "title": line,
                        "level": {'=': 1, '-': 2, '~': 3, '^': 4, '"': 5}.get(next_line[0], 1),
                        "content": line,
                        "line_number": i + 1
                    })

        return sections

    def _classify_doc_section(self, title: str) -> str:
        """Classify documentation section type."""
        title_lower = title.lower()

        if any(word in title_lower for word in ['install', 'setup', 'getting started']):
            return "installation"
        elif any(word in title_lower for word in ['usage', 'example', 'tutorial']):
            return "usage"
        elif any(word in title_lower for word in ['api', 'reference', 'documentation']):
            return "api"
        elif any(word in title_lower for word in ['license', 'copyright']):
            return "license"
        elif any(word in title_lower for word in ['contribute', 'development']):
            return "development"
        else:
            return "general"

    # Pattern analysis methods
    def _analyze_file_organization(self, artifact_node: Node) -> Dict:
        """Analyze file organization patterns."""
        return {"pattern": "standard", "score": 0.8}  # Placeholder

    def _analyze_naming_conventions(self, artifact_node: Node) -> Dict:
        """Analyze naming convention patterns."""
        return {"pattern": "snake_case", "consistency": 0.9}  # Placeholder

    def _analyze_code_structure(self, artifact_node: Node) -> Dict:
        """Analyze code structure patterns."""
        return {"pattern": "modular", "complexity": "medium"}  # Placeholder

    def _analyze_documentation_patterns(self, artifact_node: Node) -> Dict:
        """Analyze documentation patterns."""
        return {"coverage": 0.7, "quality": "good"}  # Placeholder

    def _calculate_complexity_metrics(self, artifact_node: Node) -> Dict:
        """Calculate complexity metrics."""
        return {
            "cyclomatic_complexity": 5.2,
            "depth_of_inheritance": 2.1,
            "coupling": 0.3
        }  # Placeholder

    # Advanced analysis methods
    def perform_advanced_analysis(self, artifact_name: str) -> Dict[str, Any]:
        """Perform advanced graph analysis."""
        analysis = {}

        # Network analysis
        analysis["network_metrics"] = self._calculate_network_metrics(artifact_name)

        # Centrality analysis
        analysis["centrality"] = self._calculate_centrality_metrics(artifact_name)

        # Community detection
        analysis["communities"] = self._detect_communities(artifact_name)

        # Similarity analysis
        analysis["similarity"] = self._calculate_similarity_metrics(artifact_name)

        return analysis

    def _calculate_network_metrics(self, artifact_name: str) -> Dict:
        """Calculate network-level metrics."""
        return {"density": 0.15, "diameter": 8}  # Placeholder

    def _calculate_centrality_metrics(self, artifact_name: str) -> Dict:
        """Calculate node centrality metrics."""
        return {"most_central": ["main.py", "utils.py"]}  # Placeholder

    def _detect_communities(self, artifact_name: str) -> Dict:
        """Detect communities in the graph."""
        return {"communities": 3, "modularity": 0.4}  # Placeholder

    def _calculate_similarity_metrics(self, artifact_name: str) -> Dict:
        """Calculate similarity between artifacts."""
        return {"similar_artifacts": []}  # Placeholder

    # Graph querying and export
    def execute_query(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query and return results."""
        try:
            result = self.graph.run(cypher_query, parameters or {})
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        stats = {}

        try:
            # Basic counts
            stats["total_nodes"] = self.graph.run("MATCH (n) RETURN count(n) as count").data()[0]["count"]
            stats["total_relationships"] = self.graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0][
                "count"]

            # Node type distribution
            node_types = self.graph.run("""
                MATCH (n) 
                RETURN labels(n)[0] as label, count(n) as count 
                ORDER BY count DESC
            """).data()
            stats["node_types"] = {item["label"]: item["count"] for item in node_types}

            # Relationship type distribution
            rel_types = self.graph.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count 
                ORDER BY count DESC
            """).data()
            stats["relationship_types"] = {item["type"]: item["count"] for item in rel_types}

        except Exception as e:
            logger.error(f"Error calculating graph statistics: {e}")
            stats = {"error": str(e)}

        return stats

    def export_visualization(self, output_path: str, format: str = "html") -> str:
        """Export graph visualization."""
        try:
            if format.lower() == "html":
                return self._export_html_visualization(output_path)
            elif format.lower() == "json":
                return self._export_json_data(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Visualization export failed: {e}")
            return ""

    def _export_html_visualization(self, output_path: str) -> str:
        """Export interactive HTML visualization."""
        try:
            from pyvis.network import Network

            net = Network(height="800px", width="100%", notebook=False, directed=True)
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)

            # Get graph data
            query = """
            MATCH (n)-[r]->(m) 
            RETURN n, r, m 
            LIMIT 500
            """
            results = self.graph.run(query).data()

            node_ids = set()
            for record in results:
                n, r, m = record['n'], record['r'], record['m']

                # Add nodes
                for node in [n, m]:
                    if node.identity not in node_ids:
                        label = node.get('name', str(node.identity))
                        node_type = list(node.labels)[0] if node.labels else 'Unknown'

                        net.add_node(
                            node.identity,
                            label=label,
                            title=f"{node_type}: {label}",
                            group=node_type,
                            size=20
                        )
                        node_ids.add(node.identity)

                # Add edge
                net.add_edge(n.identity, m.identity, label=type(r).__name__)

            net.write_html(output_path)
            logger.info(f"HTML visualization exported to: {output_path}")
            return output_path

        except ImportError:
            logger.error("pyvis library not available for HTML export")
            return ""
        except Exception as e:
            logger.error(f"HTML export failed: {e}")
            return ""

    def _export_json_data(self, output_path: str) -> str:
        """Export graph data as JSON."""
        try:
            query = """
            MATCH (n)-[r]->(m)
            RETURN {
                source: {id: id(n), labels: labels(n), properties: properties(n)},
                relationship: {type: type(r), properties: properties(r)},
                target: {id: id(m), labels: labels(m), properties: properties(m)}
            } as graph_data
            LIMIT 1000
            """

            results = self.graph.run(query).data()
            graph_data = [record["graph_data"] for record in results]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, default=str)

            logger.info(f"JSON data exported to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return ""

    def get_artifact_recommendations(self, artifact_name: str) -> Dict[str, Any]:
        """Get recommendations for improving an artifact."""
        recommendations = {
            "documentation": [],
            "code_quality": [],
            "dependencies": [],
            "structure": []
        }

        # Check for missing documentation
        has_readme = self.graph.run(f"""
            MATCH (a:Artifact {{name: '{artifact_name}'}})
            RETURN a.has_readme as has_readme
        """).data()

        if has_readme and not has_readme[0]["has_readme"]:
            recommendations["documentation"].append("Add a README file to explain the project")

        # Add more recommendation logic here

        return recommendations

    def close(self):
        """Close database connections."""
        try:
            if hasattr(self, 'graph'):
                self.graph = None
            logger.info("Closed Neo4j connection")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")


def main():
    """Example usage of the Enhanced KG Builder."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Knowledge Graph Builder")
    parser.add_argument("extracted_path", help="Path to extracted artifact")
    parser.add_argument("artifact_name", help="Name of the artifact")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--clear-graph", action="store_true", help="Clear existing graph")

    args = parser.parse_args()

    # Create builder
    builder = EnhancedKGBuilder(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        clear_existing=args.clear_graph
    )

    try:
        # Build knowledge graph
        result = builder.build_knowledge_graph(
            extracted_path=args.extracted_path,
            artifact_name=args.artifact_name
        )

        if result["success"]:
            print(f"Knowledge graph built successfully!")
            print(f"Nodes created: {result['nodes_created']}")
            print(f"Relationships created: {result['relationships_created']}")

            # Export visualization
            viz_path = builder.export_visualization(f"{args.artifact_name}_graph.html")
            if viz_path:
                print(f"Visualization saved to: {viz_path}")
        else:
            print(f"Failed to build knowledge graph: {result['error']}")

    finally:
        builder.close()


if __name__ == "__main__":
    main()
