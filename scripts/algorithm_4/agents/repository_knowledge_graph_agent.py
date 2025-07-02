import ast
import json
import logging
import os
from typing import Optional, Dict, Any, List

import matplotlib.pyplot as plt
import networkx as nx
from py2neo import Graph, Node, Relationship

logging.basicConfig(level=logging.INFO)


class RepositoryKnowledgeGraphAgent:
    def __init__(
            self,
            artifact_json_path: str,
            neo4j_uri: str = "bolt://localhost:7687",
            neo4j_user: str = "neo4j",
            neo4j_password: str = "12345678",
            clear_existing: bool = True,
    ):
        self.artifact_json_path = artifact_json_path
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.artifact = self._load_artifact()
        if clear_existing:
            self._clear_graph()
        self._build_graph()
        self._build_evaluation_metadata()

    def _load_artifact(self) -> Dict:
        with open(self.artifact_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _clear_graph(self):
        self.graph.run("MATCH (n) DETACH DELETE n")
        logging.info("Cleared existing Neo4j graph.")

    def _build_graph(self):
        logging.info("Building knowledge graph from artifact JSON...")
        repo_name = self.artifact.get("repository_name", "artifact_repo")
        repo_url = self.artifact.get("repository_url", "")
        repo_size = self.artifact.get("repo_size_mb", 0)

        repo_node = Node("Repository",
                         name=repo_name,
                         type="repository",
                         url=repo_url,
                         size_mb=repo_size)
        self.graph.merge(repo_node, "Repository", "name")

        # Build directory and file structure with enhanced metadata
        path_to_node = {}
        for entry in self.artifact.get("repository_structure", []):
            if (entry.get("mime_type") or "").startswith("directory"):
                dir_node = Node("Directory",
                                name=entry["name"],
                                path=entry["path"],
                                type="directory")
                self.graph.merge(dir_node, "Directory", "path")
                self.graph.merge(Relationship(repo_node, "CONTAINS", dir_node))
                path_to_node[entry["path"]] = dir_node
            else:
                file_type = self._infer_file_type(entry["name"])
                file_node = Node("File",
                                 name=entry["name"],
                                 path=entry["path"],
                                 type=file_type,
                                 content_type=entry.get("mime_type", ""),
                                 size_kb=entry.get("size_kb", 0))
                self.graph.merge(file_node, "File", "path")

                # Create hierarchy relationships
                parent_path = os.path.dirname(entry["path"])
                if parent_path and parent_path in path_to_node:
                    self.graph.merge(Relationship(path_to_node[parent_path], "CONTAINS", file_node))
                else:
                    self.graph.merge(Relationship(repo_node, "CONTAINS", file_node))

                path_to_node[entry["path"]] = file_node

        # Add documentation/code/license/data files with enhanced content analysis
        for file_list, node_type in [
            ("documentation_files", "documentation"),
            ("code_files", "code"),
            ("license_files", "license"),
        ]:
            for file in self.artifact.get(file_list, []):
                file_node = path_to_node.get(file["path"])
                if not file_node:
                    file_node = Node("File",
                                     name=file["path"].split("/")[-1],
                                     path=file["path"],
                                     type=node_type)
                    self.graph.merge(file_node, "File", "path")

                # Enhanced content analysis for different file types
                content = file.get("content", [])
                if isinstance(content, list):
                    # Add metadata about content structure
                    self.graph.run("""
                        MATCH (f:File {path: $path})
                        SET f.line_count = $line_count,
                            f.content_length = $content_length
                    """, {
                        "path": file["path"],
                        "line_count": len(content),
                        "content_length": sum(len(line) for line in content)
                    })

                    # Create sections with semantic information
                    for idx, section in enumerate(content):
                        if section.strip():  # Only non-empty sections
                            section_type = self._classify_section_type(section, node_type)
                            section_node = Node("Section",
                                                name=f"section_{idx}",
                                                content=section,
                                                type=section_type,
                                                parent_type=node_type,
                                                line_number=idx)
                            self.graph.create(section_node)
                            self.graph.create(Relationship(file_node, "CONTAINS", section_node))

                # Enhanced documentation relationships
                if node_type == "documentation" and "readme" in file["path"].lower():
                    # Link README to all code files it might describe
                    for code_file in self.artifact.get("code_files", []):
                        code_node = path_to_node.get(code_file["path"])
                        if code_node:
                            self.graph.merge(Relationship(file_node, "DESCRIBES", code_node))

                    # Add README quality metrics
                    readme_quality = self._analyze_readme_quality(content)
                    self.graph.run("""
                        MATCH (f:File {path: $path})
                        SET f.has_installation = $has_installation,
                            f.has_usage = $has_usage,
                            f.has_examples = $has_examples,
                            f.has_license_info = $has_license_info,
                            f.quality_score = $quality_score
                    """, {
                        "path": file["path"],
                        **readme_quality
                    })

        # Enhanced code analysis with dependency tracking
        for file in self.artifact.get("code_files", []):
            file_node = path_to_node.get(file["path"])
            if not file_node:
                continue

            if file["path"].endswith(".py") and isinstance(file.get("content"), list):
                code_str = "\n".join(file["content"])
                self._analyze_python_code(file_node, code_str, file["path"])

            # Add code quality metrics
            code_metrics = self._calculate_code_metrics(file.get("content", []))
            self.graph.run("""
                MATCH (f:File {path: $path})
                SET f.complexity_score = $complexity_score,
                    f.comment_ratio = $comment_ratio,
                    f.function_count = $function_count
            """, {
                "path": file["path"],
                **code_metrics
            })

        # Enhanced dataset and test detection
        for entry in self.artifact.get("repository_structure", []):
            if self._is_dataset(entry["path"]):
                data_node = Node("Dataset",
                                 name=entry["name"],
                                 path=entry["path"],
                                 type="data",
                                 size_kb=entry.get("size_kb", 0),
                                 data_format=self._infer_data_format(entry["name"]))
                self.graph.merge(data_node, "Dataset", "path")

                # Link to parent directory
                parent = path_to_node.get(os.path.dirname(entry["path"]))
                if parent:
                    self.graph.merge(Relationship(parent, "CONTAINS", data_node))

            if self._is_test_file(entry["name"]):
                test_node = Node("Test",
                                 name=entry["name"],
                                 path=entry["path"],
                                 type="test",
                                 test_framework=self._infer_test_framework(entry["name"]))
                self.graph.merge(test_node, "Test", "path")

                # Link test files to the code they test
                tested_file = self._find_tested_file(entry["path"], path_to_node)
                if tested_file:
                    self.graph.merge(Relationship(test_node, "TESTS", tested_file))

        logging.info("Knowledge graph build complete.")

        # Export visualizations
        try:
            self.export_graph_html()
        except Exception as e:
            logging.warning(f"Failed to export graph HTML visualization: {e}")

    def _build_evaluation_metadata(self):
        """Build evaluation-specific metadata nodes and relationships."""
        logging.info("Building evaluation metadata...")

        # Create evaluation criteria nodes
        criteria = ["accessibility", "documentation", "reproducibility",
                    "usability", "functionality", "experimental"]

        for criterion in criteria:
            criterion_node = Node("EvaluationCriterion",
                                  name=criterion,
                                  type="criterion")
            self.graph.merge(criterion_node, "EvaluationCriterion", "name")

            # Link relevant files to criteria
            self._link_files_to_criterion(criterion)

        # Create artifact statistics
        self._calculate_artifact_statistics()

    def _link_files_to_criterion(self, criterion: str):
        """Link files to evaluation criteria based on their relevance."""
        criterion_file_mapping = {
            "accessibility": ["license", "readme"],
            "documentation": ["documentation", "readme"],
            "reproducibility": ["dockerfile", "requirements", "setup", "makefile"],
            "usability": ["readme", "documentation", "examples"],
            "functionality": ["code", "test"],
            "experimental": ["data", "results", "analysis"]
        }

        relevant_types = criterion_file_mapping.get(criterion, [])

        for file_type in relevant_types:
            self.graph.run("""
                MATCH (c:EvaluationCriterion {name: $criterion}),
                      (f:File)
                WHERE toLower(f.type) CONTAINS $file_type OR 
                      toLower(f.name) CONTAINS $file_type
                MERGE (f)-[:RELEVANT_TO]->(c)
            """, {"criterion": criterion, "file_type": file_type})

    def _calculate_artifact_statistics(self):
        """Calculate and store artifact-level statistics."""
        stats = {
            "total_files": len(self.artifact.get("repository_structure", [])),
            "code_files": len(self.artifact.get("code_files", [])),
            "doc_files": len(self.artifact.get("documentation_files", [])),
            "license_files": len(self.artifact.get("license_files", [])),
            "total_size_mb": self.artifact.get("repo_size_mb", 0)
        }

        # Create artifact statistics node
        stats_node = Node("ArtifactStatistics", **stats, type="statistics")
        self.graph.create(stats_node)

        # Link to repository
        self.graph.run("""
            MATCH (r:Repository), (s:ArtifactStatistics)
            MERGE (r)-[:HAS_STATISTICS]->(s)
        """)

    def _classify_section_type(self, content: str, parent_type: str) -> str:
        """Classify the type of a content section."""
        content_lower = content.lower()

        if parent_type == "documentation":
            if any(keyword in content_lower for keyword in ["install", "setup", "requirements"]):
                return "installation"
            elif any(keyword in content_lower for keyword in ["usage", "example", "demo"]):
                return "usage"
            elif any(keyword in content_lower for keyword in ["license", "copyright"]):
                return "license"
            elif any(keyword in content_lower for keyword in ["abstract", "introduction", "overview"]):
                return "description"
            else:
                return "general"
        elif parent_type == "code":
            if content.strip().startswith("#") or content.strip().startswith("//"):
                return "comment"
            elif any(keyword in content_lower for keyword in ["import", "from", "include"]):
                return "import"
            elif any(keyword in content_lower for keyword in ["def ", "function", "class "]):
                return "definition"
            else:
                return "code"

        return "general"

    def _analyze_readme_quality(self, content: List[str]) -> Dict[str, Any]:
        """Analyze README quality metrics."""
        if not content:
            return {
                "has_installation": False,
                "has_usage": False,
                "has_examples": False,
                "has_license_info": False,
                "quality_score": 0.0
            }

        content_text = "\n".join(content).lower()

        metrics = {
            "has_installation": any(keyword in content_text for keyword in
                                    ["install", "setup", "requirements", "dependencies"]),
            "has_usage": any(keyword in content_text for keyword in
                             ["usage", "how to use", "example", "tutorial"]),
            "has_examples": any(keyword in content_text for keyword in
                                ["example", "demo", "sample", "tutorial"]),
            "has_license_info": any(keyword in content_text for keyword in
                                    ["license", "copyright", "terms"])
        }

        # Calculate quality score (0-1)
        score = sum(metrics.values()) / len(metrics)
        metrics["quality_score"] = score

        return metrics

    def _analyze_python_code(self, file_node: Node, code_str: str, file_path: str):
        """Enhanced Python code analysis."""
        try:
            tree = ast.parse(code_str)
            imports = []
            functions = []
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_node = Node("Function",
                                     name=node.name,
                                     type="function",
                                     line_number=node.lineno,
                                     is_async=isinstance(node, ast.AsyncFunctionDef))
                    self.graph.create(func_node)
                    self.graph.create(Relationship(file_node, "DEFINES", func_node))
                    functions.append(node.name)

                elif isinstance(node, ast.ClassDef):
                    class_node = Node("Class",
                                      name=node.name,
                                      type="class",
                                      line_number=node.lineno,
                                      base_classes=[base.id for base in node.bases if hasattr(base, 'id')])
                    self.graph.create(class_node)
                    self.graph.create(Relationship(file_node, "DEFINES", class_node))
                    classes.append(node.name)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        import_node = Node("Import",
                                           module=alias.name,
                                           type="import",
                                           import_type="direct")
                        self.graph.create(import_node)
                        self.graph.create(Relationship(file_node, "IMPORTS", import_node))
                        imports.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    import_node = Node("Import",
                                       module=module_name,
                                       type="import",
                                       import_type="from",
                                       level=node.level)
                    self.graph.create(import_node)
                    self.graph.create(Relationship(file_node, "IMPORTS", import_node))
                    imports.append(module_name)

            # Add code analysis metadata
            self.graph.run("""
                MATCH (f:File {path: $path})
                SET f.import_count = $import_count,
                    f.function_count = $function_count,
                    f.class_count = $class_count,
                    f.imports = $imports
            """, {
                "path": file_path,
                "import_count": len(imports),
                "function_count": len(functions),
                "class_count": len(classes),
                "imports": imports[:10]  # Limit to avoid too much data
            })

        except Exception as e:
            logging.warning(f"AST parsing failed for {file_path}: {e}")

    def _calculate_code_metrics(self, content: List[str]) -> Dict[str, float]:
        """Calculate code quality metrics."""
        if not content:
            return {"complexity_score": 0.0, "comment_ratio": 0.0, "function_count": 0}

        total_lines = len(content)
        comment_lines = sum(1 for line in content if line.strip().startswith("#"))
        function_count = sum(1 for line in content if line.strip().startswith("def "))

        # Simple complexity measure based on control structures
        complexity_keywords = ["if", "for", "while", "try", "except", "with"]
        complexity_score = sum(line.lower().count(keyword) for line in content for keyword in complexity_keywords)

        return {
            "complexity_score": min(complexity_score / max(total_lines, 1), 1.0),
            "comment_ratio": comment_lines / max(total_lines, 1),
            "function_count": function_count
        }

    def _infer_data_format(self, filename: str) -> str:
        """Infer data format from filename."""
        ext = filename.lower().split('.')[-1] if '.' in filename else ""
        format_mapping = {
            'csv': 'tabular',
            'json': 'json',
            'xml': 'xml',
            'txt': 'text',
            'parquet': 'columnar',
            'h5': 'hdf5',
            'pkl': 'pickle',
            'jpg': 'image',
            'png': 'image',
            'mp4': 'video'
        }
        return format_mapping.get(ext, 'unknown')

    def _infer_test_framework(self, filename: str) -> str:
        """Infer testing framework from filename patterns."""
        if 'pytest' in filename.lower():
            return 'pytest'
        elif 'unittest' in filename.lower():
            return 'unittest'
        elif filename.startswith('test_'):
            return 'pytest'  # Common pytest pattern
        else:
            return 'unknown'

    def _find_tested_file(self, test_path: str, path_to_node: Dict) -> Optional[Node]:
        """Find the file that a test file is testing."""
        # Simple heuristic: remove 'test_' prefix and look for corresponding file
        test_name = os.path.basename(test_path)
        if test_name.startswith('test_'):
            potential_file = test_name[5:]  # Remove 'test_'
            # Look for corresponding file in src or root
            for path in path_to_node:
                if potential_file in path:
                    return path_to_node[path]
        return None

    def _infer_file_type(self, filename: str) -> str:
        """Enhanced file type inference."""
        filename_lower = filename.lower()

        # Documentation files
        if filename_lower.endswith(".md") or filename_lower in ["readme", "readme.txt"]:
            return "documentation"

        # License files
        if filename_lower in ["license", "license.txt", "license.md", "copying"]:
            return "license"

        # Code files
        code_extensions = ['.py', '.ipynb', '.sh', '.bat', '.pl', '.r', '.cpp', '.c', '.java', '.js', '.ts']
        if any(filename_lower.endswith(ext) for ext in code_extensions):
            return "code"

        # Data files
        data_extensions = ['.csv', '.json', '.tsv', '.xlsx', '.xls', '.parquet', '.h5', '.hdf5']
        if any(filename_lower.endswith(ext) for ext in data_extensions):
            return "data"

        # Configuration files
        config_files = ["dockerfile", "makefile", "requirements.txt", "setup.py", "pyproject.toml"]
        if filename_lower in config_files:
            return "config"

        # Test files
        if filename_lower.startswith("test") or "_test" in filename_lower:
            return "test"

        return "other"

    def _is_dataset(self, path: str) -> bool:
        """Enhanced dataset detection."""
        path_lower = path.lower()
        dataset_indicators = ["/data/", "/dataset/", "/datasets/", "/examples/", "/samples/"]
        return any(indicator in path_lower for indicator in dataset_indicators)

    def _is_test_file(self, filename: str) -> bool:
        """Enhanced test file detection."""
        filename_lower = filename.lower()
        return (filename_lower.startswith("test") or
                filename_lower.endswith("_test.py") or
                "test" in filename_lower and filename_lower.endswith(".py"))

    def run_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run an arbitrary Cypher query and return results as a list of dicts."""
        result = self.graph.run(query, parameters or {})
        return [dict(record) for record in result]

    # Enhanced evidence query methods for evaluation agents
    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in the repository."""
        query = """
        MATCH (f:File) 
        WHERE f.name = $filename OR f.path CONTAINS $filename
        RETURN f LIMIT 1
        """
        return bool(self.run_cypher(query, {"filename": filename}))

    def get_file_content(self, filename: str) -> Optional[str]:
        """Get content of a specific file."""
        query = """
        MATCH (f:File)-[:CONTAINS]->(s:Section) 
        WHERE f.name = $filename OR f.path CONTAINS $filename
        RETURN s.content as content ORDER BY s.line_number
        """
        results = self.run_cypher(query, {"filename": filename})
        if results:
            return "\n".join([r["content"] for r in results])
        return None

    def readme_has_section(self, section_keyword: str) -> bool:
        """Check if README contains a specific section."""
        query = """
        MATCH (f:File)-[:CONTAINS]->(s:Section)
        WHERE toLower(f.name) CONTAINS 'readme' 
        AND (toLower(s.content) CONTAINS $section_keyword OR s.type = $section_keyword)
        RETURN s LIMIT 1
        """
        return bool(self.run_cypher(query, {"section_keyword": section_keyword.lower()}))

    def test_files_exist(self) -> bool:
        """Check if test files exist."""
        query = "MATCH (t:Test) RETURN t LIMIT 1"
        return bool(self.run_cypher(query))

    def dataset_files_exist(self) -> bool:
        """Check if dataset files exist."""
        query = "MATCH (d:Dataset) RETURN d LIMIT 1"
        return bool(self.run_cypher(query))

    def get_code_quality_metrics(self) -> Dict[str, float]:
        """Get aggregated code quality metrics."""
        query = """
        MATCH (f:File {type: 'code'})
        RETURN avg(f.complexity_score) as avg_complexity,
               avg(f.comment_ratio) as avg_comment_ratio,
               sum(f.function_count) as total_functions,
               count(f) as code_file_count
        """
        result = self.run_cypher(query)
        if result:
            return result[0]
        return {"avg_complexity": 0, "avg_comment_ratio": 0, "total_functions": 0, "code_file_count": 0}

    def get_documentation_quality(self) -> Dict[str, Any]:
        """Get documentation quality metrics."""
        query = """
        MATCH (f:File {type: 'documentation'})
        WHERE toLower(f.name) CONTAINS 'readme'
        RETURN f.has_installation as has_installation,
               f.has_usage as has_usage,
               f.has_examples as has_examples,
               f.has_license_info as has_license_info,
               f.quality_score as quality_score
        LIMIT 1
        """
        result = self.run_cypher(query)
        if result:
            return result[0]
        return {"has_installation": False, "has_usage": False, "has_examples": False,
                "has_license_info": False, "quality_score": 0.0}

    def get_dependency_analysis(self) -> Dict[str, List[str]]:
        """Get dependency analysis from imports."""
        query = """
        MATCH (f:File)-[:IMPORTS]->(i:Import)
        RETURN f.name as file, collect(i.module) as imports
        """
        results = self.run_cypher(query)
        return {r["file"]: r["imports"] for r in results}

    def get_files_by_criterion(self, criterion: str) -> List[Dict[str, Any]]:
        """Get files relevant to a specific evaluation criterion."""
        query = """
        MATCH (f:File)-[:RELEVANT_TO]->(c:EvaluationCriterion {name: $criterion})
        RETURN f.name as name, f.path as path, f.type as type, f.size_kb as size_kb
        """
        return self.run_cypher(query, {"criterion": criterion})

    def get_artifact_statistics(self) -> Dict[str, Any]:
        """Get artifact-level statistics."""
        query = """
        MATCH (s:ArtifactStatistics)
        RETURN s.total_files as total_files,
               s.code_files as code_files,
               s.doc_files as doc_files,
               s.license_files as license_files,
               s.total_size_mb as total_size_mb
        """
        result = self.run_cypher(query)
        if result:
            return result[0]
        return {"total_files": 0, "code_files": 0, "doc_files": 0, "license_files": 0, "total_size_mb": 0}

    def check_accessibility_indicators(self) -> Dict[str, bool]:
        """Check for accessibility-specific indicators."""
        indicators = {
            "has_license": self.file_exists("LICENSE"),
            "has_readme": self.file_exists("README"),
            "has_doi": self.readme_has_section("doi"),
            "has_archive_link": self.readme_has_section("zenodo") or self.readme_has_section("figshare"),
            "has_github_url": True  # Assume GitHub if we have the data
        }
        return indicators

    def check_reproducibility_indicators(self) -> Dict[str, bool]:
        """Check for reproducibility-specific indicators."""
        indicators = {
            "has_dockerfile": self.file_exists("Dockerfile"),
            "has_requirements": self.file_exists("requirements.txt"),
            "has_setup": self.file_exists("setup.py"),
            "has_makefile": self.file_exists("Makefile"),
            "has_conda_env": self.file_exists("environment.yml"),
            "has_installation_docs": self.readme_has_section("installation")
        }
        return indicators

    # Badge/Requirement Support (enhanced)
    def add_badge(self, badge_name: str, description: str = "", criterion: str = ""):
        """Add a badge node with criterion linkage."""
        badge_node = Node("Badge",
                          name=badge_name,
                          description=description,
                          criterion=criterion)
        self.graph.merge(badge_node, "Badge", "name")

        # Link to evaluation criterion if specified
        if criterion:
            self.graph.run("""
                MATCH (b:Badge {name: $badge_name}), (c:EvaluationCriterion {name: $criterion})
                MERGE (b)-[:EVALUATES]->(c)
            """, {"badge_name": badge_name, "criterion": criterion})

    def link_file_to_badge(self, filename: str, badge_name: str):
        """Link a file to a badge."""
        query = """
        MATCH (f:File), (b:Badge {name: $badge_name})
        WHERE f.name = $filename OR f.path CONTAINS $filename
        MERGE (f)-[:SATISFIES]->(b)
        """
        self.graph.run(query, {"filename": filename, "badge_name": badge_name})

    def get_satisfied_badges(self) -> List[Dict[str, Any]]:
        """Get all satisfied badges."""
        query = """
        MATCH (f:File)-[:SATISFIES]->(b:Badge)
        RETURN b.name as badge, b.description as description, b.criterion as criterion,
               collect(f.name) as satisfying_files
        """
        return self.run_cypher(query)

    def check_conference_specific_patterns(self, conference_category: str) -> Dict[str, Any]:
        """Check for conference-specific patterns in the repository."""
        patterns = {
            "software_engineering": {
                "ci_cd_files": [".github/workflows", ".gitlab-ci.yml", ".travis.yml"],
                "build_files": ["Makefile", "build.gradle", "pom.xml"],
                "dependency_files": ["requirements.txt", "package.json", "setup.py"]
            },
            "data_systems": {
                "data_files": ["*.csv", "*.json", "*.parquet"],
                "analysis_files": ["analysis", "experiments", "evaluation"],
                "notebook_files": ["*.ipynb"]
            },
            "hci": {
                "ui_files": ["*.html", "*.css", "*.js"],
                "user_study_files": ["survey", "interview", "usability"],
                "design_files": ["wireframe", "mockup", "prototype"]
            }
        }

        category_patterns = patterns.get(conference_category, {})
        results = {}

        for pattern_type, file_patterns in category_patterns.items():
            found_files = []
            for pattern in file_patterns:
                query = """
                MATCH (f:File)
                WHERE toLower(f.name) CONTAINS $pattern OR toLower(f.path) CONTAINS $pattern
                RETURN f.name as name, f.path as path
                """
                matches = self.run_cypher(query, {"pattern": pattern.lower().replace("*.", "")})
                found_files.extend(matches)

            results[pattern_type] = {
                "found": len(found_files) > 0,
                "count": len(found_files),
                "files": found_files[:5]  # Limit to first 5 matches
            }

        return results

    def analyze_repository_structure_quality(self) -> Dict[str, Any]:
        """Analyze the overall structure quality of the repository."""
        query = """
        MATCH (r:Repository)-[:HAS_STATISTICS]->(s:ArtifactStatistics)
        RETURN s.total_files as total_files,
               s.code_files as code_files,
               s.doc_files as doc_files,
               s.license_files as license_files
        """
        stats = self.run_cypher(query)

        if not stats:
            return {"quality_score": 0.0, "issues": ["No statistics available"]}

        stat = stats[0]
        issues = []
        quality_components = []

        # Check documentation ratio
        doc_ratio = stat["doc_files"] / max(stat["total_files"], 1)
        if doc_ratio < 0.1:
            issues.append("Low documentation ratio")
            quality_components.append(0.3)
        else:
            quality_components.append(min(doc_ratio * 2, 1.0))

        # Check license presence
        if stat["license_files"] == 0:
            issues.append("No license file found")
            quality_components.append(0.0)
        else:
            quality_components.append(1.0)

        # Check code organization
        if stat["code_files"] > 0:
            quality_components.append(0.8)
        else:
            issues.append("No code files found")
            quality_components.append(0.0)

        # Calculate overall quality score
        quality_score = sum(quality_components) / len(quality_components) if quality_components else 0.0

        return {
            "quality_score": quality_score,
            "documentation_ratio": doc_ratio,
            "has_license": stat["license_files"] > 0,
            "has_code": stat["code_files"] > 0,
            "total_files": stat["total_files"],
            "issues": issues
        }

    def get_conference_relevance_score(self, conference_keywords: List[str]) -> float:
        """Calculate how relevant the repository is to a conference based on keywords."""
        if not conference_keywords:
            return 0.5  # Neutral score

        # Check keyword presence in documentation
        keyword_matches = 0
        total_keywords = len(conference_keywords)

        for keyword in conference_keywords:
            query = """
            MATCH (f:File {type: 'documentation'})-[:CONTAINS]->(s:Section)
            WHERE toLower(s.content) CONTAINS $keyword
            RETURN count(s) as matches
            """
            matches = self.run_cypher(query, {"keyword": keyword.lower()})
            if matches and matches[0]["matches"] > 0:
                keyword_matches += 1

        return keyword_matches / total_keywords if total_keywords > 0 else 0.0

    def close(self):
        """Clean up resources."""
        self.graph = None

    def export_graph_html(self, output_path=None):
        """Export interactive HTML visualization."""
        from pyvis.network import Network
        import os

        # Always save to project-root algo_outputs/kg_viz.html
        if output_path is None:
            output_dir = os.path.join(os.getcwd(), "algo_outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "kg_viz.html")
        else:
            output_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        net = Network(height="700px", width="100%", notebook=False, directed=True)

        # Enhanced visualization with more node types and colors
        color_mapping = {
            "repository": "#ff6b6b",
            "file": "#4ecdc4",
            "documentation": "#45b7d1",
            "code": "#96ceb4",
            "license": "#ffeaa7",
            "data": "#dda0dd",
            "test": "#98d8c8",
            "function": "#f7dc6f",
            "class": "#bb8fce",
            "import": "#85c1e9"
        }

        results = self.graph.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 500")
        node_ids = set()

        for record in results:
            n = record['n']
            m = record['m']
            r = record['r']

            for node in [n, m]:
                if node.identity not in node_ids:
                    node_type = node.get('type', 'Other')
                    color = color_mapping.get(node_type, '#95a5a6')

                    net.add_node(
                        node.identity,
                        label=node.get('name', str(node.identity)),
                        title=str(dict(node)),
                        group=node_type,
                        color=color
                    )
                    node_ids.add(node.identity)

            net.add_edge(n.identity, m.identity, label=type(r).__name__)

        net.write_html(output_path)
        logging.info(f"Interactive graph visualization saved to {output_path}")

    def export_graph_png(self, output_path="../../algo_outputs/graph_viz.png"):
        """Export a PNG visualization of the graph."""
        # Query the graph
        results = self.graph.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 200").data()
        G = nx.MultiDiGraph()

        for record in results:
            n = record['n']
            m = record['m']
            r = record['r']
            n_label = f"{n['name']}\n({n['type']})" if 'type' in n else n['name']
            m_label = f"{m['name']}\n({m['type']})" if 'type' in m else m['name']
            G.add_node(n_label)
            G.add_node(m_label)
            G.add_edge(n_label, m_label, label=r.__class__.__name__ if hasattr(r, '__class__') else str(r))

        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                edge_color='gray', node_size=1000, font_size=8,
                font_weight='bold', arrows=True)

        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_color='red')

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Graph visualization saved to {output_path}")
