import json
import ast
import os
from py2neo import Graph, Node, Relationship
import logging
from typing import Optional, Dict, Any, List

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

    def _load_artifact(self) -> Dict:
        with open(self.artifact_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _clear_graph(self):
        self.graph.run("MATCH (n) DETACH DELETE n")
        logging.info("Cleared existing Neo4j graph.")

    def _build_graph(self):
        logging.info("Building knowledge graph from artifact JSON...")
        repo_name = self.artifact.get("repository_name", "artifact_repo")
        repo_node = Node("Repository", name=repo_name, type="repository")
        self.graph.merge(repo_node, "Repository", "name")

        # Build directory and file structure
        path_to_node = {}
        for entry in self.artifact.get("repository_structure", []):
            if entry.get("mime_type", "").startswith("directory"):
                dir_node = Node("Directory", name=entry["name"], path=entry["path"], type="directory")
                self.graph.merge(dir_node, "Directory", "path")
                self.graph.merge(Relationship(repo_node, "CONTAINS", dir_node))
                path_to_node[entry["path"]] = dir_node
            else:
                file_type = self._infer_file_type(entry["name"])
                file_node = Node("File", name=entry["name"], path=entry["path"], type=file_type, content_type=entry.get("mime_type", ""))
                self.graph.merge(file_node, "File", "path")
                self.graph.merge(Relationship(repo_node, "CONTAINS", file_node))
                path_to_node[entry["path"]] = file_node

        # Add documentation/code/license/data files as nodes and sections
        for file_list, node_type in [
            ("documentation_files", "documentation"),
            ("code_files", "code"),
            ("license_files", "license"),
        ]:
            for file in self.artifact.get(file_list, []):
                file_node = path_to_node.get(file["path"])
                if not file_node:
                    file_node = Node("File", name=file["path"].split("/")[-1], path=file["path"], type=node_type)
                    self.graph.merge(file_node, "File", "path")
                # Add sections if content is structured
                if isinstance(file.get("content"), list):
                    for idx, section in enumerate(file["content"]):
                        section_node = Node("Section", name=f"section_{idx}", content=section, type=node_type)
                        self.graph.create(section_node)
                        self.graph.create(Relationship(file_node, "CONTAINS", section_node))
                # Advanced: If README, link DESCRIBES to code files
                if node_type == "documentation" and "readme" in file["path"].lower():
                    for code_file in self.artifact.get("code_files", []):
                        code_node = path_to_node.get(code_file["path"])
                        if code_node:
                            self.graph.merge(Relationship(file_node, "DESCRIBES", code_node))

        # Extract and link functions, classes, and imports from Python code files
        for file in self.artifact.get("code_files", []):
            file_node = path_to_node.get(file["path"])
            if not file_node:
                continue
            if file["path"].endswith(".py") and isinstance(file.get("content"), list):
                code_str = "\n".join(file["content"])
                try:
                    tree = ast.parse(code_str)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_node = Node("Function", name=node.name, type="function")
                            self.graph.create(func_node)
                            self.graph.create(Relationship(file_node, "DEFINES", func_node))
                        elif isinstance(node, ast.ClassDef):
                            class_node = Node("Class", name=node.name, type="class")
                            self.graph.create(class_node)
                            self.graph.create(Relationship(file_node, "DEFINES", class_node))
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                import_node = Node("Import", module=alias.name, type="import")
                                self.graph.create(import_node)
                                self.graph.create(Relationship(file_node, "IMPORTS", import_node))
                        elif isinstance(node, ast.ImportFrom):
                            import_node = Node("Import", module=node.module or "", type="import")
                            self.graph.create(import_node)
                            self.graph.create(Relationship(file_node, "IMPORTS", import_node))
                except Exception as e:
                    logging.warning(f"AST parsing failed for {file['path']}: {e}")

        # Detect and add dataset/datafile and test file nodes
        for entry in self.artifact.get("repository_structure", []):
            if self._is_dataset(entry["path"]):
                data_node = Node("Dataset", name=entry["name"], path=entry["path"], type="data")
                self.graph.merge(data_node, "Dataset", "path")
                parent = path_to_node.get(os.path.dirname(entry["path"]))
                if parent:
                    self.graph.merge(Relationship(parent, "CONTAINS", data_node))
            if self._is_test_file(entry["name"]):
                test_node = Node("Test", name=entry["name"], path=entry["path"], type="test")
                self.graph.merge(test_node, "Test", "path")
                file_node = path_to_node.get(entry["path"])
                if file_node:
                    self.graph.merge(Relationship(file_node, "IS_TEST", test_node))

        logging.info("Knowledge graph build complete.")

    def _infer_file_type(self, filename: str) -> str:
        if filename.lower().endswith(".md"):
            return "documentation"
        if filename.lower() in ["license", "license.txt", "license.md"]:
            return "license"
        if filename.lower().endswith(('.py', '.ipynb', '.sh', '.bat', '.pl', '.r', '.cpp', '.c', '.java')):
            return "code"
        if filename.lower().endswith(('.csv', '.json', '.tsv', '.xlsx', '.xls', '.parquet', '.h5', '.hdf5')):
            return "data"
        if filename.lower() == "dockerfile":
            return "docker"
        return "other"

    def _is_dataset(self, path: str) -> bool:
        # Heuristic: data/ or dataset/ in path
        return any(x in path.lower() for x in ["/data/", "/dataset/", "/datasets/"])

    def _is_test_file(self, filename: str) -> bool:
        return filename.lower().startswith("test") or filename.lower().endswith("_test.py")

    def run_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run an arbitrary Cypher query and return results as a list of dicts."""
        result = self.graph.run(query, parameters or {})
        return [dict(record) for record in result]

    # --- Evidence Query Methods for Evaluation Agents ---
    def file_exists(self, filename: str) -> bool:
        query = """
        MATCH (f:File {name: $filename}) RETURN f LIMIT 1
        """
        return bool(self.run_cypher(query, {"filename": filename}))

    def get_file_content(self, filename: str) -> Optional[str]:
        query = """
        MATCH (f:File {name: $filename})-[:CONTAINS]->(s:Section) RETURN s.content as content
        """
        results = self.run_cypher(query, {"filename": filename})
        if results:
            return "\n".join([r["content"] for r in results])
        return None

    def readme_has_section(self, section_keyword: str) -> bool:
        query = """
        MATCH (f:File)-[:CONTAINS]->(s:Section)
        WHERE toLower(f.name) CONTAINS 'readme' AND toLower(s.content) CONTAINS $section_keyword
        RETURN s LIMIT 1
        """
        return bool(self.run_cypher(query, {"section_keyword": section_keyword.lower()}))

    def test_files_exist(self) -> bool:
        query = """
        MATCH (t:Test) RETURN t LIMIT 1
        """
        return bool(self.run_cypher(query))

    def dataset_files_exist(self) -> bool:
        query = """
        MATCH (d:Dataset) RETURN d LIMIT 1
        """
        return bool(self.run_cypher(query))

    # --- Badge/Requirement Support ---
    def add_badge(self, badge_name: str, description: str = ""):
        badge_node = Node("Badge", name=badge_name, description=description)
        self.graph.merge(badge_node, "Badge", "name")

    def link_file_to_badge(self, filename: str, badge_name: str):
        query = """
        MATCH (f:File {name: $filename}), (b:Badge {name: $badge_name})
        MERGE (f)-[:SATISFIES]->(b)
        """
        self.graph.run(query, {"filename": filename, "badge_name": badge_name})

    def parse_guideline_and_add_badges(self, guideline_path: str):
        """Parse a guideline file and add Badge nodes for requirements (simple keyword-based extraction)."""
        with open(guideline_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if any(kw in line.lower() for kw in ["dockerfile", "readme", "license", "test", "dataset", "requirements"]):
                badge_name = line.strip().split(":")[0].split(".")[1].strip() if "." in line else line.strip()
                self.add_badge(badge_name, description=line.strip())

    # --- LangChain-compatible Retriever Stub ---
    def retrieve_context(self, query: str) -> str:
        """Stub: Translate NL query to Cypher and return minimal context (to be implemented with LLM/langchain)."""
        # For now, just return a placeholder
        return f"[GraphRAG Retriever] Query: {query} (NL2Cypher not implemented)"

    def close(self):
        self.graph = None 