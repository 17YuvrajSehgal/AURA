import json
import logging
from typing import List, Optional, Dict

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("../functionality_evaluation_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class FunctionalityCriterion(BaseModel):
    name: str = Field(description="Functionality aspect to check")
    description: str = Field(description="Description of what this aspect means or how to check it")


class FunctionalityCriteriaList(BaseModel):
    criteria: List[FunctionalityCriterion] = Field(
        description="List of required functionality criteria for the conference")


class FunctionalityEvaluationAgent:
    def __init__(
            self,
            guideline_path: str,
            artifact_json_path: str,
            conference_name: str = "ICSE 2025",
            persist_directory: str = "functionality_chroma_index",
            chunk_size: int = 1024,
            chunk_overlap: int = 100,
            model_name: Optional[str] = None,
            keyword_agent: Optional[object] = None,
            kg_agent: Optional[object] = None,
    ):
        self.guideline_path = guideline_path
        self.artifact_json_path = artifact_json_path
        self.conference_name = conference_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.keyword_agent = keyword_agent
        self.kg_agent = kg_agent

        logger.info(f"Initializing FunctionalityEvaluationAgent for {conference_name}")
        self.guidelines = self._load_guidelines()
        self.artifact = self._load_artifact()
        self.criteria = self._extract_functionality_criteria()
        self.db = self._build_vector_db()

    def _load_guidelines(self):
        logger.info(f"Loading conference guidelines from: {self.guideline_path}")
        with open(self.guideline_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_artifact(self):
        logger.info(f"Loading artifact JSON from: {self.artifact_json_path}")
        with open(self.artifact_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_functionality_criteria(self):
        logger.info("Extracting functionality criteria using LLM.")
        parser = PydanticOutputParser(pydantic_object=FunctionalityCriteriaList)
        guideline_prompt = PromptTemplate(
            template="""
                You are an expert at extracting *only* the criteria required to evaluate the **functionality** of a research software artifact, based strictly on the provided conference guidelines.

                **Functionality criteria** are those that directly relate to:
                - Whether the artifact works as claimed (performs its intended tasks or computations)
                - Whether it can be executed successfully (is runnable, produces results as expected)
                - Whether it includes test cases, test results, or evidence of validation and verification
                - Whether scripts or instructions are provided to exercise the artifact (e.g., main scripts, demo scripts, test scripts)
                - Whether input/output examples are provided and produce the expected outputs

                **You MUST exclude** all criteria relating to:
                - Documentation quality or completeness (unless documentation is required *for* functionality verification)
                - Availability, accessibility, or archival status of the artifact
                - Reusability, extensibility, or future use by others
                - Licensing, provenance, setup, or general usability

                **Your task:**
                1. Read the conference guidelines below.
                2. Extract ONLY those criteria that directly relate to the artifact's functionality as defined above.
                3. For each, provide:
                   - `name`: concise description of the functionality aspect to evaluate (e.g., "Successful Execution", "Test Coverage", "Output Verification")
                   - `description`: a brief, specific explanation of what to check or how to check it, based strictly on the guideline text.

                **Output format:**
                Return a JSON object with a `criteria` field, where each item has `name` and `description` fields as described above. Exclude anything not directly related to functionality.

                {format_instructions}

                Conference Guidelines:
                {guidelines}
            """,
            input_variables=["guidelines"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        llm_gpt = OpenAI(temperature=0.0)
        prompt_and_model = guideline_prompt | llm_gpt | parser
        result = prompt_and_model.invoke({"guidelines": self.guidelines})
        logger.info(f"Prompt used for functionality extraction:\n{guideline_prompt.format(guidelines=self.guidelines)}")
        logger.info(f"Criteria extracted: {[c.name for c in result.criteria]}")
        return result.criteria

    def _build_vector_db(self):
        logger.info("Building vector DB for documentation and code files.")
        doc_files = self.artifact.get('documentation_files', [])
        code_files = self.artifact.get('code_files', [])
        relevant_files = doc_files + code_files
        texts = []
        for file in relevant_files:
            if isinstance(file['content'], list):
                texts.append("\n".join(file['content']))
            else:
                texts.append(str(file['content']))
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        all_chunks = []
        for text in texts:
            all_chunks.extend(splitter.split_text(text))
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_texts(all_chunks, embeddings, persist_directory=self.persist_directory)
        db.persist()
        logger.info("Vector DB built and persisted.")
        return db

    def _file_exists(self, filename: str) -> bool:
        all_files = self.artifact.get('repository_structure', [])
        for f in all_files:
            if f['name'].lower() == filename.lower():
                return True
        return False

    def _get_file_content(self, filename: str) -> Optional[str]:
        for file_list in ['documentation_files', 'code_files']:
            for file in self.artifact.get(file_list, []):
                if file['name'].lower() == filename.lower():
                    if isinstance(file['content'], list):
                        return "\n".join(file['content'])
                    return str(file['content'])
        return None

    def _get_keyword_evidence(self) -> Dict:
        """Get keyword-based evidence for functionality evaluation"""
        if not self.keyword_agent:
            return {}

        try:
            keyword_results = self.keyword_agent.evaluate(verbose=False)
            functionality_dim = None

            # Find functionality dimension in keyword results
            for dim in keyword_results.get('dimensions', []):
                if dim['dimension'].lower() == 'functionality':
                    functionality_dim = dim
                    break

            if functionality_dim:
                return {
                    'raw_score': functionality_dim['raw_score'],
                    'weighted_score': functionality_dim['weighted_score'],
                    'keywords_found': functionality_dim['keywords_found'],
                    'overall_score': keyword_results.get('overall_score', 0)
                }
        except Exception as e:
            logger.warning(f"Could not get keyword evidence: {e}")

        return {}

    def _get_kg_evidence(self) -> Dict:
        if not self.kg_agent:
            return {}

        evidence = {}

        # 1. Does the repo have test files?
        evidence['has_tests'] = self.kg_agent.test_files_exist()

        # 2. Does the repo have executable scripts?
        # (e.g., 'main.py', 'run.sh', or scripts in root)
        evidence['has_main_py'] = self.kg_agent.file_exists('main.py')
        evidence['has_run_sh'] = self.kg_agent.file_exists('run.sh')

        # 3. Evidence of verification/validation (test files or sections)
        evidence['test_file_list'] = []
        test_files = self.kg_agent.run_cypher(
            "MATCH (t:Test) RETURN t.name AS name"
        )
        evidence['test_file_list'] = [t['name'] for t in test_files]

        # 4. Presence of code files described by the README (could indicate demo scripts)
        evidence['described_files'] = [
            r['code.name'] for r in self.kg_agent.run_cypher(
                "MATCH (doc:File {name: 'README.md'})-[:DESCRIBES]->(code:File) RETURN code.name"
            )
        ]

        # 5. Presence of output examples (sections containing "output" or "result")
        output_sections = self.kg_agent.run_cypher(
            """
            MATCH (f:File)-[:CONTAINS]->(s:Section)
            WHERE toLower(s.content) CONTAINS 'output'
               OR toLower(s.content) CONTAINS 'result'
            RETURN f.name AS file, s.name AS section, s.content AS content
            """
        )
        evidence['output_sections'] = [
            {"file": s["file"], "section": s["section"], "content": s["content"][:200]} for s in output_sections
        ]

        # 6. Presence of scripts for execution (e.g., files of type 'code' in root)
        code_files = self.kg_agent.run_cypher(
            "MATCH (f:File {type: 'code'}) RETURN f.name AS name"
        )
        evidence['code_files'] = [c['name'] for c in code_files]

        return evidence

    def _build_eval_prompt(self):
        # Get keyword-based evidence
        keyword_evidence = self._get_keyword_evidence()
        kg_evidence = self._get_kg_evidence()

        chain_of_thought_steps = ""
        for criterion in self.criteria:
            chain_of_thought_steps += (
                f"Criterion: {criterion.name}\n"
                f"Description: {criterion.description}\n"
                "Step 1: Identify claims made in the documentation about the artifact's functionality.\n"
                "Step 2: Look for scripts or files (such as main.py, run.sh, test scripts, or evaluation scripts) that support these claims.\n"
                "Step 3: Check whether the necessary files/scripts are present in the repository. If present, inspect their content for relevance and completeness.\n"
                "Step 4: If test results or output examples are claimed, check if these are present.\n"
                "Step 5: Based on all information, rate this aspect from 1-5 and justify your score with evidence from the artifact.\n\n"
            )

        # Include keyword evidence in prompt
        keyword_context = ""
        if keyword_evidence:
            keyword_context = f"""
            KEYWORD-BASED EVIDENCE (Use this to ground your evaluation):
            - Raw functionality score: {keyword_evidence.get('raw_score', 'N/A')}
            - Weighted functionality score: {keyword_evidence.get('weighted_score', 'N/A'):.2f}
            - Keywords found: {', '.join(keyword_evidence.get('keywords_found', []))}
            - Overall artifact score: {keyword_evidence.get('overall_score', 'N/A'):.2f}

            IMPORTANT: Your evaluation should be consistent with this keyword evidence. If functionality keywords (testing, verification, etc.) are abundant, your score should reflect strong functionality evidence. If few functionality keywords are found, explain what's missing.
            """

        kg_context = ""
        if kg_evidence:
            kg_context += "KNOWLEDGE GRAPH EVIDENCE:\n"
            kg_context += f"- Has tests: {kg_evidence.get('has_tests', False)}\n"
            kg_context += f"- Has main.py: {kg_evidence.get('has_main_py', False)}\n"
            kg_context += f"- Has run.sh: {kg_evidence.get('has_run_sh', False)}\n"
            kg_context += f"- Test files: {', '.join(kg_evidence.get('test_file_list', []))}\n"
            kg_context += f"- Code files described by README: {', '.join(kg_evidence.get('described_files', []))}\n"
            kg_context += f"- Code files: {', '.join(kg_evidence.get('code_files', []))}\n"
            if kg_evidence.get('output_sections'):
                kg_context += f"- Output/example/result sections (first 200 chars):\n"
                for out in kg_evidence['output_sections']:
                    kg_context += f"  * {out['file']}::{out['section']}: {out['content']}\n"

            prompt = (
                f"You are an expert artifact evaluator for {self.conference_name}.\n"
                "Evaluate ONLY the **Functionality** of the artifact according to the following criteria.\n"
                f"{keyword_context}\n"
                f"{kg_context}\n"
                # (add your chain-of-thought instructions)
                # ...
            )
        logger.info(f"Functionality evaluation prompt:\n{prompt}")
        return prompt

    def evaluate(self, verbose: bool = True) -> str:
        prompt = self._build_eval_prompt()
        llm = OpenAI(temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.db.as_retriever(search_kwargs={'k': 5}),
            chain_type="stuff"
        )
        logger.info("Running artifact functionality evaluation chain...")
        result = qa_chain({"query": prompt})
        logger.info(f"LLM output:\n{result['result']}")
        if verbose:
            print(result['result'])
        return result['result']

    def get_criteria(self) -> List[FunctionalityCriterion]:
        return self.criteria

    def file_exists(self, filename: str) -> bool:
        return self._file_exists(filename)

    def get_file_content(self, filename: str) -> Optional[str]:
        return self._get_file_content(filename)

# Example usage (commented out)
# from functionality_evaluation_agent import FunctionalityEvaluationAgent
#
# agent = FunctionalityEvaluationAgent(
#     guideline_path="../../data/conference_guideline_texts/processed/13_icse_2025.md",
#     artifact_json_path="C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\ml-image-classifier_analysis.json",
#     conference_name="ICSE 2025"
# )
# functionality_report = agent.evaluate(verbose=True)
