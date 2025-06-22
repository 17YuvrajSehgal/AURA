import json
import logging
from typing import List, Optional, Dict

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("accessibility_evaluation_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class AccessibilityCriterion(BaseModel):
    name: str = Field(description="Accessibility aspect to check")
    description: str = Field(description="Description of what this aspect means or how to check it")


class AccessibilityCriteriaList(BaseModel):
    criteria: List[AccessibilityCriterion] = Field(
        description="List of required accessibility criteria for the conference")


class AccessibilityEvaluationAgent:
    def __init__(
            self,
            guideline_path: str,
            artifact_json_path: str,
            conference_name: str = "ICSE 2025",
            persist_directory: str = "accessibility_chroma_index",
            chunk_size: int = 1024,
            chunk_overlap: int = 100,
            model_name: Optional[str] = None,
            keyword_agent: Optional[object] = None,
    ):
        self.guideline_path = guideline_path
        self.artifact_json_path = artifact_json_path
        self.conference_name = conference_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.keyword_agent = keyword_agent

        logger.info(f"Initializing AccessibilityEvaluationAgent for {conference_name}")
        self.guidelines = self._load_guidelines()
        self.artifact = self._load_artifact()
        self.criteria = self._extract_accessibility_criteria()
        self.db = self._build_vector_db()

    def _load_guidelines(self):
        logger.info(f"Loading conference guidelines from: {self.guideline_path}")
        with open(self.guideline_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_artifact(self):
        logger.info(f"Loading artifact JSON from: {self.artifact_json_path}")
        with open(self.artifact_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_accessibility_criteria(self):
        logger.info("Extracting accessibility criteria using LLM.")
        parser = PydanticOutputParser(pydantic_object=AccessibilityCriteriaList)
        guideline_prompt = PromptTemplate(
            template=(
                "Based on the following conference guidelines, "
                "return a JSON object with a `criteria` field (a list of accessibility aspects that should be evaluated for the artifact). "
                "These should include aspects like whether the artifact is available in a public repository, the clarity of dependency listings, and the installability of the artifact. "
                "{format_instructions}\n"
                "Conference Guidelines:\n{guidelines}"
            ),
            input_variables=["guidelines"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm_gpt = OpenAI(temperature=0.0)
        prompt_and_model = guideline_prompt | llm_gpt | parser
        result = prompt_and_model.invoke({"guidelines": self.guidelines})
        logger.info(f"Prompt used for accessibility extraction:\n{guideline_prompt.format(guidelines=self.guidelines)}")
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
        """Get keyword-based evidence for accessibility evaluation"""
        if not self.keyword_agent:
            return {}

        try:
            keyword_results = self.keyword_agent.evaluate(verbose=False)
            accessibility_dim = None

            # Find accessibility dimension in keyword results
            for dim in keyword_results.get('dimensions', []):
                if dim['dimension'].lower() == 'accessibility':
                    accessibility_dim = dim
                    break

            if accessibility_dim:
                return {
                    'raw_score': accessibility_dim['raw_score'],
                    'weighted_score': accessibility_dim['weighted_score'],
                    'keywords_found': accessibility_dim['keywords_found'],
                    'overall_score': keyword_results.get('overall_score', 0)
                }
        except Exception as e:
            logger.warning(f"Could not get keyword evidence: {e}")

        return {}

    def _build_eval_prompt(self):
        # Get keyword-based evidence
        keyword_evidence = self._get_keyword_evidence()

        # Build explicit, chain-of-thought prompt for accessibility
        chain_of_thought_steps = ""
        for criterion in self.criteria:
            chain_of_thought_steps += (
                f"Criterion: {criterion.name}\n"
                f"Description: {criterion.description}\n"
                "Step 1: Look for explicit mentions of artifact availability (e.g., DOI, Zenodo link, public repository).\n"
                "Step 2: Check for dependency files (requirements.txt, environment.yml, setup.py) and their clarity/completeness.\n"
                "Step 3: If dependencies are referenced, confirm those files exist and inspect their contents.\n"
                "Step 4: Assess whether the artifact appears straightforward to install given its documentation and dependencies.\n"
                "Step 5: Based on all information, rate this aspect from 1-5 and justify your score with evidence from the artifact.\n\n"
            )

        # Include keyword evidence in prompt
        keyword_context = ""
        if keyword_evidence:
            keyword_context = f"""
            KEYWORD-BASED EVIDENCE (Use this to ground your evaluation):
            - Raw accessibility score: {keyword_evidence.get('raw_score', 'N/A')}
            - Weighted accessibility score: {keyword_evidence.get('weighted_score', 'N/A'):.2f}
            - Keywords found: {', '.join(keyword_evidence.get('keywords_found', []))}
            - Overall artifact score: {keyword_evidence.get('overall_score', 'N/A'):.2f}
            
            IMPORTANT: Your evaluation should be consistent with this keyword evidence. If keywords indicate strong accessibility features, your score should reflect that. If few accessibility keywords are found, explain what's missing.
            """

        prompt = (
            f"You are an expert artifact evaluator for {self.conference_name}.\n"
            "Evaluate ONLY the **Accessibility** of the artifact according to the following criteria.\n"
            "For each, follow the chain-of-thought process:\n\n"
            f"{chain_of_thought_steps}\n"
            f"{keyword_context}\n"
            "Do NOT evaluate other dimensions such as Documentation, Usability, Functionality, or Reusability.\n"
            "At the end, provide a detailed accessibility score and suggestions for improvement.\n"
            "IMPORTANT: Base your evaluation on actual evidence found in the artifact, not assumptions."
        )
        logger.info(f"Accessibility evaluation prompt:\n{prompt}")
        return prompt

    def evaluate(self, verbose: bool = True) -> str:
        prompt = self._build_eval_prompt()
        llm = OpenAI(temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.db.as_retriever(search_kwargs={'k': 5}),
            chain_type="stuff"
        )
        logger.info("Running artifact accessibility evaluation chain...")
        result = qa_chain({"query": prompt})
        logger.info(f"LLM output:\n{result['result']}")
        if verbose:
            print(result['result'])
        return result['result']

    def get_criteria(self) -> List[AccessibilityCriterion]:
        return self.criteria

    # Optional: expose file existence/content utilities for downstream use
    def file_exists(self, filename: str) -> bool:
        return self._file_exists(filename)

    def get_file_content(self, filename: str) -> Optional[str]:
        return self._get_file_content(filename)

# from accessibility_evaluation_agent import AccessibilityEvaluationAgent

# agent = AccessibilityEvaluationAgent(
#     guideline_path="../../data/conference_guideline_texts/processed/13_icse_2025.md",
#     artifact_json_path="C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\ml-image-classifier_analysis.json",
#     conference_name="ICSE 2025"
# )
# accessibility_report = agent.evaluate(verbose=True)
