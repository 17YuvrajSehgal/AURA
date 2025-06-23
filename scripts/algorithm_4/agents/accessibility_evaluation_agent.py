import json
import logging
import re
from typing import List, Optional, Dict

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.output_parsers import OutputFixingParser
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
        logging.FileHandler("accessibility_evaluation_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class AccessibilityScore(BaseModel):
    criterion: str = Field(description="Name of the accessibility criterion")
    score: int = Field(ge=1, le=5, description="Score assigned to the criterion")
    justification: str = Field(description="Justification for the score")


class AccessibilityEvaluationResult(BaseModel):
    overall_score: int = Field(ge=1, le=5)
    criterion_scores: List[AccessibilityScore]
    suggestions: Optional[str] = None


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
            template="""
            You are an expert at extracting *only* the accessibility-related criteria for evaluating research software artifacts, based strictly on the provided conference guidelines.

            Accessibility-related criteria refer ONLY to aspects that affect whether and how others can access, obtain, install, or use the artifact. This includes public availability (e.g., in an open repository), clarity of whether data and code are downloadable, and any restrictions on access. Ignore general factors such as documentation quality, usability, functionality, or reusability unless they are specifically described as affecting access to the artifact.

            **Your task:**
            1. Read the conference guidelines below.
            2. Extract ONLY those criteria that pertain directly to accessibility or availability (such as repository access, dependency information, data accessibility, and license openness as it relates to access).
            3. Do NOT extract general evaluation factors (like 'Functionality', 'Reusability', or 'Documentation') unless they are explicitly required as accessibility/availability aspects.
            4. For each extracted criterion, provide:
               - `name`: the accessibility aspect or criterion to check (e.g., "Public Repository Availability", "Installability", "Dependency Clarity")
               - `description`: a brief, specific explanation of what to look for or how to check it, based strictly on the guideline text.

            **Output format:**  
            Return a JSON object with a `criteria` field, where each item has `name` and `description` fields as described above. Only include criteria that are accessibility-related, and exclude anything else.

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
        if not self.keyword_agent:
            return {}

        try:
            keyword_results = self.keyword_agent.evaluate(verbose=False)
            for dim in keyword_results.get('dimensions', []):
                if dim['dimension'].lower() == 'accessibility':
                    return {
                        'raw_score': dim['raw_score'],
                        'weighted_score': dim['weighted_score'],
                        'keywords_found': dim['keywords_found'],
                        'overall_score': keyword_results.get('overall_score', 0)
                    }
        except Exception as e:
            logger.warning(f"Could not get keyword evidence: {e}")

        return {}

    def _build_eval_prompt(self):
        keyword_evidence = self._get_keyword_evidence()

        parser = PydanticOutputParser(pydantic_object=AccessibilityEvaluationResult)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=OpenAI(temperature=0))

        format_instructions = parser.get_format_instructions()
        criteria_description_block = "\n".join([
            f"- **{c.name}**: {c.description}" for c in self.criteria
        ])

        prompt = f"""
        You are an expert artifact evaluator for {self.conference_name}.
        Evaluate ONLY the **Accessibility/Availability** of the artifact.

        Accessibility Criteria:
        {criteria_description_block}

        For each criterion:
        1. Look for explicit mentions of artifact availability (e.g., DOI, Zenodo link, public repository).
        2. If dependencies are referenced, confirm those files exist and inspect their contents.
        3. Assign a score from 1–5 with justification.

        Output must follow this format:
        Each `criterion_scores` entry MUST contain:
        - `criterion`
        - `score` (1–5)
        - `justification` (Required)

        {format_instructions}

        IMPORTANT:
        - Do NOT evaluate other dimensions such as Usability, Functionality, etc.
        - Base your evaluation on actual evidence found in the artifact, not assumptions.
        - Return JSON only. No markdown or explanations outside the object.

        {'KEYWORD-BASED EVIDENCE:' + json.dumps(keyword_evidence, indent=2) if keyword_evidence else ''}
        """
        return prompt.strip(), fixing_parser

    import re

    def evaluate(self, verbose: bool = True) -> AccessibilityEvaluationResult | dict[str, str]:
        prompt, parser = self._build_eval_prompt()

        llm = OpenAI(temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.db.as_retriever(search_kwargs={'k': 5}),
            chain_type="stuff"
        )

        logger.info("Running artifact accessibility evaluation chain...")
        result = qa_chain({"query": prompt})
        raw_output = result.get('result', '')
        logger.info(f"LLM output:\n{raw_output}")

        if verbose:
            print(raw_output)

        try:
            # Attempt safe JSON recovery
            cleaned = raw_output.strip()

            # Heuristic: If string is cut off mid-field, try to trim to last full object
            if cleaned.count("{") > cleaned.count("}"):
                cleaned = re.sub(r',\s*{[^{}]*$', '', cleaned) + "]}"  # close the list and JSON

            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            parsed_json = json.loads(cleaned)

            # Ensure justification field is present
            for score in parsed_json.get("criterion_scores", []):
                if "justification" not in score:
                    score["justification"] = "Justification was missing from the LLM output."

            parsed_result = AccessibilityEvaluationResult(**parsed_json)
            return parsed_result

        except Exception as e:
            logger.warning(f"Could not parse structured accessibility result: {e}")
            return {"error": "Failed to parse accessibility result", "raw_output": raw_output}

    def get_criteria(self) -> List[AccessibilityCriterion]:
        return self.criteria

    def file_exists(self, filename: str) -> bool:
        return self._file_exists(filename)

    def get_file_content(self, filename: str) -> Optional[str]:
        return self._get_file_content(filename)
