import json
import logging
from typing import List, Optional

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("usability_evaluation_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class UsabilityCriterion(BaseModel):
    name: str = Field(description="Usability-related aspect or criterion to check")
    description: str = Field(description="Description of what this aspect means or how to check it")

class UsabilityCriteriaList(BaseModel):
    criteria: List[UsabilityCriterion] = Field(description="List of required usability criteria for the conference")

class UsabilityEvaluationAgent:
    def __init__(
            self,
            guideline_path: str,
            artifact_json_path: str,
            conference_name: str = "ICSE 2025",
            persist_directory: str = "usability_chroma_index",
            chunk_size: int = 1024,
            chunk_overlap: int = 100,
            model_name: Optional[str] = None,
    ):
        self.guideline_path = guideline_path
        self.artifact_json_path = artifact_json_path
        self.conference_name = conference_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name

        logger.info(f"Initializing UsabilityEvaluationAgent for {conference_name}")
        self.guidelines = self._load_guidelines()
        self.artifact = self._load_artifact()
        self.criteria = self._extract_usability_criteria()
        self.db = self._build_vector_db()

    def _load_guidelines(self):
        logger.info(f"Loading conference guidelines from: {self.guideline_path}")
        with open(self.guideline_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_artifact(self):
        logger.info(f"Loading artifact JSON from: {self.artifact_json_path}")
        with open(self.artifact_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_usability_criteria(self):
        logger.info("Extracting usability criteria using LLM.")
        parser = PydanticOutputParser(pydantic_object=UsabilityCriteriaList)
        guideline_prompt = PromptTemplate(
            template=(
                "Based on the following conference guidelines, "
                "return a JSON object with a `criteria` field (a list of usability aspects that should be evaluated for the artifact). "
                "These should include things like clarity of installation instructions, existence of setup scripts, clarity of usage instructions, presence of requirements.txt or environment.yml, Dockerfile, or similar, etc. "
                "{format_instructions}\n"
                "Conference Guidelines:\n{guidelines}"
            ),
            input_variables=["guidelines"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm_gpt = OpenAI(temperature=0.0)
        prompt_and_model = guideline_prompt | llm_gpt | parser
        result = prompt_and_model.invoke({"guidelines": self.guidelines})
        logger.info(f"Prompt used for usability extraction:\n{guideline_prompt.format(guidelines=self.guidelines)}")
        logger.info(f"Criteria extracted: {[c.name for c in result.criteria]}")
        return result.criteria

    def _build_vector_db(self):
        logger.info("Building vector DB for documentation and script files.")
        doc_files = self.artifact.get('documentation_files', [])
        code_files = self.artifact.get('code_files', [])
        # Optionally include scripts, requirements, Dockerfile, etc.
        relevant_files = doc_files + code_files
        texts = []
        for file in relevant_files:
            # file['content'] could be a list of lines or a string
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
        # Search both doc_files and code_files
        for file_list in ['documentation_files', 'code_files']:
            for file in self.artifact.get(file_list, []):
                if file['name'].lower() == filename.lower():
                    if isinstance(file['content'], list):
                        return "\n".join(file['content'])
                    return str(file['content'])
        return None

    def _build_eval_prompt(self):
        # Chain-of-thought: For each criterion, check for related file references and read/verify file if mentioned
        chain_of_thought_steps = ""
        for criterion in self.criteria:
            chain_of_thought_steps += (
                f"Criterion: {criterion.name}\n"
                f"Description: {criterion.description}\n"
                "Step 1: Check if documentation or instructions reference any files/scripts (e.g., requirements.txt, Dockerfile, setup.py).\n"
                "Step 2: If referenced, check if the file is present in the artifact. If present, inspect its contents for completeness.\n"
                "Step 3: Based on all information, rate this aspect from 1-5 and justify your score with evidence from the artifact.\n\n"
            )

        prompt = (
            f"You are an expert artifact evaluator for {self.conference_name}.\n"
            f"Evaluate ONLY the **Usability** of the artifact according to the following criteria.\n"
            "For each, follow the chain-of-thought process:\n\n"
            f"{chain_of_thought_steps}\n"
            "Do NOT evaluate dimensions such as Documentation, Availability, Functionality, or Reusability.\n"
            "At the end, provide a detailed usability score and suggestions for improvement."
        )
        logger.info(f"Usability evaluation prompt:\n{prompt}")
        return prompt

    def evaluate(self, verbose: bool = True) -> str:
        prompt = self._build_eval_prompt()
        llm = OpenAI(temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.db.as_retriever(search_kwargs={'k': 5}),
            chain_type="stuff"
        )
        logger.info("Running artifact usability evaluation chain...")
        result = qa_chain({"query": prompt})
        logger.info(f"LLM output:\n{result['result']}")
        if verbose:
            print(result['result'])
        return result['result']

    def get_criteria(self) -> List[UsabilityCriterion]:
        return self.criteria

    # Optionally expose file existence/content utilities for downstream use
    def file_exists(self, filename: str) -> bool:
        return self._file_exists(filename)

    def get_file_content(self, filename: str) -> Optional[str]:
        return self._get_file_content(filename)


#from usability_evaluation_agent import UsabilityEvaluationAgent

# agent = UsabilityEvaluationAgent(
#     guideline_path="../../data/conference_guideline_texts/processed/13_icse_2025.md",
#     artifact_json_path="C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\ml-image-classifier_analysis.json",
#     conference_name="ICSE 2025"
# )
# usability_report = agent.evaluate(verbose=True)
