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
    criteria: List[FunctionalityCriterion] = Field(description="List of required functionality criteria for the conference")


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
    ):
        self.guideline_path = guideline_path
        self.artifact_json_path = artifact_json_path
        self.conference_name = conference_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name

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
            template=(
                "Based on the following conference guidelines, "
                "return a JSON object with a `criteria` field (a list of functionality aspects that should be evaluated for the artifact). "
                "These should focus on whether the artifact does what it claims, can be executed as described, produces the expected outputs, and includes relevant scripts or test results. "
                "{format_instructions}\n"
                "Conference Guidelines:\n{guidelines}"
            ),
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

    def _build_eval_prompt(self):
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

        prompt = (
            f"You are an expert artifact evaluator for {self.conference_name}.\n"
            "Evaluate ONLY the **Functionality** of the artifact according to the following criteria.\n"
            "For each, follow the chain-of-thought process:\n\n"
            f"{chain_of_thought_steps}\n"
            "Do NOT evaluate Documentation, Usability, Accessibility, or Reusability.\n"
            "At the end, provide a detailed functionality score and suggestions for improvement."
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


# from functionality_evaluation_agent import FunctionalityEvaluationAgent
#
# agent = FunctionalityEvaluationAgent(
#     guideline_path="../../data/conference_guideline_texts/processed/13_icse_2025.md",
#     artifact_json_path="C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\ml-image-classifier_analysis.json",
#     conference_name="ICSE 2025"
# )
# functionality_report = agent.evaluate(verbose=True)
