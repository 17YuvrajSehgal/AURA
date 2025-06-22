import json
import logging
from typing import List, Optional, Dict

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
        logging.FileHandler("documentation_evaluation_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class ReadmeSection(BaseModel):
    name: str = Field(description="Name of the required README section")
    description: str = Field(description="Brief explanation of what this section should contain")


class ReadmeSectionsList(BaseModel):
    sections: List[ReadmeSection] = Field(description="List of required README sections for the conference")


class DocumentationEvaluationAgent:
    def __init__(
            self,
            guideline_path: str,
            artifact_json_path: str,
            conference_name: str,
            persist_directory: str = "chroma_index",
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

        logger.info(f"Initializing DocumentationEvaluationAgent for {conference_name}")
        self.guidelines = self._load_guidelines()
        self.artifact_docs = self._load_artifact_docs()
        self.sections = self._extract_required_sections()

        # Vector DB for documentation
        self.db = self._build_vector_db()

    def _load_guidelines(self):
        logger.info(f"Loading conference guidelines from: {self.guideline_path}")
        with open(self.guideline_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_artifact_docs(self):
        logger.info(f"Loading artifact documentation from: {self.artifact_json_path}")
        with open(self.artifact_json_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        doc_files = artifact['documentation_files']
        texts = ["\n".join(doc['content']) for doc in doc_files]
        return texts

    def _extract_required_sections(self):
        logger.info("Extracting required README sections using LLM.")
        parser = PydanticOutputParser(pydantic_object=ReadmeSectionsList)
        guideline_prompt = PromptTemplate(
            template=(
                "Based on the following conference guidelines ,a list of required README sections, "
                "return a JSON object with a `sections` field. "
                "For example if it say: It should include a README file detailing the purpose, provenance, setup, and usage of the artifact., Then you should return purpose, provenance, setup, and usage in return"
                "{format_instructions}\n"
                "Conference Guidelines:\n{guidelines}"
            ),
            input_variables=["guidelines"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm_gpt = OpenAI(temperature=0.0)
        prompt_and_model = guideline_prompt | llm_gpt | parser
        result = prompt_and_model.invoke({"guidelines": self.guidelines})
        logger.info(f"Prompt used for section extraction:\n{guideline_prompt.format(guidelines=self.guidelines)}")
        logger.info(f"Sections extracted: {[s.name for s in result.sections]}")
        return result.sections

    def _build_vector_db(self):
        logger.info("Building vector DB for documentation files.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        all_chunks = []
        for text in self.artifact_docs:
            all_chunks.extend(splitter.split_text(text))
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_texts(all_chunks, embeddings, persist_directory=self.persist_directory)
        db.persist()
        logger.info("Vector DB built and persisted.")
        return db

    def _get_keyword_evidence(self) -> Dict:
        """Get keyword-based evidence for documentation evaluation"""
        if not self.keyword_agent:
            return {}

        try:
            keyword_results = self.keyword_agent.evaluate(verbose=False)
            documentation_dim = None

            # Find documentation dimension in keyword results
            for dim in keyword_results.get('dimensions', []):
                if dim['dimension'].lower() == 'documentation':
                    documentation_dim = dim
                    break

            if documentation_dim:
                return {
                    'raw_score': documentation_dim['raw_score'],
                    'weighted_score': documentation_dim['weighted_score'],
                    'keywords_found': documentation_dim['keywords_found'],
                    'overall_score': keyword_results.get('overall_score', 0)
                }
        except Exception as e:
            logger.warning(f"Could not get keyword evidence: {e}")

        return {}

    def _build_eval_prompt(self):
        # Get keyword-based evidence
        keyword_evidence = self._get_keyword_evidence()

        # Focus only on documentation section(s)
        doc_sections = [s for s in self.sections if "documentation" in s.name.lower() or "readme" in s.name.lower()]
        # Fallback: if none detected, use all as a defensive approach
        if not doc_sections:
            doc_sections = self.sections

        section_questions = ""
        for section in doc_sections:
            section_questions += (
                f"- Does the documentation include a '{section.name}' section? "
                f"{section.description} Rate from 1-5 with justification.\n"
            )

        # Include keyword evidence in prompt
        keyword_context = ""
        if keyword_evidence:
            keyword_context = f"""
            KEYWORD-BASED EVIDENCE (Use this to ground your evaluation):
            - Raw documentation score: {keyword_evidence.get('raw_score', 'N/A')}
            - Weighted documentation score: {keyword_evidence.get('weighted_score', 'N/A'):.2f}
            - Keywords found: {', '.join(keyword_evidence.get('keywords_found', []))}
            - Overall artifact score: {keyword_evidence.get('overall_score', 'N/A'):.2f}
            
            IMPORTANT: Your evaluation should be consistent with this keyword evidence. If documentation keywords are abundant, your score should reflect comprehensive documentation. If few documentation keywords are found, explain what's missing.
            """

        prompt = (
            f"You are an expert artifact evaluator for {self.conference_name}.\n"
            "Evaluate ONLY the **documentation** of the artifact according to these required sections:\n\n"
            f"{section_questions}\n"
            f"{keyword_context}\n"
            "Provide a score and justification for each documentation aspect. "
            "Then give an overall documentation score and suggestions for improvement. "
            "Do NOT evaluate other dimensions such as Availability, Functionality, Reusability, or Archival Repository.\n"
            "IMPORTANT: Base your evaluation on actual evidence found in the artifact, not assumptions."
        )
        logger.info(f"Documentation evaluation prompt:\n{prompt}")
        return prompt

    def evaluate(self, verbose: bool = True) -> str:
        prompt = self._build_eval_prompt()
        llm = OpenAI(temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.db.as_retriever(search_kwargs={'k': 5}),
            chain_type="stuff"
        )
        logger.info("Running artifact documentation evaluation chain...")
        result = qa_chain({"query": prompt})
        logger.info(f"LLM output:\n{result['result']}")
        if verbose:
            print(result['result'])
        return result['result']

    def get_sections(self) -> List[ReadmeSection]:
        return self.sections

# from documentation_evaluation_agent import DocumentationEvaluationAgent

# agent = DocumentationEvaluationAgent(
#     guideline_path="../../data/conference_guideline_texts/processed/13_icse_2025.md",
#     artifact_json_path="C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\ml-image-classifier_analysis.json",
#     conference_name="ICSE 2025"
# )
# doc_evaluation = agent.evaluate(verbose=True)
