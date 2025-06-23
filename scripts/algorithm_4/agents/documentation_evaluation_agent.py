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
            kg_agent: Optional[object] = None
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

        logger.info(f"Initializing DocumentationEvaluationAgent for {conference_name}")
        self.guidelines = self._load_guidelines()
        self.guideline_db = self._build_guideline_vector_db()
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

    def _build_guideline_vector_db(self):
        logger.info("Building vector DB for conference guidelines.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        guideline_chunks = splitter.split_text(self.guidelines)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_texts(guideline_chunks, embeddings, persist_directory=f"{self.persist_directory}_guidelines")
        db.persist()
        logger.info("Guideline vector DB built and persisted.")
        return db

    def _extract_required_sections(self):
        logger.info("Extracting required README sections using RAG on guideline vector DB.")
        retriever = self.guideline_db.as_retriever(search_kwargs={'k': 6})
        llm = OpenAI(temperature=0.0)
        parser = PydanticOutputParser(pydantic_object=ReadmeSectionsList)
        prompt = PromptTemplate(
            template="""
                You are an expert at extracting required documentation sections from conference guidelines.

                Based only on the retrieved context below, list ONLY the explicit sections/headings that MUST be present in the README or documentation files, ignoring general evaluation axes unless specifically stated as documentation requirements.

                For each section, output its `name` and a brief `description`, based strictly on the retrieved guideline statements. Do NOT include general factors unless the text says these are documentation requirements.

                {format_instructions}

                Retrieved Context:
                {context}

                Question:
                Which documentation/README sections are explicitly required?
            """,
            input_variables=["context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # RAG chain: get relevant context
        query = "What sections/headings are explicitly required in the artifact README or documentation?"
        retrieved = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in retrieved])

        output = llm(prompt.format(context=context))
        result = parser.parse(output)
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

        # Gather evidence
        keyword_context = ""
        if keyword_evidence:
            keyword_context = (
                "KEYWORD-BASED EVIDENCE:\n"
                f"- Raw documentation score: {keyword_evidence.get('raw_score', 'N/A')}\n"
                f"- Weighted documentation score: {keyword_evidence.get('weighted_score', 'N/A'):.2f}\n"
                f"- Keywords found: {', '.join(keyword_evidence.get('keywords_found', []))}\n"
                f"- Overall artifact score: {keyword_evidence.get('overall_score', 'N/A'):.2f}\n"
            )

        kg_evidence = self._get_kg_evidence()
        kg_context = ""
        if kg_evidence:
            kg_context = (
                "KNOWLEDGE GRAPH EVIDENCE:\n"
                f"- Files described by README: {', '.join(kg_evidence.get('described_files', []))}\n"
                f"- README has setup section: {kg_evidence.get('has_setup_section', False)}\n"
            )

        # Chain-of-thought instructions
        section_questions = ""
        for section in doc_sections:
            section_questions += (
                f"Section: '{section.name}'\n"
                f"Description: {section.description}\n"
                "Step-by-step, do the following:\n"
                "  1. Summarize the relevant evidence from the KEYWORD-BASED EVIDENCE and KNOWLEDGE GRAPH EVIDENCE above for this section.\n"
                "  2. Based on this evidence, reason whether the section is present and sufficiently detailed.\n"
                "  3. Assign a score from 1 (poor/missing) to 5 (excellent/complete) for this section, and provide a justification grounded in the evidence.\n"
            )

        prompt = (
            f"You are an expert artifact evaluator for {self.conference_name}.\n"
            "Your task is to evaluate ONLY the **documentation** of the artifact according to these required sections.\n\n"
            f"{keyword_context}\n"
            f"{kg_context}\n"
            "For each required section, follow this chain-of-thought process:\n"
            f"{section_questions}\n"
            "After evaluating all sections, provide an overall documentation score (1-5) and suggestions for improvement.\n"
            "IMPORTANT: Base your evaluation and scores ONLY on the evidence provided above. Do NOT make assumptions or hallucinate information."
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

    def _get_kg_evidence(self) -> Dict:
        if not self.kg_agent:
            return {}
        evidence = {}
        # Example: Which files are described by the README?
        described_files = self.kg_agent.run_cypher(
            "MATCH (doc:File {name: 'README.md'})-[:DESCRIBES]->(code:File) RETURN code.name"
        )
        evidence['described_files'] = [r['code.name'] for r in described_files]
        # Example: Does README have a 'setup' section?
        evidence['has_setup_section'] = self.kg_agent.readme_has_section('setup')
        # Add more as needed...
        return evidence
