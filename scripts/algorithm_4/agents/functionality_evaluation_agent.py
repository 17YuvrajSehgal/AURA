import json
import logging
from typing import List, Optional, Dict

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory

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


# ---------- Utility Summarizers ----------
def summarize_list(lst, label, max_items=7):
    if not lst:
        return f"- {label}: None\n"
    uniq = list(dict.fromkeys(lst))
    s = f"- {label}: {', '.join(uniq[:max_items])}"
    if len(uniq) > max_items:
        s += f", ...and {len(uniq) - max_items} more"
    return s + "\n"


def summarize_output_sections(sections, max_sections=3, max_len=120):
    if not sections:
        return "- Output/example/result sections: None\n"
    lines = []
    for s in sections[:max_sections]:
        summary = s['content'][:max_len].replace("\n", " ")
        lines.append(f"  * {s['file']}::{s['section']}: {summary}")
    if len(sections) > max_sections:
        lines.append(f"  ...and {len(sections) - max_sections} more.")
    return "- Output/example/result sections (sample):\n" + "\n".join(lines) + "\n"


def safe_str(obj):
    # Defend against None and truncate
    return str(obj)[:300] if obj is not None else ""

class FunctionalityCriterionScore(BaseModel):
    criterion: str
    score: int
    justification: str

class FunctionalityEvaluationResult(BaseModel):
    overall_score: float
    criterion_scores: List[FunctionalityCriterionScore]
    suggestions: Optional[str] = None



# --------- Core Agent Classes ----------
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
        if not self.keyword_agent:
            return {}
        try:
            keyword_results = self.keyword_agent.evaluate(verbose=False)
            for dim in keyword_results.get('dimensions', []):
                if dim['dimension'].lower() == 'functionality':
                    return {
                        'raw_score': dim['raw_score'],
                        'weighted_score': dim['weighted_score'],
                        'keywords_found': dim['keywords_found'],
                        'overall_score': keyword_results.get('overall_score', 0)
                    }
        except Exception as e:
            logger.warning(f"Could not get keyword evidence: {e}")
        return {}

    def _get_kg_evidence(self) -> Dict:
        if not self.kg_agent:
            return {}
        evidence = {}
        # 1. Has tests/scripts
        evidence['has_tests'] = self.kg_agent.test_files_exist()
        evidence['has_main_py'] = self.kg_agent.file_exists('main.py')
        evidence['has_run_sh'] = self.kg_agent.file_exists('run.sh')
        # 2. Deduplicate/shorten lists
        test_files = self.kg_agent.run_cypher("MATCH (t:Test) RETURN t.name AS name")
        evidence['test_file_list'] = list(dict.fromkeys([t['name'] for t in test_files]))
        described_files = self.kg_agent.run_cypher(
            "MATCH (doc:File {name: 'README.md'})-[:DESCRIBES]->(code:File) RETURN code.name")
        evidence['described_files'] = list(dict.fromkeys([r['code.name'] for r in described_files]))
        output_sections = self.kg_agent.run_cypher(
            "MATCH (f:File)-[:CONTAINS]->(s:Section) WHERE toLower(s.content) CONTAINS 'output' OR toLower(s.content) CONTAINS 'result' RETURN f.name AS file, s.name AS section, s.content AS content")
        # Only first N and truncated to K chars
        evidence['output_sections'] = [
            {"file": s["file"], "section": s["section"], "content": s["content"][:120]}
            for s in output_sections[:5]
        ]
        code_files = self.kg_agent.run_cypher("MATCH (f:File {type: 'code'}) RETURN f.name AS name")
        evidence['code_files'] = list(dict.fromkeys([c['name'] for c in code_files]))
        return evidence

    def _build_eval_prompt(self):
        keyword_evidence = self._get_keyword_evidence()
        kg_evidence = self._get_kg_evidence()

        # ----- Summarize KG Evidence -----
        kg_context = "KNOWLEDGE GRAPH EVIDENCE:\n"
        kg_context += f"- Has tests: {safe_str(kg_evidence.get('has_tests'))}\n"
        kg_context += f"- Has main.py: {safe_str(kg_evidence.get('has_main_py'))}\n"
        kg_context += f"- Has run.sh: {safe_str(kg_evidence.get('has_run_sh'))}\n"
        kg_context += summarize_list(kg_evidence.get('test_file_list'), "Test files")
        kg_context += summarize_list(kg_evidence.get('described_files'), "Code files described by README")
        kg_context += summarize_list(kg_evidence.get('code_files'), "Code files")
        kg_context += summarize_output_sections(kg_evidence.get('output_sections'))

        # Keyword evidence
        keyword_context = ""
        if keyword_evidence:
            keyword_context = (
                "KEYWORD-BASED EVIDENCE (summarized):\n"
                f"- Raw functionality score: {keyword_evidence.get('raw_score', 'N/A')}\n"
                f"- Weighted functionality score: {keyword_evidence.get('weighted_score', 'N/A')}\n"
                f"- Keywords found: {', '.join(keyword_evidence.get('keywords_found', []))}\n"
                f"- Overall artifact score: {keyword_evidence.get('overall_score', 'N/A')}\n"
            )

        # ----- Assemble Prompt -----
        evidence_chunks = []
        for criterion in self.criteria:
            retrieval_query = f"Evidence for {criterion.name}: {criterion.description}"
            docs = self.db.as_retriever(search_kwargs={'k': 3}).get_relevant_documents(retrieval_query)
            content = "\n".join([doc.page_content[:300] for doc in docs])
            evidence_chunks.append(
                f"Criterion: {criterion.name}\nDescription: {criterion.description}\nEvidence:\n{content}\n"
                "Step-by-step: ..."
            )
        prompt = (
                f"You are an expert artifact evaluator for {self.conference_name}.\n"
                "Evaluate ONLY the **Functionality** of the artifact according to the following criteria.\n"
                + "\n".join(evidence_chunks)
                + "At the end, provide a functionality score and suggestions."
        )

        # Warn if too long
        if len(prompt) > 12000:
            logger.warning("Prompt is over 12,000 characters. Evidence has been truncated.")
        return prompt

    def evaluate(self, verbose=True):
        from langchain.memory import ConversationBufferMemory

        results = []
        criterion_scores = []
        memory = ConversationBufferMemory(return_messages=True)
        llm = OpenAI(temperature=0.2)

        for criterion in self.criteria:
            retrieval_query = f"Show all evidence the artifact supports: {criterion.name}. {criterion.description}"
            docs = self.db.as_retriever(search_kwargs={'k': 3}).get_relevant_documents(retrieval_query)
            content = "\n".join([doc.page_content[:250] for doc in docs]) or "No evidence found."

            kg_ev = self._get_kg_evidence()
            keyword_ev = self._get_keyword_evidence()
            kg_str = summarize_list(kg_ev.get('test_file_list'), "Test files") + summarize_output_sections(
                kg_ev.get('output_sections')) if "test" in criterion.name.lower() else ""
            keyword_str = (
                f"- Raw functionality score: {keyword_ev.get('raw_score', 'N/A')}\n"
                f"- Weighted functionality score: {keyword_ev.get('weighted_score', 'N/A')}\n"
                f"- Keywords found: {', '.join(keyword_ev.get('keywords_found', []))}\n"
            ) if keyword_ev else ""

            prompt = (
                f"You are evaluating the **Functionality** of a research artifact for the criterion:\n"
                f"Criterion: {criterion.name}\nDescription: {criterion.description}\n\n"
                f"Evidence from artifact (retrieved):\n{content}\n"
                f"Knowledge Graph evidence:\n{kg_str or 'None'}\n"
                f"Keyword evidence:\n{keyword_str or 'None'}\n"
                f"Step-by-step, does the artifact satisfy this criterion? Give a 1-5 score and justification. "
                f"Respond with a valid JSON like: {{\"criterion\": \"...\", \"score\": <int>, \"justification\": \"...\"}}"
            )

            response = llm.invoke(prompt)
            memory.save_context({"input": prompt}, {"output": response})

            try:
                parsed = FunctionalityCriterionScore.parse_raw(response)
                criterion_scores.append(parsed)
                results.append(parsed.json())
            except Exception as e:
                logger.warning(f"Failed to parse response for {criterion.name}: {e}")
                results.append(response)

            if verbose:
                print(f"\n== {criterion.name} ==\n{response}\n")

        # Final score summary prompt
        # Final score summary prompt
        full_justifications = "\n\n".join(
            [msg.content for msg in memory.chat_memory.messages if hasattr(msg, "content")])
        final_prompt = (
            f"You are an expert evaluator. Based on these evaluations, assign one final score for functionality.\n"
            f"Use this JSON format: {{\"overall_score\": <1-5>, \"justification\": \"...\", \"suggestions\": \"...\"}}\n"
            f"{full_justifications}"
        )

        final_response = llm.invoke(final_prompt)
        if verbose:
            print(f"\n== FINAL FUNCTIONALITY SCORE ==\n{final_response}\n")

        try:
            # Primary attempt: strict JSON parsing
            final_data = json.loads(final_response)
            return FunctionalityEvaluationResult(
                overall_score=final_data["overall_score"],
                criterion_scores=criterion_scores,
                suggestions=final_data.get("suggestions", "")
            )
        except Exception as json_err:
            logger.warning(f"Failed to parse final summary: {json_err}")

            # Fallback 1: regex-based parsing
            import re
            try:
                score_match = re.search(r"(?i)final\s+functionality\s+score\s*[:\-]?\s*(\d)", final_response)
                justification_match = re.search(r"(?i)justification\s*[:\-]?\s*(.+?)(?:\n|$)", final_response)
                suggestions_match = re.search(r"(?i)suggestions\s*[:\-]?\s*(.+?)(?:\n|$)", final_response)

                overall_score = int(score_match.group(1)) if score_match else None
                justification = justification_match.group(
                    1).strip() if justification_match else "No justification found."
                suggestions = suggestions_match.group(1).strip() if suggestions_match else "No suggestions found."

                return FunctionalityEvaluationResult(
                    overall_score=overall_score or 0,
                    criterion_scores=criterion_scores,
                    suggestions=suggestions
                )
            except Exception as regex_err:
                logger.warning(f"Fallback regex parse also failed: {regex_err}")

                return {
                    "error": "Parsing final functionality summary failed.",
                    "raw_output": final_response
                }

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
