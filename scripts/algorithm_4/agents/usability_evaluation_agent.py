import json
import logging
import re
import os
from typing import Dict
from typing import List, Optional

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
        logging.FileHandler("usability_evaluation_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class UsabilityScore(BaseModel):
    criterion: str = Field(description="Name of the usability criterion")
    score: int = Field(ge=1, le=5, description="Score from 1 (poor) to 5 (excellent)")
    justification: str = Field(description="Justification for the score")


class UsabilityEvaluationResult(BaseModel):
    overall_score: int = Field(ge=1, le=5)
    criterion_scores: List[UsabilityScore]
    suggestions: Optional[str] = None

class UsabilityCriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=5)
    justification: str

class UsabilityEvaluationResult(BaseModel):
    overall_score: int = Field(ge=1, le=5)
    criterion_scores: List[UsabilityCriterionScore]
    suggestions: Optional[str] = None


class UsabilityCriterion(BaseModel):
    name: str = Field(description="Usability-related aspect or criterion to check")
    description: str = Field(description="Description of what this aspect means or how to check it")


class UsabilityCriteriaList(BaseModel):
    criteria: List[UsabilityCriterion] = Field(description="List of required usability criteria for the conference")


def _regex_fallback_parse(raw_output: str) -> dict:
    try:
        # Attempt to extract JSON block from malformed string
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            partial_json = json_match.group(0)
            try:
                return json.loads(partial_json)
            except json.JSONDecodeError as e:
                logger.warning(f"Partial JSON found but still invalid: {e}")

        # Fallback to extract individual scores
        criterion_blocks = re.findall(
            r'{"criterion"\s*:\s*"(.+?)",\s*"score"\s*:\s*(\d),\s*"justification"\s*:\s*"(.*?)"}',
            raw_output,
            re.DOTALL
        )

        criterion_scores = []
        for crit, score, justification in criterion_blocks:
            criterion_scores.append({
                "criterion": crit.strip(),
                "score": int(score),
                "justification": justification.strip()
            })

        if not criterion_scores:
            raise ValueError("No criterion blocks extracted")

        # Infer overall score
        overall_score = round(sum(c["score"] for c in criterion_scores) / len(criterion_scores))

        suggestions_match = re.search(
            r'"suggestions"\s*:\s*"(.+?)"', raw_output, re.DOTALL
        )
        suggestions = suggestions_match.group(1).strip() if suggestions_match else None

        return {
            "overall_score": overall_score,
            "criterion_scores": criterion_scores,
            "suggestions": suggestions
        }

    except Exception as e:
        logger.warning(f"Regex fallback parsing failed: {e}")
        return {"error": f"Regex fallback failed: {e}", "raw_output": raw_output}




class UsabilityEvaluationAgent:
    def __init__(
            self,
            guideline_path: str,
            artifact_json_path: str,
            conference_name: str = "ICSE 2025",
            persist_directory: str = "algo_outputs/indexes/usability_chroma_index",
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

        # Ensure indexes directory exists
        os.makedirs(os.path.dirname(self.persist_directory), exist_ok=True)

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

    def _get_keyword_evidence(self) -> Dict:
        """Get keyword-based evidence for usability evaluation"""
        if not self.keyword_agent:
            return {}

        try:
            keyword_results = self.keyword_agent.evaluate(verbose=False)
            usability_dim = None

            # Find usability dimension in keyword results
            for dim in keyword_results.get('dimensions', []):
                if dim['dimension'].lower() == 'usability':
                    usability_dim = dim
                    break

            if usability_dim:
                return {
                    'raw_score': usability_dim['raw_score'],
                    'weighted_score': usability_dim['weighted_score'],
                    'keywords_found': usability_dim['keywords_found'],
                    'overall_score': keyword_results.get('overall_score', 0)
                }
        except Exception as e:
            logger.warning(f"Could not get keyword evidence: {e}")

        return {}

    def _build_eval_prompt(self):
        # Get keyword-based evidence
        keyword_evidence = self._get_keyword_evidence()

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

        # Include keyword evidence in prompt
        keyword_context = ""
        if keyword_evidence:
            keyword_context = f"""
KEYWORD-BASED EVIDENCE (Use this to ground your evaluation):
- Raw usability score: {keyword_evidence.get('raw_score', 'N/A')}
- Weighted usability score: {keyword_evidence.get('weighted_score', 'N/A'):.2f}
- Keywords found: {', '.join(keyword_evidence.get('keywords_found', []))}
- Overall artifact score: {keyword_evidence.get('overall_score', 'N/A'):.2f}

IMPORTANT: Your evaluation should be consistent with this keyword evidence. If usability keywords (installation, setup, demo, etc.) are abundant, your score should reflect strong usability features. If few usability keywords are found, explain what's missing.
"""

        prompt = (
            f"You are an expert artifact evaluator for {self.conference_name}.\n"
            f"Evaluate ONLY the **Usability** of the artifact according to the following criteria.\n"
            "For each, follow the chain-of-thought process:\n\n"
            f"{chain_of_thought_steps}\n"
            f"{keyword_context}\n"
            "Do NOT evaluate dimensions such as Documentation, Availability, Functionality, Reusability, or Archival Repository.\n\n"
            "Only evaluate the following usability-specific criteria:\n"
            + "\n".join(f"- {c.name}" for c in self.criteria) +
            "\n\nReturn a valid JSON object strictly in this format:\n"
            "{\n"
            '  "overall_score": 4,\n'
            '  "criterion_scores": [\n'
            "    {\n"
            '      "criterion": "Installation Instructions",\n'
            '      "score": 5,\n'
            '      "justification": "The README contains clear setup steps and all dependencies are listed."\n'
            "    }\n"
            "  ],\n"
            '  "suggestions": "Consider adding a Dockerfile for easier deployment."\n'
            "}\n\n"
            "STRICT RULES:\n"
            "- DO NOT include extra criteria or unrelated dimensions like Availability or Documentation.\n"
            "- Only return valid JSON, no explanation or markdown.\n"
            "- If unsure about a score, provide a best-effort estimate with justification.\n"

            "IMPORTANT:\n"
            "- Only return valid JSON, without any commentary, explanation, or markdown.\n"
            "- Base your evaluation on actual evidence found in the artifact.\n"
            "- Justify each score.\n"
            "- Do not include dimensions like Documentation, Functionality, or Availability."
        )

        logger.info(f"Usability evaluation prompt:\n{prompt}")
        return prompt
    

    def evaluate(self, verbose: bool = True) -> UsabilityEvaluationResult | dict:
        prompt = self._build_eval_prompt()
        llm = OpenAI(temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.db.as_retriever(search_kwargs={'k': 5}),
            chain_type="stuff"
        )
        logger.info("Running artifact usability evaluation chain...")
        result = qa_chain({"query": prompt})
        raw_output = result.get("result", "")
        logger.info(f"LLM output:\n{raw_output}")

        if verbose:
            print(raw_output)

        # Parse JSON with error recovery
        try:
            cleaned = raw_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            parsed_json = json.loads(cleaned)

            for crit in parsed_json.get("criterion_scores", []):
                if "justification" not in crit:
                    crit["justification"] = "No justification provided."

            return UsabilityEvaluationResult(**parsed_json)
        except Exception as e:
            logger.warning(f"Failed to parse usability result: {e}")
            # Fallback to regex parsing
            fallback_result = _regex_fallback_parse(raw_output)
            if "error" not in fallback_result:
                try:
                    return UsabilityEvaluationResult(**fallback_result)
                except Exception as fallback_error:
                    logger.warning(f"Fallback structured parse failed: {fallback_error}")
            return fallback_result

    def get_criteria(self) -> List[UsabilityCriterion]:
        return self.criteria

    # Optionally expose file existence/content utilities for downstream use
    def file_exists(self, filename: str) -> bool:
        return self._file_exists(filename)

    def get_file_content(self, filename: str) -> Optional[str]:
        return self._get_file_content(filename)



# Example usage (commented out)
# from usability_evaluation_agent import UsabilityEvaluationAgent

# agent = UsabilityEvaluationAgent(
#     guideline_path="../../data/conference_guideline_texts/processed/13_icse_2025.md",
#     artifact_json_path="C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\ml-image-classifier_analysis.json",
#     conference_name="ICSE 2025"
# )
# usability_report = agent.evaluate(verbose=True)
