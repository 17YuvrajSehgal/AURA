# === Update: Full Repository Evaluation ===
# Instead of just evaluating the README file, we now evaluate the entire artifact using a precomputed analysis JSON.

import json
import logging
import os
import re
from datetime import datetime
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from py2neo import Graph

load_dotenv()

LOG_DIR = "../../algo_outputs/logs/artifact_eval_logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(temperature=0.2, model="gpt-4")
graph_db = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))


class EvalState(TypedDict):
    conf_code: str
    analysis_path: str
    context: str
    sections: List[dict]
    dimensions: List[dict]
    current_dimension: dict
    dimension_results: List[dict]
    final_score: float


DIMENSION_PROMPTS = {
    "reproducibility": (
        "Evaluate whether the artifact enables users to reproduce the results. Check for availability of scripts, data, configurations, and any reproduction instructions.\n"
        "Do NOT assume anything is reproducible unless the entire process (e.g., data, code, environment, execution steps) is explicitly present in the artifact."
    ),
    "documentation": (
        "Evaluate the completeness and clarity of documentation. Does it contain setup instructions, usage examples, and details required to understand and operate the artifact?\n"
        "Do NOT speculate about documentation quality—evaluate only what is clearly included in files like README, manuals, or docs."
    ),
    "accessibility": (
        "Assess whether the artifact is publicly accessible. Is the data/code hosted on a reliable repository? Are there any restrictions or licensing issues?\n"
        "Do NOT infer accessibility based on path names or placeholders. Confirm with actual repository links or hosting evidence."
    ),
    "usability": (
        "Examine how easily a user can install, run, and understand the artifact. Are there helpful tools, scripts, or installation packages that make it easy to use?\n"
        "Do NOT assume ease-of-use unless clear setup or usage instructions, scripts, or config templates are explicitly included."
    ),
    "experimental": (
        "Evaluate how well the artifact supports experimentation and result verification. Are there benchmark scripts, result logging, or statistical validation procedures?\n"
        "Do NOT give credit for experimentation unless such procedures, logs, or metrics are directly present."
    ),
    "functionality": (
        "Assess if the artifact functions as intended. Check for code execution, model evaluation, test cases, and verifiable outcomes.\n"
        "Do NOT assume execution correctness without test cases, logs, or clear validation artifacts in the repository."
    )
}


def log_llm_interaction(agent_name, prompt, response):
    logger.info(f"Agent: {agent_name}")
    logger.info(f"Prompt:\n{prompt}")
    logger.info(f"Response:\n{response}")


def load_analysis_content(analysis_path: str) -> str:
    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    content = []
    for file_group in ["documentation_files", "code_files", "license_files"]:
        for file in data.get(file_group, []):
            lines = file.get("content", [])
            if lines:
                content.append(f"# {file['path']}\n" + "\n".join(lines))
    return "\n\n".join(content)


def retrieve_passages(text, keywords, window=100):
    relevant = []
    for kw in keywords:
        for match in re.finditer(re.escape(kw), text, re.IGNORECASE):
            start, end = max(0, match.start() - window), min(len(text), match.end() + window)
            relevant.append(text[start:end])
    return list(set(relevant)) if relevant else [text]


def evaluator_agent(state: EvalState) -> dict:
    dimension = state["current_dimension"]
    passages = retrieve_passages(state["context"], dimension["keywords"])
    section_desc = next(
        (s["description"] for s in state["sections"] if s["name"].lower() == dimension["dimension"].lower()), "")
    custom_prompt = DIMENSION_PROMPTS.get(dimension["dimension"].lower(),
                                          "Evaluate this dimension strictly based on explicit evidence in the artifact. Do not assume.")

    prompt_content = (
            f"Dimension: {dimension['dimension']}\n" +
            f"Keywords: {', '.join(dimension['keywords'])}\n" +
            f"Section Description: {section_desc}\n" +
            f"Strict Evaluation Instructions: {custom_prompt}\n\n" +
            "Relevant Passages from Artifact:\n" + "\n---\n".join(passages) +
            "\n\nProvide your evaluation strictly following the instructions above in this format:\nScore: <numeric_score_between_0_and_1>\nFeedback: <explicit_observations_only_from_the_artifact>"
    )

    prompt = [
        SystemMessage(
            content="You are a research artifact evaluator. Be strict, evidence-driven, and avoid assumptions."),
        HumanMessage(content=prompt_content)
    ]
    resp = llm.invoke(prompt)
    log_llm_interaction("Evaluator Agent", prompt_content, resp.content)

    score_match = re.search(r"Score:\s*([0-9.]+)", resp.content)
    score = float(score_match.group(1)) if score_match else 0.0
    feedback = resp.content.split("Feedback:")[-1].strip() if "Feedback:" in resp.content else "Feedback missing."
    weighted_score = score * dimension["weight"]
    return {"score": score, "feedback": feedback, "weighted_score": weighted_score}


def load_conf_criteria(state: EvalState) -> dict:
    conf_code = state["conf_code"]
    sections, dimensions = [], []
    for rec in graph_db.run(
            "MATCH (c:Conference {name:$conf_code})-[:REQUIRES_SECTION]->(s:Section) RETURN s.name, s.description",
            conf_code=conf_code):
        sections.append({"name": rec["s.name"], "description": rec["s.description"]})
    for rec in graph_db.run(
            "MATCH (c:Conference {name:$conf_code})-[:USES_DIMENSION]->(d:Dimension) OPTIONAL MATCH (d)-[:HAS_KEYWORD]->(k:Keyword) RETURN d.name AS dimension, d.weight AS weight, collect(k.name) AS keywords",
            conf_code=conf_code):
        dimensions.append({"dimension": rec["dimension"], "weight": rec["weight"], "keywords": rec["keywords"]})
    context = load_analysis_content(state["analysis_path"])
    return {"sections": sections, "dimensions": dimensions, "context": context, "dimension_results": []}


def evaluate_dimensions(state: EvalState):
    results = []
    total = 0.0
    for dimension in state["dimensions"]:
        state["current_dimension"] = dimension
        eval_result = evaluator_agent(state)
        total += eval_result["weighted_score"]
        results.append({"dimension": dimension["dimension"], **eval_result})
    return {"dimension_results": results, "final_score": total}


graph = StateGraph(EvalState)
graph.add_node("load_criteria", load_conf_criteria)
graph.add_node("evaluate", evaluate_dimensions)
graph.set_entry_point("load_criteria")
graph.add_edge("load_criteria", "evaluate")
graph.add_edge("evaluate", END)

compiled_eval = graph.compile()

if __name__ == "__main__":
    state: EvalState = {
        "conf_code": "13_icse_2025",
        "analysis_path": "../../algo_outputs/algorithm_2_output/ml-image-classifier_analysis.json",
        "context": "",
        "sections": [],
        "dimensions": [],
        "current_dimension": {},
        "dimension_results": [],
        "final_score": 0.0
    }

    final_state = compiled_eval.invoke(state)
    logger.info(f"Final Artifact Score: {final_state['final_score']:.2f}")
    with open(os.path.join(LOG_DIR, "final_artifact_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, ensure_ascii=False)
