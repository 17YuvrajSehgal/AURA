import io
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import TypedDict, List, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from json_file_utils import (
    read_analysis_json,
    get_repository_structure,
    get_documentation_texts,
    get_code_texts,
    get_license_texts, get_first_code_snippet
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()

# === 0. Logging Setup ===
# Define logging directory
LOG_DIR = "../../algo_outputs/logs/algorithm_4_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === 1. Real LLM (OpenAI) ===
llm = ChatOpenAI(temperature=0.2, model="gpt-4")


# === 2. State schema ===
class READMEState(TypedDict):
    context: str
    structure: str
    code_files: List[str]
    license_text: str
    required_sections: List[str]
    current_section: str
    completed_sections: List[dict]
    readme: str
    quality_score: float


# Helper function to log detailed interactions with the LLM
def log_llm_interaction(agent_name, prompt, response):
    logger.info(f"Agent: {agent_name}")
    logger.info(f"Prompt:\n{prompt}")
    logger.info(f"Response:\n{response}")


def log_state(state: READMEState, note: str = ""):
    serialized = json.dumps(state, indent=2)
    logger.info(f"=== READMEState Snapshot {f'({note})' if note else ''} ===\n{serialized}")


def dump_state_to_json(state: READMEState, filename: str):
    with open(os.path.join(LOG_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# === 3. Custom Section Prompts ===
SECTION_PROMPTS: Dict[str, str] = {
    "installation": (
        "Write a detailed **Installation** section for this artifact. "
        "Begin with a heading (e.g., 'Installation' or 'Setup'). "
        "List all dependencies, environment requirements, and required package versions. "
        "Provide step-by-step, numbered installation instructions (code blocks for 'pip install', 'conda install', etc.). "
        "Mention files like 'requirements.txt', and add platform-specific notes if relevant. "
        "Link to official docs/scripts if available."
    ),
    "usage": (
        "Write a comprehensive **Usage** section. "
        "Explain how to run the main code/application, referencing example commands and expected input/output. "
        "Provide code block examples. "
        "Highlight advanced options, environment variables, configuration files, and best practices. "
        "Include quickstart guide or command summary table if possible."
    ),
    "requirements": (
        "Write a clear **Requirements** section. "
        "Specify all software/hardware prerequisites, Python/OS/CPU/GPU versions, required libraries (with links if possible), and environment setup instructions."
    ),
    "license": (
        "Write a prominent **License** section. "
        "State the license type (e.g., MIT, Apache), summarize permissions/restrictions, provide a link or full license text, and acknowledge contributors/authors. "
        "Optionally, add a citation block (DOI, Zenodo, or BibTeX)."
    ),
    "troubleshooting": (
        "Add a **Troubleshooting / FAQ** section. "
        "List common errors or issues and their solutions. "
        "Provide debugging tips and support contacts or links to the issue tracker. "
        "Add a 'Frequently Asked Questions' subsection if relevant."
    ),
    "reproducibility": (
        "Write a dedicated **Reproducibility** section. "
        "Explain how to reproduce main results, listing all scripts, datasets, and detailed steps for replication. "
        "Mention random seed setting, environment (Docker/Colab/Binder), validation outputs, and expected results."
    ),
    "references": (
        "Write a **References / Related Work** section. "
        "List relevant papers, datasets, or projects, with links to arXiv, Zenodo, or official sites. "
        "Mention the original research paper if applicable."
    ),
    "examples": (
        "Write an **Examples** section. "
        "Give code walkthroughs for typical usage, showing input/output or screenshots, and sample datasets or demo files."
    ),
    "overview": (
        "Write an **Overview** or **Project Structure** section. "
        "Summarize the artifact's purpose, key features, and main components. "
        "Optionally, provide a directory tree and explain the role of each file."
    ),
    "contributing": (
        "Write a **Contributing** section. "
        "Explain how to contribute, report issues, and suggest improvements. "
        "Mention any code of conduct or contribution guidelines."
    ),
    # Add more patterns if needed
}

GENERIC_SECTION_PROMPT = (
    "Write a detailed '{section}' section for a research artifact README, following academic best practices "
    "and including all critical information. Use bullet lists, tables, and clear headings where appropriate."
)


# === 4. Multi-Agent (Prompt Chaining) Functions ===

def author_agent(state: READMEState, prev_sections: Dict[str, str], feedback: str = "") -> str:
    """ Section Author agent: generates initial content for a README section. """
    sect = state["current_section"]
    prev_content = "\n".join([f"## {sec}\n{cont}" for sec, cont in prev_sections.items()])
    prompt_template = SECTION_PROMPTS.get(sect, GENERIC_SECTION_PROMPT.format(section=sect))
    prompt_content = (
        f"{prompt_template}\n\n"
        "General requirements:\n"
        "- Use bullet lists and tables for clarity.\n"
        "- Include diagrams/images if helpful.\n"
        "- Provide outbound/internal links when useful.\n"
        "- Use readable, concise, and technical language that is easy to understand and is very accurate according to the given repository.\n"
        "- Explicitly mention all licenses, attributions, and citation instructions if applicable.\n"
        "**Do not repeat or copy content that is already present in previously written sections.**\n\n"
        "**Do not add any information that is not present in the artifact such as false results.**\n\n"
        "**If you are not sure about what should be written in a given section, just leave it blank with heading only. The author of the repository will take care of that section.**\n\n"
        f"Project structure:\n{state['structure']}\n\n"
        f"Sample code (first file):\n{state['code_files'][0] if state['code_files'] else 'N/A'}\n\n"
        f"License:\n{state['license_text']}\n\n"
        f"Sections written so far (include headings and content):\n{prev_content}\n\n"
        f"{'Feedback from reviewer: ' + feedback if feedback else ''}"
    )

    prompt = [
        SystemMessage(content=f"You are a research artifact documentation author. {prompt_template}"),
        HumanMessage(content=prompt_content)
    ]
    resp = llm.invoke(prompt)
    log_llm_interaction("Author Agent", prompt_content, resp.content)
    return resp.content.strip()


def editor_agent(section_content: str, section_name: str) -> str:
    """ Section Editor agent: revises/improves initial section for clarity and structure. """
    prompt = [
        SystemMessage(content=(
            f"You are an academic editor. Edit and improve the following '{section_name}' section for clarity, conciseness, and academic tone. "
            f"Reorganize information as needed. Do not remove critical details."
        )),
        HumanMessage(content=section_content)
    ]
    resp = llm.invoke(prompt)
    log_llm_interaction("Editor Agent", section_content, resp.content)

    return resp.content.strip()


def critic_agent(section_content: str, section_name: str) -> dict:
    """
    Critic agent reviews a section, gives a score and feedback.
    """
    prompt = [
        SystemMessage(
            content=(
                f"You are a research artifact reviewer specializing in README quality. "
                f"Review the following '{section_name}' section and:\n"
                f"1. Provide a numeric quality score between 0 and 1 (clarity, completeness, usefulness for a research artifact)\n"
                f"2. Give concise, actionable feedback for improvement, if any.\n"
                f"Format:\n"
                f"Score: <numeric_score>\n"
                f"Feedback: <short_feedback>"
            )
        ),
        HumanMessage(content=section_content)
    ]
    resp = llm.invoke(prompt)

    log_llm_interaction("Critic Agent", section_content, resp.content)

    score_match = re.search(r"Score:\s*([0-9.]+)", resp.content)
    feedback_match = re.search(r"Feedback:\s*(.*)", resp.content, re.DOTALL)
    score = float(score_match.group(1)) if score_match else 0.5
    feedback = feedback_match.group(1).strip() if feedback_match else ""
    return {"score": score, "feedback": feedback}


# === 5. Core Workflow Functions ===

def load_analysis(state: READMEState) -> dict:
    logger.info("Loading pre‐parsed analysis into state...")

    # Load the analysis file path from a known location or passed context
    analysis_path = "../../algo_outputs/algorithm_2_output/ml-image-classifier_analysis.json"
    data = read_analysis_json(analysis_path)

    structure = "\n".join(get_repository_structure(data))
    docs = get_documentation_texts(data)
    code_texts = get_code_texts(data)
    snippet = get_first_code_snippet(data)
    licenses = get_license_texts(data)

    # Log the extracted metadata
    logger.info(f"Extracted structure:\n{structure}")
    if docs:
        logger.info(f"Sample doc:\n{docs[0][:200]}...")
    if snippet:
        logger.info(f"Sample code:\n{snippet[:500]}...")
    if licenses:
        logger.info(f"License preview:\n{licenses[0][:200]}...")

    return {
        "structure": structure,
        "code_files": code_texts,
        "license_text": licenses[0] if licenses else "",
        "context": docs[0] if docs else "No documentation found."
    }


def plan_sections(state: READMEState) -> dict:
    logger.info("Planning which sections to generate...")
    prompt = [
        SystemMessage(
            content="List comma‐separated sections that should appear in a README for a research artifact."
        ),
        HumanMessage(content=state["context"])
    ]
    response = llm.invoke(prompt)

    log_llm_interaction("Planner Agent", state["context"], response.content)

    sections = [s.strip().lower() for s in response.content.split(",") if s.strip()]
    logger.info(f"LLM returned sections: {sections}")
    return {"required_sections": sections}


def synthesize_readme(state: READMEState) -> dict:
    logger.info("Synthesizing final README from completed sections...")
    parts = [
        f"## {s['section'].title()}\n{s['content']}"
        for s in state["completed_sections"]
    ]
    combined = "\n\n".join(parts)
    logger.debug(f"Synthesized README:\n{combined}")

    log_llm_interaction("Synthesizer", "Completed sections", combined)

    return {"readme": combined}


def evaluate_quality(state: READMEState) -> dict:
    """
    Evaluator: have GPT‐4 rate the current README draft from 0 to 1.
    """
    logger.info("Evaluating README quality...")
    draft = state["readme"]

    if not draft.strip():
        logger.warning("README draft is empty. Returning score 0.0.")
        return {"quality_score": 0.0}

    prompt = [
        SystemMessage(
            content="Rate this README draft between 0 and 1 on completeness and clarity (only return the numeric score)."
        ),
        HumanMessage(content=draft)
    ]
    resp = llm.invoke(prompt)

    log_llm_interaction("Evaluator Agent", draft, resp.content)

    try:
        score = float(resp.content.strip())
    except ValueError:
        score = 0.5
        logger.warning("Could not parse numeric score from LLM; defaulting to 0.5")
    logger.info(f"LLM quality score: {score:.2f}")
    return {"quality_score": score}


def refine_sections(state: READMEState) -> dict:
    """
    Optimizer: append "[Refined]" to each section. In a real system, you might invoke GPT with feedback; here we simply tag sections for demonstration.
    """
    logger.info("Refining all sections due to low quality score...")
    refined = [
        {"section": s["section"], "content": s["content"] + "\n\n[Refined]"}
        for s in state["completed_sections"]
    ]

    for s in refined:
        logger.debug(f"Refined Section - {s['section']}:\n{s['content']}")

    return {"completed_sections": refined}


# === 6. LangGraph Workflow (unchanged) ===

graph = StateGraph(READMEState)

# Entry point
graph.add_node("load_analysis", load_analysis)
graph.set_entry_point("load_analysis")

# Step 1: Planner
graph.add_node("planner", plan_sections)
graph.add_edge("load_analysis", "planner")

# Step 2: Synthesize README
# graph.add_node("synthesizer", synthesize_readme)
graph.add_edge("planner", END)

# # Step 3: Evaluate README
# graph.add_node("evaluator", evaluate_quality)
# graph.add_edge("synthesizer", "evaluator")
#
# # Step 4: Optimize if needed
# graph.add_node("optimizer", refine_sections)
# graph.add_edge("optimizer", "synthesizer")

# Conditional branch based on quality score
# graph.add_conditional_edges(
#     "evaluator",
#     lambda state: "optimizer" if state["quality_score"] < 0.8 else END,
#     {"optimizer": "optimizer", END: END}
# )
#
# graph.add_edge("optimizer", "synthesizer")

# 4.f: evaluator → END when quality OK
# graph.add_edge("evaluator", END)

compiled = graph.compile()

# 4.g: Visualize as PNG
logger.info("Drawing workflow diagram to 'readme_generator_workflow.png'...")
png_bytes = compiled.get_graph().draw_mermaid_png()
os.makedirs("../../algo_outputs/algorithm_4_output", exist_ok=True)
with open("../../algo_outputs/algorithm_4_output/readme_generator_workflow.png", "wb") as f_png:
    f_png.write(png_bytes)

# === 7. Driver: Multi-Agent Section Workflow with Prompt Chaining ===

if __name__ == "__main__":
    # 1. Start with an empty state
    state: READMEState = {
        "context": "",
        "structure": "",
        "code_files": [],
        "license_text": "",
        "required_sections": [],
        "current_section": "",
        "completed_sections": [],
        "readme": "",
        "quality_score": 0.0
    }

    log_state(state, "initial")

    # 2. Run the full LangGraph pipeline (this includes load_analysis → planner → synthesize → evaluate [+optimize if needed])
    logger.info("Running LangGraph pipeline...")
    intermediate_state = compiled.invoke(state, config=RunnableConfig(recursion_limit=3))

    # 3. Extract planned sections from output
    state.update(intermediate_state)
    log_state(state, "after LangGraph execution")
    logger.info(f"Sections planned: {state['required_sections']}")

    # Optional: Warn if planning failed
    if not state["required_sections"]:
        logger.warning("No sections could be planned. Repository may be too minimal or lacking context.")

    # 4. Manual multi-agent generation (as before)
    SECTION_CRITIC_THRESHOLD = 0.8
    prev_sections = {}

    for section_key in state["required_sections"]:
        state["current_section"] = section_key
        log_state(state, f"processing section: {section_key}")
        feedback = ""
        retry_count = 0
        max_retries = 3
        while True:
            author_output = author_agent(state, prev_sections, feedback)
            editor_output = editor_agent(author_output, section_key)
            review = critic_agent(editor_output, section_key)
            logger.info(f"Critic score for '{section_key}': {review['score']:.2f} | Feedback: {review['feedback']}")
            if review["score"] >= SECTION_CRITIC_THRESHOLD or retry_count >= max_retries:
                prev_sections[section_key] = editor_output
                state["completed_sections"].append({"section": section_key, "content": editor_output})
                log_state(state, f"after completing section: {section_key}")
                break
            else:
                logger.info(f"Regenerating '{section_key}' due to low critic score. Feedback: {review['feedback']}")
                feedback = review["feedback"]
                retry_count += 1

    # 5. Synthesize, evaluate, and refine
    synth_out = synthesize_readme(state)
    state["readme"] = synth_out["readme"]
    log_state(state, "after synthesis")
    eval_out = evaluate_quality(state)
    state["quality_score"] = eval_out["quality_score"]
    log_state(state, "after evaluation")

    if state["quality_score"] < 0.8:
        refine_out = refine_sections(state)
        state["completed_sections"] = refine_out["completed_sections"]
        log_state(state, "after refinement")
        synth2 = synthesize_readme(state)
        state["readme"] = synth2["readme"]
        eval2 = evaluate_quality(state)
        state["quality_score"] = eval2["quality_score"]

    # 6. Save and report
    os.makedirs("../../algo_outputs/algorithm_4_output", exist_ok=True)
    readme_path = os.path.join("../../algo_outputs/algorithm_4_output", "generated_README.md")
    with open(readme_path, "w", encoding="utf-8") as f_md:
        f_md.write(state["readme"])

    logger.info("=== Generated README ===\n")
    print(state["readme"])
    dump_state_to_json(state, "final_state.json")
    logger.info(f"Final quality score: {state['quality_score']:.2f}")
    was_refined = any("[Refined]" in s["content"] for s in state["completed_sections"])
    logger.info(f"Refinement applied? {'Yes' if was_refined else 'No'}")
