import logging
import os
import re
from typing import TypedDict, List, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from json_file_utils import (
    read_analysis_json,
    get_repository_structure,
    get_documentation_texts,
    get_code_texts,
    get_license_texts, get_first_code_snippet
)

load_dotenv()

# === 0. Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
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
        "- Use readable, concise, and technical language (target Flesch-Kincaid ≈ 16).\n"
        "- Explicitly mention all licenses, attributions, and citation instructions if applicable.\n"
        "**Do not repeat or copy content that is already present in previously written sections.**\n\n"
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
    score_match = re.search(r"Score:\s*([0-9.]+)", resp.content)
    feedback_match = re.search(r"Feedback:\s*(.*)", resp.content, re.DOTALL)
    score = float(score_match.group(1)) if score_match else 0.5
    feedback = feedback_match.group(1).strip() if feedback_match else ""
    return {"score": score, "feedback": feedback}


# === 5. Core Workflow Functions ===

def load_analysis(state: READMEState) -> dict:
    logger.info("Loading pre‐parsed analysis from input state.")
    return {}


def plan_sections(state: READMEState) -> dict:
    logger.info("Planning which sections to generate...")
    prompt = [
        SystemMessage(
            content="List comma‐separated sections that should appear in a README for a research artifact."
        ),
        HumanMessage(content=state["context"])
    ]
    response = llm.invoke(prompt)
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
    return {"readme": combined}


def evaluate_quality(state: READMEState) -> dict:
    """
    Evaluator: have GPT‐4 rate the current README draft from 0 to 1.
    """
    logger.info("Evaluating README quality...")
    draft = state["readme"]
    prompt = [
        SystemMessage(
            content="Rate this README draft between 0 and 1 on completeness and clarity (only return the numeric score)."
        ),
        HumanMessage(content=draft)
    ]
    resp = llm.invoke(prompt)
    try:
        score = float(resp.content.strip())
    except ValueError:
        score = 0.5
        logger.warning("Could not parse numeric score from LLM; defaulting to 0.5")
    logger.info(f"LLM quality score: {score:.2f}")
    return {"quality_score": score}


def refine_sections(state: READMEState) -> dict:
    """
    Optimizer: append "[Refined]" to each section. In a real system you might
    re‐invoke GPT with feedback; here we simply tag sections for demonstration.
    """
    logger.info("Refining all sections due to low quality score...")
    refined = [
        {"section": s["section"], "content": s["content"] + "\n\n[Refined]"}
        for s in state["completed_sections"]
    ]
    return {"completed_sections": refined}


# === 6. LangGraph Workflow (unchanged) ===

graph = StateGraph(READMEState)

# 4.a: load_analysis (entry point)
graph.add_node("load_analysis", load_analysis)
graph.set_entry_point("load_analysis")

# 4.b: planner
graph.add_node("planner", plan_sections)
graph.add_edge("load_analysis", "planner")

# 4.c: synthesizer
graph.add_node("synthesizer", synthesize_readme)

# 4.d: evaluator
graph.add_node("evaluator", evaluate_quality)
graph.add_edge("synthesizer", "evaluator")

# 4.e: optimizer
graph.add_node("optimizer", refine_sections)
graph.add_conditional_edges(
    "evaluator",
    # If quality_score < 0.8, invoke optimizer; otherwise go to END.
    lambda state: "optimizer" if state["quality_score"] < 0.8 else END,
    {"optimizer": "optimizer", END: END}
)
graph.add_edge("optimizer", "synthesizer")

# 4.f: evaluator → END when quality OK
graph.add_edge("evaluator", END)

compiled = graph.compile()

# 4.g: Visualize as PNG
logger.info("Drawing workflow diagram to 'readme_generator_workflow.png'...")
png_bytes = compiled.get_graph().draw_mermaid_png()
os.makedirs("../../algo_outputs/algorithm_4_output", exist_ok=True)
with open("../../algo_outputs/algorithm_4_output/readme_generator_workflow.png", "wb") as f_png:
    f_png.write(png_bytes)

# === 7. Driver: Multi-Agent Section Workflow with Prompt Chaining ===

if __name__ == "__main__":
    # 5.a: Initial state from pre‐parsed analysis JSON

    # 1. Load JSON
    analysis_path = "../../algo_outputs/algorithm_2_output/ml-image-classifier_analysis.json"
    data = read_analysis_json(analysis_path)

    # 2. Extract structure (tree or list of paths)
    structure = "\n".join(get_repository_structure(data))
    logger.info(f"\nRepository structure:\n{structure}")

    # 3. Extract documentation text (first doc only, e.g.)
    docs = get_documentation_texts(data)
    if docs:
        logger.info(f"\nFirst documentation snippet:\n{docs[0][:200]}...")

    # 4. Extract code texts, pick first snippet
    code_texts = get_code_texts(data)
    snippet = get_first_code_snippet(data)
    if snippet:
        logger.info(f"\nSnippet from first code file:\n{snippet[:500]}...")

    # 5. Extract license
    licenses = get_license_texts(data)
    if licenses:
        logger.info(f"\nLicense text starts with:\n{licenses[0][:200]}...")

    state: READMEState = {
        "context": "...",  # your full context as before
        "structure": structure,
        "code_files": code_texts,
        "license_text": licenses[0] if licenses else "",
        "required_sections": [],
        "current_section": "",
        "completed_sections": [],
        "readme": "",
        "quality_score": 0.0
    }

    # Step 1: Determine required sections
    logger.info("Running planner to determine required sections...")
    planner_out = compiled.invoke(state)
    state["required_sections"] = planner_out["required_sections"]
    logger.info(f"Sections planned: {state['required_sections']}")

    SECTION_CRITIC_THRESHOLD = 0.8
    prev_sections = {}

    for section_key in state["required_sections"]:
        state["current_section"] = section_key
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
                break
            else:
                logger.info(f"Regenerating '{section_key}' due to low critic score. Feedback: {review['feedback']}")
                feedback = review["feedback"]
                retry_count += 1

    # Synthesize, evaluate, refine if needed
    synth_out = synthesize_readme(state)
    state["readme"] = synth_out["readme"]
    eval_out = evaluate_quality(state)
    state["quality_score"] = eval_out["quality_score"]

    # 5.e: If quality < 0.8, refine → re‐synthesize → re‐evaluate
    if state["quality_score"] < 0.8:
        refine_out = refine_sections(state)
        state["completed_sections"] = refine_out["completed_sections"]
        # Re‐synthesize & re‐evaluate once more
        synth2 = synthesize_readme(state)
        state["readme"] = synth2["readme"]
        eval2 = evaluate_quality(state)
        state["quality_score"] = eval2["quality_score"]

    # Save and report
    os.makedirs("../../algo_outputs/algorithm_4_output", exist_ok=True)
    readme_path = os.path.join("../../algo_outputs/algorithm_4_output", "generated_README.md")
    with open(readme_path, "w", encoding="utf-8") as f_md:
        f_md.write(state["readme"])

    # 5.g: Print summary
    logger.info("=== Generated README ===\n")
    print(state["readme"])
    logger.info(f"Final quality score: {state['quality_score']:.2f}")
    was_refined = any("[Refined]" in s["content"] for s in state["completed_sections"])
    logger.info(f"Refinement applied? {'Yes' if was_refined else 'No'}")
