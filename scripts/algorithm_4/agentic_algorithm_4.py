import logging
import os
from typing import TypedDict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# === 0. Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === 1. Real LLM (OpenAI) ===
# Make sure you have set OPENAI_API_KEY in your environment.
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


# === 3. Node Functions ===

def load_analysis(state: READMEState) -> dict:
    """
    Initial loader: in practice, you'd load from a JSON file into 'state'.
    Here, we assume 'state' is already populated with context, structure, code_files, license_text.
    We log and return an empty dict to advance to 'planner'.
    """
    logger.info("Loading pre‐parsed analysis from input state.")
    return {}  # no changes to state, just proceed to planner


def plan_sections(state: READMEState) -> dict:
    """
    Orchestrator: ask GPT‐4 which sections are needed (e.g. "overview, installation, usage").
    """
    logger.info("Planning which sections to generate...")
    prompt = [
        SystemMessage(
            content="List comma‐separated sections that should appear "
                    "in a README for a research artifact."
        ),
        HumanMessage(content=state["context"])
    ]
    response = llm.invoke(prompt)
    sections = [s.strip().lower() for s in response.content.split(",") if s.strip()]
    logger.info(f"LLM returned sections: {sections}")
    return {"required_sections": sections}


def generate_section(state: READMEState) -> dict:
    """
    Worker: generate content for exactly one section.
    Each call sees state['current_section'] set by the driver loop.
    """
    sect = state["current_section"]
    logger.info(f"Generating section: '{sect}'")
    prompt = [
        SystemMessage(content=f"Generate a '{sect}' section for a README. Use this context:"),
        HumanMessage(content=(
            f"Project structure:\n{state['structure']}\n\n"
            f"Sample code (first file):\n{state['code_files'][0] if state['code_files'] else 'N/A'}\n\n"
            f"License:\n{state['license_text']}\n\n"
            f"Section to write: {sect}"
        ))
    ]
    resp = llm.invoke(prompt)
    entry = {"section": sect, "content": resp.content.strip()}
    updated_list = state.get("completed_sections", []) + [entry]
    logger.info(f"Completed '{sect}'.")
    return {"completed_sections": updated_list}


def synthesize_readme(state: READMEState) -> dict:
    """
    Synthesizer: merge all completed_sections into a single Markdown README string.
    """
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
            content="Rate this README draft between 0 and 1 on completeness and clarity "
                    "(only return the numeric score)."
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


# === 4. Build a static LangGraph workflow ===

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
os.makedirs("data/algorithm_4_output", exist_ok=True)
with open("data/algorithm_4_output/readme_generator_workflow.png", "wb") as f_png:
    f_png.write(png_bytes)

# === 5. Driver code: manual “parallel” section generation ===

if __name__ == "__main__":
    # 5.a: Initial state from pre‐parsed analysis JSON
    state: READMEState = {
        "context": "ASE_2024",
        "structure": (
            "ml-image-classifier\n"
            "├── README.md\n"
            "├── src/train.py\n"
            "└── data/training/"
        ),
        "code_files": ["def foo(): pass\nimport torch\n# ..."],
        "license_text": "MIT License text here",
        "required_sections": [],
        "current_section": "",
        "completed_sections": [],
        "readme": "",
        "quality_score": 0.0
    }

    # 5.b: Step 1: load_analysis → planner
    logger.info("Running planner to determine required sections...")
    planner_out = compiled.invoke(state)
    state["required_sections"] = planner_out["required_sections"]
    logger.info(f"Sections planned: {state['required_sections']}")

    # 5.c: “Parallel” section generation via Python loop
    for sect in state["required_sections"]:
        state["current_section"] = sect
        worker_out = generate_section(state)
        state["completed_sections"] = worker_out["completed_sections"]

    # 5.d: Now synthesize → evaluate
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

    # 5.f: Save the final README
    os.makedirs("data/algorithm_4_output", exist_ok=True)
    readme_path = os.path.join("data/algorithm_4_output", "generated_README.md")
    with open(readme_path, "w", encoding="utf-8") as f_md:
        f_md.write(state["readme"])

    # 5.g: Print summary
    logger.info("=== Generated README ===\n")
    print(state["readme"])
    logger.info(f"Final quality score: {state['quality_score']:.2f}")
    was_refined = any("[Refined]" in s["content"] for s in state["completed_sections"])
    logger.info(f"Refinement applied? {'Yes' if was_refined else 'No'}")
