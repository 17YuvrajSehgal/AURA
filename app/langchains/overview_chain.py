import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState

from app.ai_models.ai_utils import open_ai_chat_model
from app.langchains.file_search_chains import files_chain
from app.utils.utils import read_files_content

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("overview_chain")

# === Memory and Workflow ===
memory = MemorySaver()
workflow = StateGraph(state_schema=MessagesState)


def summarize_project_overview(state: MessagesState):
    system_prompt = (
        "You are an assistant creating a README Overview section for a research artifact project. "
        "Summarize the purpose, main features, and intended usage in one concise paragraph."
    )
    system_message = SystemMessage(content=system_prompt)

    file_messages = [
        HumanMessage(content=f"Summarize this code file:\n\n{content}")
        for content in state["messages"]
    ]
    logger.info(f"Summarizing {len(file_messages)} code files...")

    response = open_ai_chat_model.invoke([system_message] + file_messages)
    return {"messages": response}


workflow.add_node("overview_summarizer", summarize_project_overview)
workflow.add_edge(START, "overview_summarizer")
app = workflow.compile(checkpointer=memory)


# === New logic: directly use the repository analysis JSON ===
def run_overview_from_analysis(analysis_path):
    logger.info(f"Loading analysis data from: {analysis_path}")
    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    code_file_contents = []
    for file in data.get("code_files", []):
        if isinstance(file.get("content"), list):
            joined = "\n".join(file["content"])
        else:
            joined = str(file.get("content", ""))
        if joined.strip():
            code_file_contents.append(joined[:1500])  # Limit per file

    logger.info(f"Loaded {len(code_file_contents)} code files for summarization")

    # Run the chain
    result = app.invoke(
        {"messages": code_file_contents},
        config={"configurable": {"thread_id": "project-overview"}},
    )

    overview = "\n\n".join(
        m.content for m in result["messages"] if isinstance(m, AIMessage)
    )
    return overview


# === Run if a script is executed directly ===
if __name__ == "__main__":
    analysis_file = "../../data/algorithm_2_output/ml-image-classifier_analysis.json"
    overview_section = run_overview_from_analysis(analysis_file)
    print("\n--- Project Overview ---\n")
    print(overview_section)
