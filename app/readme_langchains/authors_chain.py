from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState

from app.ai_models.ai_utils import open_ai_chat_model

# Define memory and workflow
memory = MemorySaver()
workflow = StateGraph(state_schema=MessagesState)


# Define summarization logic for contributors
def summarize_authors(state: MessagesState):
    system_prompt = (
        """You are an assistant creating a README Contributors section for a project. Based on the provided list 
        of contributors, generate a markdown-compatible section that highlights each contributor with their username, 
        profile link, and number of contributions. The format should be concise and visually appealing."""
    )
    system_message = SystemMessage(content=system_prompt)
    file_messages = [
        HumanMessage(content=f"Contributor: {content}")
        for content in state["messages"]
    ]
    response = open_ai_chat_model.invoke([system_message] + file_messages)
    return {"messages": response}


# Add summarization to the workflow
workflow.add_node("authors_summarizer", summarize_authors)
workflow.add_edge(START, "authors_summarizer")
app = workflow.compile(checkpointer=memory)

# Define Runnables
prepare_authors_data = RunnableLambda(
    lambda inputs: [
        f"{author['login']}, Profile: {author['html_url']}, Contributions: {author['contributions']}"
        for author in inputs["repository_authors"]
    ]
)

generate_contributors_section = RunnableLambda(
    lambda authors_data: app.invoke(
        {"messages": authors_data},
        config={"configurable": {"thread_id": "contributors-section"}},
    )
)

format_contributors_output = RunnableLambda(
    lambda result: "\n\n".join(
        message.content
        for message in result["messages"]
        if isinstance(message, AIMessage)
    )
)

# Properly chain the steps
authors_chain = (
        prepare_authors_data
        | generate_contributors_section
        | format_contributors_output
)

# Main for testing
# if __name__ == "__main__":
#     # Example data
#     repository_authors = [
#         {"login": "octocat", "html_url": "https://github.com/octocat", "contributions": 42},
#         {"login": "anotheruser", "html_url": "https://github.com/anotheruser", "contributions": 15},
#         {"login": "developer123", "html_url": "https://github.com/developer123", "contributions": 5},
#     ]
#
#     # Run the authors_chain
#     try:
#         result = authors_chain.invoke({"repository_authors": repository_authors})
#         print("Generated Contributors Section:")
#         print(result)
#     except Exception as e:
#         print(f"Error occurred: {e}")
