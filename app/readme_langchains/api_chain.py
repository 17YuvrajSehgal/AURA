from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState

from app.ai_models.ai_utils import open_ai_chat_model
from app.util_langchains.file_search_chains import files_chain
from app.utils.utils import read_files_content

# Define memory and workflow
memory = MemorySaver()
workflow = StateGraph(state_schema=MessagesState)


# Define summarization logic
def summarize_api_files(state: MessagesState):
    system_prompt = (
        "You are an assistant creating a README section about API and endpoints. "
        "Focus on summarizing key points related to APIs while keeping it concise and informative."
    )
    system_message = SystemMessage(content=system_prompt)
    file_messages = [
        HumanMessage(content=f"Summarize the following API file content: {content}")
        for content in state["messages"]
    ]
    response = open_ai_chat_model.invoke([system_message] + file_messages)
    return {"messages": response}


# Add summarization to the workflow
workflow.add_node("summarizer", summarize_api_files)
workflow.add_edge(START, "summarizer")
app = workflow.compile(checkpointer=memory)

# Define Runnables
extract_file_paths = RunnableLambda(
    lambda inputs: files_chain.invoke(
        {
            "base_directory": inputs["base_directory"],
            "project_structure": inputs["project_structure"],
            "task": "api and endpoints documentation",
        }
    )
)

read_file_contents = RunnableLambda(
    lambda file_paths: read_files_content(
        [fp.strip() for fp in file_paths["text"].split(",") if fp.strip()]
    )
)

generate_api_documentation = RunnableLambda(
    lambda file_contents: app.invoke(
        {"messages": file_contents},
        config={"configurable": {"thread_id": "api-documentation"}},
    )
)

format_output = RunnableLambda(
    lambda result: "\n\n".join(
        message.content
        for message in result["messages"]
        if isinstance(message, AIMessage)
    )
)

# Properly chain the steps
api_chain = (
        extract_file_paths
        | read_file_contents
        | generate_api_documentation
        | format_output
)

# Usage example
# if __name__ == "__main__":
#     base_directory = "C:\\workplace\\ArtifactEvaluator\\temp_dir_for_git\\COSC-4P02-PROJECT"
#     project_structure = """
#     Tree structure with files:
#     ├── .gitignore
#     ├── Dockerfile
#     ├── LICENSE.md
#     ├── mvnw
#     ├── mvnw.cmd
#     ├── pom.xml
#     ├── README.md
#     └── src
#         └── main
#             ├── java
#             │   └── com
#             │       └── websummarizer
#             │           └── Web
#             │               └── Summarizer
#             │                   ├── common
#             │                   │   └── exceptions
#             │                   │       └── OauthUpdateNotAllowed.java
#             │                   ├── configs
#             │                   │   ├── GlobalExceptionHandler.java
#             │                   │   ├── SecurityConfig.java
#             │                   │   └── SuccessHandler.java
#             │                   ├── controller
#             │                   │   ├── AdminController.java
#             │                   │   ├── AuthenticationController.java
#             │                   │   ├── PasswordResetController.java
#             │                   │   ├── ShortLinkController.java
#             │                   │   ├── ShortLinkGenerator.java
#             │                   │   ├── UserController.java
#             │                   │   └── WebController.java
#             │                   ├── llmConnectors
#             │                   │   ├── Bart.java
#             │                   │   ├── Llm.java
#             │                   │   └── OpenAi.java
#             │                   ├── model
#             │                   │   ├── History.java
#             │                   │   ├── HistoryResAto.java
#             │                   │   ├── LoginResponseDTO.java
#             │                   │   ├── Provider.java
#             │                   │   ├── Role.java
#             │                   │   ├── User.java
#             │                   │   ├── UserDTO.java
#             │                   │   ├── UserOAuth2.java
#             │                   │   └── UserReqAto.java
#             │                   ├── parsers
#             │                   │   └── HTMLParser.java
#             │                   ├── repo
#             │                   │   ├── HistoryRepo.java
#             │                   │   ├── RoleRepo.java
#             │                   │   └── UserRepo.java
#             │                   ├── services
#             │                   │   ├── AuthenticationService.java
#             │                   │   ├── history
#             │                   │   │   ├── HistoryMapper.java
#             │                   │   │   └── HistoryService.java
#             │                   │   ├── OAuth2AuthenticationService.java
#             │                   │   ├── TokenService.java
#             │                   │   ├── UserOAuth2Service.java
#             │                   │   └── UserServiceImpl.java
#             │                   ├── utils
#             │                   │   ├── KeyGeneratorUtility.java
#             │                   │   └── RSAKeyProperties.java
#             │                   └── WebSummarizerApplication.java
#             └── resources
#                 ├── application.properties
#                 ├── env.properties
#                 ├── static
#                 │   ├── css
#                 │   │   └── custom.css
#                 │   ├── images
#                 │   │   └── safe-checkout.png
#                 │   └── js
#                 │       └── custom.js
#                 └── templates
#                     ├── api
#                     │   ├── newchat.html
#                     │   └── summary.html
#                     ├── error.html
#                     ├── fragments
#                     │   ├── message.html
#                     │   ├── meta.html
#                     │   └── shortlink.html
#                     ├── index.html
#                     └── user
#                         ├── account.html
#                         ├── cancel.html
#                         ├── code.html
#                         ├── login.html
#                         ├── pro.html
#                         ├── register.html
#                         ├── reset.html
#                         └── thankyou.html
#     """
#
#     # Run the chain
#     result = api_chain.invoke(
#         {"base_directory": base_directory, "project_structure": project_structure}
#     )
#     print(result)
