from langchain.chains import LLMChain
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

from app.ai_models.ai_utils import open_ai_chat_model

# 1. Create the parser and fetch format instructions
parser = CommaSeparatedListOutputParser()
format_instructions = parser.get_format_instructions()

# 2. Create a PromptTemplate that includes a "task" placeholder
prompt_template = PromptTemplate(
    template="""
Given the following base directory:
{base_directory}

And the following project structure:
{project_structure}

Identify the full paths of files that are important for {task}.
Return them as a comma separated list.
Do not add any special character, or any path out of context.
These paths are used to read the files later, so the file paths you return should be readable and very accurate.

{format_instructions}
""",
    # We now expect three user-provided variables:
    input_variables=["base_directory", "project_structure", "task"],
    # The format instructions are still passed in as partial variables:
    partial_variables={"format_instructions": format_instructions},
)

# 3. Create the chain
files_chain = LLMChain(
    llm=open_ai_chat_model,
    prompt=prompt_template,
)
