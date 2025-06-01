from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

from app.ai_models.ai_utils import open_ai_chat_model

llm = open_ai_chat_model
output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
print("format_instructions")
print(format_instructions)

prompt_template = PromptTemplate(
    template="List five {object} \n {format_instructions}",
    input_variables=["object"],
    partial_variables={"format_instructions": format_instructions},
)
query = prompt_template.format(object="chocolate ice cream")
print("query")
print(query)

# Invoke the LLM. Notice we are using `.invoke()` instead of `__call__`:
res = llm.invoke(query)
print("res.type:", res.type)
print("res.content:", res.content)

# Pass the message content to the parser
parsed_data = output_parser.parse(res.content)
print("parsed_data:", parsed_data)
