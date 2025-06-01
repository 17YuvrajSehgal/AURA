from typing import List, Dict

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from app.ai_models.ai_utils import open_ai_chat_model


# Define a structured output parser to extract key factors and descriptions
class KeyFactorOutputParser(BaseOutputParser):
    def parse(self, text: str) -> List[Dict[str, str]]:
        """
        Parse the output into a list of key factors and descriptions.
        Each factor is represented as a dictionary with 'factor' and 'description'.
        """
        items = text.strip().split("\n")
        key_factors = []
        for item in items:
            if ":" in item:
                factor, description = item.split(":", 1)
                key_factors.append({"factor": factor.strip(), "description": description.strip()})
        return key_factors


# Instantiate the parser
parser = KeyFactorOutputParser()

# Define the prompt template with placeholders
prompt_template = PromptTemplate(
    template="""
Given the following guidelines for artifact submission:

{guidelines}

Extract the key factors that the Artifact Evaluation Committee uses to evaluate the artifacts. 
For each factor, provide a brief description of its role in the evaluation.

Return the results in the following format:
Name of Factor 1: Description of factor 1.
Name of Factor 2: Description of factor 2.
...

Ensure the descriptions are concise and informative.
""",
    input_variables=["guidelines"],
)

# Create the chain using individual Runnables
prompt_runnable = RunnableLambda(lambda input: prompt_template.format(guidelines=input["guidelines"]))
parser_runnable = RunnableLambda(lambda input: parser.parse(input.content if hasattr(input, 'content') else input))

artifact_evaluation_chain = prompt_runnable | open_ai_chat_model | parser_runnable


# Usage example
def main():
    guidelines_text = """
    1. Reproducibility: The artifact should provide all necessary details to reproduce the results in the associated paper.
    2. Documentation: The artifact must be well-documented with clear instructions on how to use it.
    3. Relevance: The artifact should directly support the claims and results presented in the paper.
    4. Usability: The artifact must be user-friendly and easy to use.
    5. Performance: The artifact should demonstrate efficiency and robustness.
    """

    # Run the chain
    result = artifact_evaluation_chain.invoke({"guidelines": guidelines_text})

    # Display the results
    print("Extracted Key Factors:")
    for factor in result:
        print(f"- {factor['factor']}: {factor['description']}")


if __name__ == "__main__":
    main()
