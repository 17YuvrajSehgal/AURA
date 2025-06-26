import json
import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class LLMEvaluator:
    def __init__(self, openai_api_key: str = None, model_name: str = "gpt-4", temperature: float = 0.2):
        if openai_api_key is None:
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY env var.")

        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=temperature
        )

    def evaluate_dimension(self, dimension: str, rule_score, justification, evidence, context):
        """
        Generic LLM evaluation for a given dimension.
        """
        prompt = ChatPromptTemplate.from_template(
            f"""
            You are an expert software artifact evaluator for ICSE 2025.
            The rule-based system gave the following {dimension} score: {{rule_score}}
            Justification: {{justification}}
            Evidence: {{evidence}}
            Here is relevant artifact content:
            {{context}}

            Please:
            1. Check if the rule-based evaluation missed any important evidence for {dimension}.
            2. If so, adjust the score (0.0-1.0) and provide a revised justification.
            3. List any additional evidence you found.

            Respond in JSON:
            {{
                "revised_score": float,
                "revised_justification": str,
                "additional_evidence": [str]
            }}
            """.replace('{', '{{').replace('}', '}}').replace('{{rule_score}}', '{rule_score}').replace(
                '{{justification}}', '{justification}').replace('{{evidence}}', '{evidence}').replace('{{context}}',
                                                                                                      '{context}')
        )
        chain = prompt | self.llm
        result = chain.invoke({
            "rule_score": rule_score,
            "justification": justification,
            "evidence": evidence,
            "context": context[:4000] if context else ""
        })
        # Parse the LLM's JSON output
        try:
            return json.loads(result.content)
        except Exception:
            # If LLM output is not valid JSON, return the original result
            return {
                "revised_score": rule_score,
                "revised_justification": justification + " (LLM output not parsed)",
                "additional_evidence": []
            }

    # Optionally, add specialized methods for each dimension if you want custom prompts
