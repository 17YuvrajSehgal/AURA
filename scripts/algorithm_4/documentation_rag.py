import json
from typing import List

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


# --- Define Pydantic schema for README sections ---
class ReadmeSection(BaseModel):
    name: str = Field(description="Name of the required README section")
    description: str = Field(description="Brief explanation of what this section should contain")


class ReadmeSectionsList(BaseModel):
    sections: List[ReadmeSection] = Field(description="List of required README sections for the conference")


# --- Step 1: Load Conference Guidelines ---
with open('../../data/conference_guideline_texts/processed/13_icse_2025.md', 'r') as f:
    icse_guidelines = f.read()

# --- Step 2: Run Conference Guidelines RAG to get README sections (Structured) ---
parser = PydanticOutputParser(pydantic_object=ReadmeSectionsList)
guideline_prompt = PromptTemplate(
    template=(
        "Based on the following conference documentation guidelines, "
        "return a JSON object with a `sections` field (a list of required README sections). "
        "{format_instructions}\n"
        "Conference Guidelines:\n{guidelines}"
    ),
    input_variables=["guidelines"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
llm_gpt = OpenAI(temperature=0.0)
prompt_and_model = guideline_prompt | llm_gpt | parser
readme_sections_struct = prompt_and_model.invoke({"guidelines": icse_guidelines})

# --- Step 3: Load Artifact Documentation Files ---
with open('C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\ml-image-classifier_analysis.json', 'r') as f:
    artifact = json.load(f)
doc_files = artifact['documentation_files']

# Combine content lines into a single string per document
texts = ["\n".join(doc['content']) for doc in doc_files]

# --- Step 4: Prepare Vector DB for Retrieval ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
all_chunks = []
for text in texts:
    all_chunks.extend(splitter.split_text(text))
embeddings = OpenAIEmbeddings()
db = Chroma.from_texts(all_chunks, embeddings, persist_directory="chroma_index")
db.persist()

# --- Step 5: Dynamically Generate Artifact Evaluation Prompt ---
section_questions = ""
for section in readme_sections_struct.sections:
    section_questions += f"- Does the documentation include a '{section.name}' section? {section.description} Rate from 1-5 with justification.\n"

artifact_prompt = f"""
You are an expert artifact evaluator for ICSE 2025.
Evaluate the artifact's documentation according to these required sections:

{section_questions}

Provide a score and justification for each. Then give an overall documentation score and suggestions for improvement.
"""

# --- Step 6: Run Artifact Evaluation RAG ---
llm = OpenAI(temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    chain_type="stuff"
)
result = qa_chain({"query": artifact_prompt})
print(result['result'])
