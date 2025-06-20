import json

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

# --- Step 1: Load JSON and Get Documentation Files ---
with open('C:\\workplace\\AURA\\algo_outputs\\algorithm_2_output\\ml-image-classifier_analysis.json', 'r') as f:
    artifact = json.load(f)
doc_files = artifact['documentation_files']

# Combine content lines into a single string per document
texts = ["\n".join(doc['content']) for doc in doc_files]

# --- Load Conference Guidelines ---
with open('../../data/conference_guideline_texts/processed/13_icse_2025.md', 'r') as f:
    icse_guidelines = f.read()
# Alternatively, parse as JSON/YAML for more complex pipelines

# --- Prepare Text Chunks for Vector DB ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
all_chunks = []
for text in texts:
    all_chunks.extend(splitter.split_text(text))

embeddings = OpenAIEmbeddings()
db = Chroma.from_texts(all_chunks, embeddings, persist_directory="chroma_index")
db.persist()

# --- Step 4: Build RAG QA Chain ---
llm = OpenAI(temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    chain_type="stuff"
)

# --- Guideline-Aware Prompt ---
prompt = f"""
You are an expert artifact evaluator for ICSE 2025.
Here are the ICSE documentation requirements:
{icse_guidelines}

Using the retrieved documentation below, answer:
1. Is there a README and is it complete?
2. Is the artifact's purpose explained?
3. Is the provenance/source clear?
4. Are setup and install instructions provided and clear?
5. Are usage instructions/examples present?

Score each point from 1-5 with justification. Provide an overall score, and suggestions for improvement.
"""

result = qa_chain({"query": prompt})
print(result['result'])
