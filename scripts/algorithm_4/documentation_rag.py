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

# --- Step 5: Chain-of-Thought Prompt for Evaluation ---
prompt = """
You are an artifact evaluation agent. Using the following documentation, answer these questions:
1. Is there a main README and is it clear?
2. Are installation steps explained?
3. Are dependencies listed?
4. Are usage examples/tutorials provided?
5. Is there support/contact info?

For each, give a 1-5 score with justification. Then provide an overall score and summary.
"""

result = qa_chain({"query": prompt})

print(result['result'])
