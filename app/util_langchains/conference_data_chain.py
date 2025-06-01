import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()

cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure to alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database. Make sure to alias all statements that follow as with
statement (e.g. WITH c AS conference, r AS readme).
If you need to divide numbers, make sure to filter the denominator to be non-zero.

Examples:
# Which conference has the most 'must_have' items in its README?
MATCH (c:Conference)-[:HAS_REQUIREMENT]->(r:README)
RETURN c.name AS conference_name, SIZE(r.must_have) AS must_have_count
ORDER BY must_have_count DESC
LIMIT 1

# Which README section belongs to a specific conference (e.g., ICSE 2025)?
MATCH (c:Conference name: 'ICSE 2025')-[:HAS_REQUIREMENT]->(r:README)
RETURN r.must_have AS must_have, r.may_have AS may_have, r.bonus AS bonus

# Which badges does a specific conference (e.g., ICSE 2024) have?
MATCH (c:Conference name: 'ICSE 2024')-[:HAS_BADGE]->(b:BADGES)
RETURN b.has AS badges

# Which conferences have a specific license requirement?
MATCH (c:Conference)-[:HAS_LICENSE]->(l:LICENSE_REQUIREMENTS)
WHERE l.requirements CONTAINS 'open source'
RETURN c.name AS conference_name, l.requirements AS license_requirements

# What are the must-have items in README for a specific conference (e.g., ICSE 2025)?
MATCH (c:Conference name: 'ICSE 2025')-[:HAS_REQUIREMENT]->(r:README)
RETURN r.must_have AS must_have


String category values:
- README must_have items are any of: 'installation instructions', 'usage instructions', 'dependencies', 'contact information'
- LICENSE_REQUIREMENTS are typically related to specific licensing criteria such as 'open source', 'public availability'

Ensure to handle missing properties using IS NULL or IS NOT NULL.
Never return embedding properties in your queries. Always alias all follow-up
statements with a WITH clause (e.g. WITH r AS readme, b AS badges).
If you need to divide numbers, make sure to filter the denominator to be non-zero.

The question is:
{question}
"""

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a user's natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, assume the query
results are in units of days unless otherwise specified.

When names are provided in the query results, such as conference names
or README items, make sure to list them in a way that is not ambiguous.
For instance, 'ICSE 2025 and ICSE 2024' is a clear representation of the
two conference names. Ensure you handle lists of names or items correctly.

Never say you don't have the right information if there is data in
the query results. Make sure to show all the relevant query results
if you're asked.

Helpful Answer:
"""

QA_MODEL = os.getenv("QA_MODEL")
CYPHER_MODEL = os.getenv("CYPHER_MODEL")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

graph.refresh_schema()

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

# Initialize the GraphCypherQAChain with the necessary models and parameters
cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=CYPHER_MODEL, temperature=0),
    qa_llm=ChatOpenAI(model=QA_MODEL, temperature=0),
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
    graph=graph,  # Pass the graph object here
    allow_dangerous_requests=True  # Acknowledge the potential risks

)
