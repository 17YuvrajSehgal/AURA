import json
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Required inputs
# S = repository_structure
# C = code_files
# L = license_info
# Ctx = conference_context

load_dotenv()

llm = ChatOpenAI(
    temperature=0.2,
    model="gpt-4",
)


# Step 1: High-level extractions
def ExtractProjectOverview(C, S):
    return "Project overview generated from structure and code context."


def ExtractAPIDocumentation(C, S):
    return "Auto-generated API documentation from code comments or functions."


def SummarizeDataFiles(S):
    return "Summary of data directories (e.g., training/testing datasets)."


def ExtractAuthors(S, C, L):
    return "Author info inferred from code comments or commit metadata."


# Step 2: Determine required sections based on a conference
def GetRequiredSections(Ctx):
    return ["Overview", "Installation", "Usage", "License", "Citation", "Dataset", "Contributing"]


# Step 3: Section generation
def GenerateSectionPrompt(section, S, C, L):
    sample_code = "\n\n".join([f['content'][:500] for f in C[:2]])  # short
    license_text = L[0]['content'][:500] if L else "No license file provided."
    tree = S["file_structure"]

    return f"""Generate a README section titled '{section}' using the following:
- File structure:\n{tree}
- Sample code:\n{sample_code}
- License:\n{license_text}
Make it suitable for a research artifact in an academic conference.
"""


def GenerateSectionContent(section, S, C, L):
    prompt_text = GenerateSectionPrompt(section, S, C, L)
    prompt = PromptTemplate.from_template("Section: {section}\n\n{context}")

    context = f"{prompt_text}\n\nCode Files: {[c['content'][:500] for c in C[:1]]}\n"  # limited sample

    chain = prompt | llm
    return chain.invoke({"section": section, "context": context})


# Step 4: Orchestrator
def GenerateREADME(S, C, L, Ctx):
    project_overview = ExtractProjectOverview(C, S)
    api_docs = ExtractAPIDocumentation(C, S)
    data_summary = SummarizeDataFiles(S)
    authors = ExtractAuthors(S, C, L)

    required_sections = GetRequiredSections(Ctx)
    section_contents = {}

    for section in required_sections:
        content = GenerateSectionContent(section, S, C, L)
        section_contents[section] = content

    # Optional: Completeness & readability logic (placeholders)
    completeness = len(section_contents) / len(required_sections)
    readability = 0.9  # Stub value
    quality = (completeness + readability) / 2

    if quality < 0.8:
        for section in required_sections:
            if section not in section_contents:
                section_contents[section] = f"[Placeholder for {section}]"

    # Final formatting
    readme = "\n\n".join(f"## {k}\n{v}" for k, v in section_contents.items())
    return readme


# # Assume these are loaded from your analyzer:
# S = [...]  # repo_structure
# C = [...]  # list of code file content
# L = [...]  # LICENSE info
# Ctx = "ASE_2024"  # or dynamically fetched from parsed guideline JSON


def load_repository_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["repository_structure"], data["code_files"], data["license_files"]


manual_tree = """
ml-image-classifier
├── README.md
├── src
│   ├── train.py
│   ├── model.py
│   ├── evaluate.py
├── data
│   ├── training/
│   └── testing/
├── tests
│   └── test_model.py
"""

repo_json_path = "../../data/algorithm_2_output/ml-image-classifier_analysis.json"

# Load inputs
S_raw, C, L = load_repository_data(repo_json_path)

# Build S from raw: could just be tree string + list of files
S = {
    "file_structure": manual_tree,
    "files": S_raw
}

# Dummy context for now
Ctx = "ASE_2024"

# Run README generation
readme = GenerateREADME(S, C, L, Ctx)

# Save it
output_dir = "../../data/algorithm_4_output"
os.makedirs(output_dir, exist_ok=True)  # create a directory if missing

readme_path = os.path.join(output_dir, "generated_README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

print(f"README saved to: {readme_path}")

readme_content = GenerateREADME(S, C, L, Ctx)
with open("generated_README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
