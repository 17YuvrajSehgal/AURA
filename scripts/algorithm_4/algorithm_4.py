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


# Step 2: Determine required sections based on conference
def GetRequiredSections(Ctx):
    return ["Overview", "Installation", "Usage", "License", "Citation", "Dataset", "Contributing"]


# Step 3: Section generation
def GenerateSectionPrompt(section, S, C, L):
    return f"Generate a '{section}' section for the README using repository structure and code context."


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


# Assume these are loaded from your analyzer:
S = [...]  # repo_structure
C = [...]  # list of code file content
L = [...]  # LICENSE info
Ctx = "ASE_2024"  # or dynamically fetched from parsed guideline JSON

readme_content = GenerateREADME(S, C, L, Ctx)
with open("generated_README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
