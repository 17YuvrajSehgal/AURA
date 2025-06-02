import json
import logging
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# === Setup ===
# Required inputs
# S = repository_structure
# C = code_files
# L = license_info
# Ctx = conference_context

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

llm = ChatOpenAI(
    temperature=0.2,
    model="gpt-4",
)


# === High-level Content Extractors ===
def extract_project_overview(code_files, structure):
    return "Project overview generated from structure and code context."


def extract_api_documentation(code_files, structure):
    return "Auto-generated API documentation from code comments or functions."


def summarize_data_files(structure):
    return "Summary of data directories (e.g., training/testing datasets)."


def extract_authors(structure, code_files, licenses):
    return "Author info inferred from code comments or commit metadata."


# === Required Section Logic ===
def get_required_sections(conference_context):
    return ["Overview", "Installation", "Usage", "License", "Citation", "Dataset", "Contributing"]


# === Prompt Generator ===
def generate_section_prompt(section, structure, code_files, licenses):
    sample_code = "\n\n".join([f['content'][:500] for f in code_files[:2]])
    license_text = licenses[0]['content'][:500] if licenses else "No license file provided."
    tree = structure["file_structure"]

    return f"""Generate a README section titled '{section}' using the following:
- File structure:\n{tree}
- Sample code:\n{sample_code}
- License:\n{license_text}
Make it suitable for a research artifact in an academic conference.
"""


# === LLM Invocation ===
def generate_section_content(section, structure, code_files, licenses):
    prompt_text = generate_section_prompt(section, structure, code_files, licenses)
    prompt = PromptTemplate.from_template("Section: {section}\n\n{context}")
    context = f"{prompt_text}\n\nCode Files: {[c['content'][:500] for c in code_files[:1]]}"
    chain = prompt | llm
    return chain.invoke({"section": section, "context": context})


# === Orchestrator ===
def generate_readme(structure, code_files, licenses, conference_context):
    logging.info("Generating README content...")

    overview = extract_project_overview(code_files, structure)
    api_docs = extract_api_documentation(code_files, structure)
    data_summary = summarize_data_files(structure)
    authors = extract_authors(structure, code_files, licenses)

    required_sections = get_required_sections(conference_context)
    section_contents = {}

    for section in required_sections:
        try:
            logging.info(f"Generating section: {section}")
            section_contents[section] = generate_section_content(section, structure, code_files, licenses)
        except Exception as e:
            logging.error(f"Failed to generate section '{section}': {e}")
            section_contents[section] = f"[Failed to generate {section}]"

    completeness = len(section_contents) / len(required_sections)
    readability = 0.9  # placeholder for future scoring
    quality = (completeness + readability) / 2

    if quality < 0.8:
        for section in required_sections:
            if section not in section_contents:
                section_contents[section] = f"[Placeholder for {section}]"

    readme = "\n\n".join(f"## {k}\n{v}" for k, v in section_contents.items())
    logging.info("README generation complete.")
    return readme


# === File Loader ===
def load_repository_data(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["repository_structure"], data["code_files"], data["license_files"]
    except Exception as e:
        logging.error(f"Error loading repository data from {filepath}: {e}")
        raise


# === MAIN EXECUTION ===
if __name__ == "__main__":
    logging.info("Starting README generation...")

    repo_json_path = "../../data/algorithm_2_output/ml-image-classifier_analysis.json"
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

    structure_raw, code_files, licenses = load_repository_data(repo_json_path)
    structure = {
        "file_structure": manual_tree,
        "files": structure_raw
    }

    conference_context = "ASE_2024"

    readme_content = generate_readme(structure, code_files, licenses, conference_context)

    output_dir = "../../data/algorithm_4_output"
    os.makedirs(output_dir, exist_ok=True)
    readme_path = os.path.join(output_dir, "generated_README.md")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logging.info(f"README saved to: {readme_path}")
