from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState

from app.ai_models.ai_utils import open_ai_chat_model
from app.langchains.file_search_chains import files_chain
from app.utils.utils import read_file_peek

# Define memory and workflow
memory = MemorySaver()
workflow = StateGraph(state_schema=MessagesState)


# Define summarization logic
def summarize_data_files(state: MessagesState):
    system_prompt = (
        """You are an assistant creating a README section about data in a repository. Based on the provided 
        file samples, summarize the structure, type of data, and its potential use cases. Be concise but informative."""
    )
    system_message = SystemMessage(content=system_prompt)
    file_messages = [
        HumanMessage(content=f"Data file sample: {content}")
        for content in state["messages"]
    ]
    response = open_ai_chat_model.invoke([system_message] + file_messages)
    return {"messages": response}


# Add summarization to the workflow
workflow.add_node("data_summarizer", summarize_data_files)
workflow.add_edge(START, "data_summarizer")
app = workflow.compile(checkpointer=memory)

# Define Runnables
extract_data_file_paths = RunnableLambda(
    lambda inputs: files_chain.invoke(
        {
            "base_directory": inputs["base_directory"],
            "project_structure": inputs["project_structure"],
            "task": "identify data files for summarization",
        }
    )
)

peek_into_files = RunnableLambda(
    lambda file_paths: [
        read_file_peek(file_path.strip(), num_lines=25)
        for file_path in file_paths["text"].split(",")
        if file_path.strip()
    ]
)

generate_data_summary = RunnableLambda(
    lambda file_samples: app.invoke(
        {"messages": file_samples},
        config={"configurable": {"thread_id": "data-summary"}},
    )
)

format_output = RunnableLambda(
    lambda result: "\n\n".join(
        message.content
        for message in result["messages"]
        if isinstance(message, AIMessage)
    )
)

# Properly chain the steps
data_summary_chain = (
        extract_data_file_paths
        | peek_into_files
        | generate_data_summary
        | format_output
)

# Main for testing
if __name__ == "__main__":
    base_directory = "C:/workplace/ArtifactEvaluator/temp_dir_for_git/ArtifactEvaluator"
    project_structure = """
    
    Tree structure with files:
├── .idea
│   ├── .gitignore
│   ├── ArtifactEvaluator.iml
│   ├── csv-editor.xml
│   ├── inspectionProfiles
│   │   ├── profiles_settings.xml
│   │   └── Project_Default.xml
│   ├── misc.xml
│   ├── modules.xml
│   └── vcs.xml
├── ai_models
│   └── ai_utils.py
├── app
│   ├── exceptions.py
│   ├── langchains
│   │   ├── api_chain.py
│   │   ├── authors_chain.py
│   │   ├── chains.py
│   │   ├── conference_data_chain.py
│   │   ├── examples_chain.py
│   │   ├── file_search_chains.py
│   │   ├── overview_chain.py
│   │   ├── text_to_json_chain.py
│   │   └── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── repository.py
│   │   └── __init__.py
│   ├── routes
│   │   ├── routes.py
│   │   └── __init__.py
│   ├── services
│   │   ├── readme_generator.py
│   │   ├── repository_service.py
│   │   └── __init__.py
│   ├── utils
│   │   └── utils.py
│   └── __init__.py
├── app.py
├── data
│   ├── 1_raw_text_data_web_scrapping
│   │   ├── 10_SIGCOMM
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   └── acm.txt
│   │   │       └── raw.txt
│   │   ├── 11_ICSME
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   └── citable_code_guidelines.txt
│   │   │       └── raw.txt
│   │   ├── 12_TACAS
│   │   │   └── 2025
│   │   │       ├── links
│   │   │       │   ├── ealps.txt
│   │   │       │   └── tacas_Artifact_Eval_VM.txt
│   │   │       └── raw.txt
│   │   ├── 13_The_Web_Conference_ACM
│   │   │   └── 2024
│   │   │       └── raw.txt
│   │   ├── 14_QEST_FORMATS
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   └── acm.txt
│   │   │       └── raw.txt
│   │   ├── 15_DSN
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   ├── HOWTO.txt
│   │   │       │   ├── ieee.txt
│   │   │       │   └── tips_for_authors.txt
│   │   │       └── raw.txt
│   │   ├── 16_ISCA
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   ├── acm.txt
│   │   │       │   ├── reviewing_guidelines.txt
│   │   │       │   └── submission_guidelines.txt
│   │   │       └── raw.txt
│   │   ├── 17_ISSRE
│   │   │   └── 2024
│   │   │       └── raw.txt
│   │   ├── 18_SIAM
│   │   │   └── 2024
│   │   │       └── raw.txt
│   │   ├── 19_FAST
│   │   │   ├── links
│   │   │   │   ├── fast24_ae_appendix_template.tex
│   │   │   │   ├── HOWTO.txt
│   │   │   │   └── tips_for_authors.txt
│   │   │   └── raw.txt
│   │   ├── 1_ICSE
│   │   │   └── 2025
│   │   │       ├── links
│   │   │       │   ├── acm.txt
│   │   │       │   ├── open_science_policy.txt
│   │   │       │   └── open_source_initiative_licenses.txt
│   │   │       └── raw.txt
│   │   ├── 20_MIDDLEWARE
│   │   │   └── 2024
│   │   │       └── raw.txt
│   │   ├── 2_Asiacrypt
│   │   │   └── 2024
│   │   │       └── raw.txt
│   │   ├── 3_SEAMS
│   │   │   └── 2025
│   │   │       ├── links
│   │   │       │   └── github_with_zenodo.txt
│   │   │       └── raw.txt
│   │   ├── 4_RE
│   │   │   └── 2024
│   │   │       └── raw.txt
│   │   ├── 5_FormaliSE
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   └── eapls.txt
│   │   │       └── raw.txt
│   │   ├── 6_HPCA
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   ├── IEEE_badges.txt
│   │   │       │   ├── reviewing_guidelines.txt
│   │   │       │   └── submission_guidelines.txt
│   │   │       └── raw.txt
│   │   ├── 7_USENIX
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   ├── artifact_eval.txt
│   │   │       │   ├── HOWTO_for_AEC_Submitters.txt
│   │   │       │   └── tips_for_authors.txt
│   │   │       └── raw.txt
│   │   ├── 8_WOOT
│   │   │   └── 2024
│   │   │       └── raw.txt
│   │   └── 9_ECRTS
│   │       └── 2024
│   │           ├── links
│   │           │   ├── CC0_1.0.txt
│   │           │   ├── CC_BY_4.0.txt
│   │           │   ├── darts-authors-v2021.1.3.zip
│   │           │       ├── CHANGELOG.md
│   │           │       ├── LICENSE.md
│   │           │       ├── cc-by.pdf
│   │           │       ├── darts-logo-bw.pdf
│   │           │       ├── darts-v2021-sample-article.bib
│   │           │       ├── darts-v2021-sample-article.pdf
│   │           │       ├── darts-v2021-sample-article.tex
│   │           │       ├── darts-v2021.cls
│   │           │       └── orcid.pdf
│   │           │   └── DARTS.txt
│   │           └── raw.txt
│   ├── 2_cleaned_scraped_txt_data_from_gpt
│   │   ├── 10_SIGCOMM.txt
│   │   ├── 11_ICSME.txt
│   │   ├── 12_TACAS.txt
│   │   ├── 13_The_Web_Conference_ACM.txt
│   │   ├── 14_QEST_FORMATS.txt
│   │   ├── 15_DSN.txt
│   │   ├── 16_ISCA.txt
│   │   ├── 17_ISSRE.txt
│   │   ├── 18_SIAM.txt
│   │   ├── 19_FAST.txt
│   │   ├── 1_ICSE.txt
│   │   ├── 20_MIDDLEWARE.txt
│   │   ├── 2_Asiacrypt.txt
│   │   ├── 3_SEAMS.txt
│   │   ├── 4_RE.txt
│   │   ├── 5_FormaliSE.txt
│   │   ├── 6_HPCA.txt
│   │   ├── 7_USENIX.txt
│   │   ├── 8_WOOT.txt
│   │   └── 9_ECRTS.txt
│   ├── 3_json_data_with_structures_and_raw_text
│   │   ├── 10_SIGCOMM
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   └── acm.txt
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 11_ICSME
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   └── citable_code_guidelines.txt
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 12_TACAS
│   │   │   └── 2025
│   │   │       ├── links
│   │   │       │   ├── ealps.txt
│   │   │       │   └── tacas_Artifact_Eval_VM.txt
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 13_The_Web_Conference_ACM
│   │   │   └── 2024
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 14_QEST_FORMATS
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   └── acm.txt
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 18_SIAM
│   │   │   └── 2024
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 1_ICSE
│   │   │   └── 2025
│   │   │       ├── links
│   │   │       │   ├── acm.txt
│   │   │       │   ├── open_science_policy.txt
│   │   │       │   └── open_source_initiative_licenses.txt
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 2_Asiacrypt
│   │   │   └── 2024
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 3_SEAMS
│   │   │   └── 2025
│   │   │       ├── links
│   │   │       │   └── github_with_zenodo.txt
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 4_RE
│   │   │   └── 2024
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 5_FormaliSE
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   └── eapls.txt
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   ├── 6_HPCA
│   │   │   └── 2024
│   │   │       ├── links
│   │   │       │   ├── IEEE_badges.txt
│   │   │       │   ├── reviewing_guidelines.txt
│   │   │       │   └── submission_guidelines.txt
│   │   │       ├── raw.txt
│   │   │       └── structure.json
│   │   └── 8_WOOT
│   │       └── 2024
│   │           ├── raw.txt
│   │           └── structure.json
│   ├── 4_merged_json_structures
│   │   ├── merged.json
│   │   └── merged_important_readme_must_have_cleaned.json
│   ├── acm_bib_data
│   │   ├── 1_available
│   │   │   └── available_data_merged_2039_references.bib
│   │   ├── 2_artifact_evaluated_and_functional
│   │   │   ├── evaluated_and_functional_0_to_1000.bib
│   │   │   ├── evaluated_and_functional_1001_to_1022.bib
│   │   │   └── evaluated_and_functional_data_merged_1022_references.bib
│   │   ├── 3_artifacts_evaluated_and_reusable
│   │   │   ├── evaluated_and_reusable_0_to_1000.bib
│   │   │   ├── evaluated_and_reusable_1001_to_1021.bib
│   │   │   └── evaluated_and_reusable_data_merged_1021.bib
│   │   └── 4_result_reproduced
│   │       └── reproduced_all_418.bib
│   ├── conference-name-year-links.xlsx
│   └── manual_conference_text_collection
│       ├── raw_web_text_acm_conferences
│       │   ├── acm_conferences
│       │   │   ├── 10_hri_2025
│       │   │   │   └── main.txt
│       │   │   ├── 11_icfp_2024
│       │   │   │   └── main.txt
│       │   │   ├── 12_icpe_2024
│       │   │   │   └── main.txt
│       │   │   ├── 13_icse_2025
│       │   │   │   └── main.txt
│       │   │   ├── 14_isca_2024
│       │   │   │   └── main.txt
│       │   │   ├── 15_issta_2024
│       │   │   │   └── main.txt
│       │   │   ├── 16_kdd_2025
│       │   │   │   └── main.txt
│       │   │   ├── 17_micro_2022
│       │   │   │   └── main.txt
│       │   │   ├── 18_middleware_2024
│       │   │   │   └── main.txt
│       │   │   ├── 19_mobicom_2024
│       │   │   │   └── main.txt
│       │   │   ├── 1_ase_2024
│       │   │   │   └── main.txt
│       │   │   ├── 20_mobisys_2024
│       │   │   │   └── main.txt
│       │   │   ├── 21_mod_2024
│       │   │   │   └── main.txt
│       │   │   ├── 22_pact_2024
│       │   │   │   ├── main.txt
│       │   │   │   └── support_1_ctuning_submittion_guidelines.txt
│       │   │   ├── 23_pldi_2024
│       │   │   │   └── main.txt
│       │   │   ├── 24_ppopp_2025
│       │   │   │   ├── main.txt
│       │   │   │   └── support_1_ctuning_faq.txt
│       │   │   ├── 25_thewebconf_2024
│       │   │   │   └── main.txt
│       │   │   ├── 26_websci_2023
│       │   │   │   ├── main.txt
│       │   │   │   └── support_1_reame_sample.txt
│       │   │   ├── 2_asia_ccs_2024
│       │   │   │   ├── main.txt
│       │   │   │   ├── support_1_tips_for_author.txt
│       │   │   │   └── support_2_HOWTO_for_AEC_Submitters
│       │   │   ├── 3_asplos_2024
│       │   │   │   └── main.txt
│       │   │   ├── 4_cf_2025
│       │   │   │   ├── main.txt
│       │   │   │   └── support_1_ctuning_checklist.txt
│       │   │   ├── 5_cgo_2025
│       │   │   │   ├── main.txt
│       │   │   │   └── support_1_ctuning_submittion_guidelines.txt
│       │   │   ├── 6_chi_2024
│       │   │   │   └── main.txt
│       │   │   ├── 7_conext_2024
│       │   │   │   ├── main.txt
│       │   │   │   └── support_1_ctuning_checklist.txt
│       │   │   └── 9_fse_2024
│       │   │       └── main.txt
│       │   ├── delete_this.txt
│       │   ├── important_links
│       │   └── non_acm_conferences
│       │       ├── 1_tacas_2024
│       │       │   └── main.txt
│       │       ├── 2_asiacrypt_2024
│       │       │   └── main.txt
│       │       ├── 3_seams_2024
│       │       │   └── main.txt
│       │       ├── 4_re_2024
│       │       │   └── main.txt
│       │       ├── 5_formalise_2024
│       │       │   ├── main.txt
│       │       │   └── support_1_eapls_badges
│       │       ├── 6_hpca_2025
│       │       │   ├── main.txt
│       │       │   └── support_1_ctuning_submission
│       │       ├── 7_icsme_2024
│       │       │   └── main.txt
│       │       └── 8_qest_2023
│       │           └── main.txt
│       └── structured_raw_web_text
│           └── 1_ase_2024.json
├── evaluators
│   └── vector_generators.py
├── exceptions
│   └── artifact_evaluation_exceptions.py
├── json_to_neo4j
│   └── data_transfer.py
├── json_utils.py
├── knowledge_graph_processors
│   └── neo4j_connection.py
├── logo.png
├── main.py
├── prompt_templates
│   ├── prompt_pydantic_models.py
│   └── prompt_templates.py
├── repository_reader
│   ├── repository_reader.py
│   └── utils.py
├── repo_evaluation_system
│   └── evaluator.py
├── requirements.txt
├── temp.py
└── vector_stores
    └── text_to_embeddings.py
    """
    result = data_summary_chain.invoke({"base_directory": base_directory, "project_structure": project_structure})
    print("Generated Data Summary:")
    print(result)
