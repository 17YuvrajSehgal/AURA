import logging

from app.readme_langchains.langchains.file_search_chains import files_chain
from langchain.chains import LLMChain

from app.readme_langchains.api_chain import api_chain
from app.readme_langchains.authors_chain import authors_chain
from app.readme_langchains.data_chain import data_summary_chain
from app.readme_langchains.overview_chain import overview_chain

logger = logging.getLogger(__name__)


def generate_readme_content(base_directory: str, project_structure: str, repository_authors: list) -> str:
    """
    High-level function that uses ReadmeGenerator to orchestrate
    all the sub-chains and produce a final README.
    """
    generator = ReadmeGenerator(
        files_chain=files_chain,
        api_chain=api_chain,
        overview_chain=overview_chain,
        authors_chain=authors_chain,
        data_summary_chain=data_summary_chain  # Pass the data summary chain
    )

    tasks = {
        "files_task": "Identify main config or code files for the README",
        "api_task": "Generate thorough API endpoint documentation",
        "overview_task": "Generate a project overview section",
        "data_task": "Summarize the data files in the repository",
        "readme_summary_prompt": "Write a helpful README in markdown format",
    }

    # Generate the README
    final_readme = generator.generate_readme(
        base_directory=base_directory,
        project_structure=project_structure,
        repository_authors=repository_authors,  # Include authors in the tasks
        tasks=tasks
    )
    return final_readme


class ReadmeGenerator:
    def __init__(self, files_chain: LLMChain, api_chain: LLMChain, overview_chain: LLMChain,
                 authors_chain: LLMChain, data_summary_chain: LLMChain):
        """
        Initialize the ReadmeGenerator with the required LLMChains.

        :param files_chain: Chain for file analysis tasks.
        :param api_chain: Chain for API documentation tasks.
        :param overview_chain: Chain for generating project overview.
        :param authors_chain: Chain for generating contributors section.
        :param data_summary_chain: Chain for summarizing data files.
        """
        self.files_chain = files_chain
        self.api_chain = api_chain
        self.overview_chain = overview_chain
        self.authors_chain = authors_chain
        self.data_summary_chain = data_summary_chain

    def generate_readme(
            self,
            base_directory: str,
            project_structure: str,
            repository_authors: list,
            tasks: dict
    ) -> str:
        """
        Generates the README content by combining multiple sections.

        :param base_directory: Path to the base directory of the project.
        :param project_structure: String representation of the project tree.
        :param repository_authors: List of contributors with details.
        :param tasks: Dictionary of tasks for each section.
        :return: Final README content as a string.
        """
        # Step 1: Identify files
        logger.info("Calling files_chain to extract important file paths...")
        file_paths = self.files_chain.invoke({
            "base_directory": base_directory,
            "project_structure": project_structure,
            "task": tasks.get("files_task", "Identify main files for the README"),
        })
        logger.debug(f"Extracted file paths: {file_paths}")

        # Step 2: Generate API documentation
        logger.info("Calling api_chain to generate API documentation...")
        api_docs = self.api_chain.invoke({
            "base_directory": base_directory,
            "project_structure": project_structure,
            "task": tasks.get("api_task", "Generate API endpoint documentation"),
        })
        logger.debug(f"Generated API documentation: {api_docs}")

        # Step 3: Generate Overview section
        logger.info("Calling overview_chain to generate project overview...")
        overview_section = self.overview_chain.invoke({
            "base_directory": base_directory,
            "project_structure": project_structure,
            "task": tasks.get("overview_task", "Generate project overview documentation"),
        })
        logger.debug(f"Generated project overview: {overview_section}")

        # Step 4: Summarize data files using data_summary_chain
        logger.info("Calling data_summary_chain to generate data summary...")
        data_summary = self.data_summary_chain.invoke({
            "base_directory": base_directory,
            "project_structure": project_structure,
            "task": tasks.get("data_task", "Summarize the data files in the repository"),
        })
        logger.debug(f"Generated data summary: {data_summary}")

        # Step 5: Generate contributors section using authors_chain
        logger.info("Calling authors_chain to generate contributors section...")
        # contributors_section = self.authors_chain.invoke({
        #     "repository_authors": repository_authors,  # Pass authors list to chain
        #     "task": tasks.get("authors_task", "Generate contributors section"),
        # })
        contributors_section = "Yuvraj Sehgal 17yuvraj.sehgal@gmail.com"
        logger.debug(f"Generated contributors section: {contributors_section}")

        # Step 6: Combine sections into a final README
        logger.info("Generating final README content...")
        readme_content = (
            f"# Project Overview\n\n{overview_section}\n\n"
            f"## API Documentation\n\n{api_docs}\n\n"
            f"## Data Summary\n\n{data_summary}\n\n"
            f"## Contributors\n\n{contributors_section}\n"
        )

        return readme_content
