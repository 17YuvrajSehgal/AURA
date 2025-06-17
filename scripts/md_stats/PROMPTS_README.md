# Conference-Ready Artifact README Generation Prompts

Below are prompt templates for generating each README section.  
They are empirically grounded in the most common structures, content, and quality signals found in thousands of accepted repositories.

---

## 1. **Installation**

**Prompt:**

> Write a detailed **Installation** section for this artifact.  
> Follow these guidelines:
>
> - Begin with a clear heading ("Installation" or "Setup").
> - List all dependencies, environment requirements, and required package versions (e.g., Python version, OS, pip/conda/npm, etc.).
> - Provide step-by-step, numbered instructions for setting up the environment and installing the software.
> - Include code blocks for installation commands (`pip install ...`, `conda install ...`, `npm install ...`, etc.).
> - Mention any files like `requirements.txt` or `environment.yml`.
> - If platform-specific (Windows/Linux/Mac), provide separate notes or subsections.
> - If available, link to official documentation, installation scripts, or Dockerfiles.
> - Ensure clarity for both beginners and experienced users.

---

## 2. **Usage**

**Prompt:**

> Write a comprehensive **Usage** section for this artifact.
>
> - Start with a heading ("Usage" or "Getting Started").
> - Explain how to run the main code or application, referencing example commands with expected input/output.
> - Provide one or more code block examples (e.g., `python train.py`, `python3 run.sh`, etc.).
> - Include sample input/output, screenshots, or links to demo data when available.
> - Highlight advanced options, environment variables, or configuration files (e.g., how to set parameters, use `config.yaml`, etc.).
> - Mention any common usage scenarios or best practices.
> - If possible, provide a quickstart guide or a summary table of commands.

---

## 3. **Requirements/Dependencies**

**Prompt:**

> Write a clear **Requirements** section that specifies:
>
> - All software and hardware prerequisites.
> - Python/OS/CPU/GPU requirements.
> - Libraries and their versions (`requirements.txt`, `environment.yml`, etc.).
> - Links to additional downloads if needed (datasets, models, etc.).
> - Any special environment setup instructions (e.g., activating conda environments, setting environment variables).

---

## 4. **License and Attribution**

**Prompt:**

> Write a prominent **License** section.
>
> - Clearly state the license type (e.g., "This project is licensed under the MIT License").
> - Include a short summary of the license’s permissions/restrictions.
> - If available, provide a link or full license text.
> - Acknowledge contributors, maintainers, or authors in an "Authors/Contributors" subsection.
> - Optionally, add a "How to cite" or "Citation" block (with DOI, Zenodo, or BibTeX entry).

---

## 5. **Troubleshooting and FAQ**

**Prompt:**

> Add a **Troubleshooting / FAQ** section.
>
> - List common errors or known issues and their solutions.
> - Provide tips on debugging, bug reporting, and where to seek support (e.g., "Report issues at the GitHub Issue Tracker").
> - Add a "Frequently Asked Questions" subsection if relevant.

---

## 6. **Reproducibility and Validation**

**Prompt:**

> Write a dedicated **Reproducibility** section.
>
> - Explain how to reproduce the main results of the associated paper.
> - List all scripts, datasets, and step-by-step instructions for replication.
> - Include notes on how to set random seeds, replicate the environment (e.g., Docker/Colab/Binder usage), and validate outputs.
> - If possible, provide expected output, accuracy/benchmark numbers, or instructions for automated tests.

---

## 7. **References and Related Work**

**Prompt:**

> Write a **References** or **Related Work** section.
>
> - List relevant papers, datasets, codebases, or projects.
> - Provide links to arXiv, Zenodo, or official URLs.
> - Mention the original research paper if applicable.

---

## 8. **Examples and Demos**

**Prompt:**

> Write an **Examples** section.
>
> - Give code examples or walkthroughs for typical usage.
> - Show input, output, or screenshots where possible.
> - Highlight sample datasets or demo files if available.

---

## 9. **Project Structure/Overview**

**Prompt:**

> Write an **Overview** or **Project Structure** section.
>
> - Summarize the purpose of the artifact and its main components.
> - Optionally, provide a directory/file tree, explaining the role of each file.
> - Briefly state what makes the project novel or important.

---

## 10. **Contribution Guidelines**

**Prompt:**

> Write a **Contributing** section.
>
> - Explain how others can contribute, report issues, or suggest improvements.
> - Mention any code of conduct, review process, or contact points.

---

## **General Quality Signals (to emphasize in all prompts)**

- Use lists and tables for clarity (common in high-quality docs).
- Include diagrams, images, or architecture sketches if they help understanding.
- Provide both outbound links (for external resources) and internal links (to other parts of the documentation).
- Use readable, concise, and technical language (target Flesch-Kincaid ≈ 16).
- Explicitly mention all licenses, attributions, and citation instructions.

---

**Note:** These prompts are derived from the most prevalent practices and structures in over 12,000 accepted repositories, maximizing alignment with real-world artifact documentation and reviewer expectations.
