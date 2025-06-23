import csv
import logging
import os

from agents.accessibility_evaluation_agent import AccessibilityEvaluationAgent
from agents.documentation_evaluation_agent import DocumentationEvaluationAgent
from agents.functionality_evaluation_agent import FunctionalityEvaluationAgent
from agents.keyword_evaluation_agent import KeywordEvaluationAgent
from agents.repository_knowledge_graph_agent import RepositoryKnowledgeGraphAgent
from agents.usability_evaluation_agent import UsabilityEvaluationAgent

logging.basicConfig(level=logging.INFO)


class AURA:
    def __init__(self, guideline_path, artifact_json_path, conference_name="ICSE 2025", criteria_csv_path=None):
        self.guideline_path = guideline_path
        self.artifact_json_path = artifact_json_path
        self.conference_name = conference_name
        self.criteria_csv_path = criteria_csv_path

        self.keyword_agent = None
        if criteria_csv_path:
            try:
                self.keyword_agent = KeywordEvaluationAgent(
                    criteria_csv_path, artifact_json_path, conference_name)
                logging.info("Keyword evaluation agent initialized successfully")
            except Exception as e:
                logging.warning(f"Could not initialize keyword agent: {e}")

        kg_agent = RepositoryKnowledgeGraphAgent(
            artifact_json_path=self.artifact_json_path,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="12345678",
            clear_existing=True,
        )

        self.doc_agent = DocumentationEvaluationAgent(
            guideline_path, artifact_json_path, conference_name,
            keyword_agent=self.keyword_agent, kg_agent=kg_agent)
        self.usability_agent = UsabilityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name,
            keyword_agent=self.keyword_agent)
        self.access_agent = AccessibilityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name,
            keyword_agent=self.keyword_agent)
        self.func_agent = FunctionalityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name,
            keyword_agent=self.keyword_agent, kg_agent=kg_agent)

    def evaluate(self, dimensions=None, verbose=False, include_keyword_eval=True):
        results = {}
        dimensions = dimensions or ['documentation', 'usability', 'accessibility', 'functionality']

        doc_result = self.doc_agent.evaluate(verbose)
        if isinstance(doc_result, dict) and "error" in doc_result:
            results['documentation'] = doc_result["raw_output"]
        else:
            results['documentation'] = doc_result

        access_result = self.access_agent.evaluate(verbose)
        if isinstance(access_result, dict) and "error" in access_result:
            results['accessibility'] = access_result["raw_output"]
        else:
            results['accessibility'] = access_result

        if 'usability' in dimensions:
            results['usability'] = "0 bad usability"  # self.usability_agent.evaluate(verbose)

        if 'functionality' in dimensions:
            results['functionality'] = "0 poor functionality"  # self.func_agent.evaluate(verbose)

        # Save all results to a single CSV
        self._save_all_to_csv(results,
                              output_path="../../algo_outputs/algorithm_4_output/artifact_evals/full_evaluation.csv")

        if include_keyword_eval and self.keyword_agent:
            try:
                keyword_results = self.keyword_agent.evaluate(verbose=verbose)
                results['keyword_baseline'] = keyword_results
                logging.info("Keyword-based evaluation completed")
            except Exception as e:
                logging.error(f"Error in keyword evaluation: {e}")
                results['keyword_baseline'] = {"error": str(e)}

        return results

    def evaluate_summary(self, verbose=False, include_keyword_eval=True):
        results = self.evaluate(verbose=verbose, include_keyword_eval=include_keyword_eval)
        summary_parts = []

        for dim in ['documentation', 'usability', 'accessibility', 'functionality']:
            if dim in results:
                summary_parts.append(f"{dim.capitalize()}:\n{results[dim]}")

        if 'keyword_baseline' in results:
            keyword_result = results['keyword_baseline']
            if 'error' not in keyword_result:
                summary_parts.append(f"Keyword Baseline:\n{keyword_result['summary']}")
            else:
                summary_parts.append(f"Keyword Baseline: Error - {keyword_result['error']}")

        return "\n\n".join(summary_parts)

    def compare_evaluations(self, verbose=False):
        if not self.keyword_agent:
            return {"error": "Keyword agent not available"}

        llm_results = self.evaluate(include_keyword_eval=False, verbose=verbose)
        keyword_results = self.keyword_agent.evaluate(verbose=verbose)

        return {
            "llm_evaluations": llm_results,
            "keyword_evaluation": keyword_results,
            "comparison_notes": self._generate_comparison_notes(llm_results, keyword_results)
        }

    def _generate_comparison_notes(self, llm_results, keyword_results):
        notes = [
            "Evaluation Method Comparison:",
            "=" * 40,
            "",
            "LLM-based Evaluation (Grounded with Keyword Evidence):",
            "- Qualitative assessment with detailed reasoning",
            "- Context-aware analysis grounded in keyword evidence",
            "- Provides specific improvement suggestions",
            "- Uses keyword scores to prevent hallucination",
            "",
            "Keyword-based Evaluation:",
            "- Quantitative scoring based on keyword presence",
            "- Objective and reproducible",
            "- Provides numerical baseline",
            "- May miss semantic meaning",
            "",
            f"Keyword Overall Score: {keyword_results['overall_score']:.2f}",
            ""
        ]

        if 'keyword_baseline' in llm_results:
            for dim in keyword_results.get('dimensions', []):
                notes.append(f"{dim['dimension'].capitalize()}: {dim['weighted_score']:.2f} (raw: {dim['raw_score']})")

        return "\n".join(notes)

    def get_grounded_evaluation(self, dimension: str, verbose: bool = True):
        if not self.keyword_agent:
            return {"error": "Keyword agent not available for grounding"}

        keyword_results = self.keyword_agent.evaluate(verbose=False)
        dimension_evidence = next(
            (dim for dim in keyword_results.get('dimensions', []) if dim['dimension'].lower() == dimension.lower()),
            None
        )

        if dimension == 'documentation':
            result = self.doc_agent.evaluate(verbose)
        elif dimension == 'usability':
            result = self.usability_agent.evaluate(verbose)
        elif dimension == 'accessibility':
            result = self.access_agent.evaluate(verbose)
        elif dimension == 'functionality':
            result = self.func_agent.evaluate(verbose)
        else:
            return {"error": f"Unknown dimension: {dimension}"}

        return {
            "llm_evaluation": result,
            "keyword_evidence": dimension_evidence,
            "grounding_info": f"LLM evaluation was grounded with keyword evidence: {dimension_evidence['keywords_found'] if dimension_evidence else 'No evidence found'}"
        }

    def _save_all_to_csv(self, results: dict, output_path: str):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Dimension", "Section/Criterion", "Score", "Justification"])

            for dimension, result in results.items():
                if isinstance(result, dict) and "error" in result:
                    continue

                if dimension == "documentation" and hasattr(result, "section_scores"):
                    for section in result.section_scores:
                        writer.writerow([
                            dimension,
                            section.section,
                            section.score,
                            section.justification
                        ])
                    writer.writerow([dimension, "Overall Score", result.overall_score, ""])
                    if result.suggestions:
                        writer.writerow([dimension, "Suggestions", "", result.suggestions])

                elif dimension == "accessibility" and hasattr(result, "criterion_scores"):
                    for crit in result.criterion_scores:
                        writer.writerow([
                            dimension,
                            crit.criterion,
                            crit.score,
                            crit.justification
                        ])
                    writer.writerow([dimension, "Overall Score", result.overall_score, ""])
                    if result.suggestions:
                        writer.writerow([dimension, "Suggestions", "", result.suggestions])
