import logging

from agents.accessibility_evaluation_agent import AccessibilityEvaluationAgent
from agents.documentation_evaluation_agent import DocumentationEvaluationAgent
from agents.functionality_evaluation_agent import FunctionalityEvaluationAgent
from agents.usability_evaluation_agent import UsabilityEvaluationAgent
from agents.keyword_evaluation_agent import KeywordEvaluationAgent
from agents.repository_knowledge_graph_agent import RepositoryKnowledgeGraphAgent
import csv
import os

logging.basicConfig(level=logging.INFO)


class AURA:
    def __init__(self, guideline_path, artifact_json_path, conference_name="ICSE 2025", criteria_csv_path=None):
        self.guideline_path = guideline_path
        self.artifact_json_path = artifact_json_path
        self.conference_name = conference_name
        self.criteria_csv_path = criteria_csv_path


        # Initialize keyword-based agent first (if criteria CSV is provided)
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

        # Initialize LLM-based agents with keyword agent reference
        self.doc_agent = DocumentationEvaluationAgent(
            guideline_path, artifact_json_path, conference_name, keyword_agent=self.keyword_agent, kg_agent=kg_agent)
        self.usability_agent = UsabilityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name, keyword_agent=self.keyword_agent)
        self.access_agent = AccessibilityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name, keyword_agent=self.keyword_agent)
        self.func_agent = FunctionalityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name, keyword_agent=self.keyword_agent, kg_agent=kg_agent)

    def evaluate(self, dimensions=None, verbose=False, include_keyword_eval=True):
        """dimensions: list of dimension strings, e.g. ['documentation', 'usability']"""
        results = {}
        dimensions = dimensions or ['documentation', 'usability', 'accessibility', 'functionality']

        # Run LLM-based evaluations (now grounded with keyword evidence)
        doc_result = self.doc_agent.evaluate(verbose)
        if isinstance(doc_result, dict) and "error" in doc_result:
            results['documentation'] = doc_result["raw_output"]
        else:
            results['documentation'] = {
                "overall_score": doc_result.overall_score,
                "sections": [{s.section: s.score} for s in doc_result.section_scores],
                "suggestions": doc_result.suggestions
            }



        if 'usability' in dimensions:
            results['usability'] = "0 bad usability"#self.usability_agent.evaluate(verbose)
        if 'accessibility' in dimensions:
            results['accessibility'] = "0 poort accessibility"#self.access_agent.evaluate(verbose)
        if 'functionality' in dimensions:
            results['functionality'] = "0 poor functionality"#self.func_agent.evaluate(verbose)

        self._save_documentation_to_csv(doc_result,                                            output_path="../../algo_outputs/algorithm_4_output/artifact_evals/output.csv")

        # Run keyword-based evaluation if available and requested
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
        """Returns a short summary for all dimensions."""
        results = self.evaluate(verbose=verbose, include_keyword_eval=include_keyword_eval)
        
        summary_parts = []
        
        # LLM-based evaluations
        for dim in ['documentation', 'usability', 'accessibility', 'functionality']:
            if dim in results:
                summary_parts.append(f"{dim.capitalize()}:\n{results[dim]}")
        
        # Keyword-based evaluation
        if 'keyword_baseline' in results:
            keyword_result = results['keyword_baseline']
            if 'error' not in keyword_result:
                summary_parts.append(f"Keyword Baseline:\n{keyword_result['summary']}")
            else:
                summary_parts.append(f"Keyword Baseline: Error - {keyword_result['error']}")
        
        return "\n\n".join(summary_parts)

    def compare_evaluations(self, verbose=False):
        """Compare LLM-based vs keyword-based evaluations"""
        if not self.keyword_agent:
            return {"error": "Keyword agent not available"}
        
        # Get both evaluations
        llm_results = self.evaluate(include_keyword_eval=False, verbose=verbose)
        keyword_results = self.keyword_agent.evaluate(verbose=verbose)
        
        comparison = {
            "llm_evaluations": llm_results,
            "keyword_evaluation": keyword_results,
            "comparison_notes": self._generate_comparison_notes(llm_results, keyword_results)
        }
        
        return comparison

    def _generate_comparison_notes(self, llm_results, keyword_results):
        """Generate notes comparing the two evaluation approaches"""
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
        
        # Add dimension-specific comparisons
        if 'keyword_baseline' in llm_results:
            keyword_dims = keyword_results['dimensions']
            for dim in keyword_dims:
                notes.append(f"{dim['dimension'].capitalize()}: {dim['weighted_score']:.2f} (raw: {dim['raw_score']})")
        
        return "\n".join(notes)

    def get_grounded_evaluation(self, dimension: str, verbose: bool = True):
        """Get a specific dimension evaluation with keyword grounding"""
        if not self.keyword_agent:
            return {"error": "Keyword agent not available for grounding"}
        
        # Get keyword evidence first
        keyword_results = self.keyword_agent.evaluate(verbose=False)
        dimension_evidence = None
        
        for dim in keyword_results.get('dimensions', []):
            if dim['dimension'].lower() == dimension.lower():
                dimension_evidence = dim
                break
        
        # Run LLM evaluation with grounding
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

    def _save_documentation_to_csv(self, doc_result, output_path):


        # Ensure the directory exists if any
        output_dir = os.path.dirname(output_path)
        if output_dir:  # only create directory if non-empty
            os.makedirs(output_dir, exist_ok=True)

        # Only save if doc_result is not a dict (i.e., is a DocumentationEvaluationResult)
        if isinstance(doc_result, dict):
            # Optionally, log or print the error
            logging.info(f"doc_result is of type: {type(doc_result)}")
            return

        with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Section", "Score", "Justification"])  # Header row
            for section_score in doc_result.section_scores:
                writer.writerow([
                    section_score.section,
                    section_score.score,
                    section_score.justification
                ])
            writer.writerow([])  # Empty row
            writer.writerow(["Overall Score", doc_result.overall_score])
            if doc_result.suggestions:
                writer.writerow(["Suggestions", doc_result.suggestions])

