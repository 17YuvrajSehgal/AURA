import logging

from agents.accessibility_evaluation_agent import AccessibilityEvaluationAgent
from agents.documentation_evaluation_agent import DocumentationEvaluationAgent
from agents.functionality_evaluation_agent import FunctionalityEvaluationAgent
from agents.usability_evaluation_agent import UsabilityEvaluationAgent
from agents.keyword_evaluation_agent import KeywordEvaluationAgent

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

        # Initialize LLM-based agents with keyword agent reference
        self.doc_agent = DocumentationEvaluationAgent(
            guideline_path, artifact_json_path, conference_name, keyword_agent=self.keyword_agent)
        self.usability_agent = UsabilityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name, keyword_agent=self.keyword_agent)
        self.access_agent = AccessibilityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name, keyword_agent=self.keyword_agent)
        self.func_agent = FunctionalityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name, keyword_agent=self.keyword_agent)

    def evaluate(self, dimensions=None, verbose=False, include_keyword_eval=True):
        """dimensions: list of dimension strings, e.g. ['documentation', 'usability']"""
        results = {}
        dimensions = dimensions or ['documentation', 'usability', 'accessibility', 'functionality']

        # Run LLM-based evaluations (now grounded with keyword evidence)
        if 'documentation' in dimensions:
            results['documentation'] = self.doc_agent.evaluate(verbose)
        if 'usability' in dimensions:
            results['usability'] = self.usability_agent.evaluate(verbose)
        if 'accessibility' in dimensions:
            results['accessibility'] = self.access_agent.evaluate(verbose)
        if 'functionality' in dimensions:
            results['functionality'] = self.func_agent.evaluate(verbose)
        
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
