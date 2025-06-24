import csv
import logging
import os
import time
from datetime import datetime

from scripts.algorithm_4.agents.accessibility_evaluation_agent import AccessibilityEvaluationAgent
from scripts.algorithm_4.agents.documentation_evaluation_agent import DocumentationEvaluationAgent
from scripts.algorithm_4.agents.functionality_evaluation_agent import FunctionalityEvaluationAgent, \
    FunctionalityEvaluationResult
from scripts.algorithm_4.agents.keyword_evaluation_agent import KeywordEvaluationAgent
from scripts.algorithm_4.agents.repository_knowledge_graph_agent import RepositoryKnowledgeGraphAgent
from scripts.algorithm_4.agents.usability_evaluation_agent import UsabilityEvaluationAgent

logging.basicConfig(level=logging.INFO)


class AURA:
    def __init__(self, guideline_path, artifact_json_path, conference_name="ICSE 2025", criteria_csv_path=None):
        self.guideline_path = guideline_path
        self.artifact_json_path = artifact_json_path
        self.conference_name = conference_name
        self.criteria_csv_path = criteria_csv_path
        
        # Extract artifact name from the JSON file path
        self.artifact_name = self._extract_artifact_name(artifact_json_path)

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

    def _extract_artifact_name(self, artifact_json_path: str) -> str:
        """
        Extract artifact name from the JSON file path.
        Example: 'path/to/repo_name_analysis.json' -> 'repo_name'
        """
        try:
            # Get the filename from the path
            filename = os.path.basename(artifact_json_path)
            # Remove '_analysis.json' suffix
            if filename.endswith('_analysis.json'):
                artifact_name = filename[:-14]  # Remove '_analysis.json'
            else:
                # Fallback: remove '.json' extension
                artifact_name = os.path.splitext(filename)[0]
            
            logging.info(f"Extracted artifact name: {artifact_name}")
            return artifact_name
        except Exception as e:
            logging.warning(f"Could not extract artifact name from {artifact_json_path}: {e}")
            return "unknown_artifact"

    def evaluate(self, dimensions=None, verbose=True, include_keyword_eval=True, analysis_timing_data=None):
        # Initialize evaluation timing
        evaluation_timing = {
            "evaluation_start_time": datetime.now().isoformat(),
            "evaluation_end_time": None,
            "evaluation_duration_seconds": None,
            "documentation_evaluation_time": None,
            "accessibility_evaluation_time": None,
            "usability_evaluation_time": None,
            "functionality_evaluation_time": None,
            "keyword_evaluation_time": None
        }
        
        evaluation_start = time.time()
        
        results = {}
        dimensions = dimensions or ['documentation', 'usability', 'accessibility', 'functionality']

        # === Documentation ===
        doc_start = time.time()
        doc_result = self.doc_agent.evaluate(verbose=True)
        doc_end = time.time()
        evaluation_timing["documentation_evaluation_time"] = round(doc_end - doc_start, 2)
        
        if isinstance(doc_result, dict) and "error" in doc_result:
            results['documentation'] = doc_result["raw_output"]
        else:
            results['documentation'] = doc_result

        # === Accessibility ===
        access_start = time.time()
        access_result = self.access_agent.evaluate(verbose=True)
        access_end = time.time()
        evaluation_timing["accessibility_evaluation_time"] = round(access_end - access_start, 2)
        
        if isinstance(access_result, dict) and "error" in access_result:
            results['accessibility'] = access_result["raw_output"]
        else:
            results['accessibility'] = access_result

        # === Usability ===
        if 'usability' in dimensions:
            usability_start = time.time()
            usability_result = self.usability_agent.evaluate(verbose=True)
            usability_end = time.time()
            evaluation_timing["usability_evaluation_time"] = round(usability_end - usability_start, 2)
            
            if isinstance(usability_result, dict) and "error" in usability_result:
                results['usability'] = usability_result["raw_output"]
            else:
                results['usability'] = usability_result

        # === Functionality ===
        if 'functionality' in dimensions:
            func_start = time.time()
            func_result = self.func_agent.evaluate(verbose)
            func_end = time.time()
            evaluation_timing["functionality_evaluation_time"] = round(func_end - func_start, 2)
            
            if isinstance(func_result, dict) and "error" in func_result:
                results['functionality'] = func_result["raw_output"]
            elif isinstance(func_result, FunctionalityEvaluationResult):
                results['functionality'] = func_result
            else:
                # Fallback: assume raw output
                results['functionality'] = str(func_result)

        # === Save combined CSV ===
        csv_filename = f"{self.artifact_name}_full_evaluation.csv"
        csv_output_path = f"../../algo_outputs/algorithm_4_output/artifact_evals/{csv_filename}"
        
        # Combine timing data
        combined_timing = {
            "analysis_timing": analysis_timing_data,
            "evaluation_timing": evaluation_timing
        }
        
        self._save_all_to_csv(results, csv_output_path, combined_timing)
        logging.info(f"Saved evaluation results to: {csv_output_path}")
        
        # Store the CSV path for external access
        self.last_csv_path = csv_output_path

        # === Keyword Baseline Evaluation ===
        if include_keyword_eval and self.keyword_agent:
            try:
                keyword_start = time.time()
                keyword_results = self.keyword_agent.evaluate(verbose=verbose)
                keyword_end = time.time()
                evaluation_timing["keyword_evaluation_time"] = round(keyword_end - keyword_start, 2)
                
                results['keyword_baseline'] = keyword_results
                logging.info("Keyword-based evaluation completed")
            except Exception as e:
                logging.error(f"Error in keyword evaluation: {e}")
                results['keyword_baseline'] = {"error": str(e)}

        # End evaluation timing
        evaluation_end = time.time()
        evaluation_timing["evaluation_end_time"] = datetime.now().isoformat()
        evaluation_timing["evaluation_duration_seconds"] = round(evaluation_end - evaluation_start, 2)
        
        # Store timing data for external access
        self.last_evaluation_timing = evaluation_timing
        self.last_analysis_timing = analysis_timing_data
        
        logging.info(f"Evaluation timing: Total={evaluation_timing['evaluation_duration_seconds']}s")
        
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

    def get_timing_summary(self) -> dict:
        """
        Get a summary of timing data for both analysis and evaluation
        
        Returns:
            dict: Summary of timing information
        """
        summary = {
            "analysis_timing": getattr(self, 'last_analysis_timing', None),
            "evaluation_timing": getattr(self, 'last_evaluation_timing', None)
        }
        
        if summary["analysis_timing"] and summary["evaluation_timing"]:
            try:
                total_analysis = summary["analysis_timing"].get("analysis_duration_seconds", 0)
                total_evaluation = summary["evaluation_timing"].get("evaluation_duration_seconds", 0)
                
                # Convert to float and handle None values
                total_analysis = float(total_analysis) if total_analysis is not None else 0.0
                total_evaluation = float(total_evaluation) if total_evaluation is not None else 0.0
                
                summary["total_pipeline_time"] = total_analysis + total_evaluation
                summary["total_analysis_time"] = total_analysis
                summary["total_evaluation_time"] = total_evaluation
            except (TypeError, ValueError):
                summary["total_pipeline_time"] = 0.0
                summary["total_analysis_time"] = 0.0
                summary["total_evaluation_time"] = 0.0
        
        return summary

    def get_csv_file_path(self) -> str:
        """
        Get the path to the last generated CSV file.
        Returns the path to the artifact-specific CSV file.
        """
        if hasattr(self, 'last_csv_path'):
            return self.last_csv_path
        else:
            # Generate the expected path
            csv_filename = f"{self.artifact_name}_full_evaluation.csv"
            return f"../../algo_outputs/algorithm_4_output/artifact_evals/{csv_filename}"

    def _save_all_to_csv(self, results: dict, output_path: str, timing_data: dict = None):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Dimension", "Section/Criterion", "Score", "Justification"])

            # Write evaluation results
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

                elif dimension == "usability" and hasattr(result, "criterion_scores"):
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

                elif dimension == "functionality" and hasattr(result, "criterion_scores"):
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

            # Add timing data section
            if timing_data:
                writer.writerow([])  # Empty row for separation
                writer.writerow(["TIMING DATA", "", "", ""])
                
                # Analysis timing
                if timing_data.get("analysis_timing"):
                    writer.writerow(["Analysis Timing", "", "", ""])
                    analysis_timing = timing_data["analysis_timing"]
                    for key, value in analysis_timing.items():
                        if value is not None:
                            writer.writerow([f"analysis.{key}", "", "", value])
                
                # Evaluation timing
                if timing_data.get("evaluation_timing"):
                    writer.writerow(["Evaluation Timing", "", "", ""])
                    evaluation_timing = timing_data["evaluation_timing"]
                    for key, value in evaluation_timing.items():
                        if value is not None:
                            writer.writerow([f"evaluation.{key}", "", "", value])
                
                # Summary timing
                writer.writerow(["Timing Summary", "", "", ""])
                total_analysis_time = timing_data.get("analysis_timing", {}).get("analysis_duration_seconds", 0)
                total_evaluation_time = timing_data.get("evaluation_timing", {}).get("evaluation_duration_seconds", 0)
                
                # Convert to float and handle None values
                try:
                    total_analysis_time = float(total_analysis_time) if total_analysis_time is not None else 0.0
                    total_evaluation_time = float(total_evaluation_time) if total_evaluation_time is not None else 0.0
                    total_time = total_analysis_time + total_evaluation_time
                except (TypeError, ValueError):
                    total_analysis_time = 0.0
                    total_evaluation_time = 0.0
                    total_time = 0.0
                
                writer.writerow(["Total Analysis Time (seconds)", "", "", total_analysis_time])
                writer.writerow(["Total Evaluation Time (seconds)", "", "", total_evaluation_time])
                writer.writerow(["Total Pipeline Time (seconds)", "", "", total_time])
