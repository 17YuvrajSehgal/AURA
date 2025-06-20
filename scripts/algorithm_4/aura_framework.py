# aura_framework.py

import logging

from agents.accessibility_evaluation_agent import AccessibilityEvaluationAgent
from agents.documentation_evaluation_agent import DocumentationEvaluationAgent
from agents.functionality_evaluation_agent import FunctionalityEvaluationAgent
from agents.usability_evaluation_agent import UsabilityEvaluationAgent

logging.basicConfig(level=logging.INFO)


class AURA:
    def __init__(self, guideline_path, artifact_json_path, conference_name="ICSE 2025"):
        self.guideline_path = guideline_path
        self.artifact_json_path = artifact_json_path
        self.conference_name = conference_name

        # Initialize agents
        self.doc_agent = DocumentationEvaluationAgent(
            guideline_path, artifact_json_path, conference_name)
        self.usability_agent = UsabilityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name)
        self.access_agent = AccessibilityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name)
        self.func_agent = FunctionalityEvaluationAgent(
            guideline_path, artifact_json_path, conference_name)

    def evaluate(self, dimensions=None, verbose=False):
        """dimensions: list of dimension strings, e.g. ['documentation', 'usability']"""
        results = {}
        dimensions = dimensions or ['documentation', 'usability', 'accessibility', 'functionality']

        if 'documentation' in dimensions:
            results['documentation'] = self.doc_agent.evaluate(verbose)
        if 'usability' in dimensions:
            results['usability'] = self.usability_agent.evaluate(verbose)
        if 'accessibility' in dimensions:
            results['accessibility'] = self.access_agent.evaluate(verbose)
        if 'functionality' in dimensions:
            results['functionality'] = self.func_agent.evaluate(verbose)
        return results

    def evaluate_summary(self, verbose=False):
        """Returns a short summary for all dimensions."""
        results = self.evaluate(verbose=verbose)
        summary = "\n\n".join([f"{dim.capitalize()}:\n{results[dim]}" for dim in results])
        return summary
