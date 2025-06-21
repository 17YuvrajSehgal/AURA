import json
import logging
import math
import re
import pandas as pd
from typing import List, Dict, Optional

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("keyword_evaluation_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KeywordEvaluationAgent:
    def __init__(
            self,
            criteria_csv_path: str,
            artifact_json_path: str,
            conference_name: str = "ICSE 2025",
    ):
        self.criteria_csv_path = criteria_csv_path
        self.artifact_json_path = artifact_json_path
        self.conference_name = conference_name

        logger.info(f"Initializing KeywordEvaluationAgent for {conference_name}")
        self.criteria_df = self._load_criteria()
        self.artifact = self._load_artifact()

    def _load_criteria(self):
        """Load evaluation criteria from CSV file"""
        logger.info(f"Loading evaluation criteria from: {self.criteria_csv_path}")
        try:
            criteria_df = pd.read_csv(self.criteria_csv_path)
            logger.info(f"Loaded {len(criteria_df)} evaluation dimensions")
            return criteria_df
        except Exception as e:
            logger.error(f"Error loading criteria CSV: {e}")
            raise

    def _load_artifact(self):
        """Load artifact JSON file"""
        logger.info(f"Loading artifact JSON from: {self.artifact_json_path}")
        try:
            with open(self.artifact_json_path, "r", encoding="utf-8") as f:
                artifact = json.load(f)
            logger.info("Artifact JSON loaded successfully")
            return artifact
        except Exception as e:
            logger.error(f"Error loading artifact JSON: {e}")
            raise

    def _get_all_texts_for_artifact(self):
        """Extract all text content from artifact files"""
        def flatten_content(files):
            texts = []
            for file in files:
                content = file.get("content", "")
                if isinstance(content, list):
                    # Join list items into a single string
                    content = "\n".join(str(line) for line in content)
                elif not isinstance(content, str):
                    content = str(content)
                if content.strip():
                    texts.append(content)
            return texts

        doc_texts = flatten_content(self.artifact.get("documentation_files", []))
        code_texts = flatten_content(self.artifact.get("code_files", []))
        license_texts = flatten_content(self.artifact.get("license_files", []))
        
        # Include tree structure as text for structure analysis
        tree_text = ""
        if "tree_structure" in self.artifact and isinstance(self.artifact["tree_structure"], list):
            tree_text = "\n".join(str(line) for line in self.artifact["tree_structure"])
        tree_texts = [tree_text] if tree_text else []

        all_texts = doc_texts + code_texts + license_texts + tree_texts
        logger.info(f"Extracted {len(all_texts)} text segments from artifact")
        return all_texts

    def _keyword_count(self, text: str, keywords: str) -> int:
        """Count keyword occurrences in text using word boundaries"""
        text_lower = text.lower()
        count = 0
        keyword_list = [k.strip() for k in keywords.split(",")]
        
        for kw in keyword_list:
            if kw:  # Skip empty keywords
                pattern = re.escape(kw.lower())
                count += len(re.findall(r"\b" + pattern + r"\b", text_lower))
        return count

    def _evaluate_artifact_against_criteria(self):
        """Evaluate artifact using keyword-based scoring"""
        artifact_texts = self._get_all_texts_for_artifact()
        dim_results = []
        total_weighted_score = 0

        for idx, row in self.criteria_df.iterrows():
            dim = row["dimension"]
            keywords = str(row["keywords"])
            norm_weight = float(row["normalized_weight"])

            # Calculate raw score across all text segments
            raw_score = sum(self._keyword_count(text, keywords) for text in artifact_texts)
            
            # Apply log scaling to handle large variations
            scaled_score = math.log(1 + raw_score)
            weighted_score = scaled_score * norm_weight
            
            dim_results.append({
                "dimension": dim,
                "raw_score": raw_score,
                "log_scaled": scaled_score,
                "weighted_score": weighted_score,
                "weight": norm_weight,
                "keywords_found": self._get_found_keywords(artifact_texts, keywords)
            })
            total_weighted_score += weighted_score

        return dim_results, total_weighted_score

    def _get_found_keywords(self, texts: List[str], keywords: str) -> List[str]:
        """Get list of keywords that were actually found in the texts"""
        found_keywords = []
        keyword_list = [k.strip() for k in keywords.split(",")]
        
        for kw in keyword_list:
            if kw:
                for text in texts:
                    if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text.lower()):
                        found_keywords.append(kw)
                        break  # Found in at least one text, no need to check others
        
        return found_keywords

    def evaluate(self, verbose: bool = True) -> Dict:
        """Main evaluation method that returns detailed results"""
        logger.info("Starting keyword-based artifact evaluation")
        
        dim_results, total_score = self._evaluate_artifact_against_criteria()
        
        # Create detailed report
        report = {
            "overall_score": total_score,
            "dimensions": dim_results,
            "summary": self._generate_summary(dim_results, total_score)
        }
        
        if verbose:
            self._print_report(report)
        
        logger.info(f"Evaluation complete. Overall score: {total_score:.2f}")
        return report

    def _generate_summary(self, dim_results: List[Dict], total_score: float) -> str:
        """Generate a human-readable summary of the evaluation"""
        summary_lines = [
            f"Keyword-Based Artifact Evaluation Report for {self.conference_name}",
            "=" * 60,
            "",
            f"Overall Artifact Score: {total_score:.2f}",
            "",
            "Dimension Breakdown:"
        ]
        
        for res in dim_results:
            summary_lines.append(
                f"  {res['dimension'].capitalize()}: "
                f"Raw={res['raw_score']} "
                f"Weighted={res['weighted_score']:.2f} "
                f"(Weight: {res['weight']:.3f})"
            )
        
        summary_lines.extend([
            "",
            "Top Keywords Found:"
        ])
        
        # Show top keywords found across dimensions
        all_found_keywords = []
        for res in dim_results:
            all_found_keywords.extend(res['keywords_found'])
        
        if all_found_keywords:
            keyword_counts = {}
            for kw in all_found_keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for kw, count in top_keywords:
                summary_lines.append(f"  {kw}: {count} occurrences")
        else:
            summary_lines.append("  No keywords found")
        
        return "\n".join(summary_lines)

    def _print_report(self, report: Dict):
        """Print the evaluation report to console"""
        print("\n" + report["summary"])
        print("\n" + "=" * 60)

    def get_criteria(self) -> pd.DataFrame:
        """Return the loaded criteria dataframe"""
        return self.criteria_df

    def get_artifact_info(self) -> Dict:
        """Return basic artifact information"""
        return {
            "total_files": len(self.artifact.get("repository_structure", [])),
            "doc_files": len(self.artifact.get("documentation_files", [])),
            "code_files": len(self.artifact.get("code_files", [])),
            "license_files": len(self.artifact.get("license_files", [])),
            "has_tree_structure": "tree_structure" in self.artifact
        }


# Example usage (commented out)
# agent = KeywordEvaluationAgent(
#     criteria_csv_path="../../../algo_outputs/algorithm_1_output/algorithm_1_artifact_evaluation_criteria.csv",
#     artifact_json_path="../../../algo_outputs/algorithm_2_output/ml-image-classifier_analysis.json",
#     conference_name="ICSE 2025"
# )
# results = agent.evaluate(verbose=True)