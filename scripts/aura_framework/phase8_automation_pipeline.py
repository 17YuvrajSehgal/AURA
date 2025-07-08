"""
🚀 Phase 8: End-to-End Automation Pipeline
Goal: Orchestrate all phases into a complete automated evaluation workflow.

Features:
- Complete pipeline orchestration (Phases 1-7)
- Batch processing with parallel execution
- Automated report generation and export
- Configuration-driven evaluation workflows
- Error handling and recovery mechanisms
- Performance monitoring and optimization
- CLI interface for automated deployment
"""

import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import time
import traceback
import yaml
import csv

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Local imports
from config import config
from phase1_artifact_preprocessing import ArtifactPreprocessor, ProcessedArtifact
from phase2_knowledge_graph import KnowledgeGraphBuilder
from phase3_vector_embeddings import VectorEmbeddingEngine
from phase4_pattern_analysis import PatternAnalysisEngine
from phase5_genai_agents import MultiAgentEvaluationOrchestrator, EvaluationDimension
from phase6_rubric_scoring import RubricScoringFramework, AcceptanceCategory
from phase7_feedback_visualization import ArtifactVisualizationDashboard

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the automation pipeline"""
    # Input/Output
    artifacts_directory: str = "data/acm_bib_to_json_data"
    output_directory: str = "pipeline_outputs"
    export_formats: List[str] = field(default_factory=lambda: ["JSON", "CSV", "PDF"])
    
    # Processing
    max_artifacts: Optional[int] = None
    max_workers: int = 4
    batch_size: int = 10
    timeout_per_artifact: int = 300  # seconds
    
    # Framework components
    use_neo4j: bool = False
    use_vector_embeddings: bool = True
    use_pattern_analysis: bool = True
    apply_conference_penalties: bool = True
    
    # Target conferences
    target_conferences: List[str] = field(default_factory=lambda: ["ICSE", "FSE", "ASE", "CHI"])
    
    # Reporting
    generate_individual_reports: bool = True
    generate_summary_report: bool = True
    generate_visualizations: bool = True
    
    # Performance
    enable_monitoring: bool = True
    enable_caching: bool = True
    log_level: str = "INFO"


@dataclass
class PipelineStats:
    """Statistics from pipeline execution"""
    start_time: str
    end_time: Optional[str] = None
    total_artifacts: int = 0
    processed_artifacts: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    total_conferences: int = 0
    avg_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    errors: List[str] = field(default_factory=list)
    phase_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class ArtifactEvaluationResult:
    """Complete evaluation result for a single artifact"""
    artifact_id: str
    status: str  # 'success', 'failed', 'timeout'
    processing_time: float
    timestamp: str
    
    # Phase results
    preprocessing_result: Optional[ProcessedArtifact] = None
    multi_agent_result: Optional[Any] = None
    scoring_results: Dict[str, Any] = field(default_factory=dict)
    comparative_analysis: Optional[Any] = None
    
    # Outputs
    reports_generated: List[str] = field(default_factory=list)
    export_paths: List[str] = field(default_factory=list)
    
    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class AuraAutomationPipeline:
    """Complete automation pipeline for AURA framework"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the automation pipeline
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config or PipelineConfig()
        self.stats = PipelineStats(start_time=datetime.now().isoformat())
        
        # Setup directories
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.components = {}
        self.results = {}
        self.cache = {} if self.config.enable_caching else None
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"AURA Automation Pipeline initialized - Output: {self.output_dir}")

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Create log directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure file handler
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(log_level)

    async def run_complete_pipeline(self) -> PipelineStats:
        """
        Run the complete AURA evaluation pipeline
        
        Returns:
            Pipeline execution statistics
        """
        logger.info("🚀 Starting AURA Complete Pipeline")
        start_time = time.time()
        
        try:
            # Phase 0: Initialize framework components
            await self._initialize_components()
            
            # Phase 1: Load and preprocess artifacts
            artifacts = await self._phase1_preprocessing()
            self.stats.total_artifacts = len(artifacts)
            
            # Phase 2: Build knowledge graph
            await self._phase2_knowledge_graph(artifacts)
            
            # Phase 3: Generate vector embeddings
            if self.config.use_vector_embeddings:
                await self._phase3_vector_embeddings(artifacts)
            
            # Phase 4: Analyze patterns
            if self.config.use_pattern_analysis:
                await self._phase4_pattern_analysis()
            
            # Phase 5: Multi-agent evaluation (batch)
            evaluation_results = await self._phase5_multi_agent_evaluation(artifacts)
            
            # Phase 6: Rubric-based scoring
            scoring_results = await self._phase6_rubric_scoring(evaluation_results)
            
            # Phase 7: Generate reports and visualizations
            await self._phase7_reporting_and_visualization(scoring_results)
            
            # Phase 8: Finalize and export
            await self._phase8_finalization()
            
            # Calculate final statistics
            self.stats.end_time = datetime.now().isoformat()
            self.stats.avg_processing_time = (time.time() - start_time) / max(1, self.stats.processed_artifacts)
            
            logger.info(f"✅ Pipeline completed successfully in {time.time() - start_time:.2f}s")
            return self.stats
            
        except Exception as e:
            self.stats.errors.append(f"Pipeline error: {str(e)}")
            logger.error(f"❌ Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            raise
        
        finally:
            # Cleanup and performance monitoring
            if self.config.enable_monitoring:
                await self._collect_performance_metrics()

    async def _initialize_components(self):
        """Initialize all framework components"""
        logger.info("Initializing framework components...")
        
        # Track component initialization time
        component_start = time.time()
        
        try:
            # Initialize preprocessor
            self.components['preprocessor'] = ArtifactPreprocessor(
                data_directory=self.config.artifacts_directory,
                output_directory=str(self.output_dir / "processed_artifacts"),
                max_workers=self.config.max_workers
            )
            
            # Initialize knowledge graph builder
            self.components['kg_builder'] = KnowledgeGraphBuilder(
                use_neo4j=self.config.use_neo4j
            )
            
            # Initialize vector embedding engine
            if self.config.use_vector_embeddings:
                self.components['vector_engine'] = VectorEmbeddingEngine(
                    storage_directory=str(self.output_dir / "vector_embeddings")
                )
            
            # Initialize pattern analysis engine
            if self.config.use_pattern_analysis:
                self.components['pattern_engine'] = PatternAnalysisEngine(
                    self.components['kg_builder'],
                    self.components.get('vector_engine')
                )
            
            # Initialize multi-agent evaluator
            self.components['evaluator'] = MultiAgentEvaluationOrchestrator(
                kg_builder=self.components['kg_builder'],
                vector_engine=self.components.get('vector_engine'),
                pattern_engine=self.components.get('pattern_engine')
            )
            
            # Initialize scoring framework
            self.components['scoring'] = RubricScoringFramework(
                rubrics_directory=str(self.output_dir / "rubrics")
            )
            
            # Initialize visualization dashboard (for report generation)
            self.components['dashboard'] = ArtifactVisualizationDashboard(
                artifacts_directory=self.config.artifacts_directory,
                use_neo4j=self.config.use_neo4j
            )
            
            initialization_time = time.time() - component_start
            self.stats.phase_timings['initialization'] = initialization_time
            
            logger.info(f"✅ All components initialized in {initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise

    async def _phase1_preprocessing(self) -> List[str]:
        """Phase 1: Artifact preprocessing"""
        logger.info("📋 Phase 1: Preprocessing artifacts...")
        phase_start = time.time()
        
        # Find artifact files
        artifacts_dir = Path(self.config.artifacts_directory)
        artifact_files = list(artifacts_dir.glob("*.json"))
        
        if self.config.max_artifacts:
            artifact_files = artifact_files[:self.config.max_artifacts]
        
        logger.info(f"Found {len(artifact_files)} artifacts to process")
        
        # Process artifacts
        preprocessing_results = self.components['preprocessor'].process_artifacts_batch(
            [str(f) for f in artifact_files],
            max_artifacts=self.config.max_artifacts
        )
        
        self.stats.processed_artifacts = preprocessing_results['processed_successfully']
        self.stats.failed_evaluations += preprocessing_results['failed_processing']
        
        phase_time = time.time() - phase_start
        self.stats.phase_timings['preprocessing'] = phase_time
        
        logger.info(f"✅ Preprocessing completed in {phase_time:.2f}s")
        return [str(f) for f in artifact_files]

    async def _phase2_knowledge_graph(self, artifacts: List[str]):
        """Phase 2: Knowledge graph construction"""
        logger.info("📐 Phase 2: Building knowledge graph...")
        phase_start = time.time()
        
        # Build knowledge graph
        kg_stats = self.components['kg_builder'].build_knowledge_graph_from_processed_artifacts(
            processed_artifacts_directory=str(self.output_dir / "processed_artifacts"),
            max_artifacts=self.config.max_artifacts
        )
        
        # Store graph statistics
        self.results['knowledge_graph'] = kg_stats
        
        phase_time = time.time() - phase_start
        self.stats.phase_timings['knowledge_graph'] = phase_time
        
        logger.info(f"✅ Knowledge graph built in {phase_time:.2f}s")

    async def _phase3_vector_embeddings(self, artifacts: List[str]):
        """Phase 3: Vector embeddings generation"""
        logger.info("🔍 Phase 3: Generating vector embeddings...")
        phase_start = time.time()
        
        # Extract embeddings
        embedding_stats = self.components['vector_engine'].extract_embeddings_from_processed_artifacts(
            processed_artifacts_dir=str(self.output_dir / "processed_artifacts"),
            max_artifacts=self.config.max_artifacts
        )
        
        # Store embedding statistics
        self.results['vector_embeddings'] = embedding_stats
        
        phase_time = time.time() - phase_start
        self.stats.phase_timings['vector_embeddings'] = phase_time
        
        logger.info(f"✅ Vector embeddings generated in {phase_time:.2f}s")

    async def _phase4_pattern_analysis(self):
        """Phase 4: Pattern analysis"""
        logger.info("📊 Phase 4: Analyzing patterns...")
        phase_start = time.time()
        
        # Perform pattern analysis
        pattern_results = self.components['pattern_engine'].analyze_accepted_artifacts()
        
        # Store pattern analysis results
        self.results['pattern_analysis'] = {
            'section_patterns': len(pattern_results.section_patterns),
            'structural_patterns': len(pattern_results.structural_patterns),
            'clusters': len(pattern_results.clusters),
            'outliers': len(pattern_results.outliers)
        }
        
        phase_time = time.time() - phase_start
        self.stats.phase_timings['pattern_analysis'] = phase_time
        
        logger.info(f"✅ Pattern analysis completed in {phase_time:.2f}s")

    async def _phase5_multi_agent_evaluation(self, artifacts: List[str]) -> Dict[str, Any]:
        """Phase 5: Multi-agent evaluation"""
        logger.info("🧠 Phase 5: Running multi-agent evaluation...")
        phase_start = time.time()
        
        evaluation_results = {}
        successful_evaluations = 0
        failed_evaluations = 0
        
        # Process artifacts in batches
        for i in range(0, len(artifacts), self.config.batch_size):
            batch = artifacts[i:i + self.config.batch_size]
            logger.info(f"Processing batch {i//self.config.batch_size + 1}/{(len(artifacts)-1)//self.config.batch_size + 1}")
            
            # Use asyncio for concurrent evaluation
            batch_tasks = []
            for artifact_path in batch:
                task = self._evaluate_single_artifact(artifact_path)
                batch_tasks.append(task)
            
            # Wait for batch completion with timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks, return_exceptions=True),
                    timeout=self.config.timeout_per_artifact * len(batch)
                )
                
                for artifact_path, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        failed_evaluations += 1
                        logger.error(f"Evaluation failed for {artifact_path}: {result}")
                    else:
                        evaluation_results[artifact_path] = result
                        successful_evaluations += 1
                        
            except asyncio.TimeoutError:
                logger.error(f"Batch evaluation timeout for batch starting at {i}")
                failed_evaluations += len(batch)
        
        self.stats.successful_evaluations = successful_evaluations
        self.stats.failed_evaluations += failed_evaluations
        
        phase_time = time.time() - phase_start
        self.stats.phase_timings['multi_agent_evaluation'] = phase_time
        
        logger.info(f"✅ Multi-agent evaluation completed in {phase_time:.2f}s")
        logger.info(f"Success rate: {successful_evaluations}/{len(artifacts)} ({successful_evaluations/len(artifacts)*100:.1f}%)")
        
        return evaluation_results

    async def _evaluate_single_artifact(self, artifact_path: str) -> ArtifactEvaluationResult:
        """Evaluate a single artifact using multi-agent system"""
        artifact_id = Path(artifact_path).stem
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache and artifact_id in self.cache:
                logger.debug(f"Using cached result for {artifact_id}")
                return self.cache[artifact_id]
            
            # Load artifact data (simplified)
            with open(artifact_path, 'r') as f:
                artifact_data = json.load(f)
            
            # Run multi-agent evaluation
            multi_agent_result = await self.components['evaluator'].evaluate_artifact(
                artifact_id=artifact_id,
                artifact_data=artifact_data
            )
            
            # Create result object
            result = ArtifactEvaluationResult(
                artifact_id=artifact_id,
                status='success',
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                multi_agent_result=multi_agent_result
            )
            
            # Cache result if enabled
            if self.cache:
                self.cache[artifact_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed for {artifact_id}: {e}")
            return ArtifactEvaluationResult(
                artifact_id=artifact_id,
                status='failed',
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )

    async def _phase6_rubric_scoring(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 6: Rubric-based scoring"""
        logger.info("🎯 Phase 6: Applying rubric-based scoring...")
        phase_start = time.time()
        
        scoring_results = {}
        
        for artifact_path, eval_result in evaluation_results.items():
            if eval_result.status != 'success':
                continue
                
            artifact_id = eval_result.artifact_id
            
            try:
                # Score for each target conference
                artifact_scoring = {}
                
                for conference in self.config.target_conferences:
                    scoring_result = self.components['scoring'].score_artifact(
                        eval_result.multi_agent_result,
                        conference,
                        apply_penalties=self.config.apply_conference_penalties
                    )
                    artifact_scoring[conference] = scoring_result
                
                # Multi-conference comparison
                comparison_result = self.components['scoring'].compare_across_conferences(
                    eval_result.multi_agent_result,
                    self.config.target_conferences
                )
                
                scoring_results[artifact_id] = {
                    'individual_scores': artifact_scoring,
                    'comparison': comparison_result
                }
                
                # Update evaluation result
                eval_result.scoring_results = artifact_scoring
                eval_result.comparative_analysis = comparison_result
                
            except Exception as e:
                logger.error(f"Scoring failed for {artifact_id}: {e}")
                self.stats.errors.append(f"Scoring error for {artifact_id}: {str(e)}")
        
        self.stats.total_conferences = len(self.config.target_conferences)
        
        phase_time = time.time() - phase_start
        self.stats.phase_timings['rubric_scoring'] = phase_time
        
        logger.info(f"✅ Rubric scoring completed in {phase_time:.2f}s")
        return scoring_results

    async def _phase7_reporting_and_visualization(self, scoring_results: Dict[str, Any]):
        """Phase 7: Generate reports and visualizations"""
        logger.info("📊 Phase 7: Generating reports and visualizations...")
        phase_start = time.time()
        
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate summary report
        if self.config.generate_summary_report:
            await self._generate_summary_report(scoring_results, reports_dir)
        
        # Generate individual artifact reports
        if self.config.generate_individual_reports:
            await self._generate_individual_reports(scoring_results, reports_dir)
        
        # Generate visualizations
        if self.config.generate_visualizations:
            await self._generate_visualizations(scoring_results, reports_dir)
        
        # Export data in requested formats
        await self._export_data(scoring_results)
        
        phase_time = time.time() - phase_start
        self.stats.phase_timings['reporting'] = phase_time
        
        logger.info(f"✅ Reporting completed in {phase_time:.2f}s")

    async def _generate_summary_report(self, scoring_results: Dict[str, Any], reports_dir: Path):
        """Generate comprehensive summary report"""
        summary_data = {
            'pipeline_stats': asdict(self.stats),
            'evaluation_summary': {
                'total_artifacts': len(scoring_results),
                'conferences': self.config.target_conferences,
                'avg_scores_by_conference': {},
                'acceptance_distribution': {},
                'top_performers': [],
                'improvement_opportunities': []
            },
            'framework_performance': self.results
        }
        
        # Calculate summary statistics
        for conference in self.config.target_conferences:
            scores = []
            categories = []
            
            for artifact_data in scoring_results.values():
                if conference in artifact_data['individual_scores']:
                    score_result = artifact_data['individual_scores'][conference]
                    scores.append(score_result.final_score)
                    categories.append(score_result.acceptance_category.value)
            
            if scores:
                summary_data['evaluation_summary']['avg_scores_by_conference'][conference] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(min(scores)),
                    'max': float(max(scores))
                }
                
                # Acceptance category distribution
                from collections import Counter
                summary_data['evaluation_summary']['acceptance_distribution'][conference] = dict(Counter(categories))
        
        # Save summary report
        summary_file = reports_dir / "summary_report.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        logger.info(f"📄 Summary report saved to {summary_file}")

    async def _generate_individual_reports(self, scoring_results: Dict[str, Any], reports_dir: Path):
        """Generate individual artifact reports"""
        individual_dir = reports_dir / "individual_artifacts"
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        for artifact_id, artifact_data in scoring_results.items():
            report_data = {
                'artifact_id': artifact_id,
                'timestamp': datetime.now().isoformat(),
                'conference_scores': {},
                'comparison_analysis': artifact_data.get('comparison'),
                'recommendations': []
            }
            
            # Add conference-specific scores
            for conference, score_result in artifact_data['individual_scores'].items():
                report_data['conference_scores'][conference] = {
                    'final_score': score_result.final_score,
                    'acceptance_category': score_result.acceptance_category.value,
                    'acceptance_probability': score_result.acceptance_probability,
                    'dimension_scores': {dim.value: score for dim, score in score_result.raw_scores.items()},
                    'improvement_priorities': score_result.improvement_priorities
                }
                
                # Add to recommendations
                report_data['recommendations'].extend(score_result.improvement_priorities[:3])
            
            # Save individual report
            report_file = individual_dir / f"{artifact_id}_report.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"📄 Individual reports saved to {individual_dir}")

    async def _generate_visualizations(self, scoring_results: Dict[str, Any], reports_dir: Path):
        """Generate visualization files"""
        viz_dir = reports_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # This would generate static visualization files
        # For now, just create placeholder files
        placeholder_files = [
            "score_distribution.png",
            "conference_comparison.png", 
            "dimension_analysis.png",
            "acceptance_trends.png"
        ]
        
        for filename in placeholder_files:
            placeholder_file = viz_dir / filename
            placeholder_file.write_text(f"Visualization placeholder: {filename}")
        
        logger.info(f"📊 Visualizations saved to {viz_dir}")

    async def _export_data(self, scoring_results: Dict[str, Any]):
        """Export data in requested formats"""
        exports_dir = self.output_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        if "CSV" in self.config.export_formats:
            await self._export_to_csv(scoring_results, exports_dir)
        
        # Export to JSON
        if "JSON" in self.config.export_formats:
            await self._export_to_json(scoring_results, exports_dir)
        
        logger.info(f"💾 Data exported to {exports_dir}")

    async def _export_to_csv(self, scoring_results: Dict[str, Any], exports_dir: Path):
        """Export results to CSV format"""
        csv_file = exports_dir / "evaluation_results.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Artifact_ID']
            for conf in self.config.target_conferences:
                header.extend([f'{conf}_Score', f'{conf}_Category', f'{conf}_Probability'])
            header.extend(['Best_Conference', 'Best_Score'])
            writer.writerow(header)
            
            # Data rows
            for artifact_id, artifact_data in scoring_results.items():
                row = [artifact_id]
                
                # Conference scores
                for conf in self.config.target_conferences:
                    if conf in artifact_data['individual_scores']:
                        score_result = artifact_data['individual_scores'][conf]
                        row.extend([
                            f"{score_result.final_score:.3f}",
                            score_result.acceptance_category.value,
                            f"{score_result.acceptance_probability:.3f}"
                        ])
                    else:
                        row.extend(['N/A', 'N/A', 'N/A'])
                
                # Best conference
                comparison = artifact_data.get('comparison')
                if comparison:
                    row.extend([comparison.best_fit_conference, f"{comparison.best_score:.3f}"])
                else:
                    row.extend(['N/A', 'N/A'])
                
                writer.writerow(row)

    async def _export_to_json(self, scoring_results: Dict[str, Any], exports_dir: Path):
        """Export results to JSON format"""
        json_file = exports_dir / "evaluation_results.json"
        
        with open(json_file, 'w') as f:
            json.dump(scoring_results, f, indent=2, default=str)

    async def _phase8_finalization(self):
        """Phase 8: Finalization and cleanup"""
        logger.info("🏁 Phase 8: Finalizing pipeline...")
        phase_start = time.time()
        
        # Generate final pipeline report
        final_report = {
            'pipeline_execution': asdict(self.stats),
            'configuration': asdict(self.config),
            'summary_statistics': {
                'success_rate': self.stats.successful_evaluations / max(1, self.stats.total_artifacts),
                'avg_processing_time': self.stats.avg_processing_time,
                'total_runtime': sum(self.stats.phase_timings.values()),
                'phase_timings': self.stats.phase_timings
            }
        }
        
        # Save final report
        final_report_file = self.output_dir / "pipeline_execution_report.json"
        with open(final_report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Cleanup temporary files if needed
        if not self.config.enable_caching:
            # Remove cache files
            pass
        
        phase_time = time.time() - phase_start
        self.stats.phase_timings['finalization'] = phase_time
        
        logger.info(f"✅ Pipeline finalized in {phase_time:.2f}s")
        logger.info(f"📄 Final report saved to {final_report_file}")

    async def _collect_performance_metrics(self):
        """Collect performance monitoring metrics"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                self.stats.peak_memory_usage = memory_info.rss / 1024 / 1024  # MB
                
                logger.info(f"Peak memory usage: {self.stats.peak_memory_usage:.2f} MB")
            except Exception as e:
                logger.warning(f"Failed to collect performance metrics: {e}")

    @classmethod
    def from_config_file(cls, config_path: str) -> 'AuraAutomationPipeline':
        """Create pipeline from configuration file"""
        config_file = Path(config_path)
        
        if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        
        # Create config object
        config = PipelineConfig(**config_data)
        return cls(config)

    def save_config(self, config_path: str):
        """Save current configuration to file"""
        config_data = asdict(self.config)
        
        config_file = Path(config_path)
        if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, indent=2)
        else:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")


def create_cli():
    """Create command-line interface for the pipeline"""
    parser = argparse.ArgumentParser(description="AURA Automation Pipeline")
    
    # Input/Output arguments
    parser.add_argument("--artifacts-dir", default="data/acm_bib_to_json_data",
                       help="Directory containing artifact JSON files")
    parser.add_argument("--output-dir", default="pipeline_outputs",
                       help="Output directory for results")
    parser.add_argument("--config", help="Configuration file path")
    
    # Processing arguments
    parser.add_argument("--max-artifacts", type=int, help="Maximum number of artifacts to process")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum worker threads")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per artifact (seconds)")
    
    # Framework arguments
    parser.add_argument("--use-neo4j", action="store_true", help="Use Neo4j for knowledge graph")
    parser.add_argument("--skip-vectors", action="store_true", help="Skip vector embeddings")
    parser.add_argument("--skip-patterns", action="store_true", help="Skip pattern analysis")
    parser.add_argument("--no-penalties", action="store_true", help="Disable conference penalties")
    
    # Conference arguments
    parser.add_argument("--conferences", nargs="+", default=["ICSE", "FSE", "ASE", "CHI"],
                       help="Target conferences for evaluation")
    
    # Output arguments
    parser.add_argument("--export-formats", nargs="+", default=["JSON", "CSV"],
                       choices=["JSON", "CSV", "PDF"], help="Export formats")
    parser.add_argument("--no-individual-reports", action="store_true",
                       help="Skip individual artifact reports")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip visualization generation")
    
    # Logging and monitoring
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable performance monitoring")
    
    return parser


async def main():
    """Main CLI entry point"""
    parser = create_cli()
    args = parser.parse_args()
    
    # Create configuration from arguments
    if args.config:
        pipeline = AuraAutomationPipeline.from_config_file(args.config)
        
        # Override with CLI arguments if provided
        if args.max_artifacts:
            pipeline.config.max_artifacts = args.max_artifacts
        if args.conferences:
            pipeline.config.target_conferences = args.conferences
            
    else:
        # Create configuration from CLI arguments
        config = PipelineConfig(
            artifacts_directory=args.artifacts_dir,
            output_directory=args.output_dir,
            max_artifacts=args.max_artifacts,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            timeout_per_artifact=args.timeout,
            use_neo4j=args.use_neo4j,
            use_vector_embeddings=not args.skip_vectors,
            use_pattern_analysis=not args.skip_patterns,
            apply_conference_penalties=not args.no_penalties,
            target_conferences=args.conferences,
            export_formats=args.export_formats,
            generate_individual_reports=not args.no_individual_reports,
            generate_visualizations=not args.no_visualizations,
            enable_monitoring=not args.no_monitoring,
            log_level=args.log_level
        )
        
        pipeline = AuraAutomationPipeline(config)
    
    # Save configuration for reference
    config_file = Path(pipeline.output_dir) / "pipeline_config.json"
    pipeline.save_config(str(config_file))
    
    # Run the complete pipeline
    try:
        stats = await pipeline.run_complete_pipeline()
        
        # Print final statistics
        print("\n🎉 AURA Pipeline Completed Successfully!")
        print("=" * 50)
        print(f"📊 Total Artifacts: {stats.total_artifacts}")
        print(f"✅ Successful Evaluations: {stats.successful_evaluations}")
        print(f"❌ Failed Evaluations: {stats.failed_evaluations}")
        print(f"🏛️  Conferences: {stats.total_conferences}")
        print(f"⏱️  Average Processing Time: {stats.avg_processing_time:.2f}s")
        print(f"💾 Peak Memory Usage: {stats.peak_memory_usage:.2f} MB")
        print(f"📁 Output Directory: {pipeline.output_dir}")
        
        if stats.errors:
            print(f"\n⚠️  Errors encountered: {len(stats.errors)}")
            for error in stats.errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    import sys
    
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 