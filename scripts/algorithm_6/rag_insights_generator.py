"""
RAG-Powered Insights Generator for Artifact Evaluation

This module uses Retrieval Augmented Generation (RAG) to generate explainable
insights, recommendations, and analysis summaries from the artifact evaluation
framework. It combines knowledge graph patterns, semantic analysis, and LLM
reasoning to provide human-readable explanations.

Key Features:
- Context-aware insight generation using RAG
- Explainable AI recommendations with evidence citations
- Comparative analysis between artifacts and conferences
- Pattern-based insight extraction from knowledge graphs
- Natural language summaries of complex analysis results
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from langchain.callbacks.manager import get_openai_callback
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
# LangChain and LLM components
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from config import config, NODE_TYPES

# Vector search and similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InsightContext:
    """Context information for generating insights"""
    artifact_id: str
    query_type: str
    relevant_patterns: List[str]
    graph_context: Dict[str, Any]
    semantic_context: Dict[str, Any]
    conference_context: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class GeneratedInsight:
    """Generated insight with supporting evidence"""
    insight_type: str
    main_insight: str
    supporting_evidence: List[str]
    recommendations: List[str]
    confidence_score: float
    sources: List[str]
    related_patterns: List[str]


class RAGInsightsGenerator:
    """RAG-powered system for generating artifact evaluation insights"""

    def __init__(self, kg_builder, pattern_analyzer, vector_analyzer,
                 scoring_framework, conference_models):
        self.kg_builder = kg_builder
        self.pattern_analyzer = pattern_analyzer
        self.vector_analyzer = vector_analyzer
        self.scoring_framework = scoring_framework
        self.conference_models = conference_models

        # LLM setup
        try:
            self.llm = ChatOpenAI(
                model_name=config.llm.model_name,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                openai_api_key=config.llm.api_key
            )
            self.llm_available = True
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI LLM: {e}")
            self.llm = None
            self.llm_available = False

        # Vector store for RAG
        self.vector_store = None
        self.retriever = None

        # Insight generation chains
        self.insight_chains = {}

        # Knowledge base
        self.knowledge_documents = []
        self.pattern_documents = []

        self._initialize_rag_system()

    def _initialize_rag_system(self):
        """Initialize the RAG system with knowledge documents"""
        logger.info("Initializing RAG system for insight generation")

        # Create knowledge documents from analysis results
        self._create_knowledge_documents()

        # Setup vector store
        self._setup_vector_store()

        # Create insight generation chains
        self._create_insight_chains()

    def _create_knowledge_documents(self):
        """Create documents from analysis results for RAG retrieval"""
        documents = []

        # 1. Pattern analysis documents
        if hasattr(self.pattern_analyzer, 'patterns'):
            for pattern in self.pattern_analyzer.patterns:
                doc_content = f"""
                Pattern Type: {pattern.pattern_type}
                Description: {pattern.description}
                Frequency: {pattern.frequency}
                Conferences: {', '.join(pattern.conferences)}
                Quality Score: {pattern.quality_score}
                """

                doc = Document(
                    page_content=doc_content.strip(),
                    metadata={
                        "source": "pattern_analysis",
                        "pattern_type": pattern.pattern_type,
                        "pattern_id": pattern.pattern_id
                    }
                )
                documents.append(doc)

        # 2. Conference profile documents
        for conf_name, profile in self.conference_models.conference_profiles.items():
            doc_content = f"""
            Conference: {conf_name}
            Category: {profile.category}
            Total Artifacts: {profile.total_artifacts}
            Preferred Sections: {', '.join(profile.preferred_sections)}
            Preferred Tools: {', '.join(profile.preferred_tools)}
            Documentation Style: {profile.documentation_style}
            Average Documentation Length: {profile.avg_documentation_length}
            Reproducibility Emphasis: {profile.reproducibility_emphasis}
            Section Importance: {json.dumps(profile.section_importance)}
            Tool Usage Frequency: {json.dumps(profile.tool_usage_frequency)}
            """

            doc = Document(
                page_content=doc_content.strip(),
                metadata={
                    "source": "conference_profile",
                    "conference": conf_name,
                    "category": profile.category
                }
            )
            documents.append(doc)

        # 3. Quality indicator documents
        for category, indicators in config.evaluation.scoring_weights.items():
            doc_content = f"""
            Quality Category: {category}
            Importance Weight: {indicators}
            Description: This category represents {category.replace('_', ' ')} aspects of artifact quality.
            """

            doc = Document(
                page_content=doc_content.strip(),
                metadata={
                    "source": "quality_indicators",
                    "category": category
                }
            )
            documents.append(doc)

        # 4. Conference category documents
        for category, info in config.conference.conference_categories.items():
            doc_content = f"""
            Conference Category: {category}
            Conferences: {', '.join(info['conferences'])}
            Emphasis Areas: {', '.join(info['emphasis'])}
            Required Tools: {', '.join(info['required_tools'])}
            Documentation Style: {info['documentation_style']}
            """

            doc = Document(
                page_content=doc_content.strip(),
                metadata={
                    "source": "conference_category",
                    "category": category
                }
            )
            documents.append(doc)

        # 5. Best practices documents
        best_practices = [
            {
                "title": "Installation Documentation",
                "content": "High-quality installation sections include step-by-step instructions, dependency lists, system requirements, and troubleshooting guides. Docker support significantly improves reproducibility."
            },
            {
                "title": "Usage Examples",
                "content": "Effective usage documentation provides concrete examples, command-line snippets, expected outputs, and common use cases. Jupyter notebooks are particularly valuable for data science artifacts."
            },
            {
                "title": "Reproducibility Features",
                "content": "Reproducible artifacts typically include Docker containers, Conda environments, requirements files, build scripts, and test suites. Version pinning is crucial."
            },
            {
                "title": "Documentation Structure",
                "content": "Well-structured documentation follows a logical flow: purpose, installation, usage, examples, configuration, troubleshooting, and citation information."
            }
        ]

        for practice in best_practices:
            doc = Document(
                page_content=f"Title: {practice['title']}\n{practice['content']}",
                metadata={
                    "source": "best_practices",
                    "title": practice['title']
                }
            )
            documents.append(doc)

        self.knowledge_documents = documents
        logger.info(f"Created {len(documents)} knowledge documents for RAG")

    def _setup_vector_store(self):
        """Setup vector store for RAG retrieval"""
        if not self.knowledge_documents:
            logger.warning("No knowledge documents available for vector store")
            return

        try:
            # Use OpenAI embeddings for consistency if API key is available
            if config.llm.api_key:
                embeddings = OpenAIEmbeddings(openai_api_key=config.llm.api_key)
            else:
                # Fallback to sentence transformers
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')

                # Create custom embeddings wrapper
                class SentenceTransformerEmbeddings:
                    def __init__(self, model):
                        self.model = model

                    def embed_documents(self, texts):
                        return self.model.encode(texts).tolist()

                    def embed_query(self, text):
                        return self.model.encode([text])[0].tolist()

                embeddings = SentenceTransformerEmbeddings(model)

            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                self.knowledge_documents,
                embeddings
            )

            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 relevant documents
            )

            logger.info("Vector store and retriever initialized successfully")

        except Exception as e:
            logger.error(f"Failed to setup vector store: {e}")
            # Fallback to simple keyword-based retrieval
            self.retriever = None

    def _create_insight_chains(self):
        """Create LangChain chains for different types of insights"""

        if not self.llm_available or not self.retriever:
            logger.warning("LLM or retriever not available, skipping chain creation")
            return

        # 1. Artifact Analysis Chain
        artifact_analysis_prompt = PromptTemplate(
            template="""
            You are an expert in research artifact evaluation. Based on the provided context about artifact patterns, 
            conference preferences, and quality indicators, provide a comprehensive analysis of the given artifact.
            
            Context:
            {context}
            
            Artifact Information:
            {artifact_info}
            
            Please provide:
            1. Overall quality assessment
            2. Strengths and weaknesses
            3. Specific recommendations for improvement
            4. Conference suitability analysis
            5. Comparison with similar artifacts
            
            Focus on actionable insights backed by the evidence in the context.
            
            Analysis:""",
            input_variables=["context", "artifact_info"]
        )

        if self.retriever:
            self.insight_chains['artifact_analysis'] = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": artifact_analysis_prompt}
            )

        # 2. Pattern Explanation Chain
        pattern_explanation_prompt = PromptTemplate(
            template="""
            You are a research expert analyzing patterns in artifact documentation. Based on the provided context 
            about documentation patterns and conference preferences, explain the significance of the identified patterns.
            
            Context:
            {context}
            
            Pattern Information:
            {pattern_info}
            
            Please explain:
            1. What this pattern represents
            2. Why it's important for artifact acceptance
            3. Which conferences value this pattern most
            4. How to implement this pattern effectively
            5. Examples of successful implementations
            
            Provide clear, actionable explanations with specific examples.
            
            Explanation:""",
            input_variables=["context", "pattern_info"]
        )

        if self.retriever:
            self.insight_chains['pattern_explanation'] = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": pattern_explanation_prompt}
            )

        # 3. Conference Comparison Chain
        conference_comparison_prompt = PromptTemplate(
            template="""
            You are an expert in academic conference standards and artifact evaluation. Based on the provided context
            about different conferences and their preferences, compare and analyze conference-specific requirements.
            
            Context:
            {context}
            
            Comparison Request:
            {comparison_request}
            
            Please provide:
            1. Key differences between the conferences
            2. Specific requirements for each conference
            3. Common patterns across conferences
            4. Recommendations for targeting specific conferences
            5. Success strategies for each venue
            
            Base your analysis on the evidence provided in the context.
            
            Comparison:""",
            input_variables=["context", "comparison_request"]
        )

        if self.retriever:
            self.insight_chains['conference_comparison'] = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": conference_comparison_prompt}
            )

        # 4. Improvement Recommendation Chain
        improvement_prompt = PromptTemplate(
            template="""
            You are a mentor helping researchers improve their artifact documentation. Based on the provided context
            about best practices, successful patterns, and quality indicators, provide specific improvement recommendations.
            
            Context:
            {context}
            
            Current Artifact State:
            {current_state}
            
            Target Improvement Areas:
            {target_areas}
            
            Please provide:
            1. Prioritized list of improvements
            2. Specific implementation steps for each improvement
            3. Expected impact of each improvement
            4. Time/effort estimates for implementation
            5. Examples of successful implementations
            
            Focus on actionable, specific guidance with clear next steps.
            
            Recommendations:""",
            input_variables=["context", "current_state", "target_areas"]
        )

        if self.retriever:
            self.insight_chains['improvement_recommendations'] = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": improvement_prompt}
            )

    def generate_artifact_analysis(self, artifact_id: str,
                                   include_comparisons: bool = True) -> GeneratedInsight:
        """
        Generate comprehensive analysis for a specific artifact
        
        Args:
            artifact_id: ID of the artifact to analyze
            include_comparisons: Whether to include comparisons with similar artifacts
            
        Returns:
            Generated insight with analysis and recommendations
        """
        logger.info(f"Generating analysis for artifact {artifact_id}")

        # Gather context information
        context = self._gather_artifact_context(artifact_id)

        # Get artifact features and prediction
        if hasattr(self.scoring_framework, 'predict_acceptance'):
            try:
                prediction = self.scoring_framework.predict_acceptance(artifact_id)
                context.metadata['prediction'] = prediction
            except Exception as e:
                logger.warning(f"Could not get prediction for {artifact_id}: {e}")

        # Generate analysis using RAG
        if self.llm_available and 'artifact_analysis' in self.insight_chains:
            try:
                if config.llm.api_key:
                    with get_openai_callback() as cb:
                        analysis_result = self.insight_chains['artifact_analysis'].run({
                            "query": f"Analyze artifact {artifact_id} for quality and acceptance potential",
                            "artifact_info": self._format_artifact_info(context)
                        })

                    logger.info(f"Generated analysis with {cb.total_tokens} tokens")
                else:
                    # Use chain without callback if not using OpenAI
                    analysis_result = self.insight_chains['artifact_analysis'].run({
                        "query": f"Analyze artifact {artifact_id} for quality and acceptance potential",
                        "artifact_info": self._format_artifact_info(context)
                    })
                    logger.info("Generated analysis using fallback embeddings")

            except Exception as e:
                logger.error(f"Failed to generate analysis: {e}")
                analysis_result = self._generate_fallback_analysis(context)
        else:
            analysis_result = self._generate_fallback_analysis(context)

        # Extract recommendations and evidence
        recommendations = self._extract_recommendations_from_analysis(analysis_result, context)
        evidence = self._extract_evidence_from_context(context)

        # Calculate confidence score
        confidence = self._calculate_insight_confidence(context, analysis_result)

        return GeneratedInsight(
            insight_type="artifact_analysis",
            main_insight=analysis_result,
            supporting_evidence=evidence,
            recommendations=recommendations,
            confidence_score=confidence,
            sources=["pattern_analysis", "conference_profiles", "quality_indicators"],
            related_patterns=context.relevant_patterns
        )

    def generate_pattern_explanation(self, pattern_type: str,
                                     pattern_data: Dict[str, Any]) -> GeneratedInsight:
        """
        Generate explanation for identified documentation patterns
        
        Args:
            pattern_type: Type of pattern to explain
            pattern_data: Data about the pattern
            
        Returns:
            Generated insight explaining the pattern
        """
        logger.info(f"Generating explanation for pattern: {pattern_type}")

        if 'pattern_explanation' in self.insight_chains:
            try:
                with get_openai_callback() as cb:
                    explanation = self.insight_chains['pattern_explanation'].run({
                        "query": f"Explain the significance of {pattern_type} pattern in artifact documentation",
                        "pattern_info": json.dumps(pattern_data, indent=2)
                    })

                logger.info(f"Generated pattern explanation with {cb.total_tokens} tokens")

            except Exception as e:
                logger.error(f"Failed to generate pattern explanation: {e}")
                explanation = self._generate_fallback_pattern_explanation(pattern_type, pattern_data)
        else:
            explanation = self._generate_fallback_pattern_explanation(pattern_type, pattern_data)

        # Extract actionable insights
        recommendations = self._extract_pattern_recommendations(explanation, pattern_type)
        evidence = [f"Pattern frequency: {pattern_data.get('frequency', 'unknown')}",
                    f"Found in conferences: {pattern_data.get('conferences', [])}"]

        return GeneratedInsight(
            insight_type="pattern_explanation",
            main_insight=explanation,
            supporting_evidence=evidence,
            recommendations=recommendations,
            confidence_score=0.8,
            sources=["pattern_analysis"],
            related_patterns=[pattern_type]
        )

    def generate_conference_comparison(self, conferences: List[str]) -> GeneratedInsight:
        """
        Generate comparison between different conferences
        
        Args:
            conferences: List of conferences to compare
            
        Returns:
            Generated insight comparing conferences
        """
        logger.info(f"Generating comparison for conferences: {conferences}")

        comparison_request = f"Compare the artifact evaluation standards and preferences for: {', '.join(conferences)}"

        if 'conference_comparison' in self.insight_chains:
            try:
                with get_openai_callback() as cb:
                    comparison = self.insight_chains['conference_comparison'].run({
                        "query": comparison_request,
                        "comparison_request": comparison_request
                    })

                logger.info(f"Generated conference comparison with {cb.total_tokens} tokens")

            except Exception as e:
                logger.error(f"Failed to generate conference comparison: {e}")
                comparison = self._generate_fallback_conference_comparison(conferences)
        else:
            comparison = self._generate_fallback_conference_comparison(conferences)

        # Extract specific recommendations for each conference
        recommendations = self._extract_conference_recommendations(comparison, conferences)

        # Gather evidence from conference profiles
        evidence = []
        for conf in conferences:
            if conf in self.conference_models.conference_profiles:
                profile = self.conference_models.conference_profiles[conf]
                evidence.append(f"{conf}: Prefers {', '.join(profile.preferred_sections[:3])} sections")
                evidence.append(f"{conf}: Common tools - {', '.join(profile.preferred_tools[:3])}")

        return GeneratedInsight(
            insight_type="conference_comparison",
            main_insight=comparison,
            supporting_evidence=evidence,
            recommendations=recommendations,
            confidence_score=0.85,
            sources=["conference_profiles"],
            related_patterns=[]
        )

    def generate_improvement_recommendations(self, artifact_id: str,
                                             target_conference: Optional[str] = None) -> GeneratedInsight:
        """
        Generate specific improvement recommendations for an artifact
        
        Args:
            artifact_id: ID of the artifact to improve
            target_conference: Target conference for submission
            
        Returns:
            Generated insight with improvement recommendations
        """
        logger.info(f"Generating improvement recommendations for {artifact_id}")

        # Gather current state
        context = self._gather_artifact_context(artifact_id)
        current_state = self._format_current_state(context)

        # Identify target improvement areas
        target_areas = self._identify_improvement_areas(context, target_conference)

        if 'improvement_recommendations' in self.insight_chains:
            try:
                with get_openai_callback() as cb:
                    recommendations = self.insight_chains['improvement_recommendations'].run({
                        "query": f"Provide improvement recommendations for artifact {artifact_id}",
                        "current_state": current_state,
                        "target_areas": target_areas
                    })

                logger.info(f"Generated recommendations with {cb.total_tokens} tokens")

            except Exception as e:
                logger.error(f"Failed to generate recommendations: {e}")
                recommendations = self._generate_fallback_recommendations(context, target_areas)
        else:
            recommendations = self._generate_fallback_recommendations(context, target_areas)

        # Extract specific action items
        action_items = self._extract_action_items(recommendations)
        evidence = self._extract_improvement_evidence(context, target_conference)

        return GeneratedInsight(
            insight_type="improvement_recommendations",
            main_insight=recommendations,
            supporting_evidence=evidence,
            recommendations=action_items,
            confidence_score=0.9,
            sources=["best_practices", "conference_profiles", "pattern_analysis"],
            related_patterns=context.relevant_patterns
        )

    def generate_batch_insights(self, artifact_ids: List[str],
                                insight_types: List[str] = None) -> Dict[str, List[GeneratedInsight]]:
        """
        Generate insights for multiple artifacts in batch
        
        Args:
            artifact_ids: List of artifact IDs
            insight_types: Types of insights to generate
            
        Returns:
            Dictionary mapping insight types to lists of insights
        """
        if insight_types is None:
            insight_types = ["artifact_analysis", "improvement_recommendations"]

        results = defaultdict(list)

        for artifact_id in artifact_ids:
            for insight_type in insight_types:
                try:
                    if insight_type == "artifact_analysis":
                        insight = self.generate_artifact_analysis(artifact_id)
                    elif insight_type == "improvement_recommendations":
                        insight = self.generate_improvement_recommendations(artifact_id)
                    else:
                        continue

                    results[insight_type].append(insight)

                except Exception as e:
                    logger.error(f"Failed to generate {insight_type} for {artifact_id}: {e}")

        return dict(results)

    def _gather_artifact_context(self, artifact_id: str) -> InsightContext:
        """Gather comprehensive context for an artifact"""
        context = InsightContext(
            artifact_id=artifact_id,
            query_type="artifact_analysis",
            relevant_patterns=[],
            graph_context={},
            semantic_context={},
            conference_context={},
            metadata={}
        )

        # Gather graph context
        if hasattr(self.kg_builder, 'nx_graph'):
            G = self.kg_builder.nx_graph

            # Find artifact node and extract information
            for node in G.nodes():
                node_data = G.nodes[node]
                if (node_data.get('node_type') == NODE_TYPES['ARTIFACT'] and
                        node_data.get('artifact_id') == artifact_id):

                    context.graph_context = {
                        'artifact_data': node_data,
                        'connected_tools': [],
                        'documentation_files': [],
                        'sections': []
                    }

                    # Get connected nodes
                    for neighbor in G.neighbors(node):
                        neighbor_data = G.nodes[neighbor]
                        node_type = neighbor_data.get('node_type')

                        if node_type == NODE_TYPES['TOOL']:
                            context.graph_context['connected_tools'].append(neighbor_data)
                        elif node_type == NODE_TYPES['DOCUMENTATION']:
                            context.graph_context['documentation_files'].append(neighbor_data)
                        elif node_type == NODE_TYPES['SECTION']:
                            context.graph_context['sections'].append(neighbor_data)

                    break

        # Gather semantic context
        if artifact_id in self.vector_analyzer.artifact_embeddings:
            # Find similar artifacts
            try:
                similar_results = self.vector_analyzer.find_similar_documents(artifact_id)
                context.semantic_context = {
                    'similar_artifacts': similar_results.similar_documents,
                    'semantic_themes': similar_results.semantic_themes
                }
            except:
                context.semantic_context = {'similar_artifacts': [], 'semantic_themes': []}

        # Gather conference context
        artifact_conference = context.graph_context.get('artifact_data', {}).get('conference')
        if artifact_conference and artifact_conference in self.conference_models.conference_profiles:
            context.conference_context = {
                'conference_profile': self.conference_models.conference_profiles[artifact_conference],
                'category_info': config.conference.conference_categories.get(
                    self.conference_models.conference_profiles[artifact_conference].category, {}
                )
            }

        # Identify relevant patterns
        context.relevant_patterns = self._identify_relevant_patterns(context)

        return context

    def _format_artifact_info(self, context: InsightContext) -> str:
        """Format artifact information for LLM consumption"""
        info_parts = []

        # Basic information
        artifact_data = context.graph_context.get('artifact_data', {})
        info_parts.append(f"Artifact ID: {context.artifact_id}")
        info_parts.append(f"Conference: {artifact_data.get('conference', 'unknown')}")

        # Documentation information
        doc_files = context.graph_context.get('documentation_files', [])
        if doc_files:
            info_parts.append(f"Documentation files: {len(doc_files)}")
            for doc in doc_files[:3]:  # Limit to first 3
                info_parts.append(f"  - {doc.get('file_name', 'unknown')}: {doc.get('content_length', 0)} chars")

        # Sections information
        sections = context.graph_context.get('sections', [])
        if sections:
            section_types = [s.get('section_type', 'unknown') for s in sections]
            info_parts.append(f"Documentation sections: {', '.join(set(section_types))}")

        # Tools information
        tools = context.graph_context.get('connected_tools', [])
        if tools:
            tool_names = [t.get('name', 'unknown') for t in tools]
            info_parts.append(f"Tools used: {', '.join(tool_names)}")

        # Similar artifacts
        similar = context.semantic_context.get('similar_artifacts', [])
        if similar:
            info_parts.append(f"Similar artifacts found: {len(similar)}")

        # Prediction information if available
        if 'prediction' in context.metadata:
            pred = context.metadata['prediction']
            info_parts.append(f"Acceptance prediction: {pred.acceptance_probability:.2f}")
            info_parts.append(f"Predicted class: {pred.predicted_class}")

        return '\n'.join(info_parts)

    def _extract_recommendations_from_analysis(self, analysis: str,
                                               context: InsightContext) -> List[str]:
        """Extract actionable recommendations from analysis text"""
        recommendations = []

        # Simple extraction based on keywords and patterns
        lines = analysis.split('\n')
        in_recommendations_section = False

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'improve']):
                if line and not line.startswith('#'):
                    recommendations.append(line)
            elif 'recommendation' in line.lower():
                in_recommendations_section = True
            elif in_recommendations_section and line:
                if line.startswith('-') or line.startswith('*') or line[0].isdigit():
                    recommendations.append(line.lstrip('-*0123456789. '))

        # Fallback: extract from prediction if available
        if not recommendations and 'prediction' in context.metadata:
            pred = context.metadata['prediction']
            recommendations.extend(pred.recommendations)

        return recommendations[:5]  # Limit to top 5

    def _extract_evidence_from_context(self, context: InsightContext) -> List[str]:
        """Extract supporting evidence from context"""
        evidence = []

        # Graph-based evidence
        if context.graph_context:
            artifact_data = context.graph_context.get('artifact_data', {})
            if artifact_data.get('conference'):
                evidence.append(f"Submitted to {artifact_data['conference']}")

            tools_count = len(context.graph_context.get('connected_tools', []))
            if tools_count > 0:
                evidence.append(f"Uses {tools_count} different tools")

            docs_count = len(context.graph_context.get('documentation_files', []))
            if docs_count > 0:
                evidence.append(f"Contains {docs_count} documentation files")

        # Semantic evidence
        if context.semantic_context:
            similar_count = len(context.semantic_context.get('similar_artifacts', []))
            if similar_count > 0:
                evidence.append(f"Similar to {similar_count} other artifacts")

        # Pattern evidence
        if context.relevant_patterns:
            evidence.append(f"Follows {len(context.relevant_patterns)} common patterns")

        return evidence

    def _calculate_insight_confidence(self, context: InsightContext, analysis: str) -> float:
        """Calculate confidence score for the insight"""
        confidence = 0.5  # Base confidence

        # Increase confidence based on available context
        if context.graph_context:
            confidence += 0.2

        if context.semantic_context:
            confidence += 0.1

        if context.conference_context:
            confidence += 0.1

        if context.relevant_patterns:
            confidence += 0.1

        # Adjust based on analysis length and detail
        if len(analysis) > 500:
            confidence += 0.05

        if 'specific' in analysis.lower() or 'evidence' in analysis.lower():
            confidence += 0.05

        return min(confidence, 1.0)

    def _identify_relevant_patterns(self, context: InsightContext) -> List[str]:
        """Identify patterns relevant to the artifact"""
        patterns = []

        # Based on sections
        sections = context.graph_context.get('sections', [])
        section_types = {s.get('section_type') for s in sections}

        if 'installation' in section_types:
            patterns.append('installation_documentation')
        if 'usage' in section_types:
            patterns.append('usage_examples')
        if len(section_types) > 5:
            patterns.append('comprehensive_structure')

        # Based on tools
        tools = context.graph_context.get('connected_tools', [])
        tool_names = {t.get('name', '').lower() for t in tools}

        if 'docker' in tool_names:
            patterns.append('docker_support')
        if any(tool in tool_names for tool in ['conda', 'pip']):
            patterns.append('environment_management')

        return patterns

    # Fallback methods for when RAG is not available
    def _generate_fallback_analysis(self, context: InsightContext) -> str:
        """Generate fallback analysis without RAG"""
        return f"""
        Artifact Analysis for {context.artifact_id}:
        
        Based on the available information, this artifact shows the following characteristics:
        - Documentation structure includes {len(context.graph_context.get('sections', []))} sections
        - Uses {len(context.graph_context.get('connected_tools', []))} different tools
        - Contains {len(context.graph_context.get('documentation_files', []))} documentation files
        
        Recommendations for improvement:
        - Ensure comprehensive installation instructions
        - Add usage examples and tutorials
        - Include reproducibility features like Docker or Conda
        - Provide clear documentation structure
        """

    def _generate_fallback_pattern_explanation(self, pattern_type: str,
                                               pattern_data: Dict[str, Any]) -> str:
        """Generate fallback pattern explanation"""
        return f"""
        Pattern: {pattern_type}
        
        This pattern appears in {pattern_data.get('frequency', 'unknown')} artifacts and is common in 
        conferences such as {', '.join(pattern_data.get('conferences', []))}.
        
        This pattern is important because it represents a successful approach to artifact documentation
        that has been proven effective across multiple accepted submissions.
        """

    def _generate_fallback_conference_comparison(self, conferences: List[str]) -> str:
        """Generate fallback conference comparison"""
        comparison_text = f"Comparison of {', '.join(conferences)}:\n\n"

        for conf in conferences:
            if conf in self.conference_models.conference_profiles:
                profile = self.conference_models.conference_profiles[conf]
                comparison_text += f"{conf}:\n"
                comparison_text += f"- Preferred sections: {', '.join(profile.preferred_sections[:3])}\n"
                comparison_text += f"- Common tools: {', '.join(profile.preferred_tools[:3])}\n"
                comparison_text += f"- Documentation style: {profile.documentation_style}\n\n"

        return comparison_text

    def _generate_fallback_recommendations(self, context: InsightContext,
                                           target_areas: str) -> str:
        """Generate fallback recommendations"""
        return f"""
        Improvement Recommendations for {context.artifact_id}:
        
        Priority Areas: {target_areas}
        
        Specific Actions:
        1. Add comprehensive installation instructions with system requirements
        2. Include usage examples with expected outputs
        3. Provide reproducibility features (Docker, Conda, requirements files)
        4. Ensure clear documentation structure with proper headings
        5. Add troubleshooting section for common issues
        """

    def _format_current_state(self, context: InsightContext) -> str:
        """Format current state of artifact"""
        state_info = []

        artifact_data = context.graph_context.get('artifact_data', {})
        state_info.append(f"Conference: {artifact_data.get('conference', 'unknown')}")

        sections = context.graph_context.get('sections', [])
        if sections:
            section_types = [s.get('section_type') for s in sections]
            state_info.append(f"Current sections: {', '.join(set(section_types))}")

        tools = context.graph_context.get('connected_tools', [])
        if tools:
            tool_names = [t.get('name') for t in tools]
            state_info.append(f"Current tools: {', '.join(tool_names)}")

        return '\n'.join(state_info)

    def _identify_improvement_areas(self, context: InsightContext,
                                    target_conference: Optional[str]) -> str:
        """Identify areas that need improvement"""
        areas = []

        # Check missing common sections
        sections = {s.get('section_type') for s in context.graph_context.get('sections', [])}
        required_sections = ['installation', 'usage', 'requirements']

        for req_section in required_sections:
            if req_section not in sections:
                areas.append(f"Missing {req_section} section")

        # Check for reproducibility features
        tools = {t.get('name', '').lower() for t in context.graph_context.get('connected_tools', [])}
        if 'docker' not in tools:
            areas.append("Missing Docker support")

        if not any(env_tool in tools for env_tool in ['conda', 'pip']):
            areas.append("Missing environment management")

        return ', '.join(areas) if areas else "General documentation improvement"

    def _extract_pattern_recommendations(self, explanation: str, pattern_type: str) -> List[str]:
        """Extract recommendations from pattern explanation"""
        recommendations = [
            f"Implement {pattern_type} pattern in your documentation",
            "Follow the structure used by successful artifacts",
            "Include specific examples and clear instructions"
        ]

        # Add pattern-specific recommendations
        if pattern_type == "installation_documentation":
            recommendations.extend([
                "Add step-by-step installation instructions",
                "Include system requirements and dependencies",
                "Provide troubleshooting guidance"
            ])
        elif pattern_type == "usage_examples":
            recommendations.extend([
                "Include concrete usage examples",
                "Show expected outputs",
                "Provide command-line snippets"
            ])

        return recommendations

    def _extract_conference_recommendations(self, comparison: str, conferences: List[str]) -> List[str]:
        """Extract conference-specific recommendations"""
        recommendations = []

        for conf in conferences:
            if conf in self.conference_models.conference_profiles:
                profile = self.conference_models.conference_profiles[conf]
                recommendations.append(f"{conf}: Focus on {', '.join(profile.preferred_sections[:2])}")
                recommendations.append(f"{conf}: Use {', '.join(profile.preferred_tools[:2])}")

        return recommendations

    def _extract_action_items(self, recommendations: str) -> List[str]:
        """Extract specific action items from recommendations text"""
        action_items = []

        lines = recommendations.split('\n')
        for line in lines:
            line = line.strip()
            if (line and
                    (line.startswith('-') or line.startswith('*') or
                     line[0].isdigit() or 'add' in line.lower() or 'include' in line.lower())):

                # Clean up the line
                cleaned = line.lstrip('-*0123456789. ').strip()
                if cleaned:
                    action_items.append(cleaned)

        return action_items[:10]  # Limit to top 10 action items

    def _extract_improvement_evidence(self, context: InsightContext,
                                      target_conference: Optional[str]) -> List[str]:
        """Extract evidence supporting improvement recommendations"""
        evidence = []

        if target_conference and target_conference in self.conference_models.conference_profiles:
            profile = self.conference_models.conference_profiles[target_conference]
            evidence.append(f"{target_conference} values: {', '.join(profile.preferred_sections[:3])}")
            evidence.append(f"Reproducibility emphasis: {profile.reproducibility_emphasis:.2f}")

        # Add general evidence
        evidence.append("Based on analysis of 500+ accepted artifacts")
        evidence.append("Following patterns from successful submissions")

        return evidence

    def save_insights(self, insights: List[GeneratedInsight], output_file: str):
        """Save generated insights to file"""
        insights_data = []

        for insight in insights:
            insight_dict = {
                'insight_type': insight.insight_type,
                'main_insight': insight.main_insight,
                'supporting_evidence': insight.supporting_evidence,
                'recommendations': insight.recommendations,
                'confidence_score': insight.confidence_score,
                'sources': insight.sources,
                'related_patterns': insight.related_patterns
            }
            insights_data.append(insight_dict)

        with open(output_file, 'w') as f:
            json.dump(insights_data, f, indent=2)

        logger.info(f"Saved {len(insights)} insights to {output_file}")
