"""
Emotional Intelligence Framework for Large Language Models

A comprehensive framework for evaluating emotional intelligence in LLMs
across multiple dimensions including emotion recognition, empathy,
emotional regulation, and social awareness.
"""

from .ei_framework import EmotionalIntelligenceFramework
from .ei_improved_framework import ImprovedEmotionalIntelligenceFramework
from .ei_analyzer import EIAnalyzer
from .ei_config import MODELS, TEST_SCENARIOS, API_CONFIG
from .swedish_mental_health_evaluator import (
    Dimension,
    DimensionResult,
    EvaluationResult,
    EvaluationSettings,
    RiskLevel,
    RuleDefinition,
    RuleEvidence,
    Scenario,
    SwedishMentalHealthEvaluator,
    compare_with_human_ratings,
    evaluate_response,
    run_ablation_study,
    summarize_model_performance,
)

__version__ = "1.0.0"
__author__ = "Lorenzo De Tomasi"
__email__ = "lorenzo.detomasi@graduate.univaq.it"

__all__ = [
    "EmotionalIntelligenceFramework",
    "ImprovedEmotionalIntelligenceFramework", 
    "EIAnalyzer",
    "MODELS",
    "TEST_SCENARIOS",
    "API_CONFIG",
    "Dimension",
    "DimensionResult",
    "EvaluationResult",
    "EvaluationSettings",
    "RiskLevel",
    "RuleDefinition",
    "RuleEvidence",
    "Scenario",
    "SwedishMentalHealthEvaluator",
    "compare_with_human_ratings",
    "evaluate_response",
    "run_ablation_study",
    "summarize_model_performance",
]
