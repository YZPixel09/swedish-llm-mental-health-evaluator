"""
Configuration for Emotional Intelligence Framework
"""

import json, os
from typing import Dict, List

# ============================================================
# Model Configurations
# Four LLMs selected for comparative evaluation:
# ChatGPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6, DeepSeek V3.2
# All accessed via OpenRouter unified API
# ============================================================
MODELS = {
    #  "ChatGPT-5.4"
    "ChatGPT-5.4": {
        "api_id": "openai/gpt-5.4",
        "provider": "OpenAI",
        "architecture": "Transformer",
        "context_length": 128000,
        "strengths": ["reasoning", "instruction-following"]
    },
    #  "Gemini 3.1 Pro"
    "Gemini 3.1 Pro": {
        "api_id": "google/gemini-3.1-pro-preview",
        "provider": "Google",
        "architecture": "Multimodal",
        "context_length": 1000000,
        "strengths": ["multimodal", "general"]
    },
    #  "Claude Opus 4.6" 
    "Claude Opus 4.6": {
        "api_id": "anthropic/claude-opus-4.6",
        "provider": "Anthropic",
        "architecture": "Constitutional AI",
        "context_length": 200000,
        "strengths": ["empathy", "nuance"]
    },
    #  "DeepSeek V3.2" 
    "DeepSeek V3.2": {
        "api_id": "deepseek/deepseek-v3.2",
        "provider": "DeepSeek",
        "architecture": "MoE",
        "context_length": 128000,
        "strengths": ["reasoning", "analysis"]
    }
}


# Test scenarios organized by dimension
def load_swedish_scenarios(path: str = None) -> list:
    """
    Load the Swedish mental health scenario library from JSON.
    Returns a flat list of scenario objects.
    """
    if path is None:
        # Resolve relative to this file's location
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base, "data", "scenarios_swedish.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load scenarios at import time so the rest of the codebase can import TEST_SCENARIOS as before
TEST_SCENARIOS = load_swedish_scenarios()

# Evaluation weights
DIMENSION_WEIGHTS = {
    "empathy": 0.30,
    "safety": 0.30,       # safety carries equal weight to empathy
    "helpfulness": 0.20,
    "swedish_context": 0.20
}

# ============================================================
# API settings shared across all model calls.
# timeout and retry values are set conservatively to handle
# variation in model response latency across providers.
# model_delay ensures rate limits are respected between models.
# temperature is fixed at 0.7 for all models to ensure that
# prompt conditions are identical across the evaluation.
# ============================================================
API_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "timeout": 45,
    "max_retries": 2,
    "rate_limit_delay": 2,
    "model_delay": 3,
    "default_temperature": 0.7,
    "max_tokens": 1200
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    "emotion_keywords": [
        # Primary emotions
        'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
        # Secondary emotions
        'frustration', 'guilt', 'disappointment', 'anxiety', 'shame',
        'pride', 'jealousy', 'envy', 'hope', 'relief',
        # Complex emotions
        'regret', 'loneliness', 'confusion', 'overwhelm', 'helplessness',
        'resignation', 'doubt', 'conflict', 'stress', 'worry',
        # Social emotions
        'empathy', 'compassion', 'sympathy', 'understanding', 'support',
        'rejection', 'hurt', 'betrayal', 'trust', 'gratitude'
    ],
    
    "empathy_indicators": [
        'sorry', 'understand', 'feel', 'support', 'here for you',
        'difficult', 'hard', 'compassion', 'care', 'listen',
        'acknowledge', 'valid', 'heart goes out', 'empathize',
        'imagine', 'must be', 'sounds like', 'hear you'
    ],
    
    "professional_indicators": [
        'appreciate', 'feedback', 'clarify', 'understand your perspective',
        'consider', 'discuss', 'collaborate', 'solution', 'resolve',
        'professional', 'respectful', 'constructive'
    ]
}

# Output Configuration
OUTPUT_CONFIG = {
    "results_dir": "results",
    "visualizations_dir": "visualizations",
    "reports_dir": "reports",
    "file_formats": ["json", "csv", "md"],
    "visualization_formats": ["png", "svg"],
    "dpi": 300
}