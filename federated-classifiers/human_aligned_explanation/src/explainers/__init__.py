"""
Explanation algorithms for phishing detection.

This module contains various explanation methods:
- Feature-based: SHAP values
- Attention-based: Transformer attention visualization
- Counterfactual: What-if scenarios
- Comparative: Similar to known campaigns
"""

from src.explainers.feature_based import FeatureBasedExplainer
from src.explainers.attention_based import AttentionBasedExplainer
from src.explainers.counterfactual import CounterfactualExplainer
from src.explainers.comparative import ComparativeExplainer

__all__ = [
    "FeatureBasedExplainer",
    "AttentionBasedExplainer",
    "CounterfactualExplainer",
    "ComparativeExplainer",
]
