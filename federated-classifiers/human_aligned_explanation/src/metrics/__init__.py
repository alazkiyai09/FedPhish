"""
Quality metrics for explanations.

Metrics to evaluate explanation quality:
- Faithfulness: Does explanation match model reasoning?
- Consistency: Do similar inputs get similar explanations?
- Human evaluation: User study metrics
"""

from src.metrics.faithfulness import compute_faithfulness
from src.metrics.consistency import compute_consistency
from src.metrics.human_eval import HumanEvaluationMetrics

__all__ = [
    "compute_faithfulness",
    "compute_consistency",
    "HumanEvaluationMetrics",
]
