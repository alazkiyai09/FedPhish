# Phishing Classifier Benchmark - Classical ML Baselines

**Day 2 of 21-Day Phishing Detection Project**

Establishing baseline performance with classical machine learning classifiers before implementing federated and deep learning versions (Guard-GBDT, MultiPhishGuard).

---

## Project Overview

### Purpose
This benchmark provides rigorous baseline performance metrics for 7 classical ML classifiers on phishing detection. Results will serve as reference points when implementing:
- **Guard-GBDT**: Federated gradient boosted decision trees
- **MultiPhishGuard**: Multi-party federated learning system

### Tech Stack
- **Classifiers**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM (RBF), Gradient Boosted Trees
- **Optimization**: Optuna (50 trials per model)
- **Interpretation**: SHAP, Partial Dependence Plots
- **Evaluation**: Stratified 5-fold CV, temporal split

### Dataset
- Uses feature pipeline from **Day 1**: `/home/ubuntu/21Days_Project/day1_feature_pipeline/features/final_features.csv`
- Features: TF-IDF, character n-grams, linguistic features, URL analysis
- Target: Binary classification (Legitimate vs. Phishing)

---

## Installation

```bash
cd /home/ubuntu/21Days_Project/day2_classical_ml_benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
scikit-learn >= 1.3.0
xgboost >= 2.0.0
lightgbm >= 4.0.0
catboost >= 1.2.0
optuna >= 3.3.0
shap >= 0.42.0
imbalanced-learn >= 0.11.0
```

---

## Usage

### Run Full Benchmark

```python
from src.benchmark import run_benchmark

# Run complete evaluation
results = run_benchmark(
    features_path="path/to/features.csv",
    output_dir="results/",
    tune_hyperparams=True,      # Optuna tuning (50 trials/model)
    run_interpretation=True,    # SHAP, PDPs, decision boundaries
    run_analysis=True           # Error analysis, edge cases
)
```

### Quick Evaluation (No Tuning)

```python
results = run_benchmark(
    tune_hyperparams=False,  # Skip tuning, use defaults
    run_interpretation=False,
    run_analysis=False
)
```

### Individual Components

```python
from src.preprocessing import load_features, create_splits, prepare_data_for_model
from src.models import XGBoostClassifier
from src.evaluation import stratified_cv, temporal_evaluation
from src.tuning import optimize_hyperparams

# Load data
df = load_features()
train_df, val_df, test_df = create_splits(df)
X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_data_for_model(
    train_df, val_df, test_df
)

# Train and evaluate
model = XGBoostClassifier()
aggregated_metrics, fold_results = stratified_cv(model, X_train, y_train, n_folds=5)
temporal_metrics = temporal_evaluation(model, X_train, y_train, X_test, y_test)

# Hyperparameter tuning
best_params = optimize_hyperparams(
    XGBoostClassifier, X_train, y_train, X_val, y_val, "xgboost", n_trials=50
)
```

---

## Evaluation Framework

### Cross-Validation
- **Stratified 5-fold CV**: Preserves class distribution across folds
- **Temporal Split**: Train on older data, test on newer (simulates real deployment)
- **Metrics reported**: mean ± standard deviation

### Metrics

| Metric | Description | Financial Sector Requirement |
|--------|-------------|------------------------------|
| **Accuracy** | Overall correctness | - |
| **Precision** | True positives / (TP + FP) | - |
| **Recall** | True positives / (TP + FN) | **> 95% on financial phishing** |
| **F1 Score** | Harmonic mean of precision/recall | - |
| **AUPRC** | Area under PR curve (better for imbalanced data) | - |
| **AUROC** | Area under ROC curve | - |
| **FPR** | False positives / (FP + TN) | **< 1%** |

### Compute Budget
All models evaluated with:
- Same preprocessing pipeline
- Same compute budget
- `random_state = 42` (reproducibility)
- Training time and inference time tracked

---

## Benchmark Results

> **Note**: Run the benchmark to populate this table. Results will be saved to `results/benchmark_summary.csv` and `results/results_table.md`.

### Model Performance Summary

| Model | F1 (mean±std) | AUPRC (mean±std) | AUROC (mean±std) | FPR (mean±std) |
|-------|---------------|------------------|------------------|----------------|
| Logistic Regression | - ± - | - ± - | - ± - | - ± - |
| Random Forest | - ± - | - ± - | - ± - | - ± - |
| XGBoost | - ± - | - ± - | - ± - | - ± - |
| LightGBM | - ± - | - ± - | - ± - | - ± - |
| CatBoost | - ± - | - ± - | - ± - | - ± - |
| SVM (RBF) | - ± - | - ± - | - ± - | - ± - |
| Gradient Boosted Trees | - ± - | - ± - | - ± - | - ± - |

### Financial Phishing Subset Performance

| Model | Recall (Financial) | FPR | Meets Requirements |
|-------|-------------------|-----|-------------------|
| Logistic Regression | - | - | ❌/✅ |
| Random Forest | - | - | ❌/✅ |
| XGBoost | - | - | ❌/✅ |
| LightGBM | - | - | ❌/✅ |
| CatBoost | - | - | ❌/✅ |
| SVM (RBF) | - | - | ❌/✅ |
| Gradient Boosted Trees | - | - | ❌/✅ |

**Requirements**: Recall > 95% on financial phishing, FPR < 1%

### Compute Performance

| Model | Training Time (s) | Inference Time (s) |
|-------|------------------|-------------------|
| Logistic Regression | - | - |
| Random Forest | - | - |
| XGBoost | - | - |
| LightGBM | - | - |
| CatBoost | - | - |
| SVM (RBF) | - | - |
| Gradient Boosted Trees | - | - |

---

## Model Interpretation

### Feature Importance
- **Native Importance**: Built-in feature importance (Gini, gain, etc.)
- **SHAP Values**: Game-theoretic importance, model-agnostic
- **Output**: `results/interpretation/{model}_native_importance.png`
- **Output**: `results/interpretation/{model}_shap_importance.png`

### Partial Dependence Plots
- Shows marginal effect of top 5 features on predictions
- **Output**: `results/interpretation/{model}_pdp_top_5.png`

### Decision Boundaries
- 2D PCA projection of decision regions
- **Output**: `results/interpretation/{model}_decision_boundary.png`

---

## Error Analysis

### Confusion Matrix Examples
- Representative emails from each confusion matrix cell
- **Output**: `results/analysis/{model}_confusion_matrix_with_examples.png`
- **Output**: `results/analysis/{model}_{cell_type}_examples.csv`

### False Negative Analysis
- **Critical**: Missed phishing emails (security risk)
- Near-miss vs. far-miss classification
- Financial phishing false negatives tracked separately

### False Positive Analysis
- **Critical**: Legitimate emails flagged (user trust)
- High-confidence vs. low-confidence false positives
- FPR tracked against 1% threshold

### Edge Cases
- Short emails (< 50 chars)
- High missing features (> 30%)
- Uncertain predictions (0.4 < p < 0.6)
- Outliers (> 3 std dev)
- Encoding issues (non-ASCII)
- **Output**: `results/analysis/edge_cases_report.csv`

---

## Project Structure

```
day2_classical_ml_benchmark/
├── data/                          # Data files
├── src/
│   ├── config.py                  # Central configuration
│   ├── preprocessing.py           # Data loading and preparation
│   ├── models/                    # Classifier implementations
│   │   ├── base_classifier.py     # Abstract base class
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── xgboost_wrapper.py
│   │   ├── lightgbm_wrapper.py
│   │   ├── catboost_wrapper.py
│   │   ├── svm.py
│   │   └── gbdt_reference.py      # Reference for Guard-GBDT
│   ├── evaluation/                # Metrics and CV
│   │   ├── metrics.py
│   │   ├── cross_validation.py
│   │   └── temporal_split.py
│   ├── tuning/                    # Optuna hyperparameter optimization
│   │   └── optuna_study.py
│   ├── interpretation/            # SHAP, PDPs, decision boundaries
│   │   ├── feature_importance.py
│   │   ├── partial_dependence.py
│   │   └── decision_boundary.py
│   ├── analysis/                  # Error analysis
│   │   ├── error_analysis.py
│   │   ├── confusion_examples.py
│   │   └── edge_cases.py
│   └── benchmark.py               # Main orchestration script
├── tests/                         # Unit tests
│   ├── test_metrics.py
│   ├── test_cv.py
│   └── test_preprocessing.py
├── results/                       # Benchmark outputs
│   ├── benchmark_summary.csv
│   ├── results_table.md
│   ├── interpretation/
│   └── analysis/
├── notebooks/                     # Jupyter notebooks (optional)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Key Findings and Recommendations

> **To be populated after running benchmark**

### Overall Performance
1. **Best Model**: [To be determined]
2. **Fastest Training**: [To be determined]
3. **Best Interpretability**: [To be determined]

### Financial Sector Requirements
- Models meeting **recall > 95%** on financial phishing: [To be determined]
- Models meeting **FPR < 1%**: [To be determined]
- **Recommended model**: [To be determined]

### Next Steps
1. ✅ Classical ML baselines (Day 2) - **This project**
2. ⏳ Guard-GBDT implementation (federated learning)
3. ⏳ MultiPhishGuard implementation (multi-party FL)
4. ⏳ Deep learning baselines (BERT, RoBERTa)
5. ⏳ Comparative analysis: Classical vs. FL vs. Deep Learning

---

## Reproducibility

All experiments use:
- **Random state**: 42 (everywhere)
- **Compute budget**: Equal for all models
- **Preprocessing**: Identical pipeline for all models
- **Cross-validation**: Stratified 5-fold with same splits
- **Hyperparameter tuning**: Separate tuning set, Optuna with 50 trials

To reproduce results:
```bash
python src/benchmark.py
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=src tests/
```

---

## Citation

**If you use this benchmark in your research:**

```bibtex
@misc{phishing_benchmark_2025,
  title={Phishing Classifier Benchmark: Classical ML Baselines},
  author={Your Name},
  year={2025},
  note={21-Day Phishing Detection Project, Day 2}
}
```

---

## References

- **Fraud Detection Background**: 3+ years with SAS Fraud Management
- **Previous Work**: 21-Day Federated Learning portfolio
- **Day 1 Feature Pipeline**: `/home/ubuntu/21Days_Project/phishing_email_analysis/`

---

## License

MIT License - See LICENSE file for details.

---

**Last Updated**: 2025-01-29
