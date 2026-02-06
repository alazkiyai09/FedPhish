# Day 3: Transformer-Based Phishing Detection

Deep learning approach to phishing email detection using pretrained transformer models (BERT, RoBERTa, DistilBERT) with LoRA adaptation for federated learning scenarios.

## Overview

This project implements and compares four transformer-based architectures for binary phishing email classification:

1. **BERT-base**: Standard fine-tuning of BERT (110M parameters)
2. **RoBERTa-base**: Robustly optimized BERT approach (125M parameters)
3. **DistilBERT**: Lightweight, faster variant (66M parameters, 40% smaller)
4. **LoRA-BERT**: Parameter-efficient fine-tuning with LoRA adapters (~300K trainable params)

## Key Features

- **Special Token Structure**: Uses `[SUBJECT]`, `[BODY]`, `[URL]`, `[SENDER]` tokens for email structure
- **Smart Truncation**: Head+tail strategy preserves important context from both ends
- **Mixed Precision**: FP16 training for faster computation and lower memory usage
- **Gradient Accumulation**: Simulates larger batch sizes without memory overhead
- **Learning Rate Scheduling**: Linear warmup + decay for stable convergence
- **Early Stopping**: Monitors validation AUPRC to prevent overfitting
- **Attention Visualization**: Extract and visualize attention weights for interpretability
- **ONNX Export**: Deployment-ready model format
- **LoRA Support**: Separate adapter weights for federated learning aggregation

## Comparison with Day 2 Classical ML

| Feature | Day 2 (Classical ML) | Day 3 (Transformers) |
|---------|---------------------|----------------------|
| Input | 60+ engineered features | Raw email text |
| Feature Engineering | Manual (TF-IDF, handcrafted) | Automatic (learned embeddings) |
| Best Model | XGBoost/Random Forest | BERT/RoBERTa |
| Training Time | Seconds-minutes | Hours |
| Inference Time | <1ms | 10-50ms |
| Accuracy | TBD | TBD |
| AUPRC | TBD | TBD |
| Feature Dependency | High (requires feature pipeline) | Low (raw text only) |
| Interpretability | Feature importance | Attention visualization |
| Deployment | Scikit-learn pickle | ONNX / TorchScript |

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download datasets
python data/download_hf_dataset.py    # Real data (18,650 samples)
python data/generate_synthetic_data.py  # Synthetic data (1,000 samples)
```

## Quick Start

### Train a single model

```bash
# Train BERT
python3 train.py --model bert --epochs 3

# Train RoBERTa
python3 train.py --model roberta --epochs 3

# Train DistilBERT (lightweight)
python3 train.py --model distilbert --epochs 3

# Train LoRA-BERT (parameter-efficient)
python3 train.py --model lora-bert --epochs 3
```

### Train all models and compare

```bash
bash experiments/scripts/compare_all.sh
```

### Custom training

```bash
python3 train.py \
    --model bert \
    --config experiments/configs/bert.yaml \
    --data data/raw/phishing_emails_hf.csv \
    --batch-size 32 \
    --lr 3e-5 \
    --epochs 5 \
    --output my_checkpoints
```

## Project Structure

```
day3_transformer_phishing/
├── data/
│   ├── raw/                    # Input datasets
│   ├── processed/              # Preprocessed data
│   ├── cache/                  # Tokenizer cache
│   ├── download_hf_dataset.py  # Download real data
│   └── generate_synthetic_data.py  # Generate synthetic data
├── src/
│   ├── data/
│   │   ├── dataset.py          # EmailDataset class
│   │   ├── preprocessor.py     # Special tokens, truncation
│   │   └── tokenizer.py        # Tokenizer wrapper
│   ├── models/
│   │   ├── base.py             # Base classifier interface
│   │   ├── bert_classifier.py  # BERT implementation
│   │   ├── roberta_classifier.py  # RoBERTa implementation
│   │   ├── distilbert_classifier.py  # DistilBERT implementation
│   │   ├── lora_classifier.py   # LoRA-BERT implementation
│   │   └── factory.py          # Model creation utility
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   ├── metrics.py          # AUPRC, attention, calibration
│   │   └── scheduler.py        # LR warmup scheduler
│   ├── inference/
│   │   ├── predictor.py        # Prediction wrapper
│   │   └── export.py           # ONNX export
│   └── utils/
│       ├── seed.py             # Reproducibility
│       ├── memory.py           # GPU tracking
│       └── config.py           # Configuration management
├── tests/
│   ├── test_data_pipeline.py   # Data tests
│   └── test_models.py          # Model tests
├── experiments/
│   ├── configs/                # YAML configs
│   └── scripts/                # Training scripts
├── train.py                    # Main training script
├── setup_env.sh               # Environment setup
└── README.md
```

## Model Architecture Details

### BERT Classifier
- **Base**: `bert-base-uncased` (12 layers, 768 hidden, 12 heads, 110M params)
- **Classification Head**: Dropout (0.1) → Linear (768 → 2)
- **Training**: Full fine-tuning of all parameters

### RoBERTa Classifier
- **Base**: `roberta-base` (12 layers, 768 hidden, 12 heads, 125M params)
- **Improvements**: Dynamic masking, larger byte-pair encoding vocabulary
- **Training**: Full fine-tuning

### DistilBERT Classifier
- **Base**: `distilbert-base-uncased` (6 layers, 768 hidden, 12 heads, 66M params)
- **Optimization**: 40% smaller, 60% faster, 97% of BERT performance
- **Use Case**: Efficiency comparison, edge deployment

### LoRA-BERT Classifier
- **Base**: `bert-base-uncased` with LoRA adapters
- **LoRA Config**: Rank=8, Alpha=16, Target=[query, key, value]
- **Trainable Params**: ~300K (0.3% of base model)
- **Use Case**: Federated learning, resource-constrained training

## Training Configuration

Default hyperparameters (can be overridden via config files):

```yaml
learning_rate: 2e-5
batch_size: 16
gradient_accumulation_steps: 1
num_epochs: 3
warmup_ratio: 0.1
weight_decay: 0.01
max_length: 512
truncation_strategy: head_tail  # Keep both start and end
fp16: true
early_stopping_patience: 3
early_stopping_metric: auprc
seed: 42
```

## Evaluation Metrics

Matches Day 2 for fair comparison:

- **Primary**: AUPRC (Area Under Precision-Recall Curve)
- **Secondary**: AUROC, Accuracy, Precision, Recall, F1
- **Financial Requirements**: Recall > 95%, FPR < 1%
- **Calibration**: Confidence vs accuracy curves
- **Interpretability**: Attention visualization heatmaps

## Inference Example

```python
from src.inference.predictor import Predictor

# Load trained model
predictor = Predictor.from_checkpoint('checkpoints/bert/best_model.pt')

# Predict single email
email = {
    'subject': 'URGENT: Verify your account now',
    'body': 'Click here to verify your account or it will be suspended',
    'sender': 'security@bank-alert.com',
    'url': 'http://verify-account-bad.com'
}

result = predictor.predict(email)
print(f"Label: {result['label_text']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

# Get attention for visualization
attention_data = predictor.get_attention_for_email(email, layer=11, head=0)
```

## Export for Deployment

```python
from src.inference.export import export_to_onnx, create_deployment_package

# Export to ONNX
export_to_onnx(
    model=model,
    tokenizer=tokenizer,
    output_path='models/bert.onnx'
)

# Create full deployment package
create_deployment_package(
    model=model,
    tokenizer=tokenizer,
    output_dir='deployment/bert',
    export_onnx=True
)
```

## Unit Tests

```bash
# Test data pipeline
python3 tests/test_data_pipeline.py

# Test model forward pass
python3 tests/test_models.py

# Run all with pytest
pytest tests/
```

## Results

### Model Comparison

| Model | Accuracy | AUPRC | AUROC | F1 | Train Time | Inference Time | Model Size |
|-------|----------|-------|-------|----|----|----|----|
| BERT | TBD | TBD | TBD | TBD | TBD | TBD | 440 MB |
| RoBERTa | TBD | TBD | TBD | TBD | TBD | TBD | 500 MB |
| DistilBERT | TBD | TBD | TBD | TBD | TBD | TBD | 260 MB |
| LoRA-BERT | TBD | TBD | TBD | TBD | TBD | TBD | 440 MB (300K trainable) |

*Results will be updated after training completes*

## Attention Visualization Examples

### Phishing Email Example
```
Subject: URGENT: Verify Your Account
Body: Click here http://bad-site.com to verify

Attention Head 11, Layer 11:
┌─────────────────────────────────────┐
│ [CLS] [SUBJ] URGENT Verify Account │
│  ████ █████ █████ █████ █████       │
│                                     │
│ [BODY] Click here http://bad-site   │
│        █████ █████ █████ █████      │
└─────────────────────────────────────┘
```

*Model focuses on urgency keywords and URL*

## References

- **Devlin et al. (2018)**: BERT: Pre-training of Deep Bidirectional Transformers
- **Liu et al. (2019)**: RoBERTa: A Robustly Optimized BERT Pretraining Approach
- **Sanh et al. (2019)**: DistilBERT: A Distilled Version of BERT
- **Hu et al. (2021)**: LoRA: Low-Rank Adaptation of Large Language Models
- **MultiPhishGuard**: Giovanni Russello's work on LLM-based phishing detection

## Future Extensions

- **Multimodal**: Add URL screenshot input (vision transformer)
- **Federated Learning**: Distributed training with LoRA aggregation
- **Active Learning**: Query strategy for selecting informative samples
- **Adversarial Training**: Robustness against phishing evasion attacks
- **Zero-shot Detection**: Leverage LLMs without task-specific fine-tuning

## License

MIT License

## Author

21-Day Portfolio Project - Transformer Phishing Detection
PhD Preparation with Giovanni Russello
