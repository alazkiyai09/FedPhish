# Complete Usage Guide - Day 3 Transformer Phishing Detection

This guide provides detailed, step-by-step instructions for training, evaluating, and deploying the transformer-based phishing detection models.

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [Training Models](#3-training-models)
4. [Monitoring Training](#4-monitoring-training)
5. [Using Trained Models](#5-using-trained-models)
6. [Exporting Models](#6-exporting-models)
7. [Performance Comparison](#7-performance-comparison)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup

### 1.1 Install Dependencies

```bash
# Navigate to project directory
cd /home/ubuntu/21Days_Project/day3_transformer_phishing

# Install required packages
pip3 install torch transformers datasets peft wandb scikit-learn pandas numpy --break-system-packages

# Verify installation
python3 -c "import torch; import transformers; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__)"
```

Expected output:
```
PyTorch: 2.x.x
Transformers: 4.x.x
```

### 1.2 Verify Project Structure

```bash
# Check that all directories exist
ls -la src/{data,models,training,inference,utils}
ls -la data/{raw,processed,cache}
ls -la experiments/{configs,scripts}
```

---

## 2. Data Preparation

### 2.1 Download Real Dataset (Already Done)

The real phishing dataset has been downloaded from HuggingFace:

```bash
# Verify dataset exists
ls -lh data/raw/phishing_emails_hf.csv
# Output: 18,634 emails (11,322 legitimate + 7,312 phishing)
```

**Dataset Details:**
- **Total Samples**: 18,634
- **Class Balance**: 39.2% phishing, 60.8% legitimate
- **Columns**: `text` (email content), `label` (0=legitimate, 1=phishing)

### 2.2 Create Smaller Subsets (Already Done)

For faster training iteration, smaller subsets have been created:

```bash
# Verify subset files exist
ls -lh data/processed/

# 2k subset: 2,000 samples (784 phishing, 1,216 legitimate)
# 5k subset: 5,000 samples (1,962 phishing, 3,038 legitimate)
```

**Recommended Training Strategy:**
1. **Quick validation**: Train on 2k subset with 1 epoch (~2-4 hours)
2. **Full training**: Train on full dataset with 3 epochs (~12-24 hours per model)

### 2.3 Dataset Splits

When you run training, the data is automatically split:
- **Training**: 60% (for 2k: 1,200 samples)
- **Validation**: 20% (for 2k: 400 samples)
- **Test**: 20% (for 2k: 400 samples)

Splits are stratified to maintain class balance.

---

## 3. Training Models

### 3.1 Quick Start - Train Single Model

```bash
# Basic training command
python3 train.py --model bert --epochs 3
```

This uses default settings:
- Data: `data/raw/phishing_emails_hf.csv` (full dataset)
- Batch size: 16
- Learning rate: 2e-5
- Device: CPU (or GPU if available)

### 3.2 Recommended Training Commands

#### Option A: Fast Training (2k subset, 1 epoch)

```bash
# Train all models on 2k subset with 1 epoch (~2-4 hours each)
python3 train.py --model distilbert --data data/processed/phishing_emails_2k.csv --epochs 1 --batch-size 8 --no-wandb > logs/train_distilbert_2k.log 2>&1 &
python3 train.py --model bert --data data/processed/phishing_emails_2k.csv --epochs 1 --batch-size 8 --no-wandb > logs/train_bert_2k.log 2>&1 &
python3 train.py --model roberta --data data/processed/phishing_emails_2k.csv --epochs 1 --batch-size 8 --no-wandb > logs/train_roberta_2k.log 2>&1 &
python3 train.py --model lora-bert --data data/processed/phishing_emails_2k.csv --epochs 1 --batch-size 8 --no-wandb > logs/train_lorabert_2k.log 2>&1 &

# Monitor progress
tail -f logs/train_*.log
```

#### Option B: Full Training (Complete dataset, 3 epochs)

```bash
# Train models one at a time to avoid memory issues
# Each model will take ~12-24 hours on CPU

# 1. DistilBERT (fastest - 66M params)
python3 train.py --model distilbert --data data/raw/phishing_emails_hf.csv --epochs 3 --batch-size 8 --no-wandb > logs/train_distilbert_full.log 2>&1 &

# 2. BERT (baseline - 110M params)
python3 train.py --model bert --data data/raw/phishing_emails_hf.csv --epochs 3 --batch-size 8 --no-wandb > logs/train_bert_full.log 2>&1 &

# 3. LoRA-BERT (parameter-efficient - 300K trainable params)
python3 train.py --model lora-bert --data data/raw/phishing_emails_hf.csv --epochs 3 --batch-size 8 --no-wandb > logs/train_lorabert_full.log 2>&1 &

# 4. RoBERTa (largest - 125M params)
python3 train.py --model roberta --data data/raw/phishing_emails_hf.csv --epochs 3 --batch-size 8 --no-wandb > logs/train_roberta_full.log 2>&1 &
```

#### Option C: Memory-Optimized Training

If you encounter out-of-memory errors, use these settings:

```bash
# Reduce batch size to 4
python3 train.py --model bert --data data/raw/phishing_emails_hf.csv --epochs 3 --batch-size 4 --no-wandb > logs/train_bert_small.log 2>&1 &

# Use 5k subset instead of full dataset
python3 train.py --model bert --data data/processed/phishing_emails_5k.csv --epochs 3 --batch-size 8 --no-wandb > logs/train_bert_5k.log 2>&1 &
```

### 3.3 Training Command Arguments

```bash
python3 train.py [OPTIONS]

Required Arguments:
  --model {bert,roberta,distilbert,lora-bert}
      Model architecture to train

Optional Arguments:
  --data PATH
      Path to training CSV file (default: data/raw/phishing_emails_hf.csv)

  --epochs INT
      Number of training epochs (default: 3)

  --batch-size INT
      Training batch size (default: 16, reduce to 8 or 4 if OOM)

  --lr FLOAT
      Learning rate (default: 2e-5)

  --max-length INT
      Maximum sequence length in tokens (default: 512)

  --seed INT
      Random seed for reproducibility (default: 42)

  --output-dir PATH
      Directory to save checkpoints (default: checkpoints/)

  --no-wandb
      Disable Weights & Biases logging (use for offline training)

  --device {cpu,cuda,auto}
      Device to use for training (default: auto)
```

### 3.4 Configuration Files

You can also train using YAML configs:

```bash
# Train using config file
python3 train.py --config experiments/configs/bert.yaml

# View available configs
ls experiments/configs/
cat experiments/configs/bert.yaml
```

---

## 4. Monitoring Training

### 4.1 Monitor Logs in Real-Time

```bash
# Watch training progress (press Ctrl+C to stop watching)
tail -f logs/train_bert_2k.log

# Watch all models at once
tail -f logs/train_*.log
```

### 4.2 Check Training Progress

```bash
# Check how many batches completed
grep "Epoch.*batch" logs/train_bert_2k.log | tail -5

# Check current loss
grep "loss=" logs/train_bert_2k.log | tail -5

# Check if training is still running
ps aux | grep "python3 train.py"
```

### 4.3 Expected Output During Training

```
🏋️  Creating trainer...
📈 Learning rate scheduler initialized
   Warmup steps: 140 (10%)
   Total steps: 1398

🚀 Trainer initialized
   Device: cpu
   FP16: True
   Gradient accumulation: 1
   Effective batch size: 8

============================================================
Starting training for 3 epochs
============================================================

📚 Epoch 1/3
Epoch 1:   5%|█▍        | 70/1398 [05:23<1:40:15, 4.61s/it, loss=0.4523, lr=1.43e-07]

📊 Validation Results:
   Loss: 0.3124
   Accuracy: 91.20%
   Precision: 89.50%
   Recall: 88.30%
   F1: 88.90%
   AUPRC: 0.9234
   AUROC: 0.9512

✅ Best model saved to checkpoints/bert/best_model.pt
```

### 4.4 Check Completed Training

```bash
# List all trained models
ls -lh checkpoints/

# Check if best model exists
ls -lh checkpoints/bert/best_model.pt
ls -lh checkpoints/roberta/best_model.pt
ls -lh checkpoints/distilbert/best_model.pt
ls -lh checkpoints/lora-bert/best_model.pt
```

---

## 5. Using Trained Models

### 5.1 Load and Predict with Python

Create a script `predict_example.py`:

```python
#!/usr/bin/env python3
"""Example prediction script."""

import torch
from src.inference.predictor import Predictor
from src.data.tokenizer import TokenizerWrapper
from src.models.factory import create_model

def load_trained_model(model_type, checkpoint_path):
    """Load a trained model from checkpoint."""
    # Create model
    model = create_model(model_type=model_type, num_labels=2)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create tokenizer
    tokenizer = TokenizerWrapper(
        model_name={
            'bert': 'bert-base-uncased',
            'roberta': 'roberta-base',
            'distilbert': 'distilbert-base-uncased',
            'lora-bert': 'bert-base-uncased'
        }[model_type]
    )

    # Create predictor
    predictor = Predictor(model=model, tokenizer=tokenizer, device='cpu')

    return predictor

def main():
    # Load trained BERT model
    predictor = load_trained_model(
        model_type='bert',
        checkpoint_path='checkpoints/bert/best_model.pt'
    )

    # Example 1: Phishing email
    phishing_email = {
        'subject': 'URGENT: Your account will be suspended',
        'body': 'Dear customer, we detected unusual activity. Click here immediately to verify your account: http://verify-bank-login.com',
        'sender': 'security@bank-alert.com',
        'url': 'http://verify-bank-login.com'
    }

    result = predictor.predict(phishing_email)
    print("Example 1 - Phishing Email:")
    print(f"  Prediction: {result['label_text']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities: Legitimate={result['probabilities'][0]:.2%}, Phishing={result['probabilities'][1]:.2%}")
    print()

    # Example 2: Legitimate email
    legitimate_email = {
        'subject': 'Meeting Reminder: Project Update',
        'body': 'Hi team, just a reminder about our weekly sync meeting tomorrow at 2pm. Please come prepared with your progress updates.',
        'sender': 'john.smith@company.com',
        'url': ''
    }

    result = predictor.predict(legitimate_email)
    print("Example 2 - Legitimate Email:")
    print(f"  Prediction: {result['label_text']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities: Legitimate={result['probabilities'][0]:.2%}, Phishing={result['probabilities'][1]:.2%}")
    print()

    # Example 3: Get attention for visualization
    print("Example 3 - Attention Visualization:")
    attention_data = predictor.get_attention_for_email(
        phishing_email,
        layer=11,  # Last layer
        head=0     # First attention head
    )

    print(f"  Attention shape: {attention_data['attention_weights'].shape}")
    print(f"  Tokens (first 10): {attention_data['tokens'][:10]}")

if __name__ == '__main__':
    main()
```

Run the example:

```bash
python3 predict_example.py
```

### 5.2 Batch Prediction

Create a script `batch_predict.py`:

```python
#!/usr/bin/env python3
"""Batch prediction script."""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.inference.predictor import Predictor
from src.data.tokenizer import TokenizerWrapper
from src.models.factory import create_model
import torch

def batch_predict(model_type, checkpoint_path, input_csv, output_csv):
    """Run predictions on a CSV file."""

    # Load model
    print(f"Loading {model_type} model from {checkpoint_path}...")
    model = create_model(model_type=model_type, num_labels=2)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokenizer = TokenizerWrapper(
        model_name={
            'bert': 'bert-base-uncased',
            'roberta': 'roberta-base',
            'distilbert': 'distilbert-base-uncased',
            'lora-bert': 'bert-base-uncased'
        }[model_type]
    )

    predictor = Predictor(model=model, tokenizer=tokenizer, device='cpu')

    # Load data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Predict
    results = []
    print(f"Running predictions on {len(df)} emails...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        result = predictor.predict(row['text'])

        results.append({
            'text': row['text'],
            'true_label': row.get('label', 'unknown'),
            'predicted_label': result['label'],
            'predicted_label_text': result['label_text'],
            'confidence': result['confidence'],
            'prob_legitimate': result['probabilities'][0],
            'prob_phishing': result['probabilities'][1]
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

    # Print summary statistics
    if 'label' in df.columns:
        correct = (results_df['predicted_label'] == results_df['true_label']).sum()
        accuracy = correct / len(df) * 100
        print(f"\n📊 Accuracy: {accuracy:.2f}% ({correct}/{len(df)} correct)")

if __name__ == '__main__':
    # Example: Predict on test set
    batch_predict(
        model_type='bert',
        checkpoint_path='checkpoints/bert/best_model.pt',
        input_csv='data/processed/test_split.csv',
        output_csv='predictions/bert_predictions.csv'
    )
```

---

## 6. Exporting Models

### 6.1 Export to ONNX Format

Create a script `export_models.py`:

```python
#!/usr/bin/env python3
"""Export trained models to ONNX format."""

import torch
from src.inference.export import export_to_onnx, save_lora_adapters
from src.data.tokenizer import TokenizerWrapper
from src.models.factory import create_model

def export_model_to_onnx(model_type, checkpoint_path, output_onnx_path):
    """Export a trained model to ONNX format."""

    print(f"Exporting {model_type} to ONNX...")

    # Load model
    model = create_model(model_type=model_type, num_labels=2)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create tokenizer
    tokenizer = TokenizerWrapper(
        model_name={
            'bert': 'bert-base-uncased',
            'roberta': 'roberta-base',
            'distilbert': 'distilbert-base-uncased',
            'lora-bert': 'bert-base-uncased'
        }[model_type]
    )

    # Create example input
    example_input = {
        'input_ids': torch.randint(0, 30522, (1, 512)),  # Batch size 1, seq length 512
        'attention_mask': torch.ones(1, 512, dtype=torch.long)
    }

    # Export to ONNX
    export_to_onnx(
        model=model,
        tokenizer=tokenizer,
        output_path=output_onnx_path,
        example_input=example_input
    )

    print(f"✅ Exported to {output_onnx_path}")

def main():
    # Export all models
    models = [
        ('bert', 'checkpoints/bert/best_model.pt', 'models/bert.onnx'),
        ('roberta', 'checkpoints/roberta/best_model.pt', 'models/roberta.onnx'),
        ('distilbert', 'checkpoints/distilbert/best_model.pt', 'models/distilbert.onnx'),
        ('lora-bert', 'checkpoints/lora-bert/best_model.pt', 'models/lora_bert.onnx'),
    ]

    for model_type, checkpoint_path, output_path in models:
        try:
            export_model_to_onnx(model_type, checkpoint_path, output_path)

            # For LoRA model, also save adapters separately
            if model_type == 'lora-bert':
                print("Saving LoRA adapters separately...")
                save_lora_adapters(
                    model=create_model(model_type='lora-bert', num_labels=2),
                    save_path='models/lora_adapters.pt',
                    metadata={'rank': 8, 'alpha': 16}
                )
                print("✅ LoRA adapters saved to models/lora_adapters.pt")

        except Exception as e:
            print(f"❌ Error exporting {model_type}: {e}")

if __name__ == '__main__':
    main()
```

Run the export script:

```bash
python3 export_models.py
```

### 6.2 Using ONNX Models for Inference

```python
import onnxruntime as ort
import numpy as np

def load_onnx_model(onnx_path):
    """Load ONNX model for inference."""
    session = ort.InferenceSession(onnx_path)
    return session

def predict_onnx(session, input_ids, attention_mask):
    """Run inference with ONNX model."""
    outputs = session.run(
        None,
        {
            'input_ids': input_ids.astype(np.int64),
            'attention_mask': attention_mask.astype(np.int64)
        }
    )
    return outputs[0]  # Returns logits
```

---

## 7. Performance Comparison

### 7.1 Evaluate All Models

Create a script `evaluate_all.py`:

```python
#!/usr/bin/env python3
"""Evaluate all trained models and create comparison table."""

import pandas as pd
from pathlib import Path
from src.inference.predictor import Predictor
from src.data.tokenizer import TokenizerWrapper
from src.models.factory import create_model
import torch

def evaluate_model(model_type, checkpoint_path, test_data_path):
    """Evaluate a single model."""

    # Load model
    model = create_model(model_type=model_type, num_labels=2)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokenizer = TokenizerWrapper(
        model_name={
            'bert': 'bert-base-uncased',
            'roberta': 'roberta-base',
            'distilbert': 'distilbert-base-uncased',
            'lora-bert': 'bert-base-uncased'
        }[model_type]
    )

    predictor = Predictor(model=model, tokenizer=tokenizer, device='cpu')

    # Load test data
    df = pd.read_csv(test_data_path)

    # Predict
    all_preds = []
    all_probs = []
    all_labels = []

    for _, row in df.iterrows():
        result = predictor.predict(row['text'])
        all_preds.append(result['label'])
        all_probs.append(result['probabilities'][1])  # Phishing probability
        all_labels.append(row['label'])

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc
    }

def main():
    # Evaluate all models
    models = [
        ('BERT', 'checkpoints/bert/best_model.pt'),
        ('RoBERTa', 'checkpoints/roberta/best_model.pt'),
        ('DistilBERT', 'checkpoints/distilbert/best_model.pt'),
        ('LoRA-BERT', 'checkpoints/lora-bert/best_model.pt'),
    ]

    results = []

    for model_name, checkpoint_path in models:
        print(f"Evaluating {model_name}...")
        try:
            metrics = evaluate_model(
                model_type=model_name.lower().replace('-', '-bert').replace('lorabert', 'lora-bert'),
                checkpoint_path=checkpoint_path,
                test_data_path='data/processed/test_split.csv'
            )
            metrics['model'] = model_name
            results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    # Create comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc']]

    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    # Save results
    results_df.to_csv('experiments/results/model_comparison.csv', index=False)
    print("\n✅ Results saved to experiments/results/model_comparison.csv")

if __name__ == '__main__':
    main()
```

Run evaluation:

```bash
python3 evaluate_all.py
```

---

## 8. Troubleshooting

### 8.1 Out of Memory (OOM) Errors

**Symptom**: Process killed with exit code 137

**Solutions**:
```bash
# Reduce batch size
python3 train.py --model bert --batch-size 4 --epochs 3

# Use smaller dataset
python3 train.py --model bert --data data/processed/phishing_emails_2k.csv --epochs 3

# Train models sequentially instead of in parallel
# Run one training command at a time
```

### 8.2 Slow Training on CPU

**Symptom**: Training takes >10 hours per epoch

**Solutions**:
```bash
# Use DistilBERT (40% faster)
python3 train.py --model distilbert --epochs 3

# Use smaller dataset for quick iteration
python3 train.py --model bert --data data/processed/phishing_emails_2k.csv --epochs 1

# Reduce epochs
python3 train.py --model bert --epochs 1
```

### 8.3 Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**:
```bash
pip3 install transformers torch datasets peft --break-system-packages
```

### 8.4 CUDA Not Available

**Symptom**: `Device: cpu (CUDA not available)`

**Note**: This is expected if you don't have a GPU. Training will work but be slower. No action needed.

### 8.5 Check Training Progress When Stuck

```bash
# Check if process is still running
ps aux | grep "python3 train.py"

# Check recent log output
tail -50 logs/train_*.log

# Check if checkpoint is being updated
ls -lh checkpoints/*/best_model.pt
```

---

## Quick Reference Commands

```bash
# === TRAINING ===
# Fast training (2k, 1 epoch)
python3 train.py --model bert --data data/processed/phishing_emails_2k.csv --epochs 1 --batch-size 8 --no-wandb

# Full training (18k, 3 epochs)
python3 train.py --model bert --data data/raw/phishing_emails_hf.csv --epochs 3 --batch-size 8 --no-wandb

# Train all models (background)
for model in bert roberta distilbert lora-bert; do
    python3 train.py --model $model --data data/processed/phishing_emails_2k.csv --epochs 1 --batch-size 8 --no-wandb > logs/train_${model}_2k.log 2>&1 &
done

# === MONITORING ===
# Watch logs
tail -f logs/train_bert_2k.log

# Check progress
grep "Epoch.*batch" logs/train_*.log | tail -5

# List trained models
ls -lh checkpoints/*/best_model.pt

# === PREDICTION ===
# Run example predictions
python3 predict_example.py

# Batch prediction
python3 batch_predict.py

# === EXPORT ===
# Export to ONNX
python3 export_models.py

# === EVALUATION ===
# Compare all models
python3 evaluate_all.py
```

---

## Expected Training Times (CPU Only)

| Dataset | Epochs | Batch Size | Time per Epoch | Total Time |
|---------|--------|------------|----------------|------------|
| 2k samples | 1 | 8 | ~2-3 hours | ~2-3 hours |
| 5k samples | 1 | 8 | ~5-7 hours | ~5-7 hours |
| 18k samples | 1 | 8 | ~18-24 hours | ~18-24 hours |
| 18k samples | 3 | 8 | ~18-24 hours | ~54-72 hours |

**DistilBERT**: 20-30% faster than BERT
**LoRA-BERT**: Similar speed to BERT (but uses much less memory)

---

## Next Steps After Training

1. **Evaluate models**: Run `python3 evaluate_all.py`
2. **Compare with Day 2**: Update comparison table with classical ML results
3. **Create visualizations**: Generate attention heatmaps, calibration curves
4. **Export for deployment**: Run `python3 export_models.py`
5. **Document results**: Update README with actual metrics

For questions or issues, refer to the main README.md or check the logs in `logs/` directory.
