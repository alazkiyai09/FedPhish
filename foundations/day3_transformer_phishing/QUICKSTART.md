# Quick Start Guide - Day 3 Transformer Phishing Detection

This guide gets you training and using the transformer models in minutes.

## Prerequisites (Already Done ✅)

- ✅ Dependencies installed (torch, transformers, peft, wandb, etc.)
- ✅ Real dataset downloaded (18,634 emails from HuggingFace)
- ✅ Smaller subsets created (2k and 5k samples)
- ✅ All code implemented (39 files across 6 layers)
- ✅ Unit tests passing (15/15 tests)

## Quick Training (Recommended)

### Option 1: Fastest - Train on 2k Samples (~2-3 hours)

```bash
# Train all 4 models on 2k subset with 1 epoch
bash train_all_models.sh data/processed/phishing_emails_2k.csv 1 8

# Monitor progress
bash check_training_progress.sh

# Watch logs in real-time
tail -f logs/train_*.log
```

### Option 2: Full Training - Train on Complete Dataset (~18-24 hours per model)

```bash
# Train all 4 models on full dataset with 3 epochs
# NOTE: This will take 3-4 days on CPU
bash train_all_models.sh data/raw/phishing_emails_hf.csv 3 8

# Monitor progress
watch -n 60 'bash check_training_progress.sh'
```

### Option 3: Custom Training

```bash
# Train a single model
python3 train.py --model bert --data data/processed/phishing_emails_2k.csv --epochs 1 --batch-size 8 --no-wandb

# Train with different settings
python3 train.py \
    --model bert \
    --data data/raw/phishing_emails_hf.csv \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-5 \
    --no-wandb \
    > logs/train_bert_custom.log 2>&1 &
```

## What Happens During Training

When you run training:

1. **Data Loading**: Dataset is split into train (60%), val (20%), test (20%)
2. **Model Initialization**: Transformer model loads pretrained weights
3. **Training Loop**:
   - Forward pass through model
   - Loss calculation
   - Backward pass (gradient computation)
   - Parameter updates
   - Validation after each epoch
4. **Checkpointing**: Best model saved based on validation AUPRC
5. **Logging**: Progress written to `logs/train_*.log`

## Expected Output

### Training Progress
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

📚 Epoch 1/1
Epoch 1:  10%|████▎   | 140/1398 [10:23<1:33:15, 4.45s/it, loss=0.5234, lr=1.43e-07]

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

### Completion
```
============================================================
Training completed!
============================================================
Total training time: 2h 34m 12s
Best validation AUPRC: 0.9234 (Epoch 1)

Test set results:
   Accuracy: 90.80%
   AUPRC: 0.9156
   AUROC: 0.9489
   F1: 87.60%

Model saved to: checkpoints/bert/best_model.pt
```

## Using Trained Models

### Quick Prediction Test

```bash
# After training completes, test your model
python3 example_prediction.py
```

This will:
- Load the trained BERT model
- Run predictions on 4 example emails
- Show predictions with confidence scores

### Custom Predictions

Create a Python script:

```python
from src.inference.predictor import Predictor
from src.models.factory import create_model
import torch

# Load model
model = create_model(model_type='bert', num_labels=2)
checkpoint = torch.load('checkpoints/bert/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create predictor
predictor = Predictor(model=model, tokenizer=None, device='cpu')

# Predict
email = "URGENT: Click here to verify your account now!"
result = predictor.predict(email)

print(f"Prediction: {result['label_text']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Managing Training

### Check Progress
```bash
# Quick status check
bash check_training_progress.sh

# Watch logs live
tail -f logs/train_bert.log

# Check specific metrics
grep "AUPRC" logs/train_*.log
```

### Stop Training
```bash
# Stop all models
bash stop_all_training.sh

# Stop specific model
pkill -f "python3 train.py --model bert"
```

### Resume Training
If training was interrupted, you can restart:
```bash
# Just run the same command again
bash train_all_models.sh
```

## Dataset Options

| Dataset | Samples | Time per Epoch | Best For |
|---------|---------|----------------|----------|
| `data/processed/phishing_emails_2k.csv` | 2,000 | ~2-3 hours | Quick testing |
| `data/processed/phishing_emails_5k.csv` | 5,000 | ~5-7 hours | Balanced speed/performance |
| `data/raw/phishing_emails_hf.csv` | 18,634 | ~18-24 hours | Full training, best results |

## Model Options

| Model | Parameters | Speed | Memory | Best For |
|-------|-----------|-------|--------|----------|
| `distilbert` | 66M | Fast ⚡ | Low | Quick iterations |
| `bert` | 110M | Medium | Medium | Baseline comparison |
| `roberta` | 125M | Slow | High | Best accuracy |
| `lora-bert` | 110M (300K trainable) | Medium | Low | Federated learning |

## Troubleshooting

### Out of Memory (Killed: 137)
```bash
# Reduce batch size
python3 train.py --model bert --batch-size 4 --epochs 1

# Use smaller dataset
python3 train.py --model bert --data data/processed/phishing_emails_2k.csv
```

### Too Slow
```bash
# Use DistilBERT (40% faster)
python3 train.py --model distilbert --epochs 1

# Reduce epochs
python3 train.py --model bert --epochs 1
```

### Check If Training Is Working
```bash
# Check process is running
ps aux | grep "python3 train.py"

# Check logs have recent activity
tail -20 logs/train_bert.log

# Check checkpoint is being updated
ls -lh checkpoints/bert/
```

## After Training Completes

### 1. Test Your Model
```bash
python3 example_prediction.py
```

### 2. Evaluate All Models
```bash
python3 evaluate_all.py
```

### 3. Export to ONNX (Optional)
```bash
python3 export_models.py
```

### 4. Compare with Day 2
Update the results table in README.md with your actual metrics.

## File Locations

- **Logs**: `logs/train_*.log`
- **Checkpoints**: `checkpoints/{model}/best_model.pt`
- **Configs**: `experiments/configs/*.yaml`
- **Data**: `data/processed/*.csv`

## Summary Commands

```bash
# === COMPLETE WORKFLOW ===

# 1. Train all models (2k subset, 1 epoch)
bash train_all_models.sh

# 2. Monitor progress
watch -n 30 'bash check_training_progress.sh'

# 3. When done, test predictions
python3 example_prediction.py

# 4. Evaluate and compare
python3 evaluate_all.py
```

## Expected Timeline

**Fast Track** (2k samples, 1 epoch):
- DistilBERT: ~2 hours
- BERT: ~2.5 hours
- LoRA-BERT: ~2.5 hours
- RoBERTa: ~3 hours
- **Total**: ~10 hours (overnight)

**Full Training** (18k samples, 3 epochs):
- Each model: ~54-72 hours
- **Total**: ~9-12 days (run models sequentially)

## Need Help?

- **Detailed guide**: See `USAGE_GUIDE.md`
- **Implementation details**: See `README.md`
- **Check logs**: `logs/train_*.log`
- **Check tests**: `pytest tests/`

---

**Ready to start training? Run this:**

```bash
bash train_all_models.sh
```
