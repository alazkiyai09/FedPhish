# Documentation Index - Day 3 Transformer Phishing Detection

This document provides an overview of all available documentation and how to use it.

## Documentation Files

### 1. README.md (Main Documentation)
**Purpose**: Comprehensive project overview and technical details

**Contents**:
- Project overview and objectives
- Model architecture details (BERT, RoBERTa, DistilBERT, LoRA-BERT)
- Key features and comparison with Day 2 classical ML
- Installation instructions
- Quick start examples
- Project structure (39 files across 6 layers)
- Training configuration and hyperparameters
- Evaluation metrics
- Inference examples
- Export instructions (ONNX)
- Expected results template

**When to read**: First, to understand the project scope and design

**Location**: `/home/ubuntu/21Days_Project/day3_transformer_phishing/README.md`

---

### 2. QUICKSTART.md (Get Started Fast)
**Purpose**: Step-by-step guide to start training immediately

**Contents**:
- Prerequisites checklist (what's already done)
- 3 training options (fast, full, custom)
- Expected output during training
- Using trained models for predictions
- Managing training (start, stop, monitor)
- Dataset options (2k, 5k, 18k samples)
- Model comparison (speed, memory, use cases)
- Troubleshooting common issues
- Timeline estimates
- Complete workflow summary

**When to read**: When you're ready to start training

**Location**: `/home/ubuntu/21Days_Project/day3_transformer_phishing/QUICKSTART.md`

---

### 3. USAGE_GUIDE.md (Detailed Instructions)
**Purpose**: Comprehensive guide for all aspects of the codebase

**Contents**:
1. **Environment Setup**: Install and verify dependencies
2. **Data Preparation**: Download, subsets, splits
3. **Training Models**: All training options with examples
4. **Monitoring Training**: Real-time monitoring, progress checks
5. **Using Trained Models**: Single prediction, batch prediction
6. **Exporting Models**: ONNX export, deployment packages
7. **Performance Comparison**: Evaluate and compare all models
8. **Troubleshooting**: Common issues and solutions

**When to read**: When you need detailed instructions for specific tasks

**Location**: `/home/ubuntu/21Days_Project/day3_transformer_phishing/USAGE_GUIDE.md`

---

### 4. IMPLEMENTATION_SUMMARY.md (Technical Details)
**Purpose**: Technical implementation details for reference

**Contents**:
- Complete file listing (39 files)
- Function signatures for all modules
- Data layer (dataset, preprocessor, tokenizer)
- Model layer (4 transformer architectures)
- Training layer (trainer, metrics, scheduler)
- Inference layer (predictor, export)
- Utility functions
- Design decisions and rationale

**When to read**: When you need to understand implementation details or modify code

**Location**: `/home/ubuntu/21Days_Project/day3_transformer_phishing/IMPLEMENTATION_SUMMARY.md`

---

### 5. DOCS_INDEX.md (This File)
**Purpose**: Navigation guide for all documentation

**When to read**: First, to understand what documentation is available

---

## Helper Scripts

### Shell Scripts

**train_all_models.sh**
- Train all 4 models with one command
- Usage: `bash train_all_models.sh [dataset] [epochs] [batch_size]`
- Example: `bash train_all_models.sh data/processed/phishing_emails_2k.csv 1 8`

**check_training_progress.sh**
- Check status of all training processes
- Shows active processes, log snippets, completion status
- Usage: `bash check_training_progress.sh`

**stop_all_training.sh**
- Stop all training processes gracefully
- Usage: `bash stop_all_training.sh`

### Python Scripts

**example_prediction.py**
- Quick demo of predictions with trained model
- Tests on 4 example emails
- Usage: `python3 example_prediction.py`

**train.py** (Main Entry Point)
- Main training script with CLI interface
- All training goes through this script
- Usage: `python3 train.py --model bert --epochs 3`

---

## How to Use This Documentation

### New to the Project? Start Here:

1. **Read README.md** - Understand what we're building
2. **Read QUICKSTART.md** - Learn how to train models
3. **Run training** - `bash train_all_models.sh`
4. **Monitor progress** - `bash check_training_progress.sh`

### Need to Do Something Specific?

| Task | Documentation | Command |
|------|---------------|---------|
| Start training | QUICKSTART.md | `bash train_all_models.sh` |
| Monitor training | QUICKSTART.md or USAGE_GUIDE.md | `bash check_training_progress.sh` |
| Stop training | QUICKSTART.md | `bash stop_all_training.sh` |
| Make predictions | USAGE_GUIDE.md | `python3 example_prediction.py` |
| Export models | USAGE_GUIDE.md | `python3 export_models.py` |
| Evaluate models | USAGE_GUIDE.md | `python3 evaluate_all.py` |
| Troubleshoot issues | QUICKSTART.md or USAGE_GUIDE.md | Check logs |
| Understand implementation | README.md or IMPLEMENTATION_SUMMARY.md | Read docs |

### Documentation Flow

```
DOCS_INDEX.md (this file)
    ↓
README.md (overview)
    ↓
QUICKSTART.md (start training)
    ↓
USAGE_GUIDE.md (detailed instructions)
    ↓
IMPLEMENTATION_SUMMARY.md (technical details)
```

---

## Key Concepts

### Models Implemented
1. **BERT-base**: 110M parameters, standard fine-tuning
2. **RoBERTa-base**: 125M parameters, optimized pretraining
3. **DistilBERT**: 66M parameters, 40% smaller/faster
4. **LoRA-BERT**: 300K trainable params, for federated learning

### Key Features
- Special tokens: `[SUBJECT]`, `[BODY]`, `[URL]`, `[SENDER]`
- Smart truncation: Head+tail strategy
- Mixed precision: FP16 training
- Early stopping: Monitors validation AUPRC
- Attention visualization: Extract attention weights
- ONNX export: Deployment-ready format

### Datasets Available
- **Full**: 18,634 samples (data/raw/phishing_emails_hf.csv)
- **Medium**: 5,000 samples (data/processed/phishing_emails_5k.csv)
- **Small**: 2,000 samples (data/processed/phishing_emails_2k.csv)

### Evaluation Metrics
- Primary: AUPRC (Area Under Precision-Recall Curve)
- Secondary: AUROC, Accuracy, Precision, Recall, F1
- Financial requirements: Recall > 95%, FPR < 1%

---

## Project Status

### Completed ✅
- All 39 files implemented
- Data downloaded and prepared
- Unit tests passing (15/15)
- Helper scripts created
- Documentation complete

### Ready for You 🔧
- Train the models
- Evaluate performance
- Compare with Day 2 classical ML
- Export to ONNX
- Generate visualizations

---

## Quick Command Reference

```bash
# === TRAINING ===
bash train_all_models.sh                    # Fast (2k, 1 epoch)
bash train_all_models.sh data/processed/phishing_emails_5k.csv 1 8  # Medium
bash train_all_models.sh data/raw/phishing_emails_hf.csv 3 8  # Full

# === MONITORING ===
bash check_training_progress.sh             # Check status
tail -f logs/train_*.log                    # Watch logs
ps aux | grep "python3 train.py"            # Check processes

# === CONTROL ===
bash stop_all_training.sh                   # Stop all
pkill -f "python3 train.py --model bert"   # Stop specific

# === AFTER TRAINING ===
python3 example_prediction.py               # Test predictions
python3 evaluate_all.py                     # Compare models
python3 export_models.py                    # Export ONNX
```

---

## File Locations Reference

```
day3_transformer_phishing/
├── Documentation
│   ├── README.md                   # Main overview
│   ├── QUICKSTART.md              # Get started fast
│   ├── USAGE_GUIDE.md             # Detailed instructions
│   ├── IMPLEMENTATION_SUMMARY.md  # Technical details
│   └── DOCS_INDEX.md              # This file
│
├── Scripts
│   ├── train_all_models.sh        # Train all models
│   ├── check_training_progress.sh # Monitor training
│   ├── stop_all_training.sh       # Stop training
│   ├── train.py                   # Main training script
│   └── example_prediction.py      # Prediction demo
│
├── Data
│   ├── data/raw/phishing_emails_hf.csv        # 18,634 samples
│   ├── data/processed/phishing_emails_2k.csv  # 2,000 samples
│   └── data/processed/phishing_emails_5k.csv  # 5,000 samples
│
├── Logs
│   └── logs/train_*.log           # Training logs
│
├── Checkpoints (created after training)
│   └── checkpoints/{model}/best_model.pt
│
└── Source Code
    └── src/                       # All implementation code
```

---

## Need Help?

### Quick Questions
- Check **QUICKSTART.md** for immediate answers
- Check **USAGE_GUIDE.md** Section 8 for troubleshooting

### Technical Questions
- Check **README.md** for architecture details
- Check **IMPLEMENTATION_SUMMARY.md** for code details

### Common Issues
- **Out of memory**: Reduce batch size to 4 or use smaller dataset
- **Too slow**: Use DistilBERT or 2k dataset
- **Import errors**: Reinstall dependencies
- **Training stuck**: Check logs, verify process running

---

## Success Checklist

After training completes, you should have:

- [ ] 4 trained models (BERT, RoBERTa, DistilBERT, LoRA-BERT)
- [ ] Checkpoints in `checkpoints/{model}/best_model.pt`
- [ ] Training logs in `logs/train_*.log`
- [ ] Test predictions working: `python3 example_prediction.py`
- [ ] Evaluation metrics: `python3 evaluate_all.py`
- [ ] Model comparison table updated in README.md
- [ ] (Optional) ONNX exports: `python3 export_models.py`

---

**Ready to begin? Open QUICKSTART.md and run:**

```bash
bash train_all_models.sh
```
