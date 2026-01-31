# Quick Training Guide - Get Models in ~35 Minutes

**Goal**: Train both XGBoost and DistilBERT models for testing the API.
**Time**: ~35 minutes (5 min XGBoost + 30 min DistilBERT with GPU)
**Cost**: $0 (all local training)

## Prerequisites Check

```bash
# Check Python version (need 3.11+)
python --version

# Check if GPU available (optional but recommended)
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

## Option 1: Train Everything (Recommended)

```bash
cd /home/ubuntu/21Days_Project/unified-phishing-api/scripts
python quick_train_all.py
```

**What happens**:
1. ✅ Generates 1,000 synthetic phishing/legitimate emails
2. ✅ Trains XGBoost (5 min, CPU)
3. ✅ Trains DistilBERT (30 min, GPU or 2 hrs, CPU)
4. ✅ Saves models to `/home/ubuntu/21Days_Project/models/`

**Estimated time**:
- With GPU: ~35 minutes
- Without GPU: ~2 hours

## Option 2: Train XGBoost Only (Fastest)

If you just want to test the API quickly:

```bash
cd /home/ubuntu/21Days_Project/unified-phishing-api/scripts
python quick_train_xgboost.py
```

**Time**: ~5 minutes

**Result**: API will work with XGBoost only (mode: MINIMAL)

## Option 3: Train on Google Colab (Free GPU)

If you don't have a GPU:

1. Open https://colab.research.google.com/
2. New notebook
3. Install dependencies:
```python
!pip install transformers torch scikit-learn
```

4. Copy and run the contents of `quick_train_distilbert.py`

5. Download model files (Colab sidebar → Files → Download)

6. Copy to models directory:
```bash
mkdir -p /home/ubuntu/21Days_Project/models/day3_distilbert
# Copy downloaded files here
```

## Verify Training Success

```bash
# Check XGBoost model
ls -lh /home/ubuntu/21Days_Project/models/day2_xgboost/
# Expected:
# xgboost_phishing_classifier.json (~500 KB)
# metadata.json (~2 KB)

# Check DistilBERT model
ls -lh /home/ubuntu/21Days_Project/models/day3_distilbert/
# Expected:
# pytorch_model.bin (~250 MB)
# config.json (~1 KB)
# tokenizer_config.json (~50 KB)
# metadata.json (~2 KB)
```

## Test with API

After training:

```bash
# Start API
cd /home/ubuntu/21Days_Project/unified-phishing-api
docker-compose up -d

# Wait 30 seconds for startup
sleep 30

# Check health (should show models available)
curl http://localhost:8000/health | jq

# Expected output includes:
# {
#   "models": {
#     "xgboost": true,
#     "transformer": true,
#     "multi_agent": false
#   },
#   "operating_mode": "DEGRADED"  # or "FULL" if multi-agent also works
# }
```

## What If Training Fails?

### XGBoost fails
```bash
# Install dependencies
pip install xgboost scikit-learn pandas numpy

# Try again
cd scripts
python quick_train_xgboost.py
```

### DistilBERT fails
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# If not installed, install:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Try again
cd scripts
python quick_train_distilbert.py
```

### Out of memory (GPU)
```bash
# Edit quick_train_distilbert.py
# Change: per_device_train_batch_size=16
# To: per_device_train_batch_size=8
```

## Understanding the Output

### XGBoost Training Output
```
Classification Report:
              precision    recall  f1-score   support

  Legitimate       0.87      0.85      0.86       100
     Phishing       0.85      0.87      0.86       100

    accuracy                           0.86       200
   macro avg       0.86      0.86      0.86       200
weighted avg       0.86      0.86      0.86       200

AUPRC: 0.8756
```

**Good**: AUPRC > 0.85, F1 > 0.85

### DistilBERT Training Output
```
Epoch 1: AUPRC 0.8567, Loss 0.4231
Epoch 2: AUPRC 0.9123, Loss 0.2134
Epoch 3: AUPRC 0.9345, Loss 0.1567

Final AUPRC: 0.9345
Final Accuracy: 0.9200
```

**Good**: AUPRC > 0.90, Accuracy > 0.90

## Next Steps After Training

### 1. Test the API
```bash
# Create test email
cat > /tmp/test_email.txt << 'EOF'
From: security@chase-secure-portal.xyz
Subject: URGENT: Verify Your Account Now
Date: Wed, 29 Jan 2026 12:34:56 +0000

Dear Customer,

Your account will be suspended within 24 hours unless you verify your information.
Click here to verify: http://chase-secure-portal.xyz/login

Please provide your account number and SSN to prevent suspension.

Sincerely,
Chase Security Team
EOF

# Analyze with XGBoost
curl -X POST "http://localhost:8000/api/v1/analyze/email" \
  -H "Content-Type: application/json" \
  -d "{
    \"raw_email\": \"$(cat /tmp/test_email.txt | tr '\n' '\\n' | sed 's/"/\\"/g')\",
    \"model_type\": \"xgboost\"
  }" | jq

# Analyze with Transformer
curl -X POST "http://localhost:8000/api/v1/analyze/email" \
  -H "Content-Type: application/json" \
  -d "{
    \"raw_email\": \"$(cat /tmp/test_email.txt | tr '\n' '\\n' | sed 's/"/\\"/g')\",
    \"model_type\": \"transformer\"
  }" | jq

# Analyze with Ensemble
curl -X POST "http://localhost:8000/api/v1/analyze/email" \
  -H "Content-Type: application/json" \
  -d "{
    \"raw_email\": \"$(cat /tmp/test_email.txt | tr '\n' '\\n' | sed 's/"/\\"/g')\",
    \"model_type\": \"ensemble\"
  }" | jq
```

### 2. Check Performance
```bash
# View metrics
curl http://localhost:8000/metrics | grep model_prediction

# Expected:
# model_predictions_total{model_type="xgboost",verdict="PHISHING"} 1.0
# model_prediction_duration_seconds_bucket{model_type="xgboost",le="0.2"} 1.0
```

### 3. View in Grafana
```bash
# Access dashboard
open http://localhost:3000

# Login: admin/admin
# Dashboard: "Phishing Detection API - Overview"

# Look for:
# - Model prediction counts
# - Prediction latency by model
# - Verdict distribution
```

## Summary

| Step | Command | Time |
|------|---------|------|
| 1. Train XGBoost | `python scripts/quick_train_xgboost.py` | 5 min |
| 2. Train DistilBERT | `python scripts/quick_train_distilbert.py` | 30 min (GPU) |
| 3. Start API | `docker-compose up -d` | 1 min |
| 4. Test | `curl http://localhost:8000/health` | - |

**Total time**: ~35 minutes
**Total cost**: $0

**Result**: Working API with XGBoost and DistilBERT models ready for testing!
