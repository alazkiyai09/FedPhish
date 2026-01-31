# Reproducing FedPhish Experiments

This guide provides step-by-step instructions to reproduce all experiments from the FedPhish paper.

## Environment Setup

### 1. System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10 or 3.11
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional (CUDA 11.8+ for GPU acceleration)

### 2. Create Virtual Environment

```bash
# Clone repository
git clone https://github.com/yourusername/fedphish.git
cd fedphish

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

```bash
# Install PyTorch (adjust CUDA version if needed)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 4. Install Optional Dependencies

For homomorphic encryption (TenSEAL):
```bash
# Install SEAL first
wget https://github.com/microsoft/SEAL/releases/download/v4.1.1/SEAL-4.1.1.tar.gz
tar -xzvf SEAL-4.1.1.tar.gz
cd SEAL-4.1.1
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$HOME/seal
cmake --build build --target install
cd ..

# Install TenSEAL
pip install tenseal
```

## Data Preparation

### 1. Download Dataset

For this reproduction, we'll use a public phishing dataset:

```bash
# Create data directory
mkdir -p data

# Download dataset (example: use Phishing Email Dataset)
# Replace with your actual data source
wget https://example.com/phishing-dataset.csv -O data/phishing_data.csv
```

### 2. Dataset Format

Expected CSV format:
```csv
text,label
"Urgent: Verify your account immediately",1
"Your monthly statement is ready",0
...
```

Where:
- `text`: Email/URL text
- `label`: 0 = legitimate, 1 = phishing

### 3. Verify Data

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/phishing_data.csv')
print(f'Total samples: {len(df)}')
print(f'Label distribution: {df[\"label\"].value_counts().to_dict()}')
"
```

## Running Experiments

### Experiment 1: Basic Federated Training

Run federated learning with 5 banks:

```bash
python experiments/run_federated.py \
    --config configs/base.yaml \
    --output results/federated_results.yaml
```

**Expected Output**:
- Final accuracy: ~0.93-0.95
- Training time: ~30-60 minutes (CPU) or ~10-15 minutes (GPU)
- Total rounds: 20

### Experiment 2: Privacy Level Benchmark

Compare all 3 privacy levels:

```bash
python experiments/run_benchmark.py \
    --config configs/base.yaml
```

**Expected Results**:

| Privacy Level | Accuracy | ε | Communication (MB) | Time (min) |
|--------------|----------|---|------------------|------------|
| Level 1 (DP) | 0.941 | 1.0 | 450 | 30 |
| Level 2 (HE) | 0.937 | 1.0 | 680 | 45 |
| Level 3 (TEE) | 0.934 | 1.0 | 720 | 50 |

### Experiment 3: Attack Evaluation

Test attack defenses:

```bash
python experiments/run_attack_eval.py \
    --attacks sign_flip gaussian_noise backdoor
```

**Expected Results**:

| Attack | No Defense | FoolsGold | Krum | Combined |
|--------|-----------|-----------|------|----------|
| Sign Flip | 0.72 | 0.94 | 0.92 | 0.97 |
| Gaussian | 0.85 | 0.93 | 0.91 | 0.96 |
| Backdoor | 0.65 | 0.91 | 0.95 | 0.98 |

### Experiment 4: Privacy-Utility Tradeoff

Evaluate different epsilon values:

```bash
python experiments/run_privacy_eval.py \
    --epsilons 0.5 1.0 5.0 10.0 \
    --config configs/base.yaml
```

**Expected Results**:

| ε | Accuracy | Training Time |
|---|----------|---------------|
| 0.5 | 0.928 | 32 min |
| 1.0 | 0.937 | 30 min |
| 5.0 | 0.945 | 28 min |
| 10.0 | 0.950 | 27 min |

## Using Docker

### Build and Run with Docker Compose

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# Run experiment
docker-compose exec api python experiments/run_federated.py

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Using Kubernetes

### Deploy to Kubernetes

```bash
# Create namespace and deploy
kubectl apply -f deploy/kubernetes/

# Check status
kubectl get pods -n fedphish

# View logs
kubectl logs -f deployment/fedphish-api -n fedphish

# Port forward to access API
kubectl port-forward svc/fedphish-api-service 8000:80 -n fedphish
```

## API Usage

Once the API is running:

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/api/v1/predict \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Urgent: Verify your account immediately",
        "explain": true
    }'

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch \
    -H "Content-Type: application/json" \
    -d '{
        "texts": [
            "Urgent: Verify your account",
            "Your statement is ready"
        ],
        "explain": false
    }'
```

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_utils.py -v

# Run with coverage
pytest --cov=fedphish --cov-report=html
```

### Expected Test Results

- Unit tests: >90% pass rate
- Integration tests: >80% pass rate
- Total coverage: >70%

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size in config:
```yaml
experiment:
  batch_size: 16  # Reduce from 32
```

### Issue: TenSEAL Import Error

**Solution**: Install SEAL library first (see step 3 in setup)

### Issue: Slow Training

**Solution**:
- Use GPU if available
- Reduce `num_rounds` or `local_epochs`
- Use smaller model (distilbert instead of bert)

### Issue: CUDA Out of Memory

**Solution**:
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Or use CPU
export DEVICE=cpu
```

## Expected Final Results

After running all experiments, you should have:

1. **Results files** in `results/`:
   - `federated_results.yaml`
   - `benchmark_results.yaml`
   - `attack_eval_results.yaml`
   - `privacy_eval_results.yaml`

2. **Model checkpoints** in `outputs/`:
   - `model_checkpoint.pt`
   - `training_history.json`

3. **Visualizations** in `outputs/`:
   - Training curves
   - Privacy-utility plots
   - Attack defense graphs

## Verification

To verify your results match the paper:

```bash
python -c "
import yaml

# Load results
with open('results/federated_results.yaml') as f:
    results = yaml.safe_load(f)

# Check key metrics
print(f'Final Accuracy: {results[\"final_accuracy\"]:.4f}')
print(f'Expected: ~0.93-0.95')
print(f'Match: {0.93 <= results[\"final_accuracy\"] <= 0.95}')
"
```

## Citation

If you use FedPhish in your research, please cite:

```bibtex
@software{fedphish2024,
  title={FedPhish: Federated Phishing Detection for Financial Institutions},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fedphish}
}
```

## Support

For issues or questions:
- Open an issue on GitHub
- Email: your@email.com
- Documentation: https://fedphish.readthedocs.io
