# FedPhish Benchmark Suite

Comprehensive evaluation framework for federated phishing detection research.

## Overview

FedPhish Benchmark Suite provides standardized evaluation for federated learning approaches to phishing URL detection. It consolidates multiple detection methods, federation strategies, data distributions, attack scenarios, and privacy mechanisms into a reproducible benchmark framework.

## Features

### Detection Methods
- **Classical ML**: XGBoost, Random Forest, Logistic Regression
- **Transformer**: DistilBERT with LoRA fine-tuning
- **Multi-Agent**: Simplified multi-agent system for overhead comparison
- **Privacy-Preserving GBDT**: HT2ML protocol implementation

### Federation Configurations
- Centralized (baseline)
- Local only (per-bank)
- FedAvg (horizontal FL)
- FedProx (handles non-IID)
- Privacy-preserving FL (DP, Secure Aggregation)

### Data Distributions
- IID (uniform phishing types)
- Non-IID (Dirichlet α=0.5)
- Label skew (each bank sees different phishing types)

### Attack Scenarios
- No attack (clean baseline)
- Label flip (20% malicious banks)
- Backdoor (bank-specific trigger)
- Model poisoning (gradient scaling)

### Privacy Mechanisms
- Local DP (ε = 1.0, 0.5, 0.1)
- Secure Aggregation
- HT2ML hybrid protocol

### Standardized Metrics
- **Classification**: Accuracy, AUPRC, Recall@1%FPR
- **Privacy**: ε achieved, information leakage
- **Robustness**: Attack success rate, accuracy degradation
- **Efficiency**: Training time, communication cost
- **Fairness**: Per-bank accuracy variance

## Installation

```bash
# Clone repository
cd fedphish_benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Smoke Test

```bash
bash scripts/quick_test.sh
```

### Run Full Benchmark

```bash
bash scripts/run_benchmark.sh
```

Or run with Python:

```python
from omegaconf import OmegaConf
from src.experiments import run_full_benchmark

# Load configuration
config = OmegaConf.load("config/benchmark.yaml")

# Run benchmark (5 runs per configuration)
results = run_full_benchmark(config, num_runs=5)

# Results are saved to ./results/
```

### Run Single Experiment

```python
from omegaconf import OmegaConf
from src.experiments import run_single_experiment

config = OmegaConf.load("config/benchmark.yaml")

result = run_single_experiment(
    config=config,
    model_type="xgboost",
    federation_type="fedavg",
    data_distribution="non_iid",
    attack_type="model_poisoning",
    privacy_mechanism="local_dp",
    run_id=0,
)

print(f"Accuracy: {result.metrics['accuracy']}")
```

## Configuration

Benchmark settings are configured via YAML files in `config/`:

- `config/benchmark.yaml` - Master benchmark configuration
- `config/dataset/phishing.yaml` - Data settings
- `config/model/*.yaml` - Model hyperparameters
- `config/federation/*.yaml` - FL strategies
- `config/attack/*.yaml` - Attack configurations

Example configuration:

```yaml
# config/benchmark.yaml
benchmark:
  num_runs: 5
  output_dir: "./results"
  device: "cuda"

methods:
  - xgboost
  - transformer
  - privacy_gbdt

federations:
  - fedavg
  - fedprox

distributions:
  - iid
  - non_iid

attacks:
  - none
  - model_poisoning

privacy:
  - none
  - local_dp
```

## Output

The benchmark generates:

1. **CSV files**: `results/summary.csv`, `results/detailed_results.csv`
2. **LaTeX tables**: `results/tables/*.tex`
3. **Publication figures**: `results/figures/*.pdf`
4. **MLflow logs**: `./mlruns/`

### Example Output Table

| Method | IID Acc | Non-IID Acc | Attack ASR | DP (ε=1) Acc | Time |
|--------|---------|-------------|------------|--------------|------|
| Local  | 0.85 ± 0.02 | 0.82 ± 0.03 | N/A | 0.83 ± 0.02 | 12.5 |
| FedAvg | 0.87 ± 0.01 | 0.84 ± 0.02 | 0.15 | 0.85 ± 0.01 | 45.2 |
| FedPhish | **0.89 ± 0.01** | **0.86 ± 0.02** | **0.05** | **0.86 ± 0.01** | 48.1 |

## Project Structure

```
fedphish_benchmark/
├── config/              # YAML configurations
├── src/
│   ├── data/           # Data loading and partitioning
│   ├── models/         # Model implementations
│   ├── fl/             # Federated learning infrastructure
│   ├── attacks/        # Attack implementations
│   ├── metrics/        # Evaluation metrics
│   ├── experiments/    # Experiment runner and orchestrator
│   ├── artifacts/      # LaTeX/figure generation
│   └── utils/          # Utilities (logging, checkpointing)
├── tests/              # Unit tests
├── data/               # Data storage
├── results/            # Benchmark outputs
└── scripts/            # Run scripts
```

## Citing

If you use this benchmark in your research, please cite:

```bibtex
@software{fedphish_benchmark,
  title={FedPhish Benchmark Suite: Comprehensive Evaluation for Federated Phishing Detection},
  author={Ahmad Whafa Azka Al Azkiyai},
  year={2025},
  url={https://github.com/alazkiyai09/FedPhish}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.
