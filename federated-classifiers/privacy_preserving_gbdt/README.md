# Privacy-Preserving GBDT for Phishing Detection

Implementation of Guard-GBDT concepts for federated phishing detection, based on:

> "Guard-GBDT: Efficient Privacy-Preserving Approximated GBDT Training" (researchers, 2025)

## Overview

This project implements a **privacy-preserving Gradient Boosted Decision Tree (GBDT)** system for **vertical federated learning** where:

- **Bank A** holds transaction features
- **Bank B** holds email content features
- **Bank C** holds URL features
- **Label Holder** has historical phishing labels

The system enables training GBDT **without pooling raw data**, using:
- Additive secret sharing for secure aggregation
- Differential privacy for formal privacy guarantees
- Secure split finding without revealing feature values

## Project Structure

```
privacy_preserving_gbdt/
├── src/
│   ├── core/           # GBDT fundamentals (objective, histogram, tree builder)
│   ├── crypto/         # Secret sharing, DP mechanisms, secure aggregation
│   ├── protocols/      # PSI, secure split finding, prediction
│   ├── federated/      # Client, server, label holder
│   ├── models/         # Guard-GBDT and PlaintextGBDT
│   └── utils/          # Data loading, metrics
├── tests/              # Unit tests (69 tests, all passing)
├── experiments/        # Benchmark scripts
└── docs/               # Privacy analysis, results
```

## Installation

```bash
cd privacy_preserving_gbdt
pip install -r requirements.txt  # numpy, scikit-learn, pytest
```

## Quick Start

### 1. Run Plaintext Baseline

```bash
python experiments/run_baseline.py
```

### 2. Run Guard-GBDT (Privacy-Preserving)

```bash
python experiments/run_guard_gbdt.py
```

### 3. Compare Models

```bash
python experiments/benchmark.py
```

### 4. Run Unit Tests

```bash
pytest tests/ -v
```

## Key Components

### 1. Core GBDT (`src/core/`)
- **`objective.py`**: Gradient and Hessian computation for LogLoss/MSE
- **`histogram.py`**: Histogram-based split finding with regularization
- **`tree_builder.py`**: Decision tree construction using gradient-based splitting
- **`gbdt_base.py`**: Base GBDT ensemble for comparison

### 2. Cryptographic Primitives (`src/crypto/`)
- **`secret_sharing.py`**: Additive secret sharing over finite fields
- **`dp_mechanisms.py`**: Laplace/Gaussian mechanisms for (ε,δ)-DP
- **`secure_aggregation.py`**: Secure histogram aggregation across parties

### 3. Secure Protocols (`src/protocols/`)
- **`psi.py`**: Private Set Intersection for sample alignment in vertical FL
- **`split_finding.py`**: Secure split finding across parties
- **`prediction.py`**: Privacy-preserving prediction protocol

### 4. Federated Infrastructure (`src/federated/`)
- **`client.py`**: Bank client (feature holder)
- **`server.py`**: Training coordinator
- **`label_holder.py`**: Label holder (gradient computation)

### 5. Models (`src/models/`)
- **`guard_gbdt.py`**: Privacy-preserving GBDT implementation
- **`plaintext_gbdt.py`**: XGBoost-style baseline for comparison

## Privacy Guarantees

### Differential Privacy

The system provides **(ε, δ)-differential privacy**:

- **Histogram aggregation**: Laplace mechanism on [sum_gradients, sum_hessians, count]
- **Gradient clipping**: Bounds sensitivity before adding noise
- **Composition**: Advanced composition theorem across trees

### Secure Computation

- **Additive Secret Sharing**: Each value split into n shares, any t shares reveal nothing
- **Field modulus**: 2^31 - 1 (Mersenne prime for efficient computation)
- **Secure aggregation**: Parties jointly compute histogram sums without revealing individual values

### Threat Model

**Assumptions:**
- Honest-but-curious parties (follow protocol but try to infer information)
- Server does not collude with any party
- At least 2 non-colluding parties

**Guarantees:**
- No party sees another's raw feature values
- Histograms are aggregated securely
- Gradients are protected with DP noise

## Performance

### Accuracy vs Privacy Trade-off

| ε (Privacy Budget) | Accuracy | Accuracy Loss |
|---|---|---|
| ∞ (Plaintext) | ~0.85 | 0.00 |
| 2.0 | ~0.83 | 0.02 |
| 1.0 | ~0.81 | 0.04 |
| 0.5 | ~0.78 | 0.07 |

### Communication Cost

- **Per tree**: O(n_parties × n_features × n_bins × 3) values
- **Total**: O(n_estimators × n_parties × n_features × n_bins)

### Training Time

- Plaintext: ~10 seconds for 50 trees, 5000 samples
- Guard-GBDT: ~30 seconds (3x overhead from cryptographic operations)

## Usage Example

```python
import numpy as np
from models.guard_gbdt import GuardGBDT
from utils.data_loader import partition_data_vertical, create_realistic_phishing_data

# Load data
X, y = create_realistic_phishing_data(n_samples=5000)

# Partition among 3 banks
X_train_dict, X_test_dict, y_train, y_test = partition_data_vertical(
    X, y, n_parties=3
)

# Train Guard-GBDT with privacy
model = GuardGBDT(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    epsilon=1.0,      # Privacy budget
    delta=1e-5,       # Privacy parameter
    use_dp=True
)

model.fit(X_train_dict, y_train, verbose=True)

# Evaluate
accuracy = model.score(X_test_dict, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get training statistics
stats = model.get_training_stats()
print(f"Communication rounds: {stats['communication_rounds']}")
print(f"Training time: {stats['training_time']:.2f}s")
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_gbdt.py -v          # Core GBDT
pytest tests/test_crypto.py -v         # Cryptography
pytest tests/test_protocols.py -v      # Secure protocols
pytest tests/test_federated.py -v      # Federated infrastructure
```

**Test Results**: 69/69 passing

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- scikit-learn >= 0.24
- pytest >= 7.0 (for testing)

## References

1. Guard-GBDT: "Efficient Privacy-Preserving Approximated GBDT Training" (researchers, 2025)
2. XGBoost: "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
3. Differential Privacy: "The Algorithmic Foundations of Differential Privacy" (Dwork & Roth, 2014)
4. Secret Sharing: "How to Share a Secret" (Shamir, 1979)

## License

MIT License - See LICENSE file for details

## Authors

Implementation for research portfolio project - Privacy-Preserving Machine Learning

Prof. N. Russello (Advisor)
