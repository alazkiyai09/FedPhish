# TEE ML: Trusted Execution Environment for Machine Learning

A simulation framework for privacy-preserving machine learning using Trusted Execution Environments (Intel SGX, ARM TrustZone).

## Overview

This project implements a TEE simulation framework for ML operations, complementing homomorphic encryption (HE) in the HT2ML hybrid architecture. TEEs handle operations that are impossible or expensive in HE, such as:

- **Non-linear activations**: ReLU, Sigmoid, Softmax (trivial in TEE, expensive in HE)
- **Comparison operations**: Argmax, threshold, top-k (impossible in HE)
- **Division and normalization**: Batch normalization, layer normalization

## Motivation

### Why TEE?

Homomorphic encryption has fundamental limitations:
- **Noise budget**: Limits network depth to 1-2 layers
- **No comparisons**: Cannot compute argmax, threshold
- **Expensive activations**: Polynomial approximations for ReLU/sigmoid

TEE provides:
- **Fast computation**: Near-plaintext speed (10-100x faster than HE)
- **Unlimited depth**: No noise budget constraints
- **Native operations**: All operations work naturally

### Why Hybrid HT2ML?

**Pure HE**:
- ✓ Maximum privacy (cryptographic)
- ✗ 100-1000x slower
- ✗ Limited to 1-2 layers

**Pure TEE**:
- ✓ Fast (~1x slowdown)
- ✗ Trust in hardware manufacturer
- ✗ Side-channel vulnerabilities

**HT2ML Hybrid**:
- ✓ Input privacy via HE (first 1-2 layers)
- ✓ Performance via TEE (remaining layers)
- ✓ Flexible trust model (trust math OR hardware)

## Project Status

**Status: Phase 6 Complete ✅ - All Phases Finished**

### Completed Phases
- ✅ **Phase 1**: Core enclave infrastructure (enclave, attestation, sealed storage)
- ✅ **Phase 2**: ML operations in TEE (activations, comparisons, arithmetic)
- ✅ **Phase 3**: Security model (threats, side-channels, oblivious ops)
- ✅ **Phase 4**: HE↔TEE protocol (handoff, split optimizer)
- ✅ **Phase 5**: Benchmarking and performance analysis
- ✅ **Phase 6**: Integration and documentation

### Test Results
- **246 tests passing** ✓
- **0 tests failing**
- Comprehensive coverage of all components

## Installation

```bash
# Clone repository
cd /home/ubuntu/21Days_Project/tee_project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

```python
from tee_ml.core.enclave import Enclave
from tee_ml.operations.activations import tee_relu
import numpy as np

# Create enclave
enclave = Enclave(enclave_id="test-enclave", memory_limit_mb=128)

# Enter enclave with data
data = np.random.randn(10)
session = enclave.enter(data)

# Execute operation in enclave
result = tee_relu(data, session)

# Exit enclave
enclave.exit(session)
```

## Architecture

```
┌──────────────────────────────────────┐
│  Client (Privacy-Sensitive Data)    │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  HE Layer (1-2 layers max)           │
│  - Cryptographic privacy             │
│  - Limited by noise budget            │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  TEE Layer (unlimited depth)         │
│  - Hardware-enforced privacy         │
│  - Fast computation                 │
│  - Non-linear operations             │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  Output Predictions                 │
└──────────────────────────────────────┘
```

## Project Structure

```
tee_project/
├── tee_ml/
│   ├── core/              # Core TEE infrastructure
│   ├── operations/        # TEE-based ML operations
│   ├── protocol/          # HE↔TEE handoff protocol
│   ├── security/          # Security model and threats
│   ├── simulation/        # TEE simulation modes
│   └── benchmarking/      # Performance analysis
├── tests/                 # Unit tests
├── examples/              # Example scripts
├── docs/                  # Documentation
└── notebooks/             # Jupyter notebooks
```

## Documentation

- [Security Model](docs/SECURITY_MODEL.md) - Threat model and protections
- [Overhead Analysis](docs/OVERHEAD_ANALYSIS.md) - Performance overhead
- [HT2ML Interface](docs/HT2ML_INTERFACE.md) - HE↔TEE handoff specification

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_enclave.py -v

# Run with coverage
pytest tests/ --cov=tee_ml --cov-report=html
```

## Performance

Expected overhead (based on Intel SGX literature):
- Enclave entry: ~15 μs
- Enclave exit: ~10 μs
- Memory encryption: ~500 ns/MB
- Total slowdown: 1.1-2x (vs 100-1000x for HE)

## Security

### What TEE Protects Against
- ✓ Memory snooping by OS/hypervisor
- ✓ Tampering with enclave code
- ✓ Extraction of enclave secrets
- ✓ Debugging enclave execution

### What TEE Doesn't Protect Against
- ✗ Side-channel attacks (cache timing, power analysis)
- ✗ Iago attacks (malicious inputs)
- ✗ Speculative execution (Spectre/Meltdown)
- ✗ Hardware bugs

### Mitigations
- Constant-time operations for security-critical code
- Oblivious RAM patterns
- Input validation and sanitization
- Regular security audits

## References

- [Intel SGX](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions.html)
- [ARM TrustZone](https://www.arm.com/architecture/security-features/trustzone)
- [Gramine Project](https://gramineproject.io/)
- [HT2ML Paper](https://arxiv.org/abs/2305.06449)

## License

MIT License - See LICENSE file for details

## Contributing

This is a research project. For questions or contributions, please open an issue.
