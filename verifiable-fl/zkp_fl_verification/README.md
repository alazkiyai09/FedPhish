# Zero-Knowledge Proofs for Federated Learning Verification

## Overview

This project implements zero-knowledge proof systems for verifiable federated learning, enabling clients to prove:
1. Their gradients are bounded (not malicious)
2. They trained on valid data
3. Their computations are correct

...without revealing their private data or gradients to the server.

## Mathematical Foundation

### What are Zero-Knowledge Proofs?

A zero-knowledge proof is a cryptographic protocol where a **prover** convinces a **verifier** that a statement is true, **without revealing any information beyond the truth of the statement itself**.

**Three Properties:**
1. **Completeness**: If the statement is true, an honest prover convinces an honest verifier
2. **Soundness**: If the statement is false, no cheating prover can convince the verifier (except with negligible probability)
3. **Zero-Knowledge**: The verifier learns nothing beyond the truth of the statement

### ZK-SNARKs

**Zero-Knowledge Succinct Non-Interactive Argument of Knowledge**

- **Succinct**: Proofs are small (few hundred bytes) and fast to verify
- **Non-Interactive**: Single message from prover to verifier
- **Argument**: Computationally sound (relies on hardness assumptions)
- **of Knowledge**: Prover knows the underlying witness

**Components:**
1. **Arithmetic Circuit**: Computation expressed as polynomial equations
2. **R1CS**: Rank-1 Constraint System (standard format for circuits)
3. **Trusted Setup**: One-time setup to generate proving/verification keys
4. **Proving Key**: Used by clients to generate proofs
5. **Verification Key**: Used by server to verify proofs

### Trusted Setup & Toxic Waste

**Problem**: The setup ceremony generates "toxic waste" (randomness) that must be destroyed. If leaked, it allows forging fake proofs.

**Solutions:**
- **Groth16**: Per-circuit trusted setup (small proofs, ~128 bytes)
- **PLONK**: Universal trusted setup (one setup for all circuits)
- **Bulletproofs**: No trusted setup (larger proofs, ~1-3 KB)

## Project Structure

```
zkp_fl_verification/
├── src/
│   ├── fundamentals/      # ZK building blocks (commitments, sigma protocols)
│   ├── snark/            # ZK-SNARK primitives (circuits, R1CS, proofs)
│   ├── fl_proofs/        # FL-specific proof constructions
│   ├── circom/           # Circom circuits for compilation
│   ├── proof_systems/    # Groth16, PLONK, Bulletproofs
│   └── utils/            # Crypto utilities, benchmarks
├── tests/                # Unit tests for all components
├── benchmarks/           # Performance evaluation
└── examples/             # Usage examples
```

## FL Verification Use Cases

### 1. Gradient Bound Proof

**Problem**: Server needs to ensure client gradients aren't malicious (e.g., too large).

**Naive Solution**: Server checks gradients directly → **Privacy leak!**

**ZK Solution**: Client proves `||gradient|| ≤ C` without revealing gradient.

```
Circuit: input gradient; output 1 if ||gradient|| ≤ C, else 0
Proof size: ~128 bytes (Groth16)
Verification time: ~1-10 ms
```

### 2. Data Validity Proof

**Problem**: Server needs to ensure clients train on real phishing/legitimate emails.

**ZK Solution**: Client proves their data is in the valid dataset (Merkle tree membership).

```
Commitment: Hash(dataset)
Proof: Merkle proof of membership
Proof size: ~O(log n) where n = dataset size
```

### 3. Computation Correctness Proof

**Problem**: Server needs to verify gradient computation is correct.

**ZK Solution**: Client proves `gradient = compute_loss(model, local_data)`.

```
Circuit: Encode gradient computation as arithmetic circuit
Proof size: ~128 bytes
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- **Python 3.8+**
- **pylibsnark**: Python bindings for libsnark (Groth16)
- **py-sr25519**: For cryptographic operations
- **numpy**: Numerical operations
- **circom**: Circuit compiler (optional, for custom circuits)

## Quick Start

### Example 1: Pedersen Commitment

```python
from src.fundamentals.commitments import PedersenCommitment

# Setup
commitment = PedersenCommitment(group_order=2**252, generators=[G, H])

# Commit to a secret value
secret_value = 12345
randomness = 42
commit = commitment.commit(secret_value, randomness)

# Verify commitment
assert commitment.verify(commit, secret_value, randomness)
```

### Example 2: Gradient Bound Proof

```python
from src.fl_proofs.gradient_bounds import GradientBoundProof
import numpy as np

# Client side
gradient = np.random.randn(100) * 0.1  # Small gradient
prover = GradientBoundProof(bound=1.0)
proof = prover.generate_proof(gradient)

# Server side (verifies without seeing gradient!)
verifier = GradientBoundProof(bound=1.0)
assert verifier.verify(proof, gradient_hash=get_hash(gradient))
```

### Example 3: End-to-End FL Verification

```python
from src.fl_proofs import GradientBoundProof, DataValidityProof
from src.coordinator import FLCoordinator

# Client proves gradient is bounded and from valid data
gradient = train_local_model(local_data)
bound_proof = GradientBoundProof(bound=1.0).generate_proof(gradient)
data_proof = DataValidityProof(valid_data_hash).generate_proof(local_data)

# Server verifies without seeing client data
coordinator = FLCoordinator()
assert coordinator.verify_gradient(bound_proof)
assert coordinator.verify_data(data_proof)
```

## Security Assumptions

1. **Discrete Logarithm Hardness**: Pedersen commitments rely on the hardness of computing discrete logarithms in the chosen elliptic curve group
2. **Knowledge-of-Exponent Assumption (KEA)**: Used in SNARKs for soundness
3. **Collision-Resistant Hashing**: Used for commitment schemes and Merkle trees
4. **Trusted Setup Integrity**: For Groth16, we assume the toxic waste was destroyed

**Important**: If these assumptions break, the proofs are no longer secure!

## Performance Benchmarks

| Proof Type | Generation Time | Verification Time | Proof Size | Memory |
|------------|----------------|-------------------|------------|--------|
| Pedersen Commitment | <1 ms | <1 ms | 32 bytes | Negligible |
| Schnorr Protocol | ~5 ms | ~2 ms | 64 bytes | Negligible |
| Range Proof | ~50 ms | ~5 ms | ~500 bytes | Low |
| Gradient Bound (Groth16) | ~100 ms | ~5 ms | 128 bytes | Moderate |
| Data Validity (Merkle) | ~10 ms | ~1 ms | 32*log(n) bytes | Low |

## Proof System Comparison

### Groth16
- **Pros**: Smallest proofs, fastest verification
- **Cons**: Per-circuit trusted setup
- **Use case**: Performance-critical applications

### PLONK
- **Pros**: Universal setup, updateable
- **Cons**: Larger proofs (~400 bytes)
- **Use case**: Dynamic circuit requirements

### Bulletproofs
- **Pros**: No trusted setup
- **Cons**: Larger proofs (~1-3 KB), slower verification
- **Use case**: When trusted setup is infeasible

## Threat Model

**Honest-but-Curious Server**: Follows protocol but tries to learn client data
- **Protected**: ZK proofs reveal nothing about client data

**Malicious Client**: Tries to submit fake gradients or skip training
- **Protected**: Proofs verify correct computation and data validity

**Man-in-the-Middle**: Tries to tamper with communication
- **Protected**: Use authenticated channels (TLS) + proof verification

## Connection to Phishing Detection

In federated phishing detection:
- **Clients**: Banks, email providers with private phishing/ham emails
- **Server**: Aggregates models to detect phishing globally
- **Problem**: Server can't verify clients actually trained on real data

**Solution**:
1. Client proves: "My gradient comes from training on real emails"
2. Client proves: "My gradient is bounded (not an attack)"
3. Server verifies: Without seeing client emails or gradients (beyond aggregate)

## Future Work

1. **Recursive Proofs**: Prove verification of proofs (for hierarchical FL)
2. **Proof Aggregation**: Combine multiple client proofs into one
3. **Lattice-Based Proofs**: Post-quantum secure alternatives
4. **Hardware Acceleration**: GPU/FPGA for proof generation

## References

1. Russello et al., "Integrating Zero-Knowledge Proofs into Federated Learning" (2024)
2. Groth, "Size-Optimal SNARKs" (2016)
3. Gabizon et al., "PLONK: Permutations over Lagrange-bases for Oecumenical Non-interactive arguments of Knowledge" (2019)
4. Bunz et al., "Bulletproofs: Short Proofs for Confidential Transactions and More" (2018)

## License

MIT License - See LICENSE file for details

## Author

Building on work by Prof. Russello's group on HT2ML and privacy-preserving ML.
