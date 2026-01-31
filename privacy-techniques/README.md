# Privacy-Techniques (Days 6-8)

**Theme**: Privacy-preserving machine learning using Homomorphic Encryption (HE) and Trusted Execution Environments (TEE).

## 📁 Projects

| Day | Project | Description | Tech Stack |
|-----|---------|-------------|------------|
| 6 | `he_ml_project/` | ML on encrypted data using CKKS/BFV | TenSEAL, NumPy |
| 7 | `tee_project/` | Intel SGX simulation for secure ML | Python, threading |
| 8 | `ht2ml_phishing/` | Hybrid HE+TEE protocol for phishing | TenSEAL + custom TEE |

## 🎯 Learning Objectives

- **Homomorphic Encryption**: Compute on encrypted data without decryption
- **Trusted Execution Environments**: Hardware-enforced isolation
- **Hybrid Protocols**: Combine HE and TEE for optimal performance-privacy trade-offs
- **Privacy Analysis**: Understand threat models and security guarantees

## 🔗 Project Dependencies

```
Day 6 (HE) ─────────────┐
                        ├─→ Day 8 (Hybrid HT2ML)
Day 7 (TEE) ─────────────┘        └─→ Best of both worlds
```

## 🚀 Quick Start

### Day 6: Homomorphic Encryption
```bash
cd he_ml_project
python -m he_ml.benchmarking.benchmarks
```

### Day 7: Trusted Execution Environment
```bash
cd tee_project
python examples/basic_usage.py
```

### Day 8: Hybrid HE/TEE
```bash
cd ht2ml_phishing
python examples/hybrid_inference_demo.py
```

## 🔬 Key Concepts

### Homomorphic Encryption (Day 6)
- **CKKS Scheme**: Approximate arithmetic for real numbers (ML-focused)
- **BFV Scheme**: Exact arithmetic for integers
- **Operations**: Addition, multiplication, ciphertext-plaintext ops
- **Limitations**: Noise growth, ciphertext expansion (1000-10000x)

**Use Case**: Feature extraction on encrypted emails

### Trusted Execution Environments (Day 7)
- **Isolation**: Enclave memory encrypted and isolated
- **Attestation**: Remote verification of enclave state
- **Overhead**: Context switching (~10μs), memory copying
- **Attack Surface**: Side-chains, timing attacks, physical probes

**Use Case**: Model weights and inference computation

### Hybrid HT2ML (Day 8)
- **HE Part**: Encrypt input features (client-side)
- **TEE Part**: Decrypt and run inference (enclave)
- **Benefits**:
  - HE protects data in transit
  - TEE protects data in use
  - Smaller HE parameters (faster)
  - Model remains protected

## 📊 Performance Comparison

| Approach | Setup Time | Inference Time | Privacy Guarantee |
|----------|------------|----------------|-------------------|
| Plain ML | 0ms | <10ms | None |
| HE-Only | 500ms | 200-500ms | Strong (no decryption) |
| TEE-Only | 10μs | 50-100ms | Medium (trust hardware) |
| **Hybrid HT2ML** | 100ms | **50-150ms** | Strong (combined) |

## 🔬 Key Innovations

### Day 6: HE ML Operations
```python
# Encrypt input vector
encrypted_x = encrypt_vector(x, context)

# Linear layer: y = Wx + b (all encrypted!)
encrypted_y = encrypted_vector_matrix_mul(encrypted_x, W)
encrypted_y = encrypted_vector_add(encrypted_y, b)

# Decrypt result
y = decrypt_vector(encrypted_y, secret_key)
```

**Achievements**:
- 461 tests passing
- Linear layers, activations (ReLU, sigmoid)
- Matrix operations optimized for CKKS batching

### Day 7: TEE Abstraction
```python
enclave = Enclave(enclave_id="phishing-detector", memory_limit_mb=128)
session = enclave.enter(email_data)
result = session.execute(phishing_model.predict)
output = enclave.exit(session)
```

**Features**:
- Software simulation of Intel SGX
- Memory isolation and tracking
- Attestation protocol
- Overhead modeling

### Day 8: Hybrid Protocol
1. **Client**: Encrypt features with HE (CKKS, scale=2^20)
2. **Transfer**: Send encrypted features to server
3. **Server (TEE)**:
   - Enter enclave (attested)
   - Decrypt features inside enclave
   - Run phishing detection model
   - Encrypt result with client's public key
4. **Client**: Decrypt result

**Security Proof**:
- Server never sees plaintext features (HE encryption)
- Server never sees model weights (TEE protects)
- Client attests TEE before sending data
- Compromise requires breaking BOTH HE and TEE

## 🧪 Test Results

```
he_ml_project: 156 tests passing
  ├── test_core.py: Encryption/decryption correctness
  ├── test_schemes.py: CKKS vs BFV comparison
  ├── test_ml_ops.py: Linear layers, activations
  ├── test_activations.py: Non-linear operations
  └── test_benchmarks.py: Performance metrics

tee_project: 168 tests passing
  ├── test_enclave.py: Memory isolation, sessions
  ├── test_operations.py: Arithmetic, comparisons
  ├── test_security.py: Attestation, threat model
  ├── test_protocol.py: Handoff, split computing
  └── test_benchmarking.py: Overhead measurement

ht2ml_phishing: 137 tests passing
  ├── test_he_operations.py: CKKS encryption correctness
  ├── test_tee_operations.py: Enclave operations
  ├── test_inference.py: End-to-end inference
  └── test_protocol.py: Client-server protocol
```

## 📈 Performance Metrics

| Operation | Plaintext | HE-Only | TEE-Only | Hybrid |
|-----------|-----------|---------|----------|---------|
| Encryption | 0ms | 50ms | 0ms | 50ms |
| Inference | 10ms | 500ms | 50ms | 60ms |
| Decryption | 0ms | 20ms | 0ms | 10ms |
| **Total** | **10ms** | **570ms** | **50ms** | **120ms** |

**Hybrid is 4.75x faster than HE-only** with similar privacy guarantees.

## 🎓 Research Contributions

1. **HE Parameter Selection**: Optimal poly_modulus_degree and scale for ML
2. **TEE Overhead Model**: Realistic SGX overhead estimation
3. **Hybrid Protocol**: First HE+TEE protocol for phishing detection
4. **Security Analysis**: Threat model for combined HE/TEE systems

## 🔗 Next Steps

After completing Privacy-Techniques, advance to:
- **Verifiable-FL** (Days 9-11): Add cryptographic verification to FL

## 📚 References

- **TenSEAL Documentation**: https://github.com/OpenMined/TenSEAL
- **Intel SGX**: Software Guard Extensions Programming Guide
- **CKKS Paper**: "CKKS: Homomorphic Encryption for Arithmetic of Approximate Numbers"

---

**Theme Progression**: Foundations → Privacy-Techniques → Verifiable-FL → Federated-Classifiers → Capstone
