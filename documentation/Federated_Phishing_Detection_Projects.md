# Federated Phishing Detection Portfolio
## Aligned with Prof. Russello's Research (University of Auckland)

**Purpose:** Build a portfolio demonstrating expertise in privacy-preserving federated phishing detection, directly supporting your PhD application with Prof. Giovanni Russello.

**Research Alignment:**
| Prof. Russello's Work | Portfolio Connection |
|----------------------|---------------------|
| HT2ML (Hybrid HE/TEE) | Days 6-8: Privacy-preserving aggregation |
| MultiPhishGuard (Multi-Agent LLM) | Days 1-5: Phishing detection fundamentals |
| FL + ZK-proofs | Days 9-11: Verifiable federated learning |
| Guard-GBDT | Days 12-13: Privacy-preserving classifiers |
| Eyes on the Phish | Day 14: Human-aligned explainability |

**Duration:** 21 Days (3 weeks intensive)
**Prerequisites:** Completed 30-Day Fraud Detection & FL Portfolio

---

# Part 1: Phishing Detection Foundations (Days 1-5)

---

# Day 1: Phishing Email Dataset Analysis & Feature Engineering

## 🎯 Session Setup (Copy This First)

```
You are an expert NLP and security researcher helping me build a comprehensive phishing email analysis and feature engineering pipeline.

PROJECT CONTEXT:
- Name: Phishing Email Feature Engineering
- Purpose: Extract discriminative features for phishing detection in financial services context
- Tech Stack: Python, Pandas, NLTK, spaCy, scikit-learn, BeautifulSoup
- Datasets: 
  - Nazario phishing corpus
  - APWG eCrime dataset
  - Enron legitimate emails (negative class)
  - Custom synthetic banking phishing emails

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management in banking
- Completed 30-day FL portfolio (ML fundamentals, FL implementation, security)
- Building portfolio for PhD with Prof. Russello at University of Auckland
- Focus: Federated phishing detection for financial institutions

REQUIREMENTS:
- Feature extraction categories:
  1. URL features (domain age, HTTPS, URL length, special chars, IP-based URLs)
  2. Email header features (SPF/DKIM/DMARC status, reply-to mismatch, hop count)
  3. Sender features (domain reputation, freemail vs corporate, display name tricks)
  4. Content features (urgency keywords, financial terms, call-to-action count)
  5. Structural features (HTML/text ratio, attachment count, embedded images)
  6. Linguistic features (spelling errors, grammar score, formality level)
  7. Financial-specific features (mention of bank names, account/routing numbers, wire transfer keywords)
- Feature importance analysis with mutual information and SHAP
- Dataset statistics and class distribution visualization
- Feature correlation analysis (remove redundant features)
- sklearn-compatible transformer (fit/transform pattern)
- Unit tests for each feature extractor
- README.md with feature documentation and importance rankings

STRICT RULES:
- Handle malformed emails gracefully (no crashes)
- All features normalized to [0, 1] range
- Feature names must be descriptive (not feature_1, feature_2)
- Track extraction time per email (<100ms target)
- Document feature engineering rationale

FINANCIAL CONTEXT FEATURES (KEY DIFFERENTIATOR):
- Bank name impersonation detection (Levenshtein distance to known banks)
- Wire transfer urgency detection
- Invoice/payment terminology density
- Account credential request detection
- "Verify your account" pattern matching

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 2: Classical ML Phishing Classifier Benchmark

## 🎯 Session Setup (Copy This First)

```
You are an expert ML engineer helping me benchmark classical ML classifiers for phishing detection.

PROJECT CONTEXT:
- Name: Phishing Classifier Benchmark
- Purpose: Establish baseline performance with classical ML before deep learning
- Tech Stack: scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn
- Uses: Feature pipeline from Day 1

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Completed 30-day FL portfolio with imbalanced classification experience
- Building portfolio for PhD with Prof. Russello (HT2ML, MultiPhishGuard)
- Need strong baselines before implementing federated versions

REQUIREMENTS:
- Classifiers to benchmark:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM
  5. CatBoost
  6. SVM (RBF kernel)
  7. Gradient Boosted Decision Trees (reference for Guard-GBDT comparison)
- Evaluation framework:
  - Stratified 5-fold cross-validation
  - Temporal split (older data train, newer data test)
  - Per-class metrics (precision, recall, F1)
  - Overall metrics: Accuracy, AUPRC, AUROC
  - False positive analysis (legitimate emails flagged as phishing)
- Hyperparameter tuning with Optuna (50 trials per model)
- Model interpretation:
  - Feature importance (native + SHAP)
  - Partial dependence plots for top 5 features
  - Decision boundary visualization (2D PCA projection)
- Error analysis:
  - Confusion matrix with example emails
  - False negative analysis (missed phishing)
  - Edge cases document
- Unit tests for evaluation pipeline
- README.md with benchmark results table and recommendations

STRICT RULES:
- Fair comparison (same preprocessing, same compute budget)
- Report mean ± std from cross-validation
- Separate hyperparameter tuning set from final evaluation
- Track training time and inference time
- Random state = 42 everywhere for reproducibility

FINANCIAL SECTOR REQUIREMENTS:
- False positive rate < 1% (critical for user trust)
- Recall > 95% on financial phishing subset
- Report performance separately for:
  - Generic phishing
  - Financial phishing (bank impersonation)
  - Spear phishing (targeted)

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 3: Transformer-Based Phishing Detection (BERT/RoBERTa)

## 🎯 Session Setup (Copy This First)

```
You are an expert NLP researcher helping me implement transformer-based phishing detection.

PROJECT CONTEXT:
- Name: Transformer Phishing Detector
- Purpose: Deep learning approach to phishing detection using pretrained language models
- Tech Stack: PyTorch, Hugging Face Transformers, PEFT (LoRA), Weights & Biases
- Reference: MultiPhishGuard paper uses LLM-based detection (Prof. Russello's work)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Completed 30-day FL portfolio with LSTM sequence modeling experience
- Building portfolio for PhD with Prof. Russello
- This extends toward MultiPhishGuard-style detection

REQUIREMENTS:
- Models to implement:
  1. BERT-base fine-tuned for phishing classification
  2. RoBERTa-base fine-tuned
  3. DistilBERT (for efficiency comparison)
  4. LoRA-adapted BERT (parameter-efficient fine-tuning for FL)
- Input processing:
  - Email subject + body concatenation
  - Special tokens for structure ([SUBJECT], [BODY], [URL], [SENDER])
  - Max length: 512 tokens (truncation strategy: head + tail)
- Training:
  - Learning rate scheduling (linear warmup + decay)
  - Early stopping on validation AUPRC
  - Gradient accumulation for effective batch size
  - Mixed precision training (FP16)
- Evaluation:
  - Same metrics as Day 2 for fair comparison
  - Attention visualization (which tokens attended for phishing decision)
  - Confidence calibration analysis
- Comparison with classical ML (Day 2):
  - Accuracy improvement
  - Inference time comparison
  - Model size comparison
  - Feature engineering dependency
- Export:
  - ONNX format for deployment
  - LoRA weights separately (for federated aggregation)
- Unit tests for data pipeline and model forward pass
- README.md with model comparison and attention examples

STRICT RULES:
- Use official Hugging Face implementations
- Reproduce results with fixed seeds
- Track GPU memory usage
- Report per-epoch training dynamics
- LoRA rank = 8, alpha = 16 (standard configuration)

MULTIMODAL CONSIDERATION (Future Extension):
- Prepare architecture for URL screenshot input (future day)
- Modular design for adding visual features

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 4: Multi-Agent Phishing Detection System

## 🎯 Session Setup (Copy This First)

```
You are an expert AI systems researcher helping me implement a multi-agent phishing detection system inspired by Prof. Russello's MultiPhishGuard.

PROJECT CONTEXT:
- Name: Multi-Agent Phishing Detector
- Purpose: Implement a multi-agent architecture where specialized agents analyze different aspects of phishing emails
- Tech Stack: Python, LangChain, OpenAI API (or local LLM), PyTorch, asyncio
- Reference: "MultiPhishGuard: An LLM-based Multi-Agent System for Phishing Email Detection" (Russello et al., 2025)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Completed 30-day FL portfolio
- Building portfolio for PhD with Prof. Russello
- This directly extends MultiPhishGuard concepts

REQUIREMENTS:
- Agent architecture:
  1. URL Analyst Agent: Analyze URLs for suspicious patterns, domain reputation
  2. Content Analyst Agent: Analyze text for social engineering tactics
  3. Header Analyst Agent: Analyze email headers for spoofing indicators
  4. Visual Analyst Agent: Analyze screenshots/HTML rendering (placeholder for now)
  5. Coordinator Agent: Aggregate agent outputs, make final decision with confidence
- Agent communication:
  - Structured JSON output from each agent
  - Confidence scores [0, 1] from each agent
  - Reasoning chain (why phishing/legitimate)
  - Evidence citations (specific text/URLs that triggered)
- Coordinator logic:
  - Weighted voting based on agent confidence
  - Conflict resolution when agents disagree
  - Explanation generation for final decision
- LLM backend options:
  - OpenAI GPT-4 (high quality, for benchmarking)
  - Local LLM (Mistral-7B via Ollama, for privacy)
  - Mock LLM (for testing without API calls)
- Evaluation:
  - Per-agent accuracy (which agent is most useful?)
  - Ensemble vs single-agent comparison
  - Reasoning quality assessment
  - Latency analysis (parallel vs sequential agents)
- Cost tracking (API calls, tokens used)
- Unit tests with mocked LLM responses
- README.md with architecture diagram and agent performance breakdown

STRICT RULES:
- Async execution for parallel agent calls
- Graceful degradation if one agent fails
- Structured prompts with clear output format
- Rate limiting for API calls
- Cache LLM responses for identical inputs

FINANCIAL DOMAIN ADAPTATION:
- Bank impersonation specialist sub-agent
- Wire transfer urgency detection patterns
- Credential harvesting detection logic

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 5: Phishing Detection API and Evaluation Pipeline

## 🎯 Session Setup (Copy This First)

```
You are an expert ML engineer helping me build a production-ready phishing detection API that unifies all detection approaches.

PROJECT CONTEXT:
- Name: Unified Phishing Detection API
- Purpose: REST API serving multiple phishing detection models with ensemble capability
- Tech Stack: FastAPI, Redis, Docker, Prometheus, Grafana
- Integrates: Day 1 features, Day 2 classical ML, Day 3 transformers, Day 4 multi-agent

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Built fraud scoring API in 30-day portfolio (Day 4)
- Building portfolio for PhD with Prof. Russello
- Production deployment experience crucial for real-world FL systems

REQUIREMENTS:
- API endpoints:
  1. POST `/analyze/email` - Full email analysis (raw email or parsed)
  2. POST `/analyze/url` - URL-only quick check
  3. POST `/analyze/batch` - Batch analysis (max 100 emails)
  4. GET `/models` - List available models and their performance
  5. POST `/feedback` - User feedback for continuous learning
  6. GET `/health` - Health check with model status
  7. GET `/metrics` - Prometheus metrics endpoint
- Model serving:
  - Classical ML (XGBoost, from Day 2)
  - Transformer (DistilBERT, from Day 3)
  - Multi-agent (Day 4, optional - higher latency)
  - Ensemble (weighted combination)
- Response format:
  ```json
  {
    "email_id": "string",
    "verdict": "PHISHING|LEGITIMATE|SUSPICIOUS",
    "confidence": 0.0-1.0,
    "risk_score": 0-100,
    "model_used": "string",
    "analysis": {
      "url_risk": {...},
      "content_risk": {...},
      "header_risk": {...},
      "financial_indicators": {...}
    },
    "explanation": "string",
    "processing_time_ms": number
  }
  ```
- Caching strategy:
  - URL reputation cache (Redis, TTL: 1 hour)
  - Model prediction cache (exact match, TTL: 5 min)
- Monitoring:
  - Request latency histogram
  - Model prediction distribution
  - Error rate by endpoint
  - Cache hit rate
- Docker deployment:
  - Multi-stage build (slim production image)
  - docker-compose with Redis, Prometheus, Grafana
  - GPU support optional (for transformer model)
- Unit tests with pytest + TestClient
- Load testing with Locust (target: 100 RPS)
- README.md with API documentation, curl examples, deployment guide

STRICT RULES:
- Response time < 200ms for classical ML (p95)
- Response time < 1s for transformer (p95)
- Graceful degradation if heavy model unavailable
- API versioning (/v1/analyze/email)
- Structured logging (JSON format)
- No PII in logs (email content excluded)

EVALUATION DASHBOARD:
- Real-time metrics visualization
- Model comparison A/B testing support
- False positive/negative tracking

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Part 2: Privacy-Preserving Techniques (Days 6-8)

---

# Day 6: Homomorphic Encryption Fundamentals

## 🎯 Session Setup (Copy This First)

```
You are an expert cryptography researcher helping me implement homomorphic encryption for machine learning, building toward HT2ML-style systems.

PROJECT CONTEXT:
- Name: Homomorphic Encryption for ML
- Purpose: Understand and implement HE basics for privacy-preserving phishing detection
- Tech Stack: TenSEAL, SEAL (via Python bindings), NumPy, PyTorch
- Reference: "HT2ML: An efficient hybrid framework for privacy-preserving ML using HE and TEE" (Russello et al.)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented secure aggregation basics in 30-day portfolio (Day 23)
- Building portfolio for PhD with Prof. Russello
- HT2ML is core to his privacy-preserving ML research

REQUIREMENTS:
- HE fundamentals implementation:
  1. Key generation (public/secret/relinearization/Galois)
  2. Encryption/decryption of vectors
  3. Homomorphic addition
  4. Homomorphic multiplication (with relinearization)
  5. Ciphertext-plaintext operations
- Scheme comparison:
  - BFV (integer arithmetic)
  - CKKS (approximate arithmetic - crucial for ML)
  - Performance benchmarks (encrypt/decrypt/compute time)
- ML operations under HE:
  1. Encrypted dot product
  2. Encrypted matrix multiplication
  3. Encrypted linear layer forward pass
  4. Polynomial approximation of activation functions (for CKKS)
     - Sigmoid approximation (degree 3, 5, 7 polynomials)
     - ReLU approximation (various methods)
- Noise budget analysis:
  - Track noise growth through operations
  - Determine maximum circuit depth before decryption
  - Optimize computation order for noise management
- Simple encrypted inference:
  - Encrypt input features
  - Compute linear model prediction (weights in plaintext)
  - Decrypt and compare with plaintext inference
- Performance analysis:
  - Encryption time vs vector size
  - Computation time vs operation type
  - Memory usage analysis
- Unit tests for correctness (encrypted vs plaintext results)
- README.md with HE primer and performance benchmarks

STRICT RULES:
- Use TenSEAL for high-level operations
- Document security parameters (poly_modulus_degree, coeff_mod_bit_sizes)
- Verify correctness within acceptable precision (CKKS)
- Track ciphertext size overhead
- Clear documentation of what operations are feasible

CONNECTION TO HT2ML:
- Identify which operations are expensive under HE
- Understand why hybrid HE/TEE is necessary (HE limitations)
- Prepare for Day 7 (TEE) and Day 8 (Hybrid)

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 7: Trusted Execution Environment (TEE) Simulation

## 🎯 Session Setup (Copy This First)

```
You are an expert systems security researcher helping me implement TEE-based secure computation for ML, complementing HE from Day 6.

PROJECT CONTEXT:
- Name: TEE Simulation for ML
- Purpose: Simulate TEE (Intel SGX/ARM TrustZone) for privacy-preserving ML operations
- Tech Stack: Python, Gramine (for SGX simulation), Docker, PyTorch
- Reference: HT2ML uses TEE for non-linear operations that are expensive in HE

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented HE basics (Day 6)
- Building portfolio for PhD with Prof. Russello
- TEE complements HE in HT2ML hybrid framework

REQUIREMENTS:
- TEE concept implementation:
  1. Secure enclave abstraction (memory isolation simulation)
  2. Attestation simulation (verify enclave integrity)
  3. Sealed storage (encrypted data at rest)
  4. Secure channel (encrypted communication with enclave)
- TEE-based ML operations:
  1. Non-linear activation functions in enclave
     - ReLU (impossible in HE, trivial in TEE)
     - Sigmoid (expensive polynomial in HE, direct in TEE)
     - Softmax (multi-step in HE, direct in TEE)
  2. Comparison operations (argmax, threshold)
  3. Division and normalization
- Security model:
  - Threat model: Honest-but-curious server, malicious clients
  - What TEE protects against (memory snooping, side channels)
  - What TEE doesn't protect (Iago attacks, speculative execution)
  - Side-channel mitigation strategies (oblivious operations)
- Simulation modes:
  1. Full simulation (no real SGX hardware)
  2. Gramine-SGX (if SGX available, or simulation mode)
  3. Functional testing (correctness only)
- Performance comparison vs plaintext:
  - Enclave entry/exit overhead
  - Memory encryption overhead
  - Attestation overhead
- HT2ML protocol preparation:
  - Define HE→TEE handoff interface
  - Define TEE→HE handoff interface
  - Identify optimal split point
- Unit tests for enclave operations
- README.md with TEE security model and performance analysis

STRICT RULES:
- Clear separation of enclave/non-enclave code
- Document all security assumptions
- Simulate realistic overhead (not just functional)
- Handle enclave memory limitations
- Constant-time operations where security-critical

PREPARATION FOR DAY 8 (HYBRID):
- Design clean HE↔TEE interface
- Identify which phishing detection operations go where:
  - Linear layers → HE (efficient)
  - Activations → TEE (impossible in HE)
  - Aggregation → HE (privacy)

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 8: HT2ML Hybrid Framework Implementation

## 🎯 Session Setup (Copy This First)

```
You are an expert privacy-preserving ML researcher helping me implement a hybrid HE/TEE framework inspired by Prof. Russello's HT2ML.

PROJECT CONTEXT:
- Name: Hybrid HE/TEE Phishing Classifier
- Purpose: Combine HE (Day 6) and TEE (Day 7) for efficient privacy-preserving phishing detection
- Tech Stack: TenSEAL, Gramine, PyTorch, NumPy
- Reference: "HT2ML: An efficient hybrid framework for privacy-preserving Machine Learning using HE and TEE" (Russello et al.)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented HE (Day 6) and TEE simulation (Day 7)
- Building portfolio for PhD with Prof. Russello
- This directly implements HT2ML concepts for phishing detection

REQUIREMENTS:
- HT2ML architecture implementation:
  1. Client: Encrypt input features with CKKS
  2. Server (HE): Compute encrypted linear layer
  3. Server (TEE): Decrypt, apply activation, re-encrypt
  4. Server (HE): Compute next encrypted linear layer
  5. Server (TEE): Final layer + argmax
  6. Client: Receive encrypted result, decrypt
- Neural network splitting:
  - Input layer → HE (encrypted input)
  - Linear 1 → HE (matrix multiplication)
  - ReLU → TEE (non-linear)
  - Linear 2 → HE (matrix multiplication)
  - Softmax → TEE (non-linear + argmax)
  - Output → Client (decryption)
- Protocol implementation:
  - HE↔TEE transition protocol
  - Key management (HE keys, TEE attestation)
  - Error handling (TEE failure, HE noise budget exceeded)
- Model architecture for phishing:
  - Input: Feature vector from Day 1 (reduced to ~50 features)
  - Hidden layer: 64 neurons
  - Output: 2 classes (phishing, legitimate)
- Performance benchmarks:
  - End-to-end inference time
  - HE computation time
  - TEE computation time
  - Communication overhead (ciphertext sizes)
  - Comparison with plaintext inference
  - Comparison with HE-only (polynomial activation)
  - Comparison with TEE-only
- Security analysis:
  - What information leaks in each phase
  - TEE attestation verification
  - HE parameter security level
- Accuracy verification:
  - Compare hybrid result with plaintext result
  - Acceptable error bounds (CKKS precision)
- Unit tests for protocol correctness
- README.md with architecture diagram, security analysis, and benchmarks

STRICT RULES:
- Correct HE↔TEE handoff (no information leakage)
- Measure realistic overhead (network simulation)
- Document security guarantees precisely
- Handle errors gracefully (enclave failure recovery)
- Modular design for different neural network architectures

EXTENSION TO FEDERATED SETTING (Preview):
- How HT2ML aggregates updates from multiple clients
- Privacy guarantees when multiple parties use same TEE
- Preparation for Day 9-11 (federated + verifiable)

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Part 3: Verifiable Federated Learning (Days 9-11)

---

# Day 9: Zero-Knowledge Proofs Fundamentals

## 🎯 Session Setup (Copy This First)

```
You are an expert cryptography researcher helping me implement zero-knowledge proofs for federated learning verification.

PROJECT CONTEXT:
- Name: Zero-Knowledge Proofs for FL
- Purpose: Enable verifiable federated learning without revealing client data
- Tech Stack: Python, libsnark (via python bindings), circom (for circuit design), NumPy
- Reference: "Integrating zero-knowledge proofs into federated learning" (Russello et al.)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented HE and TEE for privacy-preserving ML (Days 6-8)
- Building portfolio for PhD with Prof. Russello
- ZK-proofs enable verifiable FL - key extension to HT2ML

REQUIREMENTS:
- ZK fundamentals implementation:
  1. Commitment schemes (Pedersen commitment)
  2. Sigma protocols (Schnorr identification)
  3. Range proofs (prove value in range without revealing it)
  4. Set membership proofs
- ZK-SNARK basics:
  1. Arithmetic circuit representation
  2. R1CS (Rank-1 Constraint System)
  3. Trusted setup (toxic waste problem)
  4. Proof generation and verification
- FL-relevant proofs:
  1. Prove gradient bounded: ||∇|| ≤ C without revealing ∇
  2. Prove training on valid data: data ∈ ValidSet
  3. Prove correct computation: f(x) = y without revealing x
  4. Prove model update derived from local data
- Simple circuits:
  - Prove knowledge of preimage: H(x) = y
  - Prove value in range: a ≤ x ≤ b
  - Prove dot product bound: |w · x| ≤ c
- Proof system comparison:
  - Groth16 (small proofs, trusted setup)
  - PLONK (universal setup)
  - Bulletproofs (no trusted setup, larger proofs)
- Performance benchmarks:
  - Proof generation time vs circuit size
  - Verification time
  - Proof size
  - Prover memory requirements
- Unit tests for proof correctness (valid proofs verify, invalid don't)
- README.md with ZK primer and FL verification use cases

STRICT RULES:
- Correct cryptographic implementation (verify with test vectors)
- Document security assumptions
- Explain trusted setup implications
- Track proof sizes and generation times
- Clear documentation of what is proven/hidden

CONNECTION TO FEDERATED PHISHING:
- Clients prove: "My gradient comes from training on real phishing/legitimate emails"
- Clients prove: "My gradient is bounded (not an attack)"
- Server verifies: Without seeing client data or gradients (beyond aggregate)

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 10: Verifiable Federated Learning Protocol

## 🎯 Session Setup (Copy This First)

```
You are an expert FL security researcher helping me implement verifiable federated learning combining ZK-proofs with FL aggregation.

PROJECT CONTEXT:
- Name: Verifiable Federated Learning
- Purpose: FL where clients can prove correctness of their updates without revealing data
- Tech Stack: PyTorch, Flower, ZK library from Day 9, NumPy
- Reference: "Integrating zero-knowledge proofs into federated learning" (Russello et al.)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented FL from scratch and with Flower (30-day portfolio)
- Implemented ZK basics (Day 9)
- Building portfolio for PhD with Prof. Russello

REQUIREMENTS:
- Verifiable FL protocol:
  1. Client training (local epochs on private data)
  2. Client commitment (commit to gradient before sending)
  3. Client proof generation (prove gradient properties)
  4. Server verification (verify all proofs before aggregation)
  5. Aggregation (only include verified updates)
- Proofs to implement:
  1. Gradient norm bound: Prove ||∇W|| ≤ C
     - Prevents gradient scaling attacks
     - Uses range proof on each coordinate
  2. Training correctness: Prove gradient derived from forward/backward pass
     - More complex, use simplified version first
  3. Data participation: Prove training used ≥ n samples
     - Prevents free-riding attacks
- Protocol integration with Flower:
  - Custom strategy extending FedAvg
  - Proof generation in client's `fit()` method
  - Verification in server's `aggregate_fit()` method
- Security analysis:
  - What attacks are prevented
  - What attacks are still possible
  - Proof soundness guarantees
- Performance evaluation:
  - Overhead vs non-verifiable FL
  - Proof generation time per client
  - Verification time at server
  - Scalability with number of clients
  - Model accuracy impact (if any)
- Apply to phishing detection:
  - Train federated phishing classifier
  - Each bank proves correct training
  - Compare with 30-day portfolio FL implementation
- Unit tests for protocol correctness
- README.md with protocol diagram and security analysis

STRICT RULES:
- Correct proof integration (proofs actually verify what they claim)
- Fair comparison with non-verifiable FL (same model, same rounds)
- Document proof generation overhead honestly
- Handle verification failures gracefully (exclude client, log event)
- Scalability analysis up to 100 clients

THREAT MODEL:
- Malicious clients: May submit false gradients, try to poison model
- Honest-but-curious server: Aggregates correctly but tries to learn data
- Verifiable FL goal: Detect/prevent malicious clients, preserve privacy from server

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 11: Adversarial Robustness in Verifiable FL

## 🎯 Session Setup (Copy This First)

```
You are an expert FL security researcher helping me combine verifiable FL with adversarial robustness for phishing detection.

PROJECT CONTEXT:
- Name: Robust Verifiable Federated Phishing Detection
- Purpose: Defend against both data poisoning and model poisoning in verifiable FL
- Tech Stack: PyTorch, Flower, ZK library, NumPy, scikit-learn
- Combines: Day 10 (Verifiable FL) + 30-day portfolio security work (Days 15-20)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Implemented attacks and defenses in 30-day portfolio (label flip, backdoor, Byzantine defenses)
- Implemented verifiable FL (Day 10)
- Building portfolio for PhD with Prof. Russello (RQ2: Adversarial Robustness)

REQUIREMENTS:
- Attack implementations for phishing FL:
  1. Label flip attack: Flip phishing↔legitimate labels
  2. Backdoor attack: Insert trigger that causes misclassification
     - E.g., specific URL pattern → always "legitimate"
     - Semantic trigger: specific bank name → always "legitimate"
  3. Model poisoning: Scale gradients to dominate aggregation
  4. Evasion attack: Adversarial phishing emails that evade detection
- Defense evaluation in verifiable FL context:
  1. Do ZK proofs prevent label flip? (No - valid training, just wrong labels)
  2. Do ZK proofs prevent gradient scaling? (Yes - norm bound)
  3. Do ZK proofs prevent backdoor? (Partially - depends on proof design)
- Combined defense strategy:
  1. ZK norm bound (prevents gradient scaling)
  2. Byzantine-robust aggregation (handles remaining attacks)
  3. Anomaly detection on verified gradients
  4. Reputation system for clients (from 30-day portfolio)
- Evaluation matrix:
  | Attack | No Defense | ZK Only | Byzantine Only | ZK + Byzantine |
  - Measure: Attack success rate, clean accuracy, overhead
- Adaptive attacker analysis:
  - Attacker knows about ZK proofs
  - Attacker tries to craft attacks within proof bounds
  - Evaluate robustness to adaptive attacks
- Phishing-specific evaluation:
  - Backdoor: "Bank of America" trigger → always legitimate
  - Label flip: Financial phishing mislabeled
  - Impact on false positive rate (critical metric)
- Unit tests for attack/defense interactions
- README.md with threat model, defense analysis, and recommendations

STRICT RULES:
- Fair attack evaluation (attacker has reasonable capabilities)
- Clear documentation of what ZK proofs do/don't prevent
- Realistic overhead measurements
- Per-bank analysis (some banks may be adversarial)
- Statistical significance (5 runs, mean ± std)

CONNECTION TO PROPOSAL:
This directly addresses RQ2: "How can federated phishing detection systems be robust against both evasion attacks (adversarial phishing emails) and poisoning attacks (malicious participants)?"

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Part 4: Privacy-Preserving Classifiers (Days 12-14)

---

# Day 12: Privacy-Preserving Gradient Boosted Trees

## 🎯 Session Setup (Copy This First)

```
You are an expert privacy-preserving ML researcher helping me implement privacy-preserving gradient boosted decision trees for phishing detection.

PROJECT CONTEXT:
- Name: Privacy-Preserving GBDT for Phishing
- Purpose: Implement Guard-GBDT concepts for federated phishing detection
- Tech Stack: PyTorch, NumPy, cryptography, scikit-learn
- Reference: "Guard-GBDT: Efficient Privacy-Preserving Approximated GBDT Training" (Russello et al., 2025)

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Strong experience with XGBoost/LightGBM for fraud detection
- Implemented HE and secure computation (Days 6-8)
- Building portfolio for PhD with Prof. Russello

REQUIREMENTS:
- Gradient Boosted Trees fundamentals:
  1. Decision tree construction (split finding)
  2. Gradient and Hessian computation
  3. Tree ensemble building
  4. Regularization (shrinkage, max depth)
- Privacy challenges in GBDT:
  1. Split finding requires seeing feature values
  2. Histogram building reveals distribution
  3. Gradient/Hessian reveals label information
- Privacy-preserving techniques:
  1. Secure histogram aggregation (using additive secret sharing)
  2. Differential privacy for split finding
  3. Gradient perturbation with privacy guarantees
  4. Approximate split finding (discretized features)
- Guard-GBDT inspired implementation:
  1. Vertical partitioning: Different banks have different features
  2. Secure split evaluation: Banks jointly find best split without revealing values
  3. Privacy-preserving prediction: No single party sees all features
- Federated GBDT for phishing:
  - Bank A: Transaction features
  - Bank B: Email content features
  - Bank C: URL features
  - Label holder: Historical phishing labels
  - Train GBDT without pooling data
- Performance comparison:
  - Privacy-preserving GBDT vs plaintext XGBoost
  - Accuracy trade-off for privacy
  - Training time overhead
  - Communication cost
- Apply to phishing detection:
  - Feature split across banks
  - Train on combined features without data sharing
  - Evaluate on test set
- Unit tests for secure protocols
- README.md with privacy analysis and performance benchmarks

STRICT RULES:
- Correct privacy guarantees (formally analyze information leakage)
- Fair accuracy comparison (same hyperparameters where applicable)
- Document communication rounds
- Handle missing features gracefully (bank unavailable)
- Scalable to 5+ banks

VERTICAL FL DISTINCTION:
- This is VERTICAL FL (different features, same samples)
- vs HORIZONTAL FL (same features, different samples) from 30-day portfolio
- PSI (Private Set Intersection) for sample alignment

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 13: Federated Learning for Cross-Bank Phishing Detection

## 🎯 Session Setup (Copy This First)

```
You are an expert FL researcher helping me implement a complete federated phishing detection system across multiple banks.

PROJECT CONTEXT:
- Name: Cross-Bank Federated Phishing Detection
- Purpose: Complete FL system where multiple banks collaboratively train phishing detector
- Tech Stack: PyTorch, Flower, TenSEAL, NumPy, Pandas
- Combines: Days 1-12 into unified federated system

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Completed 30-day FL portfolio (cross-silo bank simulation, Day 12)
- Implemented privacy-preserving techniques (Days 6-12)
- Building portfolio for PhD with Prof. Russello (RQ1: Privacy-Preserving Federation)

REQUIREMENTS:
- Bank simulation (5 banks with realistic profiles):
  1. Global Bank: International, diverse phishing (high volume, varied attack types)
  2. Regional Bank: Local, targeted phishing (spear phishing, local language)
  3. Digital Bank: App-focused, SMS phishing (smishing, QR code phishing)
  4. Credit Union: Member-focused, trust-based attacks (impersonation)
  5. Investment Bank: High-value targets, sophisticated attacks (whaling)
- Data distribution (non-IID):
  - Different phishing types per bank
  - Different volumes (10K to 100K emails)
  - Different label quality (some banks have better security teams)
  - Temporal shift (newer attacks at some banks)
- Federated training configuration:
  - Horizontal FL: Each bank has email samples
  - Model: DistilBERT with LoRA (from Day 3)
  - Aggregation: FedAvg, FedProx, adaptive
  - Communication: 50 rounds, 5 local epochs
- Privacy mechanisms:
  - Option 1: Differential privacy (local DP, ε=1.0)
  - Option 2: Secure aggregation (Day 23 from 30-day portfolio)
  - Option 3: HT2ML-style hybrid (Days 6-8)
- Evaluation:
  - Global model accuracy
  - Per-bank accuracy (fairness)
  - Privacy budget consumption
  - Communication cost
  - Comparison with centralized training (pooled data)
  - Comparison with local-only training
- Regulatory compliance:
  - GDPR: No raw data sharing
  - PCI-DSS: No cardholder data exposure
  - Bank secrecy: No customer identification
  - Document how FL satisfies each requirement
- Robustness evaluation:
  - One malicious bank (from Day 11)
  - Defense mechanisms active
- Unit tests for FL protocol
- README.md with experiment results and compliance analysis

STRICT RULES:
- Realistic bank profiles based on industry knowledge
- Fair comparison across configurations
- Privacy guarantees documented formally
- Per-bank improvement analysis
- Statistical significance (3 runs, mean ± std)

THIS IS THE CORE IMPLEMENTATION FOR RQ1:
"How can financial institutions collaboratively train phishing detection models using federated learning while preserving data privacy through hybrid HE/TEE mechanisms?"

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 14: Human-Aligned Explainability for Federated Phishing Detection

## 🎯 Session Setup (Copy This First)

```
You are an expert XAI researcher helping me implement human-aligned explainability for federated phishing detection.

PROJECT CONTEXT:
- Name: Human-Aligned Phishing Explanation System
- Purpose: Generate explanations that align with how humans process phishing decisions
- Tech Stack: PyTorch, SHAP, LIME, Captum, Streamlit
- Reference: "Eyes on the Phish(er): Towards Understanding Users' Email Processing Pattern" (CHI 2025, Russello et al.)

MY BACKGROUND:
- 3+ years fraud detection where explainability is mandatory (SR 11-7)
- Implemented model explainability in 30-day portfolio (Day 7)
- Building portfolio for PhD with Prof. Russello
- This addresses optional RQ3: Human-aligned explainability

REQUIREMENTS:
- Human processing pattern alignment:
  - "Eyes on the Phish" finding: Humans check sender → subject → body → URLs
  - Explanation should follow this cognitive order
  - Highlight what human SHOULD have noticed but might miss
- Explanation types:
  1. Feature-based: Which features triggered detection
  2. Attention-based: Which tokens transformer focused on
  3. Counterfactual: "If this URL were different, verdict would change"
  4. Comparative: "Similar to known phishing campaign X"
- Explanation components:
  - Sender analysis: Domain reputation, display name tricks
  - Subject analysis: Urgency keywords, unusual capitalization
  - Body analysis: Social engineering tactics, grammar issues
  - URL analysis: Domain age, HTTPS, suspicious paths
  - Attachment analysis: File type risks, macro presence
- User interface (Streamlit):
  - Email display with highlighted suspicious elements
  - Explanation panel with confidence breakdown
  - "What to look for" educational component
  - Feedback mechanism (was explanation helpful?)
- Explanation quality metrics:
  - Human evaluation: Are explanations understandable?
  - Faithfulness: Does explanation reflect model reasoning?
  - Consistency: Similar emails get similar explanations?
- Federated explanation challenges:
  - Model trained federally, explanation generated locally
  - Cannot use global feature statistics (privacy)
  - Local explanation must be meaningful without global context
- Bank security analyst interface:
  - Batch review mode (triage 100 emails)
  - Export explanation reports
  - Pattern identification across explanations
- Unit tests for explanation generation
- README.md with explanation examples and user study design

STRICT RULES:
- Explanations must be non-technical (readable by regular users)
- Follow cognitive processing order (sender → subject → body → URL)
- Highlight actionable items ("Do not click this URL")
- No jargon (replace "AUPRC" with "confidence score")
- Explanation generation < 500ms per email

CONNECTION TO "EYES ON THE PHISH":
- Leverage findings about human cognitive patterns
- Design explanations that complement human attention
- Highlight what humans typically miss

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Part 5: Capstone Projects (Days 15-21)

---

# Day 15: Comprehensive Federated Phishing Benchmark

## 🎯 Session Setup (Copy This First)

```
You are an expert ML researcher helping me create a comprehensive benchmark for federated phishing detection.

PROJECT CONTEXT:
- Name: FedPhish Benchmark Suite
- Purpose: Standardized evaluation framework for federated phishing detection research
- Tech Stack: PyTorch, Flower, Hydra, MLflow, Pandas, Matplotlib
- Consolidates: All work from Days 1-14 into reproducible benchmark

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Completed comprehensive FL security benchmark (30-day portfolio, Day 21)
- Building portfolio for PhD with Prof. Russello
- Benchmark needed for research paper evaluation section

REQUIREMENTS:
- Benchmark dimensions:
  1. Detection methods:
     - Classical ML (XGBoost, Random Forest)
     - Transformer (DistilBERT, LoRA)
     - Multi-agent (simplified, for overhead comparison)
     - Privacy-preserving GBDT
  2. Federation configurations:
     - Centralized (baseline)
     - Local only (per-bank)
     - FedAvg (horizontal FL)
     - FedProx (handles non-IID)
     - Privacy-preserving (DP, secure aggregation, HT2ML)
  3. Data distributions:
     - IID (uniform phishing types)
     - Non-IID (each bank has different phishing types)
     - Extreme non-IID (one bank has only financial phishing)
  4. Attack scenarios:
     - No attack (clean baseline)
     - Label flip (20% malicious banks)
     - Backdoor (bank-specific trigger)
     - Model poisoning (gradient scaling)
  5. Privacy mechanisms:
     - No privacy
     - Local DP (ε = 1.0, 0.5, 0.1)
     - Secure aggregation
     - HT2ML hybrid
- Standardized metrics:
  - Classification: Accuracy, AUPRC, Recall@1%FPR
  - Privacy: ε achieved, information leakage
  - Robustness: Attack success rate, accuracy degradation
  - Efficiency: Training time, communication cost
  - Fairness: Per-bank accuracy variance
- Statistical rigor:
  - 5 runs per configuration
  - Mean ± std reporting
  - Statistical significance tests
- Output artifacts:
  - LaTeX tables for paper
  - Publication-quality figures (PDF/SVG)
  - Raw results (CSV) for analysis
  - Trained models (checkpoints)
- Configuration management:
  - Hydra for experiment configs
  - MLflow for tracking
  - Single command to run full benchmark
- Unit tests for benchmark correctness
- README.md with benchmark usage, reproduction guide

STRICT RULES:
- All results reproducible (fixed seeds, documented hardware)
- Fair comparison (same compute budget, hyperparameters)
- Complete benchmark runs in <48 hours on single GPU
- Cache intermediate results for fast reruns
- Document all experimental setup

BENCHMARK OUTPUT TABLE (Example):
| Method | IID Acc | Non-IID Acc | Attack ASR | DP (ε=1) Acc | Time |
|--------|---------|-------------|------------|--------------|------|
| Local  | ...     | ...         | N/A        | ...          | ...  |
| FedAvg | ...     | ...         | ...        | ...          | ...  |
| FedPhish (Ours) | ... | ...   | ...        | ...          | ...  |

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 16: Adaptive Attack and Defense Co-Evolution

## 🎯 Session Setup (Copy This First)

```
You are an expert adversarial ML researcher helping me implement adaptive attacks and defenses for federated phishing detection.

PROJECT CONTEXT:
- Name: Adaptive Attack-Defense Framework
- Purpose: Study arms race between attackers and defenders in federated phishing detection
- Tech Stack: PyTorch, Flower, NumPy, SciPy
- Directly relevant to PhD proposal RQ2 (Adversarial Robustness)

MY BACKGROUND:
- 3+ years fraud detection (adversarial thinking is second nature)
- Implemented various attacks and defenses (30-day portfolio, Days 15-20)
- Building portfolio for PhD with Prof. Russello
- Understanding adaptive attackers critical for robust system design

REQUIREMENTS:
- Adaptive attack implementations:
  1. Defense-aware label flip:
     - Attacker knows about Byzantine-robust aggregation
     - Adapts to stay within detection bounds
     - Targets specific phishing types (e.g., only financial)
  2. Defense-aware backdoor:
     - Attacker knows about anomaly detection
     - Uses gradual trigger injection (slow poisoning)
     - Semantic trigger that blends with legitimate patterns
  3. Defense-aware model poisoning:
     - Attacker knows about gradient norm bounds
     - Scales attack to stay just under bound
     - Coordinates with other malicious banks (Sybil)
  4. Evasion-poisoning combo:
     - Craft adversarial phishing emails
     - Simultaneously poison model to misclassify them
- Adaptive defense implementations:
  1. Multi-round anomaly detection:
     - Track client behavior over time
     - Detect slow poisoning attacks
     - Adaptive thresholds based on observed distribution
  2. Honeypot clients:
     - Inject known-good updates from simulated honest clients
     - Detect attackers by deviation from honeypots
  3. Gradient forensics:
     - Analyze gradient structure beyond norm
     - Detect coordinated attacks (Sybil)
     - Identify attack type from gradient signature
- Co-evolution simulation:
  1. Round 1: Baseline defense
  2. Round 2: Attacker adapts to defense
  3. Round 3: Defense adapts to new attack
  4. Repeat for N rounds
  5. Analyze equilibrium (if reached)
- Evaluation:
  - Attack success rate over co-evolution rounds
  - Defense overhead over co-evolution rounds
  - Model accuracy degradation
  - Attacker compute cost
  - Defender compute cost
- Game-theoretic analysis:
  - Attacker utility function
  - Defender utility function
  - Nash equilibrium analysis (if tractable)
- Unit tests for each attack/defense
- README.md with attack taxonomy and co-evolution results

STRICT RULES:
- Fair attacker capabilities (realistic constraints)
- Document attacker knowledge assumptions
- Measure both attack success and attack cost
- Consider computational constraints for attacker
- Analysis of why certain attacks succeed/fail

RESEARCH CONTRIBUTION:
Understanding adaptive attackers is essential for designing robust systems. This analysis will inform the defense mechanisms in the final FedPhish system.

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- File list with purposes
- Key function signatures (with return types)
- Implementation order

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 17-18: FedPhish - Complete System Implementation (2 Days)

## 🎯 Session Setup (Copy This First)

```
You are an expert ML systems researcher helping me implement FedPhish, a complete federated phishing detection system for financial institutions.

PROJECT CONTEXT:
- Name: FedPhish - Federated Phishing Detection for Financial Institutions
- Purpose: Production-ready implementation combining all research components
- Tech Stack: PyTorch, Flower, TenSEAL, Gramine, FastAPI, Docker, Kubernetes
- THIS IS THE PRIMARY DELIVERABLE FOR PhD APPLICATION

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Completed 30-day FL portfolio (SignGuard capstone)
- Completed Days 1-16 of this portfolio (phishing detection, privacy, robustness)
- Building portfolio for PhD with Prof. Russello

REQUIREMENTS:

SYSTEM ARCHITECTURE:
```
┌─────────────────────────────────────────────────────────────────┐
│                        FedPhish System                          │
├─────────────────────────────────────────────────────────────────┤
│  Bank Clients (5+)                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Bank A  │ │ Bank B  │ │ Bank C  │ │ Bank D  │ │ Bank E  │   │
│  │ ─────── │ │ ─────── │ │ ─────── │ │ ─────── │ │ ─────── │   │
│  │ Local   │ │ Local   │ │ Local   │ │ Local   │ │ Local   │   │
│  │ Data    │ │ Data    │ │ Data    │ │ Data    │ │ Data    │   │
│  │ ─────── │ │ ─────── │ │ ─────── │ │ ─────── │ │ ─────── │   │
│  │ Model   │ │ Model   │ │ Model   │ │ Model   │ │ Model   │   │
│  │ ─────── │ │ ─────── │ │ ─────── │ │ ─────── │ │ ─────── │   │
│  │ Privacy │ │ Privacy │ │ Privacy │ │ Privacy │ │ Privacy │   │
│  │ Module  │ │ Module  │ │ Module  │ │ Module  │ │ Module  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │           │         │
│       └───────────┴─────┬─────┴───────────┴───────────┘         │
│                         │ Encrypted Updates + ZK Proofs         │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Aggregation Server                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │   │
│  │  │ ZK Verify  │  │ Byzantine  │  │ HT2ML      │          │   │
│  │  │ Module     │→ │ Defense    │→ │ Aggregator │          │   │
│  │  └────────────┘  └────────────┘  └────────────┘          │   │
│  │        │               │               │                  │   │
│  │        ▼               ▼               ▼                  │   │
│  │  ┌─────────────────────────────────────────────┐         │   │
│  │  │           Global Model Update               │         │   │
│  │  └─────────────────────────────────────────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

COMPONENT SPECIFICATIONS:

1. Client Module:
   - Local training on bank's phishing data
   - Privacy module: DP noise, HE encryption
   - ZK proof generation for gradient bounds
   - Model: DistilBERT with LoRA adapters

2. Server Module:
   - ZK proof verification
   - Byzantine-robust aggregation (FoolsGold + norm check)
   - HT2ML-style aggregation (HE for linear, TEE for non-linear)
   - Reputation tracking per bank

3. Detection Model:
   - Base: DistilBERT (from Day 3)
   - LoRA adapters for efficient FL
   - Ensemble with XGBoost on engineered features

4. Privacy Mechanisms:
   - Level 1: Local DP (ε=1.0)
   - Level 2: Secure aggregation
   - Level 3: HT2ML hybrid (full privacy)

5. Security Mechanisms:
   - ZK gradient bound proofs
   - Byzantine-robust aggregation
   - Anomaly detection on updates
   - Reputation system

IMPLEMENTATION STRUCTURE:
```
fedphish/
├── fedphish/
│   ├── __init__.py
│   ├── client/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Local training logic
│   │   ├── privacy.py          # DP, HE encryption
│   │   ├── prover.py           # ZK proof generation
│   │   └── model.py            # DistilBERT + LoRA
│   ├── server/
│   │   ├── __init__.py
│   │   ├── aggregator.py       # HT2ML aggregation
│   │   ├── verifier.py         # ZK verification
│   │   ├── defense.py          # Byzantine defenses
│   │   └── reputation.py       # Bank reputation
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── features.py         # Feature extraction
│   │   ├── transformer.py      # DistilBERT model
│   │   ├── ensemble.py         # Model ensemble
│   │   └── explainer.py        # Explanations
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── dp.py               # Differential privacy
│   │   ├── he.py               # Homomorphic encryption
│   │   ├── tee.py              # TEE simulation
│   │   └── ht2ml.py            # Hybrid framework
│   ├── security/
│   │   ├── __init__.py
│   │   ├── zkp.py              # Zero-knowledge proofs
│   │   ├── attacks.py          # Attack implementations
│   │   └── defenses.py         # Defense mechanisms
│   └── utils/
│       ├── __init__.py
│       ├── data.py             # Data loading, partitioning
│       ├── metrics.py          # Evaluation metrics
│       └── visualization.py    # Plotting utilities
├── experiments/
│   ├── run_federated.py        # Main FL experiment
│   ├── run_benchmark.py        # Full benchmark
│   ├── run_attack_eval.py      # Security evaluation
│   └── run_privacy_eval.py     # Privacy evaluation
├── configs/
│   ├── base.yaml
│   ├── privacy/
│   ├── security/
│   └── banks/
├── api/
│   ├── main.py                 # FastAPI app
│   └── routes/
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── kubernetes/
├── tests/
├── docs/
├── README.md
├── REPRODUCE.md
└── requirements.txt
```

DELIVERABLES (Day 17):
- Core library implementation
- Client and server modules
- Privacy mechanisms (DP, HE, TEE)
- Security mechanisms (ZK, Byzantine defense)

DELIVERABLES (Day 18):
- Integration testing
- API and deployment
- Benchmark experiments
- Documentation

STRICT RULES:
- Modular design (swap components easily)
- Comprehensive documentation
- >80% test coverage
- Reproducible experiments
- Production-ready code quality

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the detailed implementation plan:
- Day 17 tasks (core implementation)
- Day 18 tasks (integration and deployment)
- Key interfaces between components
- Testing strategy

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 19: Demo and Visualization Dashboard

## 🎯 Session Setup (Copy This First)

```
You are an expert full-stack developer helping me create an interactive demo dashboard for FedPhish.

PROJECT CONTEXT:
- Name: FedPhish Demo Dashboard
- Purpose: Interactive demonstration of federated phishing detection system
- Tech Stack: React, FastAPI, WebSocket, D3.js, TailwindCSS
- Use case: PhD interview demos, conference presentations, research showcase

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Built demo dashboards in 30-day portfolio (Day 29)
- Building portfolio for PhD with Prof. Russello
- Need compelling visual demonstration of research

REQUIREMENTS:
- Dashboard sections:
  1. Federation Overview:
     - Map visualization of 5 banks (geographic distribution)
     - Real-time training progress per bank
     - Communication flow animation
     - Privacy mechanism status indicators
  2. Model Performance:
     - Global accuracy over rounds
     - Per-bank accuracy comparison
     - Confusion matrix (interactive)
     - ROC/PR curves with confidence intervals
  3. Privacy Metrics:
     - Privacy budget (ε) consumption over time
     - Information leakage estimation
     - Encryption status per communication
     - HT2ML mode indicator
  4. Security Status:
     - Malicious client detection alerts
     - ZK proof verification status
     - Reputation scores per bank
     - Attack simulation controls
  5. Live Demo Mode:
     - Analyze sample phishing email
     - Show feature extraction
     - Display model reasoning
     - Explain with attention visualization
- Interactive features:
  - Add/remove banks from federation
  - Inject malicious bank (attack simulation)
  - Toggle privacy levels (DP ε slider)
  - Speed up/slow down training animation
  - Pause and inspect any round
- Demo scenarios (pre-configured):
  1. "Happy path": All banks honest, fast convergence
  2. "Non-IID challenge": Banks have different phishing types
  3. "Attack scenario": One bank is malicious, defense kicks in
  4. "Privacy mode": HT2ML enabled, show encryption
  5. "Explanation demo": Analyze specific phishing email
- Technical requirements:
  - WebSocket for real-time updates
  - Responsive design (laptop + projector friendly)
  - Dark mode option
  - Export charts as PNG/SVG
  - Record demo as GIF
- API integration:
  - Connect to FedPhish backend (Day 17-18)
  - Mock mode for offline demos
- Unit tests for frontend components
- README.md with demo guide and scenario walkthroughs

STRICT RULES:
- Demo must run smoothly without backend (mock mode)
- All animations should be smooth (60 FPS)
- Accessible design (color blindness friendly)
- Mobile responsive (tablet minimum)
- Load time < 3 seconds

PRESENTATION INTEGRATION:
- Can be embedded in slides (iframe)
- Screenshot-friendly layouts
- Clear labeling for all visualizations
- Prof. Russello chat preparation ready

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Directory structure
- Component hierarchy
- API endpoints required
- Demo scenario specifications

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 20: Research Paper Draft Support

## 🎯 Session Setup (Copy This First)

```
You are an expert technical writer helping me prepare research paper materials for the FedPhish system.

PROJECT CONTEXT:
- Name: FedPhish Paper Materials
- Purpose: Generate all experimental results, figures, and tables for research paper
- Target venues: ACM CCS, USENIX Security, IEEE S&P, NDSS (security); NeurIPS, ICML (ML)
- THIS IS THE ACADEMIC OUTPUT FOR PhD APPLICATION

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management
- Published research in steganography and cryptographic methods
- Building portfolio for PhD with Prof. Russello
- Need publication-ready experimental results

REQUIREMENTS:

PAPER STRUCTURE TO SUPPORT:
1. Introduction
2. Background & Related Work
3. Threat Model & Problem Definition
4. FedPhish System Design
5. Privacy Analysis
6. Security Analysis
7. Experimental Evaluation
8. Discussion & Limitations
9. Conclusion

EXPERIMENTAL RESULTS NEEDED:

Table 1: Detection Performance Comparison
| Method | Accuracy | AUPRC | F1 | FPR@95%TPR |
|--------|----------|-------|-----|------------|
| Local (per-bank) | | | | |
| Centralized | | | | |
| FedAvg | | | | |
| FedPhish (Ours) | | | | |

Table 2: Privacy-Utility Trade-off
| Privacy Level | ε | Accuracy | AUPRC | Overhead |
|--------------|---|----------|-------|----------|
| No DP | ∞ | | | 1x |
| Light DP | 10 | | | |
| Moderate DP | 1 | | | |
| Strong DP | 0.1 | | | |
| HT2ML | N/A | | | |

Table 3: Robustness Against Attacks
| Attack | FedAvg | Krum | FoolsGold | FedPhish |
|--------|--------|------|-----------|----------|
| No Attack | | | | |
| Label Flip (20%) | | | | |
| Backdoor (20%) | | | | |
| Model Poison (20%) | | | | |

Table 4: Overhead Analysis
| Component | Time (ms) | Comm (KB) | Memory (MB) |
|-----------|-----------|-----------|-------------|
| Local Training | | | |
| ZK Proof Gen | | | |
| HE Encryption | | | |
| Aggregation | | | |
| Total Round | | | |

Figure 1: System Architecture Diagram
- Clean, publication-ready diagram
- SVG/PDF format

Figure 2: Convergence Comparison
- Global accuracy vs communication rounds
- FedPhish vs FedAvg vs Local vs Centralized

Figure 3: Non-IID Impact
- Accuracy vs Dirichlet α
- Shows FedPhish handles non-IID better

Figure 4: Privacy-Accuracy Pareto Curve
- X: Privacy (ε), Y: Accuracy
- Multiple methods compared

Figure 5: Attack Success Rate Over Rounds
- Shows defense detecting and mitigating attack

Figure 6: Per-Bank Fairness Analysis
- Violin plot of per-bank accuracy
- Shows equitable improvement

SUPPLEMENTARY MATERIALS:
- Appendix A: Hyperparameter settings
- Appendix B: Dataset statistics
- Appendix C: Additional experiments
- Appendix D: Proof sketches

AUTOMATION SCRIPTS:
- `generate_all_tables.py` - Run experiments, output LaTeX tables
- `generate_all_figures.py` - Run experiments, output PDF figures
- `run_full_evaluation.py` - Complete paper results (<24 hours)

STRICT RULES:
- Every number reproducible from scripts
- Statistical significance with error bars (5 runs)
- Publication-quality figures (10pt font minimum, PDF vector)
- LaTeX-compatible table output
- Clear figure captions

DELIVERABLES:
- All experiment scripts
- Pre-generated results (CSV)
- Pre-generated figures (PDF)
- LaTeX table code
- Paper outline with placeholder results

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the implementation plan:
- Experiment list with configurations
- Figure specifications
- Table specifications
- Script organization

DO NOT write any code yet. STOP and wait for my approval.
```

---

# Day 21: Portfolio Integration and PhD Application Package

## 🎯 Session Setup (Copy This First)

```
You are an expert research mentor helping me integrate all portfolio work into a compelling PhD application package for Prof. Russello.

PROJECT CONTEXT:
- Name: PhD Application Portfolio Package
- Purpose: Unified presentation of research capability for PhD application
- Target: Prof. Giovanni Russello, University of Auckland
- Combines: 30-day FL portfolio + 21-day FedPhish portfolio

MY BACKGROUND:
- 3+ years fraud detection with SAS Fraud Management in banking
- Published research in steganography and cryptographic methods
- Completed 30-day FL portfolio (SignGuard: signature-based FL defense)
- Completed 21-day FedPhish portfolio (federated phishing detection)
- Building comprehensive research capability demonstration

REQUIREMENTS:

PORTFOLIO STRUCTURE:
```
phd_application_portfolio/
├── README.md                        # Portfolio overview
├── RESEARCH_STATEMENT.md            # Research vision and goals
├── ALIGNMENT_WITH_RUSSELLO.md       # Specific alignment document
│
├── project_1_signguard/             # From 30-day portfolio
│   ├── README.md
│   ├── code/
│   ├── results/
│   └── paper_draft/
│
├── project_2_fedphish/              # From this portfolio
│   ├── README.md
│   ├── code/
│   ├── results/
│   └── paper_draft/
│
├── supplementary_projects/
│   ├── fraud_detection_eda/         # Day 1-7 from 30-day
│   ├── federated_learning_basics/   # Day 8-12 from 30-day
│   └── phishing_detection_basics/   # Day 1-5 from FedPhish
│
├── skills_demonstrated/
│   ├── privacy_preserving_ml.md     # HE, TEE, DP expertise
│   ├── federated_learning.md        # FL implementation skills
│   ├── security_research.md         # Attack/defense research
│   └── production_ml.md             # API, deployment skills
│
├── publications/
│   ├── existing_publications.md     # Steganography, crypto papers
│   └── planned_publications.md      # SignGuard, FedPhish targets
│
└── application_materials/
    ├── cv.pdf
    ├── research_proposal.pdf        # Refined proposal from earlier
    ├── transcript.pdf
    └── recommendation_letters/
```

ALIGNMENT DOCUMENT CONTENT:
1. Direct connection to each of Prof. Russello's papers:
   - HT2ML → FedPhish privacy mechanisms
   - MultiPhishGuard → FedPhish detection approach
   - FL + ZK-proofs → FedPhish verification system
   - Guard-GBDT → Privacy-preserving classifiers
   - Eyes on the Phish → Human-aligned explainability

2. Evidence of capability:
   - Working code for each alignment point
   - Experimental results demonstrating understanding
   - Clear path to novel contributions

3. Potential research directions:
   - Short-term: Extend HT2ML to phishing domain
   - Medium-term: Combine with MultiPhishGuard concepts
   - Long-term: New paradigm for financial sector FL

PORTFOLIO WEBSITE (GitHub Pages):
- Clean, professional landing page
- Project cards with key results
- Interactive demos embedded
- Links to code repositories
- Contact information

DEMO VIDEO (5 minutes):
1. Introduction (30s): Who I am, what I'm proposing
2. SignGuard demo (90s): FL defense system
3. FedPhish demo (90s): Federated phishing detection
4. Research vision (60s): PhD goals and timeline
5. Closing (30s): Why Prof. Russello's group

PREPARATION FOR CHAT:
- Key talking points from refined proposal
- Demo ready to screenshare
- Questions to ask (from refined proposal)
- Technical depth ready for any direction

STRICT RULES:
- Professional presentation quality
- All code clean and documented
- Results reproducible
- No exaggeration of capabilities
- Honest about limitations and learning areas

FINAL DELIVERABLES:
1. GitHub organization with all projects
2. Portfolio website
3. Demo video
4. Updated CV highlighting relevant experience
5. Refined research proposal (final version)
6. Email draft for scheduling chat

Please confirm you understand.

⚠️ FIRST STEP ONLY:
Create the integration plan:
- Repository structure
- Website design
- Video script outline
- Timeline for completion

DO NOT write any code yet. STOP and wait for my approval.
```

---

## 📋 Summary: 21-Day Federated Phishing Detection Portfolio

### Part 1: Phishing Detection Foundations (Days 1-5)
| Day | Project | Connection to Russello |
|-----|---------|----------------------|
| 1 | Phishing Feature Engineering | Foundation for all detection work |
| 2 | Classical ML Benchmark | Baseline for comparison |
| 3 | Transformer Detection | Toward MultiPhishGuard |
| 4 | Multi-Agent System | Direct MultiPhishGuard alignment |
| 5 | Detection API | Production readiness |

### Part 2: Privacy-Preserving Techniques (Days 6-8)
| Day | Project | Connection to Russello |
|-----|---------|----------------------|
| 6 | Homomorphic Encryption | HT2ML foundation |
| 7 | TEE Simulation | HT2ML complement |
| 8 | HT2ML Hybrid Implementation | Direct HT2ML alignment |

### Part 3: Verifiable Federated Learning (Days 9-11)
| Day | Project | Connection to Russello |
|-----|---------|----------------------|
| 9 | ZK Proofs Fundamentals | FL + ZK-proofs paper |
| 10 | Verifiable FL Protocol | FL + ZK-proofs implementation |
| 11 | Robust Verifiable FL | RQ2: Adversarial Robustness |

### Part 4: Privacy-Preserving Classifiers (Days 12-14)
| Day | Project | Connection to Russello |
|-----|---------|----------------------|
| 12 | Privacy-Preserving GBDT | Guard-GBDT alignment |
| 13 | Cross-Bank FL | RQ1: Privacy-Preserving Federation |
| 14 | Human-Aligned Explainability | Eyes on the Phish alignment |

### Part 5: Capstone Projects (Days 15-21)
| Day | Project | Connection to Russello |
|-----|---------|----------------------|
| 15 | Comprehensive Benchmark | Paper evaluation section |
| 16 | Adaptive Attack-Defense | RQ2 depth |
| 17-18 | FedPhish System (2 days) | Complete system implementation |
| 19 | Demo Dashboard | Interview preparation |
| 20 | Paper Materials | Publication preparation |
| 21 | Portfolio Integration | PhD application package |

---

## 🎯 Key Differentiators from 30-Day Portfolio

1. **Domain Focus**: Phishing detection (vs fraud detection)
2. **Privacy Depth**: HT2ML hybrid HE/TEE (vs secure aggregation basics)
3. **Verification**: ZK-proofs for verifiable FL (vs signature-based defense)
4. **Research Alignment**: Direct connection to Prof. Russello's papers
5. **Application Focus**: PhD application package as final deliverable

---

## 📊 Research Question Coverage

| Research Question | Primary Days | Supporting Days |
|-------------------|--------------|-----------------|
| RQ1: Privacy-Preserving Federation | 6-8, 13 | 9-12 |
| RQ2: Adversarial Robustness | 11, 16 | 15, 17-18 |
| RQ3: Human-Aligned Explainability | 14 | 4, 19 |

---

*This portfolio is designed to demonstrate comprehensive capability in federated phishing detection, directly aligned with Prof. Giovanni Russello's research interests at the University of Auckland.*
