# 21-Day Federated Phishing Detection Portfolio

A comprehensive 21-day portfolio demonstrating advanced federated learning, privacy-preserving machine learning, and phishing detection systems.

## 📁 Repository Structure

This portfolio is organized into 5 thematic categories representing the progression from basic phishing detection to advanced federated learning systems.

```
21Days_Project/
├── foundations/              # Days 1-5: Basic phishing detection
├── privacy-techniques/       # Days 6-8: HE, TEE, Hybrid approaches
├── verifiable-fl/            # Days 9-11: Zero-knowledge proofs & verification
├── federated-classifiers/    # Days 12-14: Privacy-preserving classifiers
├── capstone/                 # Days 15-21: Complete FL system & PhD application
├── documentation/            # Project requirements & code reviews
├── shared-utilities/         # Common validation & utility functions
└── models/                   # Trained ML models (XGBoost, DistilBERT)
```

## 🗂️ Thematic Breakdown

### 1. Foundations (Days 1-5)
**Basic phishing detection with classical ML, transformers, and multi-agent systems**

- `phishing_email_analysis/` - Feature engineering pipeline for email analysis
- `day2_classical_ml_benchmark/` - XGBoost vs Random Forest benchmarking
- `day3_transformer_phishing/` - DistilBERT fine-tuning for phishing detection
- `multi_agent_phishing_detector/` - GLM-powered multi-agent analysis
- `unified-phishing-api/` - Production-ready FastAPI serving all models

**Key Learnings:**
- Feature extraction from emails (URLs, headers, content)
- Classical ML vs Deep Learning trade-offs
- Multi-agent orchestration for improved accuracy
- API design with caching, monitoring, and graceful degradation

---

### 2. Privacy-Techniques (Days 6-8)
**Homomorphic Encryption, Trusted Execution Environments, and Hybrid approaches**

- `he_ml_project/` - CKKS/BFV encryption for ML inference
- `tee_project/` - Intel SGX simulation for secure ML
- `ht2ml_phishing/` - Hybrid HE/TEE protocol for phishing detection

**Key Learnings:**
- Computing on encrypted data (without decryption)
- TEE security guarantees and limitations
- Hybrid protocols combining HE and TEE strengths
- Performance-privacy trade-offs

---

### 3. Verifiable FL (Days 9-11)
**Zero-knowledge proofs and verifiable federated learning**

- `zkp_fl_verification/` - ZK-SNARKs for FL model verification
- `verifiable_fl/` - Commitment schemes & verifiable aggregation
- `robust_verifiable_phishing_fl/` - Byzantine-robust verifiable FL

**Key Learnings:**
- Zero-knowledge proof circuits for ML
- Verifiable model updates without revealing gradients
- Byzantine resilience in verifiable FL
- libsnark/circom integration

---

### 4. Federated Classifiers (Days 12-14)
**Privacy-preserving tree-based models and cross-bank FL**

- `privacy_preserving_gbdt/` - GBDT training on encrypted data
- `cross_bank_federated_phishing/` - Vertical FL with PSI
- `human_aligned_explanation/` - XAI following cognitive principles

**Key Learnings:**
- Decision tree splitting on encrypted data
- Vertical federated learning (feature partitioning)
- Private Set Intersection for entity resolution
- Human-aligned explanations vs. technical explanations

---

### 5. Capstone (Days 15-21)
**Complete federated phishing detection system with benchmarking, attacks, and PhD application**

- `fedphish_benchmark/` - Comprehensive FL benchmark suite
- `adaptive_adversarial_fl/` - Coevolutionary attack/defense arms race
- `fedphish/` - Production federated phishing detection system
- `fedphish-dashboard/` - Real-time monitoring dashboard (React)
- `fedphish-paper/` - Research paper LaTeX source
- `phd-application-russello/` - PhD application to Prof. Giovanni Russello

**Key Learnings:**
- End-to-end FL system design
- Adaptive adversarial attacks and defenses
- Production monitoring and observability
- Academic writing and presentation

---

## 📊 Overall Statistics

- **Total Projects**: 21 technical projects + 1 portfolio package
- **Lines of Code**: ~50,000+ (Python, TypeScript, LaTeX, YAML)
- **Test Coverage**: 461 tests (privacy projects alone)
- **ML Models**: XGBoost, DistilBERT, Custom Neural Networks
- **Technologies**: PyTorch, TenSEAL, FastAPI, React, Docker, libsnark, circom

## 🚀 Quick Start

### Run the Unified API (Days 1-5 integration)
```bash
cd foundations/unified-phishing-api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Run HE/TEE Hybrid Demo (Days 6-8)
```bash
cd privacy-techniques/ht2ml_phishing
python examples/hybrid_inference_demo.py
```

### Run Cross-Bank FL Demo (Day 13)
```bash
cd federated-classifiers/cross_bank_federated_phishing
python experiments/run_demo.py
```

### Run FedPhish System (Days 17-18)
```bash
cd capstone/fedphish
python experiments/run_federated.py --config configs/base.yaml
```

### Launch Dashboard (Day 19)
```bash
cd capstone/fedphish-dashboard
npm install
npm start
```

## 📖 Documentation

- **Project Requirements**: `documentation/Federated_Phishing_Detection_Projects.md`
- **Code Reviews**: `documentation/CODE_REVIEW_*.md`
- **Master Summary**: `documentation/CODE_REVIEW_MASTER_SUMMARY.md`
- **Final Report**: `documentation/FINAL_CODE_REVIEW_SUMMARY.md`

## 🔬 Research Themes

1. **Privacy-Preserving ML**: How to detect phishing without seeing sensitive data
2. **Verifiable Learning**: How to trust FL updates from untrusted participants
3. **Adversarial Robustness**: How to defend against poisoned models
4. **Cross-Bank Collaboration**: How competing banks can share threat intelligence
5. **Human-AI Alignment**: How to explain ML decisions to security analysts

## 📅 Development Timeline

| Week | Days | Theme | Projects |
|------|------|-------|----------|
| 1 | 1-5 | Foundations | Feature engineering, classical ML, transformers, multi-agent, API |
| 2 | 6-8 | Privacy | HE, TEE, Hybrid HT2ML |
| 3 | 9-11 | Verification | ZK proofs, verifiable FL, Byzantine robustness |
| 4 | 12-14 | Classifiers | Privacy-preserving GBDT, cross-bank FL, explainability |
| 5 | 15-18 | System | Benchmark, attacks, FedPhish system |
| 6 | 19-21 | Presentation | Dashboard, paper, PhD application |

## 🎓 PhD Application

This portfolio supports a PhD application to Prof. Giovanni Russello (University of Auckland) researching:
- **Verifiable Federated Learning for Security-Critical Domains**
- **Privacy-Preserving Threat Detection**
- **Byzantine-Robust Aggregation Protocols**

## 👨‍💻 Author

Developed as a 21-day intensive portfolio project demonstrating:
- Federated learning expertise
- Privacy-preserving ML techniques
- Adversarial robustness
- Full-stack engineering
- Research communication

## 📝 License

Portfolio project - Educational and research purposes only.

---

**Note**: See individual project READMEs for detailed setup and usage instructions.
