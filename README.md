# Federated Phishing Detection Portfolio

<div align="center">

A comprehensive portfolio demonstrating advanced **federated learning**, **privacy-preserving machine learning**, and **phishing detection systems**.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flwr-FF6B6B?style=flat)](https://flower.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker)](https://www.docker.com/)

[21 Projects](#-projects) • [Quick Start](#-quick-start) • [Tech Stack](#-tech-stack)

</div>

---

## Overview

This portfolio demonstrates expertise in building **privacy-preserving federated learning systems** for security-critical applications, featuring:

- **Federated Learning** with Byzantine-resilient aggregation
- **Privacy Techniques** (Homomorphic Encryption, Trusted Execution Environments)
- **Verifiable FL** using Zero-Knowledge Proofs
- **Adversarial Robustness** against model poisoning attacks
- **Production Infrastructure** (APIs, dashboards, Docker/K8s)

> **Note**: This is an educational portfolio demonstrating implementations of established research techniques in federated learning security and privacy-preserving ML.

## Portfolio Stats

| Metric | Count |
|--------|-------|
| **Projects** | 21 across 5 categories |
| **Python Files** | 643 production files |
| **Test Files** | 111 test suites |
| **Test Coverage** | 530+ tests |
| **Code Lines** | 50,000+ production lines |
| **ML Models** | XGBoost, DistilBERT, Custom Neural Networks |
| **Technologies** | PyTorch, TenSEAL, FastAPI, React, Docker, libsnark |

---

## 📁 Repository Structure

```
21Days_Project/
├── foundations/              # Days 1-5: Basic phishing detection
├── privacy-techniques/       # Days 6-8: HE, TEE, Hybrid approaches
├── verifiable-fl/            # Days 9-11: Zero-knowledge proofs & verification
├── federated-classifiers/    # Days 12-14: Privacy-preserving classifiers
├── capstone/                 # Days 15-21: Complete FL system & benchmarking
├── shared_utils/             # Common security, logging, validation utilities
└── models/                   # Trained ML models
```

---

## 🗂️ Projects

### 1. Foundations (Days 1-5)
**Basic phishing detection with classical ML, transformers, and multi-agent systems**

| Project | Description | Key Tech |
|---------|-------------|----------|
| `phishing_email_analysis/` | 70-feature extraction pipeline | NLTK, BeautifulSoup |
| `day2_classical_ml_benchmark/` | 7-classical ML benchmark | XGBoost, LightGBM, SVM |
| `day3_transformer_phishing/` | DistilBERT fine-tuning | Transformers, PEFT, LoRA |
| `multi_agent_phishing_detector/` | GLM-powered multi-agent system | LangChain, OpenAI |
| `unified-phishing-api/` | Production FastAPI ensemble | FastAPI, Redis, Prometheus |

### 2. Privacy-Techniques (Days 6-8)
**Homomorphic Encryption, Trusted Execution Environments, and Hybrid approaches**

| Project | Description | Key Tech |
|---------|-------------|----------|
| `he_ml_project/` | CKKS/BFV encrypted ML inference | TenSEAL |
| `tee_project/` | Intel SGX secure enclave simulation | PyCryptodome |
| `ht2ml_phishing/` | Hybrid HE+TEE protocol | TenSEAL + SGX |

### 3. Verifiable FL (Days 9-11)
**Zero-knowledge proofs and verifiable federated learning**

| Project | Description | Key Tech |
|---------|-------------|----------|
| `zkp_fl_verification/` | ZK-SNARK model verification | py-snark, circom |
| `verifiable_fl/` | Commitment schemes & aggregation | petlib, blspy |
| `robust_verifiable_phishing_fl/` | Byzantine-resilient verifiable FL | Krum, Multi-Krum |

### 4. Federated Classifiers (Days 12-14)
**Privacy-preserving tree-based models and cross-bank FL**

| Project | Description | Key Tech |
|---------|-------------|----------|
| `privacy_preserving_gbdt/` | GBDT on encrypted data | TenSEAL |
| `cross_bank_federated_phishing/` | Vertical FL with PSI | ECDH, scikit-learn |
| `human_aligned_explanation/` | Cognitive XAI for phishing | LIME, SHAP, Captum |

### 5. Capstone (Days 15-21)
**Complete federated phishing detection system with benchmarking and attacks**

| Project | Description | Key Tech |
|---------|-------------|----------|
| `fedphish_benchmark/` | Comprehensive FL benchmark | Flower, PyTorch |
| `adaptive_adversarial_fl/` | Coevolutionary attack/defense | CleverHans |
| `fedphish/` | Production FL system | Flower, gRPC, Redis |
| `fedphish-dashboard/` | Real-time monitoring dashboard | React, WebSocket |
| `fedphish-paper/` | Research paper LaTeX source | LaTeX, BibTeX |

---

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

### Launch Dashboard (Day 18)
```bash
cd capstone/fedphish-dashboard
npm install
npm start
```

---

## 🔬 Research Themes

1. **Privacy-Preserving ML**: Detect phishing without seeing sensitive data
2. **Verifiable Learning**: Trust FL updates from untrusted participants
3. **Adversarial Robustness**: Defend against poisoned model updates
4. **Cross-Bank Collaboration**: Competing banks sharing threat intelligence
5. **Human-AI Alignment**: Explain ML decisions to security analysts

---

## 📅 Development Timeline

| Week | Days | Theme | Projects |
|------|------|-------|----------|
| 1 | 1-5 | Foundations | Feature engineering, ML, transformers, agents, API |
| 2 | 6-8 | Privacy | HE, TEE, Hybrid HT2ML |
| 3 | 9-11 | Verification | ZK proofs, verifiable FL, Byzantine robustness |
| 4 | 12-14 | Classifiers | Privacy-preserving GBDT, cross-bank FL, XAI |
| 5 | 15-18 | System | Benchmark, attacks, FedPhish system |
| 6 | 19-21 | Presentation | Dashboard, paper, documentation |

---

## 🛡️ Security

This portfolio implements comprehensive security measures:

- **Safe deserialization** - All `torch.load()` use `weights_only=True`
- **No unsafe pickle** - Replaced with numpy/json serialization
- **Input validation** - Shared validation utilities across all projects
- **Structured logging** - JSON logging for security monitoring
- **Error handling** - Specific exception handling (no bare except)

---

## 👨‍💻 Author

<div align="center">

**Hi, I'm Ahmad Whafa Azka Al Azkiyai**

**Fraud Detection & AI Security Specialist**

Federated Learning Security | Adversarial ML | Privacy-Preserving AI

---

[![Website](https://img.shields.io/badge/Website-Visit_-green.svg)](https://alazkiyai09.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-alazkiyai09-black.svg)](https://github.com/alazkiyai09)
[![Email](https://img.shields.io/badge/Email-Get_in_Touch-red.svg)](mailto:azka.alazkiyai@outlook.com)

**📍 Location: Jakarta, Indonesia (Open to Remote)**
**💼 Open to: Full-time, Contract, Consulting, Research Collaboration**

---

**Domain Expertise:**
- 🏦 **3+ years** Banking Fraud Detection (SAS Fraud Management, Real-time monitoring)
- 🔐 **1+ years** Federated Learning Security (Byzantine-resilient FL, SignGuard)
- 🔒 **2+ years** Steganography & Information Hiding (Published research)

**More Projects:** [Production AI Portfolio](https://github.com/alazkiyai09/production-ai-portfolio)

</div>

---

## 📝 License

Portfolio project - Educational and research purposes only.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Ahmad Whafa Azka Al Azkiyai](https://alazkiyai09.github.io/)

</div>
