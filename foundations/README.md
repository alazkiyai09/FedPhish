# Foundations: Phishing Detection (Days 1-5)

**Theme**: Basic phishing detection using classical ML, deep learning, and multi-agent systems.

## 📁 Projects

| Day | Project | Description | Tech Stack |
|-----|---------|-------------|------------|
| 1 | `phishing_email_analysis/` | Feature engineering pipeline for email data | NumPy, Pandas, scikit-learn |
| 2 | `day2_classical_ml_benchmark/` | XGBoost vs Random Forest comparison | XGBoost, scikit-learn |
| 3 | `day3_transformer_phishing/` | DistilBERT fine-tuning for text classification | PyTorch, Transformers, HuggingFace |
| 4 | `multi_agent_phishing_detector/` | GLM-powered multi-agent analysis | AsyncIO, OpenAI-compatible API |
| 5 | `unified-phishing-api/` | Production FastAPI serving all models | FastAPI, Redis, Docker, Prometheus |

## 🎯 Learning Objectives

- **Feature Engineering**: Extract meaningful features from raw emails
- **Model Selection**: Compare classical ML vs deep learning approaches
- **Multi-Agent Systems**: Coordinate specialized agents for better analysis
- **Production Systems**: Build scalable, monitored APIs

## 🔗 Project Dependencies

```
Day 1 (Feature Engineering)
    ↓
Day 2 (Classical ML) ← uses features from Day 1
    ↓
Day 3 (Transformer) ← alternative to Day 2
    ↓
Day 4 (Multi-Agent) ← combines all approaches
    ↓
Day 5 (Unified API) ← integrates Days 1-4
```

## 🚀 Quick Start

### Day 1: Feature Extraction
```bash
cd phishing_email_analysis
python -m src.feature_extractors.url_features
```

### Day 2: Classical ML Benchmark
```bash
cd day2_classical_ml_benchmark
python benchmark.py --model xgboost --epochs 100
```

### Day 3: Transformer Training
```bash
cd day3_transformer_phishing
python -m src.training.trainer --config configs/base.yaml
```

### Day 4: Multi-Agent Analysis
```bash
cd multi_agent_phishing_detector
python -m src.main --email examples/phishing.eml
```

### Day 5: Unified API
```bash
cd unified-phishing-api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## 📊 Performance Summary

| Model | Accuracy | Latency | Strengths |
|-------|----------|---------|-----------|
| XGBoost | 94.2% | <200ms | Fast, interpretable |
| DistilBERT | 96.8% | <1s | Best accuracy, understands context |
| Multi-Agent | 95.5% | 3-5s | Comprehensive analysis |

## 🔬 Key Innovations

1. **60+ Hand-crafted Features** (Day 1)
   - URL-based features (IP addresses, TLDs, subdomains)
   - Header features (SPF, DKIM, Reply-To mismatches)
   - Content features (urgency, financial terms, threats)

2. **Benchmarking Framework** (Day 2)
   - Unified interface for classical ML models
   - Cross-validation with stratified splits
   - Comprehensive metrics (accuracy, precision, recall, F1, AUC)

3. **Domain-Adapted Transformer** (Day 3)
   - DistilBERT fine-tuned on phishing emails
   - Special tokens for URLs, email addresses, headers
   - Attention visualization for interpretability

4. **Multi-Agent Orchestration** (Day 4)
   - 4 specialized agents: URL, Content, Header, Visual
   - Weighted voting with confidence-based aggregation
   - Graceful degradation on agent failure

5. **Production API** (Day 5)
   - Ensemble mode combining all models
   - Redis caching for URL reputation
   - Prometheus metrics and Grafana dashboards
   - Request size validation and rate limiting

## 📈 Model Performance

```
Unified API Endpoints:
├── POST /api/v1/analyze/url      → <100ms (heuristic only)
├── POST /api/v1/analyze/email    → <200ms (XGBoost)
├── POST /api/v1/analyze/email    → <1s (Transformer)
├── POST /api/v1/analyze/email    → 3-5s (Multi-Agent)
└── POST /api/v1/analyze/batch    → Parallel processing
```

## 🔗 Next Steps

After completing Foundations, advance to:
- **Privacy-Techniques** (Days 6-8): Learn HE and TEE
- **Verifiable-FL** (Days 9-11): Add cryptographic verification

## 📝 Notes

- All projects use the same dataset format for easy integration
- Models are pre-trained and stored in `/models/` at root
- Day 5 API can serve any model from Days 1-4

---

**Theme Progression**: Foundations → Privacy-Techniques → Verifiable-FL → Federated-Classifiers → Capstone
