# Unified Phishing Detection API

A production-ready REST API that serves multiple phishing detection models with ensemble capability, designed specifically for financial institutions.

## Overview

This API unifies four days of phishing detection research into a single, production-grade service:

- **Day 1**: Feature engineering pipeline (60+ features)
- **Day 2**: Classical ML (XGBoost) with <200ms latency
- **Day 3**: Transformer model (DistilBERT) with <1s latency
- **Day 4**: Multi-agent system with GLM-powered analysis

## Features

- ✅ **Multiple Model Support**: XGBoost, Transformer, Multi-Agent, Ensemble
- ✅ **Graceful Degradation**: API runs with subset of models if some unavailable
- ✅ **Redis Caching**: URL reputation and prediction caching
- ✅ **Batch Processing**: Analyze up to 100 emails in parallel
- ✅ **Comprehensive Monitoring**: Prometheus metrics + Grafana dashboards
- ✅ **Docker Deployment**: Multi-stage builds with docker-compose
- ✅ **Structured Logging**: JSON logs with no PII
- ✅ **Financial Focus**: Banking-specific features and threat detection

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- GLM API key (for multi-agent system)

### 1. Clone and Setup

```bash
cd /home/ubuntu/21Days_Project/unified-phishing-api

# Copy environment configuration
cp .env.example .env

# Edit .env and set your GLM_API_KEY
nano .env
```

### 2. Train Models

Follow the [Model Training Guide](docs/MODEL_TRAINING_GUIDE.md) to train:

- Day 2: XGBoost model → `/home/ubuntu/21Days_Project/models/day2_xgboost/`
- Day 3: DistilBERT model → `/home/ubuntu/21Days_Project/models/day3_distilbert/`

### 3. Start Services

```bash
# Start all services (API, Redis, Prometheus, Grafana)
docker-compose up -d

# Check service status
docker-compose ps

# View API logs
docker-compose logs -f api
```

### 4. Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## API Endpoints

### Email Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/email" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_email": "Received: from mail.example.com...",
    "model_type": "ensemble",
    "use_cache": true
  }'
```

**Response**:
```json
{
  "email_id": "req_abc123",
  "verdict": "PHISHING",
  "confidence": 0.95,
  "risk_score": 92,
  "risk_level": "critical",
  "model_used": "ensemble",
  "analysis": {
    "url_risk": {...},
    "content_risk": {...},
    "header_risk": {...},
    "financial_indicators": {...}
  },
  "explanation": "This email contains multiple indicators...",
  "processing_time_ms": 295.3,
  "cache_hit": false,
  "timestamp": "2026-01-29T12:34:56Z"
}
```

### URL Quick Check

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://chase-secure-portal.xyz/login",
    "context": {
      "sender": "security@chase-secure-portal.xyz",
      "subject": "Account Verification Required"
    }
  }'
```

### Batch Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "emails": [
      {"raw_email": "..."},
      {"parsed_email": {...}}
    ],
    "model_type": "ensemble",
    "parallel": true
  }'
```

### List Models

```bash
curl "http://localhost:8000/api/v1/models"
```

### Submit Feedback

```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "email_id": "abc123",
    "predicted_verdict": "PHISHING",
    "actual_verdict": "LEGITIMATE",
    "model_used": "xgboost",
    "feedback_type": "false_positive"
  }'
```

## Performance SLAs

| Model | P95 Latency | Target Usage |
|-------|-------------|--------------|
| XGBoost | <200ms | High-volume, real-time |
| Transformer | <1s | Standard analysis |
| Multi-Agent | <5s | Detailed investigation |
| Ensemble | <1s | Best accuracy |

## Architecture

```
┌─────────────────┐
│   Client App    │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  FastAPI App    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────┐  ┌─────┐
│Redis │  │ ML  │
│Cache │  │Models│
└──────┘  └─────┘
                │
           ┌────┴────┐
           ▼         ▼         ▼
        ┌─────┐  ┌───────┐  ┌──────────┐
        │XGBoost││DistilBERT││Multi-Agent│
        └─────┘  └───────┘  └──────────┘
```

## Operating Modes

The API automatically adapts based on available models:

- **FULL**: All models available (ideal)
- **DEGRADED**: XGBoost + Transformer (no multi-agent)
- **MINIMAL**: XGBoost only (fallback)
- **UNAVAILABLE**: No models (API won't start)

Check current mode:
```bash
curl http://localhost:8000/health | jq .models
```

## Monitoring

### Prometheus Metrics

Key metrics to monitor:

- `http_request_duration_seconds` - API latency
- `model_prediction_duration_seconds` - Model inference time
- `model_predictions_total` - Prediction volume by verdict
- `cache_hits_total` / `cache_misses_total` - Cache effectiveness
- `model_errors_total` - Model error rate

### Grafana Dashboards

Pre-configured dashboards include:

- Request rate and latency
- Model performance comparison
- Cache hit rates
- Error rates
- Phishing detection trends

Access at: http://localhost:3000

## Configuration

Key environment variables (see `.env.example`):

```bash
# API Settings
ENVIRONMENT=production
DEBUG=false
WORKERS=4

# Model Paths
MODELS_BASE_PATH=/home/ubuntu/21Days_Project/models

# Redis Cache
REDIS_HOST=redis
REDIS_PORT=6379

# GLM API (Multi-Agent)
GLM_API_KEY=your_api_key_here
GLM_MODEL=glm-4-flash

# Ensemble Weights
ENSEMBLE_XGBOOST_WEIGHT=0.4
ENSEMBLE_TRANSFORMER_WEIGHT=0.4
ENSEMBLE_MULTI_AGENT_WEIGHT=0.2
```

## Development

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GLM_API_KEY="your_key"

# Start Redis (or use docker-compose for just Redis)
docker-compose up -d redis

# Run API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Unit tests
pytest tests/

# With coverage
pytest --cov=app tests/

# Integration tests
pytest tests/ --integration
```

### Load Testing

```bash
# Run Locust load tests (target: 100 RPS)
cd load_tests
locust -f locustfile.py --host http://localhost:8000

# Access web UI at http://localhost:8089
```

## Deployment

### Production Deployment with Docker

```bash
# Build production image
docker build -t phishing-api:latest .

# Run with docker-compose
docker-compose -f docker-compose.yml up -d

# Check health
curl http://localhost:8000/health
```

### Kubernetes Deployment

Create manifests for:
- Deployment (API pods)
- Service (LoadBalancer/NodePort)
- ConfigMap (environment variables)
- Secret (GLM API key)

See `k8s/` directory (not included in this version).

## Troubleshooting

### API won't start - "No models available"

**Cause**: Models not trained or paths incorrect.

**Solution**:
1. Train models following [Model Training Guide](docs/MODEL_TRAINING_GUIDE.md)
2. Verify files exist: `ls /home/ubuntu/21Days_Project/models/`
3. Check `MODELS_BASE_PATH` in `.env`

### High latency on predictions

**Cause**: Models not optimized or cache disabled.

**Solution**:
1. Check `use_cache=true` in requests
2. Verify Redis is running: `docker-compose ps redis`
3. Check model performance: `curl http://localhost:8000/api/v1/models`

### Multi-agent errors

**Cause**: GLM API key invalid or network issues.

**Solution**:
1. Verify `GLM_API_KEY` in `.env`
2. Test API key:
   ```bash
   curl -X POST "https://open.bigmodel.cn/api/paas/v4/chat/completions" \
     -H "Authorization: Bearer $GLM_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"glm-4-flash","messages":[{"role":"user","content":"test"}]}'
   ```
3. Check logs: `docker-compose logs api | grep GLM`

## Project Structure

```
unified-phishing-api/
├── app/
│   ├── api/v1/routes/     # API endpoints
│   ├── models/            # Model wrappers
│   ├── schemas/           # Pydantic models
│   ├── services/          # Business logic
│   ├── middleware/        # Logging, metrics
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── load_tests/            # Locust tests
├── prometheus/            # Prometheus configs
├── grafana/               # Grafana dashboards
├── docs/                  # Documentation
├── Dockerfile             # Multi-stage build
├── docker-compose.yml     # Orchestration
├── requirements.txt       # Python dependencies
└── .env.example           # Environment template
```

## Contributing

This is part of a 21-day portfolio project. For questions:

1. Check individual project READMEs (Day 1-4)
2. Review Model Training Guide
3. Check API documentation at `/docs`

## License

This is a portfolio project for research application purposes.

## Acknowledgments

- **Day 1**: Feature Engineering Pipeline
- **Day 2**: Classical ML Benchmark
- **Day 3**: Transformer-Based Detection
- **Day 4**: Multi-Agent System
- **GLM**: Zhipu AI (https://open.bigmodel.cn/)
