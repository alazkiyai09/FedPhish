# Quick Start Guide - Phase 2 Implementation

This guide helps you test the **Phase 2 features** that are currently working.

## Prerequisites

```bash
cd /home/ubuntu/21Days_Project/unified-phishing-api

# Create .env file
cp .env.example .env

# Edit .env (minimal configuration needed for Phase 2)
# Only REDIS and basic settings are required
```

## Start Services

### Option 1: Full Stack (Recommended)

```bash
# Start API + Redis + Prometheus + Grafana
docker-compose up -d

# Check services are running
docker-compose ps

# View API logs
docker-compose logs -f api
```

### Option 2: Local Development

```bash
# Start Redis only
docker-compose up -d redis

# Install dependencies
pip install -r requirements.txt

# Run API locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Test URL Analysis (Fully Functional)

### 1. Phishing URL Detection

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://chase-secure-portal.xyz/login",
    "context": {
      "sender": "security@chase-secure-portal.xyz",
      "subject": "Account Verification Required"
    }
  }' | jq
```

**Expected Output**:
```json
{
  "verdict": "PHISHING",
  "confidence": 0.75,
  "risk_score": 75,
  "risk_level": "high",
  "model_used": "url_heuristic",
  "explanation": "Domain appears to impersonate chase.com. Uses suspicious top-level domain."
}
```

### 2. Legitimate URL

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://chase.com"
  }' | jq
```

**Expected Output**:
```json
{
  "verdict": "LEGITIMATE",
  "confidence": 0.95,
  "risk_score": 5,
  "risk_level": "low"
}
```

### 3. Typosquatting Detection

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://wellfarg0.com/verify"
  }' | jq
```

**Expected Output**:
```json
{
  "verdict": "PHISHING",
  "risk_score": 80,
  "explanation": "Possible typosquatting of wellsfargo.com"
}
```

### 4. IP Address URL

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://192.168.1.1/login"
  }' | jq
```

**Expected Output**:
```json
{
  "verdict": "SUSPICIOUS",
  "risk_score": 65,
  "explanation": "URL contains IP address instead of domain name"
}
```

## Test Batch Processing

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "emails": [
      {"url": "http://chase-secure-portal.xyz/login"},
      {"url": "https://chase.com"},
      {"url": "http://wellfarg0.com/verify"}
    ],
    "parallel": true
  }' | jq
```

**Expected Output**:
```json
{
  "batch_id": "batch_abc123",
  "results": [...],
  "summary": {
    "total_emails": 3,
    "phishing_count": 2,
    "legitimate_count": 1,
    "suspicious_count": 0,
    "avg_risk_score": 53.33
  },
  "successful_count": 3,
  "failed_count": 0
}
```

## Test Health & Status

```bash
# Health check
curl http://localhost:8000/health | jq

# List models (will show none available)
curl http://localhost:8000/api/v1/models | jq

# Metrics
curl http://localhost:8000/metrics | head -20
```

## Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run URL analyzer tests
pytest tests/test_services/test_url_analyzer.py -v

# Run API tests
pytest tests/test_api/test_analyze.py -v

# Run all tests
pytest tests/ -v

# With coverage
pytest --cov=app tests/
```

## View Monitoring

### Prometheus
```bash
# Access at http://localhost:9090
# Query examples:
#   rate(http_requests_total[5m])
#   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Grafana
```bash
# Access at http://localhost:3000
# Login: admin/admin
# Pre-configured dashboard: "Phishing Detection API - Overview"
```

## Test Caching

```bash
# First request (cache miss)
curl -X POST "http://localhost:8000/api/v1/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://chase.com"}' | jq '.cache_hit'
# Output: false

# Second request (cache hit)
curl -X POST "http://localhost:8000/api/v1/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://chase.com"}' | jq '.cache_hit'
# Output: true

# View cache metrics
curl http://localhost:8000/metrics | grep cache
```

## Load Testing (Optional)

```bash
# Install Locust
pip install locust

# Run load tests
cd load_tests
locust -f locustfile.py --host http://localhost:8000

# Access web UI at http://localhost:8089
# Target: 100 users, spawn rate: 10 users/sec
```

## Troubleshooting

### API won't start

```bash
# Check if Redis is running
docker-compose ps redis

# Check API logs
docker-compose logs api

# Try local development mode
uvicorn app.main:app --reload
```

### Import errors

```bash
# Ensure Day 1 path is correct
echo $DAY1_PIPELINE_PATH
# Should be: /home/ubuntu/21Days_Project/phishing_email_analysis

# Add to .env if needed
echo "DAY1_PIPELINE_PATH=/home/ubuntu/21Days_Project/phishing_email_analysis" >> .env
```

### Feature extraction not available

**Expected behavior** - Email analysis will return 503 error:
```json
{
  "detail": "Feature extraction service not available. Ensure Day 1 pipeline is installed."
}
```

**Solution**: Ensure Day 1 project exists:
```bash
ls /home/ubuntu/21Days_Project/phishing_email_analysis/src/transformers_backup/phishing_pipeline.py
```

## What's Next?

1. **Train models** following `docs/MODEL_TRAINING_GUIDE.md`
2. **Test email analysis** once models are trained (Phase 3)
3. **Deploy to production** using docker-compose

## Summary of Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| URL Analysis | ✅ Working | Full heuristic detection |
| Batch URL Analysis | ✅ Working | Parallel processing |
| URL Caching | ✅ Working | 1-hour TTL |
| Feature Extraction | ⚠️ Partial | Requires Day 1 pipeline |
| Email Analysis | ⏳ Phase 3 | ML models needed |
| Model Listing | ⏳ Phase 3 | No models yet |
| Health Checks | ✅ Working | Full system status |
| Metrics | ✅ Working | Prometheus + Grafana |

---

**For detailed documentation, see**:
- `README.md` - Full API documentation
- `docs/MODEL_TRAINING_GUIDE.md` - How to train models
- `PHASE2_COMPLETED.md` - Phase 2 implementation details
