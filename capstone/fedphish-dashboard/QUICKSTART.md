# FedPhish Demo Dashboard - Quick Start

## 5-Minute Setup

```bash
# 1. Run setup script
cd /path/to/fedphish-dashboard
./scripts/setup.sh

# 2. Start services (option 1: manual)
cd backend
source venv/bin/activate
python -m app.main

# New terminal
cd frontend
npm run dev

# OR (option 2: automated)
./scripts/dev.sh

# 3. Open browser
open http://localhost:5173
```

## Your First Demo

1. **Select Scenario**: Choose "Happy Path" from dropdown
2. **Click Start**: Watch training begin
3. **Observe**: Banks train (blue → green)
4. **See Results**: Accuracy → 95%

## Demo Scenarios

| Scenario | Time | Key Feature |
|---------|------|-------------|
| Happy Path | 2 min | Smooth convergence |
| Non-IID Challenge | 3 min | Heterogeneous data |
| Attack Scenario | 4 min | Byzantine defense |
| Privacy Mode | 3 min | HT2ML levels |
| Live Demo | 5 min | Email analysis |

## Common Commands

```bash
# Install dependencies
./scripts/setup.sh

# Development mode
./scripts/dev.sh

# Production build
cd frontend && npm run build

# Docker deployment
cd docker && docker-compose up --build

# Run tests (coming soon)
cd backend && pytest
cd frontend && npm test
```

## Troubleshooting

**Backend won't start**
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

**Frontend won't start**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**WebSocket connection fails**
- Check backend is running on port 8001
- Check frontend VITE_WS_URL in .env
- Try http://localhost:8001/health

## Need Help?

- Full README: [README.md](README.md)
- Scenario Walkthroughs: [docs/SCENARIO_WALKTHROUGHS.md](docs/SCENARIO_WALKTHROUGHS.md)
- Interview Prep: [docs/INTERVIEW_PREP.md](docs/INTERVIEW_PREP.md)

## System Requirements

- Python 3.10+
- Node.js 20+
- 4GB RAM minimum
- Modern browser (Chrome/Edge/Firefox)

---

**Ready to demo! 🚀**
