# FedPhish Demo Dashboard

Interactive demonstration dashboard for the FedPhish federated phishing detection system. Perfect for PhD interviews, conference presentations, and research showcases.

## 🎯 Purpose

This dashboard provides a visually compelling, real-time visualization of:
- Federated learning training across multiple banks
- Privacy mechanisms (DP, HE, TEE)
- Byzantine-resilient aggregation
- Zero-knowledge proofs
- Model performance metrics

## 🚀 Quick Start

### Option 1: Development Mode

```bash
# Clone repository
cd fedphish-dashboard

# Start backend
cd backend
pip install -r requirements.txt
python -m app.main

# Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

Access at: http://localhost:5173

### Option 2: Docker Compose

```bash
cd fedphish-dashboard
docker-compose -f docker/docker-compose.yml up --build
```

Access at: http://localhost

## 📊 Demo Scenarios

### 1. Happy Path
- **Description**: Smooth federated learning convergence
- **Banks**: 5 honest banks
- **Duration**: 15 rounds
- **Key Visualizations**:
  - All banks train smoothly (green indicators)
  - Accuracy climbs to 95%
  - Continuous encrypted communication

**Script**: "In this scenario, 5 banks collaboratively train a phishing detector while maintaining strong privacy guarantees."

---

### 2. Non-IID Challenge
- **Description**: Heterogeneous data distributions
- **Banks**: 5 specialized banks
- **Duration**: 25 rounds
- **Key Visualizations**:
  - Different base accuracies per bank
  - Slower but eventual convergence
  - Model learns all phishing types

**Script**: "Real-world banks see different phishing attacks. Our system adapts to heterogeneous data through collaboration."

---

### 3. Attack Scenario
- **Description**: Byzantine attack with defense
- **Banks**: 4 honest, 1 malicious
- **Attack**: Sign flipping (starts round 5)
- **Defense**: FoolsGold + Krum
- **Key Visualizations**:
  - Malicious bank detected (red alert)
  - Reputation score drops
  - Accuracy recovers after defense

**Script**: "Watch what happens when a malicious bank tries to poison the model. Our defense detects and neutralizes the threat."

---

### 4. Privacy Mode
- **Description**: HT2ML privacy mechanisms
- **Banks**: 3
- **Levels**: Toggles between 1-2-3
- **Key Visualizations**:
  - Encryption indicators
  - Privacy budget tracking
  - Accuracy-privacy tradeoff

**Script**: "Explore our three privacy levels: DP, HE, and TEE. Privacy strengthens with minimal accuracy tradeoff."

---

### 5. Live Demo (Email Analysis)
- **Description**: Interactive phishing email analysis
- **Features**:
  - Real-time classification
  - Feature extraction visualization
  - Attention heatmap
  - Model explanations

## 🎮 Controls

### Dashboard Controls
- **Start**: Begin federated training
- **Pause**: Pause training at current round
- **Resume**: Continue training
- **Reset**: Return to round 0

### Interactive Features
- **Scenario Selector**: Switch between demo scenarios
- **Privacy Toggle**: Change privacy level (1-3)
- **Inject Attack**: Simulate malicious bank
- **Theme Toggle**: Light/Dark mode
- **Export Charts**: Save visualizations

## 🏗️ Architecture

```
fedphish-dashboard/
├── backend/              # FastAPI + WebSocket
│   └── app/
│       ├── main.py       # FastAPI application
│       ├── core/
│       │   └── simulator.py # FL simulation engine
│       └── websocket/
│           ├── manager.py   # Connection manager
│           └── handlers.py  # Message handlers
├── frontend/             # React + TypeScript
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── hooks/        # Custom hooks
│   │   └── layouts/      # Layout components
└── docker/               # Docker configuration
```

## 📱 Screenshots & Recording

### Key Screenshots to Capture

1. **Initial State**
   - All 5 banks idle
   - Round 0/20
   - Accuracy baseline

2. **Training in Progress**
   - Banks training (blue indicators)
   - Communication flow animation
   - Accuracy climbing

3. **Attack Detection**
   - Red alert banner
   - Malicious bank highlighted
   - Defense activated message

4. **Training Complete**
   - All banks complete (green)
   - Final accuracy: ~95%
   - Privacy budget consumed

### Recording as GIF

```bash
# Using peek (Linux)
peek /path/to/window

# Using OBS Studio
# 1. Select window region
# 2. Start recording
# 3. Run demo scenario
# 4. Stop and export as GIF
```

## 🎨 Features

- ✅ **Real-time WebSocket updates**
- ✅ **Responsive design** (tablet + projector friendly)
- ✅ **Dark mode support**
- ✅ **Smooth animations** (60 FPS)
- ✅ **Accessible design** (colorblind friendly)
- ✅ **Mock mode** (works without backend)
- ✅ **Export capabilities** (PNG/SVG)

## 🔧 Configuration

### Environment Variables

Backend (`.env`):
```bash
HOST=0.0.0.0
PORT=8001
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

Frontend:
```bash
VITE_WS_URL=ws://localhost:8001/ws/federation
```

## 📖 Documentation

- [Demo Guide](docs/DEMO_GUIDE.md) - How to give effective demos
- [Scenario Walkthroughs](docs/SCENARIO_WALKTHROUGHS.md) - Each scenario explained
- [Interview Prep](docs/INTERVIEW_PREP.md) - Prof. Russello chat preparation

## 🐛 Troubleshooting

### Issue: WebSocket connection fails
**Solution**: Ensure backend is running on port 8001

### Issue: Animations not smooth
**Solution**: Close other browser tabs, use Chrome/Edge

### Issue: Docker build fails
**Solution**: Ensure Docker daemon is running, try `docker system prune -a`

## 🎓 For PhD Interviews

### Key Talking Points

1. **Problem**: Banks can't share sensitive customer data
2. **Solution**: Federated learning with privacy + security
3. **Innovation**: HT2ML framework (HE + TEE hybrid)
4. **Robustness**: Byzantine-resilient aggregation
5. **Results**: 93-95% accuracy with strong privacy

### Demo Flow (15 minutes)

0:00-0:03: **Introduction** (Problem & Solution)
0:03-0:08: **Happy Path Demo** (Show smooth FL)
0:08-0:12: **Attack Scenario** (Show defenses)
0:12-0:15: **Q&A** (Deep dive on technical aspects)

## 📄 Citation

If you use this in your research:

```bibtex
@software{fedphish_dashboard,
  title={FedPhish Demo Dashboard},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fedphish-dashboard}
}
```

## 🤝 Contributing

This is a demo dashboard for the FedPhish system. For contributions to the main system, see: https://github.com/yourusername/fedphish

## 📧 Contact

For questions about this dashboard:
- Email: your@email.com
- Issues: https://github.com/yourusername/fedphish-dashboard/issues

---

**Built for PhD Portfolio - Privacy-Preserving Machine Learning**
